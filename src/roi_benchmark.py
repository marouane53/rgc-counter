from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.io_ome import load_any_image
from src.roi_data import RoiRecord, crop_2d_or_yxc, filter_roi_manifest_by_split, iter_roi_records, load_roi_manifest
from src.run_service import RuntimeOptions, build_runtime, run_array
from src.validation import (
    build_benchmark_quality_table,
    load_manual_points,
    match_points,
    point_matching_metrics,
)


SENSITIVITY_TOLERANCES_PX = (6.0, 8.0, 10.0)
PRIMARY_TOLERANCE_PX = 8.0
BENCHMARK_MIN_ROIS = 20


@dataclass(frozen=True)
class RoiBenchmarkConfig:
    config_id: str
    backend: str | None = None
    segmentation_preset: str | None = None
    diameter: float | None = None
    min_size: int | None = None
    max_size: int | None = None
    apply_clahe: bool = False
    modality: str = "flatmount"
    modality_channel_index: int | None = 0
    segmenter_config: dict[str, Any] | None = None
    object_filters: dict[str, Any] | None = None
    use_gpu: bool = False
    notes: str = ""


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, divider]
    for row in frame.to_dict("records"):
        values: list[str] = []
        for column in columns:
            value = row.get(column)
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def predicted_points_from_context(ctx: Any) -> np.ndarray:
    if getattr(ctx, "object_table", None) is None or ctx.object_table.empty:
        return np.empty((0, 2), dtype=float)
    frame = ctx.object_table.copy()
    if "kept" in frame.columns:
        frame = frame.loc[frame["kept"].fillna(True).astype(bool)]
    if frame.empty:
        return np.empty((0, 2), dtype=float)
    return frame[["centroid_y_px", "centroid_x_px"]].to_numpy(dtype=float)


def runtime_options_from_config(config: RoiBenchmarkConfig) -> RuntimeOptions:
    return RuntimeOptions(
        backend=config.backend,
        segmentation_preset=config.segmentation_preset,
        diameter=config.diameter,
        min_size=config.min_size,
        max_size=config.max_size,
        apply_clahe=config.apply_clahe,
        modality=config.modality,
        modality_channel_index=config.modality_channel_index,
        segmenter_config=config.segmenter_config,
        object_filters=config.object_filters,
        focus_mode="none",
        save_debug=False,
        write_html_report=False,
        write_object_table=False,
        write_provenance=False,
        use_gpu=config.use_gpu,
    )


def run_single_roi_case(
    record: RoiRecord,
    *,
    config: RoiBenchmarkConfig,
    runtime_builder: Callable[..., Any] = build_runtime,
    runtime_runner: Callable[..., Any] = run_array,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    image, meta = load_any_image(str(record.image_path))
    crop = crop_2d_or_yxc(image, x0=record.x0, y0=record.y0, width=record.width, height=record.height)
    manual_points = load_manual_points(record.manual_points_path) if record.manual_points_path else np.empty((0, 2), dtype=float)

    runtime = runtime_builder(runtime_options_from_config(config))
    started = time.perf_counter()
    ctx = runtime_runner(
        runtime,
        image=crop,
        source_path=f"{record.image_path}#roi:{record.roi_id}",
        meta=meta,
    )
    runtime_seconds = float(time.perf_counter() - started)
    predicted_points = predicted_points_from_context(ctx)

    per_tolerance_rows: list[dict[str, Any]] = []
    for tolerance_px in SENSITIVITY_TOLERANCES_PX:
        metrics = point_matching_metrics(manual_points, predicted_points, tolerance_px=float(tolerance_px))
        per_tolerance_rows.append(
            {
                "config_id": config.config_id,
                "roi_id": record.roi_id,
                "marker": record.marker,
                "modality": record.modality,
                "image_path": str(record.image_path),
                "manual_points_path": str(record.manual_points_path) if record.manual_points_path is not None else None,
                "backend": runtime.backend,
                "model_label": runtime.model_spec.model_label,
                "segmentation_preset": config.segmentation_preset,
                "runtime_seconds": runtime_seconds,
                **metrics,
            }
        )

    primary = next(row for row in per_tolerance_rows if np.isclose(float(row["match_tolerance_px"]), PRIMARY_TOLERANCE_PX))
    primary["predicted_points_yx_json"] = json.dumps(predicted_points.tolist())
    primary["manual_points_yx_json"] = json.dumps(manual_points.tolist())
    return primary, per_tolerance_rows


def save_roi_match_overlay(
    *,
    roi_image: np.ndarray,
    manual_points_yx: np.ndarray,
    predicted_points_yx: np.ndarray,
    tolerance_px: float,
    destination: str | Path,
    title: str,
) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    match = match_points(manual_points_yx, predicted_points_yx, tolerance_px=tolerance_px)

    fig, ax = plt.subplots(figsize=(6, 6))
    if roi_image.ndim == 2:
        ax.imshow(roi_image, cmap="gray")
    else:
        ax.imshow(roi_image)

    if len(match.matched_manual_indices):
        manual_matched = np.asarray(manual_points_yx, dtype=float)[match.matched_manual_indices]
        predicted_matched = np.asarray(predicted_points_yx, dtype=float)[match.matched_pred_indices]
        ax.scatter(manual_matched[:, 1], manual_matched[:, 0], s=40, marker="o", facecolors="none", edgecolors="lime", label="manual matched")
        ax.scatter(predicted_matched[:, 1], predicted_matched[:, 0], s=40, marker="+", c="cyan", label="pred matched")
        for (manual_y, manual_x), (pred_y, pred_x) in zip(manual_matched, predicted_matched):
            ax.plot([manual_x, pred_x], [manual_y, pred_y], linewidth=0.8, color="white", alpha=0.8)

    if len(match.unmatched_manual_indices):
        unmatched_manual = np.asarray(manual_points_yx, dtype=float)[match.unmatched_manual_indices]
        ax.scatter(unmatched_manual[:, 1], unmatched_manual[:, 0], s=40, marker="x", c="red", label="manual missed")

    if len(match.unmatched_pred_indices):
        unmatched_pred = np.asarray(predicted_points_yx, dtype=float)[match.unmatched_pred_indices]
        ax.scatter(unmatched_pred[:, 1], unmatched_pred[:, 0], s=40, marker="+", c="orange", label="false positive")

    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)
    return destination


def summarize_config_results(results_frame: pd.DataFrame) -> pd.DataFrame:
    if results_frame.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for config_id, frame in results_frame.groupby("config_id", dropna=False):
        primary = frame.loc[np.isclose(frame["match_tolerance_px"].astype(float), PRIMARY_TOLERANCE_PX)].copy()
        if primary.empty:
            primary = frame.copy()
        row: dict[str, Any] = {
            "config_id": str(config_id),
            "backend": str(primary["backend"].dropna().iloc[0]) if "backend" in primary and primary["backend"].notna().any() else None,
            "segmentation_preset": (
                str(primary["segmentation_preset"].dropna().iloc[0])
                if "segmentation_preset" in primary and primary["segmentation_preset"].notna().any()
                else None
            ),
            "model_label": str(primary["model_label"].dropna().iloc[0]) if "model_label" in primary and primary["model_label"].notna().any() else None,
            "n_rois": int(primary["roi_id"].nunique()) if "roi_id" in primary else int(len(primary)),
            "marker": str(primary["marker"].dropna().iloc[0]) if "marker" in primary and primary["marker"].notna().any() else None,
            "modality": str(primary["modality"].dropna().iloc[0]) if "modality" in primary and primary["modality"].notna().any() else None,
            "precision_mean_8px": float(primary["precision"].mean()),
            "recall_mean_8px": float(primary["recall"].mean()),
            "f1_mean_8px": float(primary["f1"].mean()),
            "count_mae_mean_8px": float(primary["count_mae"].mean()),
            "count_bias_mean_8px": float(primary["count_bias"].mean()),
            "runtime_seconds_mean": float(primary["runtime_seconds"].mean()) if "runtime_seconds" in primary else float("nan"),
            "median_manual_count": float(primary["manual_count"].median()) if "manual_count" in primary else float("nan"),
        }
        for tolerance_px in SENSITIVITY_TOLERANCES_PX:
            tolerance_frame = frame.loc[np.isclose(frame["match_tolerance_px"].astype(float), float(tolerance_px))].copy()
            if tolerance_frame.empty:
                continue
            row[f"precision_mean_{int(tolerance_px)}px"] = float(tolerance_frame["precision"].mean())
            row[f"recall_mean_{int(tolerance_px)}px"] = float(tolerance_frame["recall"].mean())
            row[f"f1_mean_{int(tolerance_px)}px"] = float(tolerance_frame["f1"].mean())

        median_manual = row["median_manual_count"]
        mae = row["count_mae_mean_8px"]
        row["pass_threshold"] = bool(
            row["n_rois"] >= BENCHMARK_MIN_ROIS
            and np.isfinite(row["f1_mean_8px"])
            and np.isfinite(row["recall_mean_8px"])
            and np.isfinite(mae)
            and np.isfinite(median_manual)
            and row["f1_mean_8px"] >= 0.75
            and row["recall_mean_8px"] >= 0.75
            and (median_manual <= 0 or mae <= 0.10 * median_manual)
        )
        rows.append(row)

    out = pd.DataFrame(rows)
    out = out.sort_values(
        ["f1_mean_8px", "recall_mean_8px", "count_mae_mean_8px", "runtime_seconds_mean", "config_id"],
        ascending=[False, False, True, True, True],
    ).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def build_roi_benchmark_report(
    *,
    roi_manifest: pd.DataFrame,
    config_summary: pd.DataFrame,
    per_roi_primary: pd.DataFrame,
) -> str:
    marker = str(roi_manifest["marker"].dropna().astype(str).iloc[0]) if not roi_manifest.empty else ""
    modality = str(roi_manifest["modality"].dropna().astype(str).iloc[0]) if not roi_manifest.empty else ""
    lines = [
        "# Real ROI Benchmark Report",
        "",
        f"- ROIs: `{len(roi_manifest)}`",
        f"- Marker: `{marker}`",
        f"- Modality: `{modality}`",
        f"- Tolerances reported: `{list(SENSITIVITY_TOLERANCES_PX)}`",
        f"- Pass threshold requires at least `{BENCHMARK_MIN_ROIS}` ROIs.",
        "",
    ]
    if config_summary.empty:
        lines.extend(["No config results were produced.", ""])
        return "\n".join(lines)

    winner = config_summary.iloc[0]
    lines.extend(
        [
            "## Winning Config",
            "",
            f"- Config ID: `{winner['config_id']}`",
            f"- Backend: `{winner.get('backend')}`",
            f"- Segmentation preset: `{winner.get('segmentation_preset')}`",
            f"- Mean F1 @ 8 px: `{winner['f1_mean_8px']:.3f}`",
            f"- Mean Recall @ 8 px: `{winner['recall_mean_8px']:.3f}`",
            f"- Mean Precision @ 8 px: `{winner['precision_mean_8px']:.3f}`",
            f"- Mean Count MAE @ 8 px: `{winner['count_mae_mean_8px']:.3f}`",
            f"- Pass threshold: `{bool(winner['pass_threshold'])}`",
            "",
        ]
    )
    if bool(winner["pass_threshold"]):
        lines.extend(
            [
                "## Plain-language conclusion",
                "",
                (
                    "This benchmark passed the project acceptance threshold for one narrow matched-modality ROI counting use-case. "
                    "It does not automatically generalize beyond the current marker/modality slice."
                ),
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Plain-language conclusion",
                "",
                (
                    "This benchmark did not pass the project acceptance threshold. "
                    "The workflow remains useful as software, but it is not yet validated for scientific counting claims on this benchmark slice."
                ),
                "",
            ]
        )

    lines.extend(["## Config comparison", "", markdown_table(config_summary), ""])
    hardest = per_roi_primary.sort_values(["f1", "count_mae", "roi_id"], ascending=[True, False, True]).head(10)
    if not hardest.empty:
        lines.extend(["## Hardest ROIs", "", markdown_table(hardest), ""])
    return "\n".join(lines)


def build_roi_benchmark_suite_report(
    *,
    roi_manifest: pd.DataFrame,
    comparison_frame: pd.DataFrame,
    best_payload: dict[str, Any],
    quality_frame: pd.DataFrame,
) -> str:
    marker = str(roi_manifest["marker"].dropna().astype(str).iloc[0]) if not roi_manifest.empty else ""
    modality = str(roi_manifest["modality"].dropna().astype(str).iloc[0]) if not roi_manifest.empty else ""
    lines = [
        "# ROI Benchmark Suite Report",
        "",
        f"- ROIs: `{len(roi_manifest)}`",
        f"- Marker: `{marker}`",
        f"- Modality: `{modality}`",
        "- Ranking rule: `F1@8 desc, Recall@8 desc, MAE asc, runtime asc`",
        "- Baseline-beat rule: `F1 > baseline, Recall >= baseline, MAE <= baseline`",
        "",
    ]
    if best_payload:
        if bool(best_payload.get("pass_threshold")) and bool(best_payload.get("beats_baseline")):
            lines.extend(
                [
                    "## Verdict",
                    "",
                    (
                        "A config beat the `cellpose_default` baseline and passed the project acceptance threshold. "
                        f"Supported narrow use-case: `{marker}` matched-modality `{modality}` ROI detection with `{best_payload.get('config_id')}`."
                    ),
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    "## Verdict",
                    "",
                    (
                        "No configuration both beat the `cellpose_default` baseline and passed the project acceptance threshold. "
                        "Do not promote any configuration to a validated scientific default yet."
                    ),
                    "",
                ]
            )
    lines.extend(["## Config Comparison", "", markdown_table(comparison_frame), "", "## Benchmark Quality Table", "", markdown_table(quality_frame), ""])
    return "\n".join(lines)


def _write_config_outputs(
    *,
    output_dir: Path,
    primary_frame: pd.DataFrame,
    all_frame: pd.DataFrame,
    config_summary: pd.DataFrame,
    quality_frame: pd.DataFrame,
    roi_manifest: pd.DataFrame,
) -> dict[str, Path]:
    results_dir = output_dir / "results"
    report_dir = output_dir / "report"
    results_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    primary_frame.to_csv(results_dir / "per_roi_metrics.csv", index=False)
    all_frame.to_csv(results_dir / "per_roi_tolerance_metrics.csv", index=False)
    config_summary.to_csv(results_dir / "config_comparison.csv", index=False)
    if not config_summary.empty:
        payload = {key: _json_ready(value) for key, value in config_summary.iloc[0].to_dict().items()}
        (results_dir / "best_config.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    quality_frame.to_csv(report_dir / "benchmark_quality.csv", index=False)
    (report_dir / "benchmark_report.md").write_text(
        build_roi_benchmark_report(
            roi_manifest=roi_manifest,
            config_summary=config_summary,
            per_roi_primary=primary_frame,
        )
        + "\n",
        encoding="utf-8",
    )

    # Backward-compatible aliases from the earlier script-only runner.
    primary_frame.to_csv(results_dir / "roi_metrics.csv", index=False)
    config_summary.to_csv(results_dir / "benchmark_summary.csv", index=False)
    if not all_frame.empty:
        sensitivity = (
            all_frame.groupby("match_tolerance_px", dropna=False)[["precision", "recall", "f1"]]
            .mean()
            .reset_index()
            .rename(columns={"match_tolerance_px": "tolerance_px"})
        )
    else:
        sensitivity = pd.DataFrame(columns=["tolerance_px", "precision", "recall", "f1"])
    sensitivity.to_csv(results_dir / "tolerance_sensitivity.csv", index=False)
    quality_frame.to_csv(results_dir / "benchmark_quality.csv", index=False)

    return {
        "results_dir": results_dir,
        "report_dir": report_dir,
        "report_path": report_dir / "benchmark_report.md",
        "quality_path": report_dir / "benchmark_quality.csv",
        "comparison_path": results_dir / "config_comparison.csv",
        "best_path": results_dir / "best_config.json",
    }


def run_roi_benchmark_config(
    *,
    roi_manifest: str | Path,
    output_dir: str | Path,
    config: RoiBenchmarkConfig,
    save_overlays: bool = False,
    runtime_builder: Callable[..., Any] = build_runtime,
    runtime_runner: Callable[..., Any] = run_array,
    include_splits: list[str] | None = None,
    exclude_splits: list[str] | None = None,
) -> dict[str, Any]:
    manifest = filter_roi_manifest_by_split(
        load_roi_manifest(roi_manifest),
        include_splits=include_splits,
        exclude_splits=exclude_splits,
    )
    records = iter_roi_records(manifest, manifest_path=roi_manifest)
    output_dir = Path(output_dir)

    primary_rows: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    overlay_dir = output_dir / "results" / "overlays"

    for record in records:
        primary, per_tolerance_rows = run_single_roi_case(
            record,
            config=config,
            runtime_builder=runtime_builder,
            runtime_runner=runtime_runner,
        )
        primary_rows.append(primary)
        all_rows.extend(per_tolerance_rows)

        if save_overlays:
            image, _ = load_any_image(str(record.image_path))
            crop = crop_2d_or_yxc(image, x0=record.x0, y0=record.y0, width=record.width, height=record.height)
            manual_points = np.asarray(json.loads(primary["manual_points_yx_json"]), dtype=float)
            predicted_points = np.asarray(json.loads(primary["predicted_points_yx_json"]), dtype=float)
            save_roi_match_overlay(
                roi_image=crop,
                manual_points_yx=manual_points,
                predicted_points_yx=predicted_points,
                tolerance_px=PRIMARY_TOLERANCE_PX,
                destination=overlay_dir / f"{record.roi_id}__{config.config_id}__tol8.png",
                title=f"{record.roi_id} | {config.config_id}",
            )

    primary_frame = pd.DataFrame(primary_rows)
    all_frame = pd.DataFrame(all_rows)
    config_summary = summarize_config_results(all_frame)
    quality_frame = pd.DataFrame()
    if not config_summary.empty:
        winner = config_summary.iloc[0]
        quality_frame = build_benchmark_quality_table(
            benchmark_kind="roi_point_matching",
            matched_modality=True,
            n_rois=int(winner["n_rois"]),
            precision=float(winner["precision_mean_8px"]),
            recall=float(winner["recall_mean_8px"]),
            f1=float(winner["f1_mean_8px"]),
            mae=float(winner["count_mae_mean_8px"]),
            pass_threshold=bool(winner["pass_threshold"]),
        )

    paths = _write_config_outputs(
        output_dir=output_dir,
        primary_frame=primary_frame,
        all_frame=all_frame,
        config_summary=config_summary,
        quality_frame=quality_frame,
        roi_manifest=manifest,
    )
    return {
        "manifest": manifest,
        "records": records,
        "primary_frame": primary_frame,
        "all_frame": all_frame,
        "config_summary": config_summary,
        "quality_frame": quality_frame,
        "summary_row": config_summary.iloc[0].to_dict() if not config_summary.empty else {},
        **paths,
    }


def default_config_manifest_for_marker(marker: str) -> pd.DataFrame:
    normalized = str(marker).strip().upper()
    if normalized not in {"RBPMS", "BRN3A"}:
        raise ValueError(f"Unsupported benchmark marker: {marker}")
    preset = "flatmount_rgc_rbpms_demo" if normalized == "RBPMS" else "flatmount_rgc_brn3a_demo"
    prefix = "rbpms" if normalized == "RBPMS" else "brn3a"
    return pd.DataFrame(
        [
            {
                "config_id": "cellpose_default",
                "backend": "cellpose",
                "segmentation_preset": None,
                "diameter": 30.0,
                "min_size": 20,
                "max_size": 400,
                "apply_clahe": False,
                "modality": "flatmount",
                "modality_channel_index": 0,
                "segmenter_config_json": "{}",
                "object_filters_json": "{}",
                "notes": "legacy baseline",
            },
            {
                "config_id": f"cellpose_{prefix}_preset",
                "backend": "cellpose",
                "segmentation_preset": preset,
                "diameter": np.nan,
                "min_size": np.nan,
                "max_size": np.nan,
                "apply_clahe": True,
                "modality": "flatmount",
                "modality_channel_index": 0,
                "segmenter_config_json": "{}",
                "object_filters_json": "{}",
                "notes": "preset-forced cellpose",
            },
            {
                "config_id": f"blob_watershed_{prefix}_preset",
                "backend": "blob_watershed",
                "segmentation_preset": preset,
                "diameter": np.nan,
                "min_size": np.nan,
                "max_size": np.nan,
                "apply_clahe": True,
                "modality": "flatmount",
                "modality_channel_index": 0,
                "segmenter_config_json": "{}",
                "object_filters_json": "{}",
                "notes": "preset baseline",
            },
            {
                "config_id": f"{prefix}_blob_tight",
                "backend": "blob_watershed",
                "segmentation_preset": preset,
                "diameter": np.nan,
                "min_size": 20.0,
                "max_size": 300.0,
                "apply_clahe": True,
                "modality": "flatmount",
                "modality_channel_index": 0,
                "segmenter_config_json": json.dumps({"threshold_rel": 0.18, "min_distance": 6}),
                "object_filters_json": json.dumps({"min_mean_intensity": 0.05, "min_circularity": 0.15}),
                "notes": "tighter peak threshold",
            },
            {
                "config_id": f"{prefix}_blob_loose",
                "backend": "blob_watershed",
                "segmentation_preset": preset,
                "diameter": np.nan,
                "min_size": 15.0,
                "max_size": 400.0,
                "apply_clahe": True,
                "modality": "flatmount",
                "modality_channel_index": 0,
                "segmenter_config_json": json.dumps({"threshold_rel": 0.10, "min_distance": 5}),
                "object_filters_json": json.dumps({"min_mean_intensity": 0.03}),
                "notes": "more recall-oriented",
            },
        ]
    )


def config_from_manifest_row(row: dict[str, Any], *, use_gpu: bool = False) -> RoiBenchmarkConfig:
    return RoiBenchmarkConfig(
        config_id=str(row["config_id"]),
        backend=None if pd.isna(row.get("backend")) or not str(row.get("backend")).strip() else str(row.get("backend")),
        segmentation_preset=(
            None
            if pd.isna(row.get("segmentation_preset")) or not str(row.get("segmentation_preset")).strip()
            else str(row.get("segmentation_preset"))
        ),
        diameter=float(row["diameter"]) if pd.notna(row.get("diameter")) else None,
        min_size=int(row["min_size"]) if pd.notna(row.get("min_size")) else None,
        max_size=int(row["max_size"]) if pd.notna(row.get("max_size")) else None,
        apply_clahe=bool(row.get("apply_clahe", False)) if not isinstance(row.get("apply_clahe"), str) else str(row.get("apply_clahe")).strip().lower() in {"1", "true", "yes", "y"},
        modality=str(row.get("modality") or "flatmount"),
        modality_channel_index=int(row["modality_channel_index"]) if pd.notna(row.get("modality_channel_index")) else 0,
        segmenter_config=json.loads(str(row["segmenter_config_json"])) if pd.notna(row.get("segmenter_config_json")) and str(row.get("segmenter_config_json")).strip() else None,
        object_filters=json.loads(str(row["object_filters_json"])) if pd.notna(row.get("object_filters_json")) and str(row.get("object_filters_json")).strip() else None,
        use_gpu=bool(use_gpu),
        notes=str(row.get("notes") or ""),
    )


def configs_from_manifest(config_manifest: pd.DataFrame, *, use_gpu: bool = False) -> list[RoiBenchmarkConfig]:
    return [config_from_manifest_row(row, use_gpu=use_gpu) for row in config_manifest.to_dict("records")]


def run_benchmark_suite(
    *,
    roi_manifest: str | Path,
    config_manifest: pd.DataFrame,
    output_dir: str | Path,
    save_overlays: bool = False,
    use_gpu: bool = False,
    runtime_builder: Callable[..., Any] = build_runtime,
    runtime_runner: Callable[..., Any] = run_array,
    include_splits: list[str] | None = None,
    exclude_splits: list[str] | None = None,
) -> dict[str, Any]:
    manifest = filter_roi_manifest_by_split(
        load_roi_manifest(roi_manifest),
        include_splits=include_splits,
        exclude_splits=exclude_splits,
    )
    output_dir = Path(output_dir)
    results_root = output_dir / "results"
    report_root = output_dir / "report"
    results_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)

    primary_frames: list[pd.DataFrame] = []
    all_frames: list[pd.DataFrame] = []
    comparison_rows: list[dict[str, Any]] = []

    for config in configs_from_manifest(config_manifest, use_gpu=use_gpu):
        run_result = run_roi_benchmark_config(
            roi_manifest=roi_manifest,
            output_dir=results_root / config.config_id,
            config=config,
            save_overlays=save_overlays,
            runtime_builder=runtime_builder,
            runtime_runner=runtime_runner,
            include_splits=include_splits,
            exclude_splits=exclude_splits,
        )
        primary_frames.append(run_result["primary_frame"])
        all_frames.append(run_result["all_frame"])
        comparison_rows.append(run_result["summary_row"])

    all_primary = pd.concat(primary_frames, ignore_index=True) if primary_frames else pd.DataFrame()
    all_tolerance = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    comparison = summarize_config_results(all_tolerance)

    baseline = comparison.loc[comparison["config_id"] == "cellpose_default"].iloc[0] if not comparison.empty and (comparison["config_id"] == "cellpose_default").any() else None
    if baseline is not None:
        comparison["beats_baseline"] = comparison.apply(
            lambda row: False
            if row["config_id"] == "cellpose_default"
            else (
                float(row["f1_mean_8px"]) > float(baseline["f1_mean_8px"])
                and float(row["recall_mean_8px"]) >= float(baseline["recall_mean_8px"])
                and float(row["count_mae_mean_8px"]) <= float(baseline["count_mae_mean_8px"])
            ),
            axis=1,
        )
    else:
        comparison["beats_baseline"] = False

    comparison = comparison.sort_values(
        ["f1_mean_8px", "recall_mean_8px", "count_mae_mean_8px", "runtime_seconds_mean", "config_id"],
        ascending=[False, False, True, True, True],
    ).reset_index(drop=True)
    comparison["rank"] = np.arange(1, len(comparison) + 1)

    best_payload = comparison.iloc[0].to_dict() if not comparison.empty else {}
    quality_frame = pd.DataFrame()
    if best_payload:
        quality_frame = build_benchmark_quality_table(
            benchmark_kind="roi_point_matching",
            matched_modality=True,
            n_rois=int(best_payload["n_rois"]),
            precision=float(best_payload["precision_mean_8px"]),
            recall=float(best_payload["recall_mean_8px"]),
            f1=float(best_payload["f1_mean_8px"]),
            mae=float(best_payload["count_mae_mean_8px"]),
            pass_threshold=bool(best_payload["pass_threshold"]),
        )

    all_primary.to_csv(results_root / "per_roi_metrics.csv", index=False)
    all_tolerance.to_csv(results_root / "per_roi_tolerance_metrics.csv", index=False)
    comparison.to_csv(results_root / "config_comparison.csv", index=False)
    (results_root / "best_config.json").write_text(
        json.dumps(
            {
                "ranking_rule": "f1_desc_then_recall_desc_then_mae_asc_then_runtime_asc",
                **{key: _json_ready(value) for key, value in best_payload.items()},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    quality_frame.to_csv(report_root / "benchmark_quality.csv", index=False)
    (report_root / "benchmark_report.md").write_text(
        build_roi_benchmark_suite_report(
            roi_manifest=manifest,
            comparison_frame=comparison,
            best_payload=best_payload,
            quality_frame=quality_frame,
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "primary_frame": all_primary,
        "all_frame": all_tolerance,
        "comparison_frame": comparison,
        "best_payload": best_payload,
        "quality_frame": quality_frame,
        "comparison_path": results_root / "config_comparison.csv",
        "best_path": results_root / "best_config.json",
        "quality_path": report_root / "benchmark_quality.csv",
        "report_path": report_root / "benchmark_report.md",
    }
