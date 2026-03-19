from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.micro_roi_benchmark import (
    load_or_build_projection_lab_config_manifest,
    moderate_locked_eval_gate,
    summarize_split_metrics,
)
from src.point_detection import detect_dog_peaks, detect_hmax_peaks, detect_log_peaks
from src.roi_benchmark import PRIMARY_TOLERANCE_PX, SENSITIVITY_TOLERANCES_PX, save_roi_match_overlay, summarize_truth_provenance
from src.roi_data import crop_2d_or_yxc, iter_roi_records, load_roi_manifest
from src.validation import load_manual_points, point_matching_metrics


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
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


def _detect_points(image: np.ndarray, exclude_mask: np.ndarray | None, backend: str, config: dict) -> pd.DataFrame:
    normalized = str(backend).strip().lower()
    if normalized == "log":
        return detect_log_peaks(image, exclude_mask=exclude_mask, **config)
    if normalized == "dog":
        return detect_dog_peaks(image, exclude_mask=exclude_mask, **config)
    if normalized == "hmax":
        return detect_hmax_peaks(image, exclude_mask=exclude_mask, **config)
    raise ValueError(f"Unsupported predictor backend: {backend}")


def _load_detector_input(path: Path) -> np.ndarray:
    return np.asarray(tifffile.imread(str(path)))


def _load_exclude_mask(path: str | Path | None) -> np.ndarray | None:
    if path is None or (isinstance(path, float) and np.isnan(path)) or not str(path).strip():
        return None
    resolved = Path(path)
    if not resolved.exists():
        return None
    return np.asarray(tifffile.imread(str(resolved))).astype(bool)


def _matching_view_row(views: pd.DataFrame, roi_id: str, projection_recipe_json: str, preprocess_json: str) -> pd.Series:
    matches = views.loc[
        (views["roi_id"].astype(str) == roi_id)
        & (views["projection_recipe_json"].astype(str) == str(projection_recipe_json))
        & (views["preprocess_json"].astype(str) == str(preprocess_json))
    ].copy()
    if matches.empty:
        raise ValueError(f"No projection-lab view found for ROI {roi_id} with projection={projection_recipe_json} preprocess={preprocess_json}")
    return matches.iloc[0]


def _build_report(comparison: pd.DataFrame, quality: pd.DataFrame) -> str:
    lines = [
        "# Micro ROI Projection Lab Benchmark",
        "",
        f"- Configs evaluated: `{len(comparison)}`",
        f"- Quality rows: `{len(quality)}`",
        "",
    ]
    if comparison.empty:
        lines.extend(["No results were produced.", ""])
        return "\n".join(lines)
    winner = comparison.iloc[0]
    lines.extend(
        [
            "## Best Config",
            "",
            f"- Config ID: `{winner['config_id']}`",
            f"- Truth provenance: `{winner['truth_provenance_status']}`",
            f"- Dev F1 @ 8 px: `{winner['dev_f1_8px']:.3f}`",
            f"- Dev Recall @ 8 px: `{winner['dev_recall_8px']:.3f}`",
            f"- Locked Eval F1 @ 8 px: `{winner['locked_eval_f1_8px']:.3f}`",
            f"- Locked Eval Recall @ 8 px: `{winner['locked_eval_recall_8px']:.3f}`",
            f"- Pass threshold: `{bool(winner['pass_threshold'])}`",
            "",
            "## Config Comparison",
            "",
            _markdown_table(comparison),
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the private micro-ROI projection lab benchmark.")
    parser.add_argument("--roi-manifest", "--manifest", dest="roi_manifest", required=True, type=Path)
    parser.add_argument("--view-manifest", required=True, type=Path)
    parser.add_argument("--config-manifest", default=None, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--include-splits", nargs="*", default=["dev", "locked_eval"])
    parser.add_argument("--exclude-splits", nargs="*", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = load_roi_manifest(args.roi_manifest)
    records = iter_roi_records(
        manifest,
        manifest_path=args.roi_manifest,
        include_splits=args.include_splits,
        exclude_splits=args.exclude_splits,
    )
    views = pd.read_csv(args.view_manifest)
    configs = load_or_build_projection_lab_config_manifest(str(args.config_manifest) if args.config_manifest is not None else None)
    output_dir = args.output_dir.resolve()
    results_dir = output_dir / "results"
    report_dir = output_dir / "report"
    results_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    configs.to_csv(output_dir / "resolved_config_manifest.csv", index=False)
    total_runs = len(configs) * len(records)
    _log(
        f"[run_micro_roi_benchmark_suite] Starting micro benchmark with {len(configs)} config(s) x "
        f"{len(records)} ROI(s) = {total_runs} detector runs"
    )

    primary_rows: list[dict[str, object]] = []
    all_rows: list[dict[str, object]] = []
    provenance_summary = summarize_truth_provenance(records)
    completed_runs = 0

    for config_index, config in enumerate(configs.to_dict("records"), start=1):
        predictor_backend = str(config["predictor_backend"])
        predictor_config = json.loads(str(config["predictor_config_json"]))
        projection_recipe_json = str(config["projection_recipe_json"])
        preprocess_json = str(config["preprocess_json"])
        _log(f"[run_micro_roi_benchmark_suite] [{config_index}/{len(configs)}] {config['config_id']}")
        for record_index, record in enumerate(records, start=1):
            _log(
                f"[run_micro_roi_benchmark_suite] run {completed_runs + 1}/{total_runs}: "
                f"{config['config_id']} on ROI {record.roi_id} ({record_index}/{len(records)} in config)"
            )
            view_row = _matching_view_row(views, record.roi_id, projection_recipe_json, preprocess_json)
            image = _load_detector_input(Path(str(view_row["detector_input_path"])))
            exclude_mask = _load_exclude_mask(view_row.get("exclude_mask_path"))
            predicted = _detect_points(image, exclude_mask, predictor_backend, predictor_config)
            points_dir = results_dir / str(config["config_id"]) / "points"
            overlay_dir = results_dir / str(config["config_id"]) / "overlays"
            points_dir.mkdir(parents=True, exist_ok=True)
            overlay_dir.mkdir(parents=True, exist_ok=True)
            predicted_path = points_dir / f"{record.roi_id}__predicted_points.csv"
            predicted.to_csv(predicted_path, index=False)

            manual_points = load_manual_points(record.manual_points_path) if record.manual_points_path is not None else np.empty((0, 2), dtype=float)
            predicted_points = predicted[["y_px", "x_px"]].to_numpy(dtype=float) if not predicted.empty else np.empty((0, 2), dtype=float)
            for tolerance in SENSITIVITY_TOLERANCES_PX:
                metrics = point_matching_metrics(manual_points, predicted_points, tolerance_px=float(tolerance))
                row = {
                    "config_id": str(config["config_id"]),
                    "roi_id": record.roi_id,
                    "split": record.split,
                    "marker": record.marker,
                    "modality": record.modality,
                    "image_marker": record.image_marker,
                    "image_source_channel": record.image_source_channel,
                    "truth_marker": record.truth_marker,
                    "truth_source_channel": record.truth_source_channel,
                    "truth_derivation": record.truth_derivation,
                    "truth_provenance_status": record.truth_provenance_status,
                    "truth_provenance_valid": bool(record.truth_provenance_valid),
                    "projection_recipe_json": projection_recipe_json,
                    "preprocess_json": preprocess_json,
                    "predictor_backend": predictor_backend,
                    "predictor_config_json": str(config["predictor_config_json"]),
                    "predicted_points_path": str(predicted_path),
                    **metrics,
                }
                all_rows.append(row)
                if np.isclose(float(tolerance), PRIMARY_TOLERANCE_PX):
                    primary_rows.append(row)

            image_source, _ = tifffile.imread(str(record.image_path)), None
            crop = crop_2d_or_yxc(image_source, x0=record.x0, y0=record.y0, width=record.width, height=record.height)
            save_roi_match_overlay(
                roi_image=crop,
                manual_points_yx=manual_points,
                predicted_points_yx=predicted_points,
                tolerance_px=PRIMARY_TOLERANCE_PX,
                destination=overlay_dir / f"{record.roi_id}__{config['config_id']}__tol8.png",
                title=f"{record.roi_id} | {config['config_id']}",
            )
            completed_runs += 1
            _log(
                f"[run_micro_roi_benchmark_suite] completed {completed_runs}/{total_runs}: "
                f"{config['config_id']} ROI {record.roi_id} -> predicted={len(predicted_points)} truth={len(manual_points)}"
            )

    primary = pd.DataFrame(primary_rows)
    all_metrics = pd.DataFrame(all_rows)
    comparison_rows: list[dict[str, object]] = []
    if not all_metrics.empty:
        for config_id, frame in all_metrics.groupby("config_id", dropna=False):
            summary_row = {
                "config_id": str(config_id),
                "predictor_backend": str(frame["predictor_backend"].dropna().iloc[0]),
                "projection_recipe_json": str(frame["projection_recipe_json"].dropna().iloc[0]),
                "preprocess_json": str(frame["preprocess_json"].dropna().iloc[0]),
                "truth_provenance_status": provenance_summary["truth_provenance_status"],
                "truth_provenance_valid": bool(provenance_summary["truth_provenance_valid"]),
            }
            summary_row.update(summarize_split_metrics(frame, "dev"))
            summary_row.update(summarize_split_metrics(frame, "locked_eval"))
            summary_row["pass_threshold"] = bool(summary_row["truth_provenance_valid"]) and moderate_locked_eval_gate(summary_row)
            comparison_rows.append(summary_row)
    comparison = pd.DataFrame(comparison_rows)
    if not comparison.empty:
        comparison = comparison.sort_values(
            ["pass_threshold", "dev_f1_8px", "dev_recall_8px", "locked_eval_f1_8px", "locked_eval_recall_8px", "config_id"],
            ascending=[False, False, False, False, False, True],
        ).reset_index(drop=True)
        comparison["rank"] = np.arange(1, len(comparison) + 1)
    quality = pd.DataFrame()
    if not comparison.empty:
        winner = comparison.iloc[0].to_dict()
        quality = pd.DataFrame(
            [
                {
                    "benchmark_kind": "micro_roi_projection_lab",
                    "truth_provenance_valid": bool(winner["truth_provenance_valid"]),
                    "truth_provenance_status": winner["truth_provenance_status"],
                    "dev_n_rois": int(winner["dev_n_rois"]),
                    "locked_eval_n_rois": int(winner["locked_eval_n_rois"]),
                    "dev_f1_8px": float(winner["dev_f1_8px"]),
                    "dev_recall_8px": float(winner["dev_recall_8px"]),
                    "locked_eval_f1_8px": float(winner["locked_eval_f1_8px"]),
                    "locked_eval_recall_8px": float(winner["locked_eval_recall_8px"]),
                    "pass_threshold": bool(winner["pass_threshold"]),
                }
            ]
        )

    primary.to_csv(results_dir / "per_roi_metrics.csv", index=False)
    all_metrics.to_csv(results_dir / "per_roi_tolerance_metrics.csv", index=False)
    comparison.to_csv(results_dir / "config_comparison.csv", index=False)
    (results_dir / "best_config.json").write_text(
        json.dumps(comparison.iloc[0].to_dict() if not comparison.empty else {}, indent=2) + "\n",
        encoding="utf-8",
    )
    quality.to_csv(report_dir / "benchmark_quality.csv", index=False)
    (report_dir / "benchmark_report.md").write_text(_build_report(comparison, quality) + "\n", encoding="utf-8")
    _log(
        f"[run_micro_roi_benchmark_suite] Finished: wrote comparison/report outputs to {output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
