from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.io_ome import load_any_image
from src.model_registry import model_summary_fields
from src.run_service import RuntimeOptions, build_runtime, run_array


REQUIRED_MODEL_MANIFEST_COLUMNS = [
    "run_id",
    "image_path",
    "label_path",
    "backend",
    "model_type",
    "cellpose_model",
    "stardist_weights",
    "sam_checkpoint",
    "model_alias",
    "diameter",
    "channel_index",
    "notes",
]


def load_model_manifest(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in REQUIRED_MODEL_MANIFEST_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[REQUIRED_MODEL_MANIFEST_COLUMNS + [c for c in frame.columns if c not in REQUIRED_MODEL_MANIFEST_COLUMNS]]


def count_objects(labels: np.ndarray) -> int:
    ids = np.unique(np.asarray(labels))
    ids = ids[ids != 0]
    return int(len(ids))


def _flatten_label_image(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    arr = np.squeeze(arr)
    while arr.ndim > 2:
        if arr.shape[-1] <= 4:
            arr = arr[..., 0]
        else:
            arr = arr[0]
    return arr.astype(np.uint32, copy=False)


def load_label_image(path: str | Path) -> np.ndarray:
    image, _ = load_any_image(str(path))
    return _flatten_label_image(image)


def overlap_metrics(pred_labels: np.ndarray, ref_labels: np.ndarray) -> dict[str, float]:
    pred_mask = np.asarray(pred_labels) > 0
    ref_mask = np.asarray(ref_labels) > 0
    inter = float(np.logical_and(pred_mask, ref_mask).sum())
    pred_area = float(pred_mask.sum())
    ref_area = float(ref_mask.sum())
    union = float(np.logical_or(pred_mask, ref_mask).sum())
    dice = (2.0 * inter / (pred_area + ref_area)) if (pred_area + ref_area) > 0 else 1.0
    iou = (inter / union) if union > 0 else 1.0
    return {
        "dice_score": float(dice),
        "iou_score": float(iou),
    }


def _rmse(values: pd.Series) -> float:
    if values.empty:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(values.to_numpy(dtype=float)))))


def summarize_model_runs(per_run_frame: pd.DataFrame) -> pd.DataFrame:
    if per_run_frame.empty:
        return pd.DataFrame()

    group_cols = [
        "backend",
        "model_label",
        "model_source",
        "model_alias",
        "model_asset_path",
        "model_builtin_name",
        "model_trust_mode",
    ]
    rows: list[dict[str, Any]] = []
    for keys, frame in per_run_frame.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["n_runs"] = int(len(frame))
        row["n_overlap_runs"] = int(frame["has_overlap_reference"].fillna(False).sum())
        row["mean_dice"] = float(frame["dice_score"].dropna().mean()) if frame["dice_score"].notna().any() else float("nan")
        row["median_dice"] = float(frame["dice_score"].dropna().median()) if frame["dice_score"].notna().any() else float("nan")
        row["mean_iou"] = float(frame["iou_score"].dropna().mean()) if frame["iou_score"].notna().any() else float("nan")
        row["mae_count"] = float(frame["count_abs_error"].dropna().mean()) if frame["count_abs_error"].notna().any() else float("nan")
        row["mean_bias"] = float(frame["count_bias"].dropna().mean()) if frame["count_bias"].notna().any() else float("nan")
        row["rmse_count"] = _rmse(frame["count_bias"].dropna())
        rows.append(row)
    return pd.DataFrame(rows)


def rank_model_summary(summary_frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if summary_frame.empty:
        return summary_frame.copy(), "none"

    ranked = summary_frame.copy()
    if int(ranked["n_overlap_runs"].fillna(0).max()) > 0:
        ranked["_dice_rank"] = ranked["mean_dice"].fillna(-1.0)
        ranked["_mae_rank"] = ranked["mae_count"].fillna(np.inf)
        ranked = ranked.sort_values(["_dice_rank", "_mae_rank", "model_label"], ascending=[False, True, True]).drop(columns=["_dice_rank", "_mae_rank"])
        ranking_rule = "dice_then_mae"
    else:
        ranked["_mae_rank"] = ranked["mae_count"].fillna(np.inf)
        ranked = ranked.sort_values(["_mae_rank", "model_label"], ascending=[True, True]).drop(columns=["_mae_rank"])
        ranking_rule = "mae_only"

    ranked = ranked.reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    return ranked, ranking_rule


def build_evaluation_report(
    *,
    ranked_summary: pd.DataFrame,
    ranking_rule: str,
    per_run_frame: pd.DataFrame,
) -> str:
    lines = [
        "# Model Evaluation Report",
        "",
        f"- Ranking rule: `{ranking_rule}`",
        f"- Evaluated runs: `{len(per_run_frame)}`",
        f"- Distinct models: `{ranked_summary['model_label'].nunique() if not ranked_summary.empty else 0}`",
        "",
    ]
    if not ranked_summary.empty:
        winner = ranked_summary.iloc[0]
        lines.extend(
            [
                "## Best Model",
                "",
                f"- Model label: `{winner['model_label']}`",
                f"- Alias: `{winner.get('model_alias')}`",
                f"- Backend: `{winner['backend']}`",
                f"- Asset path: `{winner.get('model_asset_path')}`",
                f"- Mean Dice: `{winner.get('mean_dice')}`",
                f"- MAE count: `{winner.get('mae_count')}`",
                "",
            ]
        )
    return "\n".join(lines)


def _resolve_reference_count(row: pd.Series, reference_labels: np.ndarray | None) -> tuple[float | None, str | None]:
    if reference_labels is not None:
        return float(count_objects(reference_labels)), "label_path"
    if "manual_count" in row and pd.notna(row["manual_count"]):
        return float(row["manual_count"]), "manual_count"
    return None, None


def evaluate_model_manifest(
    manifest_df: pd.DataFrame,
    *,
    use_gpu: bool = False,
    strict_schemas: bool = False,
    runtime_builder: Callable[..., Any] = build_runtime,
    runtime_runner: Callable[..., Any] = run_array,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for item in manifest_df.to_dict("records"):
        row = pd.Series(item)
        image_path = Path(str(row["image_path"]))
        if not image_path.exists():
            raise FileNotFoundError(f"Evaluation image not found: {image_path}")

        label_path = None
        if pd.notna(row.get("label_path")) and str(row.get("label_path")).strip():
            label_path = Path(str(row["label_path"]))
            if strict_schemas and not label_path.exists():
                raise FileNotFoundError(f"Evaluation label not found: {label_path}")

        image, meta = load_any_image(str(image_path))
        runtime = runtime_builder(
            RuntimeOptions(
                backend=str(row["backend"]) if pd.notna(row["backend"]) else "cellpose",
                model_type=str(row["model_type"]) if pd.notna(row["model_type"]) and str(row["model_type"]).strip() else None,
                cellpose_model=str(row["cellpose_model"]) if pd.notna(row["cellpose_model"]) and str(row["cellpose_model"]).strip() else None,
                stardist_weights=str(row["stardist_weights"]) if pd.notna(row["stardist_weights"]) and str(row["stardist_weights"]).strip() else None,
                sam_checkpoint=str(row["sam_checkpoint"]) if pd.notna(row["sam_checkpoint"]) and str(row["sam_checkpoint"]).strip() else None,
                model_alias=str(row["model_alias"]) if pd.notna(row["model_alias"]) and str(row["model_alias"]).strip() else None,
                diameter=float(row["diameter"]) if pd.notna(row["diameter"]) else None,
                modality_channel_index=int(row["channel_index"]) if pd.notna(row["channel_index"]) else 0,
                use_gpu=use_gpu,
                focus_mode="none",
                save_debug=False,
                write_html_report=False,
                write_object_table=False,
                write_provenance=False,
            )
        )
        ctx = runtime_runner(runtime, image=image, source_path=image_path, meta=meta)
        predicted_labels = np.asarray(ctx.labels) if ctx.labels is not None else np.zeros(image.shape[:2], dtype=np.uint32)
        reference_labels = load_label_image(label_path) if label_path is not None and label_path.exists() else None
        reference_count, reference_source = _resolve_reference_count(row, reference_labels)

        metric_row: dict[str, Any] = {
            "run_id": row["run_id"],
            "image_path": str(image_path),
            "label_path": str(label_path) if label_path is not None else None,
            "backend": runtime.model_spec.backend,
            **model_summary_fields(runtime.model_spec),
            "predicted_count": float(ctx.metrics.get("cell_count", count_objects(predicted_labels))),
            "reference_count": reference_count,
            "reference_source": reference_source,
            "has_overlap_reference": bool(reference_labels is not None),
            "dice_score": np.nan,
            "iou_score": np.nan,
            "count_bias": np.nan,
            "count_abs_error": np.nan,
            "notes": row.get("notes"),
        }
        if reference_labels is not None:
            metric_row.update(overlap_metrics(predicted_labels, reference_labels))
        if reference_count is not None:
            bias = float(metric_row["predicted_count"]) - float(reference_count)
            metric_row["count_bias"] = bias
            metric_row["count_abs_error"] = abs(bias)
        rows.append(metric_row)

    per_run_frame = pd.DataFrame(rows)
    summary_frame = summarize_model_runs(per_run_frame)
    ranked_summary, ranking_rule = rank_model_summary(summary_frame)
    best_model = ranked_summary.iloc[0].to_dict() if not ranked_summary.empty else {}
    metadata = {
        "ranking_rule": ranking_rule,
        "best_model": best_model,
        "best_model_json": {
            "ranking_rule": ranking_rule,
            "model_label": best_model.get("model_label"),
            "alias": best_model.get("model_alias"),
            "backend": best_model.get("backend"),
            "asset_path": best_model.get("model_asset_path"),
            "summary_metrics": {
                "mean_dice": best_model.get("mean_dice"),
                "median_dice": best_model.get("median_dice"),
                "mean_iou": best_model.get("mean_iou"),
                "mae_count": best_model.get("mae_count"),
                "mean_bias": best_model.get("mean_bias"),
                "rmse_count": best_model.get("rmse_count"),
            },
        },
        "report_markdown": build_evaluation_report(
            ranked_summary=ranked_summary,
            ranking_rule=ranking_rule,
            per_run_frame=per_run_frame,
        ),
    }
    return per_run_frame, ranked_summary, metadata


def write_evaluation_outputs(
    *,
    output_dir: str | Path,
    per_run_frame: pd.DataFrame,
    ranked_summary: pd.DataFrame,
    metadata: dict[str, Any],
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_run_path = output_dir / "per_run_metrics.csv"
    summary_path = output_dir / "model_summary.csv"
    best_path = output_dir / "best_model.json"
    report_path = output_dir / "evaluation_report.md"

    per_run_frame.to_csv(per_run_path, index=False)
    ranked_summary.to_csv(summary_path, index=False)
    best_path.write_text(json.dumps(metadata["best_model_json"], indent=2), encoding="utf-8")
    report_path.write_text(str(metadata["report_markdown"]), encoding="utf-8")

    return {
        "per_run_metrics": per_run_path,
        "model_summary": summary_path,
        "best_model": best_path,
        "evaluation_report": report_path,
    }
