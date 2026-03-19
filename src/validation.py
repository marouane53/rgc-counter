from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def count_labels(path: str | Path) -> int:
    labels = tifffile.imread(str(path))
    object_ids = np.unique(labels)
    object_ids = object_ids[object_ids != 0]
    return int(len(object_ids))


def load_manual_points(path: str | Path) -> np.ndarray:
    frame = pd.read_csv(path)
    required = {"x_px", "y_px"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Manual points file is missing required columns: {missing}")
    if frame.empty:
        return np.empty((0, 2), dtype=float)
    return frame[["y_px", "x_px"]].to_numpy(dtype=float)


@dataclass(frozen=True)
class PointMatchResult:
    matched_pred_indices: np.ndarray
    matched_manual_indices: np.ndarray
    matched_distances_px: np.ndarray
    unmatched_pred_indices: np.ndarray
    unmatched_manual_indices: np.ndarray


def match_points(
    manual_points_yx: np.ndarray,
    predicted_points_yx: np.ndarray,
    *,
    tolerance_px: float = 8.0,
) -> PointMatchResult:
    manual = np.asarray(manual_points_yx, dtype=float).reshape(-1, 2)
    predicted = np.asarray(predicted_points_yx, dtype=float).reshape(-1, 2)
    tolerance = float(tolerance_px)

    if len(manual) == 0 or len(predicted) == 0:
        return PointMatchResult(
            matched_pred_indices=np.empty(0, dtype=int),
            matched_manual_indices=np.empty(0, dtype=int),
            matched_distances_px=np.empty(0, dtype=float),
            unmatched_pred_indices=np.arange(len(predicted), dtype=int),
            unmatched_manual_indices=np.arange(len(manual), dtype=int),
        )

    distances = cdist(predicted, manual, metric="euclidean")
    row_ind, col_ind = linear_sum_assignment(distances)

    matched_pred: list[int] = []
    matched_manual: list[int] = []
    matched_distances: list[float] = []
    for pred_idx, manual_idx in zip(row_ind, col_ind):
        distance = float(distances[pred_idx, manual_idx])
        if distance <= tolerance:
            matched_pred.append(int(pred_idx))
            matched_manual.append(int(manual_idx))
            matched_distances.append(distance)

    matched_pred_arr = np.asarray(matched_pred, dtype=int)
    matched_manual_arr = np.asarray(matched_manual, dtype=int)
    matched_distances_arr = np.asarray(matched_distances, dtype=float)
    unmatched_pred = np.setdiff1d(np.arange(len(predicted), dtype=int), matched_pred_arr, assume_unique=False)
    unmatched_manual = np.setdiff1d(np.arange(len(manual), dtype=int), matched_manual_arr, assume_unique=False)

    return PointMatchResult(
        matched_pred_indices=matched_pred_arr,
        matched_manual_indices=matched_manual_arr,
        matched_distances_px=matched_distances_arr,
        unmatched_pred_indices=unmatched_pred,
        unmatched_manual_indices=unmatched_manual,
    )


def point_matching_metrics(
    manual_points_yx: np.ndarray,
    predicted_points_yx: np.ndarray,
    *,
    tolerance_px: float = 8.0,
) -> dict[str, float]:
    manual = np.asarray(manual_points_yx, dtype=float).reshape(-1, 2)
    predicted = np.asarray(predicted_points_yx, dtype=float).reshape(-1, 2)
    tolerance = float(tolerance_px)

    if len(manual) == 0 and len(predicted) == 0:
        return {
            "manual_count": 0.0,
            "predicted_count": 0.0,
            "true_positive": 0.0,
            "false_positive": 0.0,
            "false_negative": 0.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "count_bias": 0.0,
            "count_mae": 0.0,
            "match_tolerance_px": tolerance,
            "mean_match_distance_px": float("nan"),
        }

    matches = match_points(manual, predicted, tolerance_px=tolerance)
    true_positive = int(len(matches.matched_distances_px))
    false_positive = int(max(len(predicted) - true_positive, 0))
    false_negative = int(max(len(manual) - true_positive, 0))
    precision = float(true_positive / (true_positive + false_positive)) if (true_positive + false_positive) > 0 else 0.0
    recall = float(true_positive / (true_positive + false_negative)) if (true_positive + false_negative) > 0 else 0.0
    f1 = float((2.0 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0
    count_bias = float(len(predicted) - len(manual))

    return {
        "manual_count": float(len(manual)),
        "predicted_count": float(len(predicted)),
        "true_positive": float(true_positive),
        "false_positive": float(false_positive),
        "false_negative": float(false_negative),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "count_bias": count_bias,
        "count_mae": float(abs(count_bias)),
        "match_tolerance_px": tolerance,
        "mean_match_distance_px": float(np.mean(matches.matched_distances_px)) if len(matches.matched_distances_px) else float("nan"),
    }


def validate_roi_benchmark_manifest(manifest_df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "roi_id",
        "image_path",
        "marker",
        "modality",
        "x0",
        "y0",
        "width",
        "height",
        "annotator",
        "manual_points_path",
        "split",
        "notes",
    ]
    frame = manifest_df.copy()
    for column in required:
        if column not in frame.columns:
            frame[column] = pd.NA
    missing = sorted(set(required[:10]) - set(manifest_df.columns))
    if missing:
        raise ValueError(f"ROI benchmark manifest is missing required columns: {missing}")
    ordered_columns = required + [column for column in frame.columns if column not in required]
    frame = frame[ordered_columns].copy()
    markers = [str(value).strip() for value in frame["marker"].dropna().tolist() if str(value).strip()]
    if len(set(markers)) > 1:
        raise ValueError("ROI benchmark manifest mixes markers. Use one matched marker family per run.")
    modalities = [str(value).strip() for value in frame["modality"].dropna().tolist() if str(value).strip()]
    if len(set(modalities)) > 1:
        raise ValueError("ROI benchmark manifest mixes modalities. Use one matched modality per run.")
    roi_ids = [str(value).strip() for value in frame["roi_id"].fillna("").tolist()]
    if len(roi_ids) != len(set(roi_id for roi_id in roi_ids if roi_id)):
        raise ValueError("ROI benchmark manifest contains duplicate roi_id values.")
    frame["split"] = frame["split"].fillna("benchmark")
    frame["notes"] = frame["notes"].fillna("")
    return frame


def summarize_roi_benchmark(results_frame: pd.DataFrame) -> pd.DataFrame:
    if results_frame.empty:
        return pd.DataFrame()
    median_manual = float(results_frame["manual_count"].median()) if "manual_count" in results_frame.columns else float("nan")
    mae = float(results_frame["count_mae"].mean()) if "count_mae" in results_frame.columns else float("nan")
    precision = float(results_frame["precision"].mean()) if "precision" in results_frame.columns else float("nan")
    recall = float(results_frame["recall"].mean()) if "recall" in results_frame.columns else float("nan")
    f1 = float(results_frame["f1"].mean()) if "f1" in results_frame.columns else float("nan")
    pass_threshold = bool(
        np.isfinite(f1)
        and np.isfinite(recall)
        and np.isfinite(mae)
        and np.isfinite(median_manual)
        and f1 >= 0.75
        and recall >= 0.75
        and (median_manual <= 0 or mae <= 0.10 * median_manual)
    )
    return pd.DataFrame(
        [
            {
                "benchmark_kind": "roi_point_matching",
                "matched_modality": True,
                "n_rois": int(len(results_frame)),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mae": mae,
                "pass_threshold": pass_threshold,
                "median_manual_count": median_manual,
                "runtime_seconds_mean": float(results_frame["runtime_seconds"].mean()) if "runtime_seconds" in results_frame.columns else float("nan"),
            }
        ]
    )


def build_benchmark_quality_table(
    *,
    benchmark_kind: str,
    matched_modality: bool,
    n_rois: int,
    precision: float | None = None,
    recall: float | None = None,
    f1: float | None = None,
    mae: float | None = None,
    pass_threshold: bool | None = None,
    **extra: Any,
) -> pd.DataFrame:
    row: dict[str, Any] = {
        "benchmark_kind": benchmark_kind,
        "matched_modality": bool(matched_modality),
        "n_rois": int(n_rois),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mae": mae,
        "pass_threshold": pass_threshold,
    }
    row.update(extra)
    return pd.DataFrame([row])


def build_validation_table(sample_table: pd.DataFrame) -> pd.DataFrame:
    if sample_table.empty:
        return pd.DataFrame()

    rows = []
    for row in sample_table.to_dict("records"):
        manual_count = None
        source = None
        label_count = None
        declared_manual_count = None
        label_path = row.get("label_path")
        if label_path:
            resolved_label_path = Path(str(label_path))
            if resolved_label_path.exists():
                label_count = float(count_labels(resolved_label_path))
        if row.get("manual_count") is not None and not pd.isna(row.get("manual_count")):
            declared_manual_count = float(row["manual_count"])

        if label_count is not None and declared_manual_count is not None and abs(label_count - declared_manual_count) > 1e-6:
            raise ValueError(
                f"Conflicting references for {row.get('sample_id')}: "
                f"label_path={int(label_count) if float(label_count).is_integer() else label_count}, "
                f"manual_count={int(declared_manual_count) if float(declared_manual_count).is_integer() else declared_manual_count}"
            )

        if label_count is not None:
            manual_count = label_count
            source = "label_path"
        elif declared_manual_count is not None:
            manual_count = declared_manual_count
            source = "manual_count"
        if manual_count is None and row.get("expected_total_objects") is not None and not pd.isna(row.get("expected_total_objects")):
            manual_count = int(row["expected_total_objects"])
            source = "expected_total_objects"
        if manual_count is None:
            continue
        predicted = float(row["cell_count"])
        rows.append(
            {
                **row,
                "manual_count": float(manual_count),
                "validation_source": source,
                "count_bias": predicted - float(manual_count),
                "count_abs_error": abs(predicted - float(manual_count)),
            }
        )
    return pd.DataFrame(rows)


def _icc_2_1(data: np.ndarray) -> float:
    n, k = data.shape
    if n < 2 or k != 2:
        return float("nan")
    mean_targets = data.mean(axis=1, keepdims=True)
    mean_raters = data.mean(axis=0, keepdims=True)
    grand = data.mean()

    ssr = k * np.sum((mean_targets - grand) ** 2)
    ssc = n * np.sum((mean_raters - grand) ** 2)
    sse = np.sum((data - mean_targets - mean_raters + grand) ** 2)
    msr = ssr / max(n - 1, 1)
    msc = ssc / max(k - 1, 1)
    mse = sse / max((n - 1) * (k - 1), 1)
    denom = msr + (k - 1) * mse + (k * (msc - mse) / n)
    if abs(denom) <= 1e-12:
        return float("nan")
    return float((msr - mse) / denom)


def summarize_validation(validation_table: pd.DataFrame) -> pd.DataFrame:
    if validation_table.empty:
        return pd.DataFrame()
    predicted = validation_table["cell_count"].to_numpy(dtype=float)
    manual = validation_table["manual_count"].to_numpy(dtype=float)
    rmse = float(np.sqrt(np.mean((predicted - manual) ** 2)))
    bias = float(np.mean(predicted - manual))
    mae = float(np.mean(np.abs(predicted - manual)))
    corr = float(np.corrcoef(predicted, manual)[0, 1]) if len(validation_table) >= 2 else float("nan")
    icc = _icc_2_1(np.column_stack([predicted, manual]))
    return pd.DataFrame(
        [
            {
                "n_samples": int(len(validation_table)),
                "mean_bias": bias,
                "mae": mae,
                "rmse": rmse,
                "pearson_r": corr,
                "icc_2_1": icc,
            }
        ]
    )


def save_bland_altman_plot(validation_table: pd.DataFrame, destination: str | Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    predicted = validation_table["cell_count"].to_numpy(dtype=float)
    manual = validation_table["manual_count"].to_numpy(dtype=float)
    mean = (predicted + manual) / 2.0
    diff = predicted - manual
    bias = diff.mean()
    std = diff.std(ddof=1) if len(diff) > 1 else 0.0

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(mean, diff, color="#1f77b4")
    ax.axhline(bias, color="black", linestyle="--", label="Bias")
    ax.axhline(bias + 1.96 * std, color="crimson", linestyle=":", label="95% LoA")
    ax.axhline(bias - 1.96 * std, color="crimson", linestyle=":")
    ax.set_xlabel("Mean of manual and automated counts")
    ax.set_ylabel("Automated - manual")
    ax.set_title("Bland-Altman Plot")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(destination, dpi=200)
    plt.close(fig)
    return destination


def save_agreement_scatter_plot(validation_table: pd.DataFrame, destination: str | Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    predicted = validation_table["cell_count"].to_numpy(dtype=float)
    manual = validation_table["manual_count"].to_numpy(dtype=float)
    lo = min(predicted.min(), manual.min())
    hi = max(predicted.max(), manual.max())

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(manual, predicted, color="#2ca02c")
    ax.plot([lo, hi], [lo, hi], color="black", linestyle="--")
    ax.set_xlabel("Manual count")
    ax.set_ylabel("Automated count")
    ax.set_title("Manual vs Automated Counts")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(destination, dpi=200)
    plt.close(fig)
    return destination
