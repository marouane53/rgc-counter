from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile


def count_labels(path: str | Path) -> int:
    labels = tifffile.imread(str(path))
    object_ids = np.unique(labels)
    object_ids = object_ids[object_ids != 0]
    return int(len(object_ids))


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
