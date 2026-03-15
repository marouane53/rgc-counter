from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.schema import OBJECT_TABLE_COLUMNS, OBJECT_TABLE_VERSION, order_columns, validate_object_table


def build_object_table(
    path: str | Path,
    labels: np.ndarray,
    focus_mask: np.ndarray | None,
    gray_image: np.ndarray | None = None,
    meta: dict[str, Any] | None = None,
) -> pd.DataFrame:
    source_path = str(path)
    filename = Path(path).name
    image_id = filename.rsplit(".", 1)[0]
    reader = (meta or {}).get("reader", "")

    object_ids = np.unique(labels)
    object_ids = object_ids[object_ids != 0]
    rows: list[dict[str, Any]] = []

    for oid in object_ids:
        ys, xs = np.where(labels == oid)
        if len(ys) == 0:
            continue

        area_px = int(len(ys))
        focus_overlap_px = int(focus_mask[ys, xs].sum()) if focus_mask is not None else area_px
        intensities = gray_image[ys, xs] if gray_image is not None else None

        row = {
            "object_table_version": OBJECT_TABLE_VERSION,
            "image_id": image_id,
            "source_path": source_path,
            "filename": filename,
            "reader": reader,
            "object_id": int(oid),
            "kept": True,
            "area_px": area_px,
            "centroid_y_px": float(ys.mean()),
            "centroid_x_px": float(xs.mean()),
            "bbox_ymin_px": int(ys.min()),
            "bbox_xmin_px": int(xs.min()),
            "bbox_ymax_px": int(ys.max()) + 1,
            "bbox_xmax_px": int(xs.max()) + 1,
            "focus_overlap_px": focus_overlap_px,
            "focus_overlap_fraction": float(focus_overlap_px / area_px) if area_px else 0.0,
            "mean_intensity": float(np.mean(intensities)) if intensities is not None else None,
            "max_intensity": float(np.max(intensities)) if intensities is not None else None,
            "phenotype": "unclassified",
        }
        rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=OBJECT_TABLE_COLUMNS)
        frame["image_id"] = pd.Series(dtype="string")
        frame["source_path"] = pd.Series(dtype="string")
        frame["filename"] = pd.Series(dtype="string")
        frame["reader"] = pd.Series(dtype="string")
        frame["phenotype"] = pd.Series(dtype="string")
        frame["kept"] = pd.Series(dtype="boolean")
    return order_columns(frame, OBJECT_TABLE_COLUMNS)


def add_uncertainty_summary_columns(
    object_table: pd.DataFrame,
    labels: np.ndarray,
    foreground_probability: np.ndarray | None,
) -> pd.DataFrame:
    if foreground_probability is None or object_table.empty:
        return object_table.copy()

    rows: list[dict[str, float | int]] = []
    for object_id in object_table["object_id"].astype(int).tolist():
        pixels = foreground_probability[labels == object_id]
        if pixels.size == 0:
            rows.append(
                {
                    "object_id": object_id,
                    "uncertainty_fg_prob_mean": float("nan"),
                    "uncertainty_fg_prob_std": float("nan"),
                    "uncertainty_fg_prob_p10": float("nan"),
                }
            )
            continue
        rows.append(
            {
                "object_id": object_id,
                "uncertainty_fg_prob_mean": float(np.mean(pixels)),
                "uncertainty_fg_prob_std": float(np.std(pixels)),
                "uncertainty_fg_prob_p10": float(np.quantile(pixels, 0.10)),
            }
        )

    extra = pd.DataFrame(rows)
    merged = object_table.merge(extra, on="object_id", how="left")
    return order_columns(merged, OBJECT_TABLE_COLUMNS)


def object_table_path_for(output_dir: str | Path, source_path: str | Path) -> Path:
    filename = Path(source_path).name.rsplit(".", 1)[0] + "_objects.parquet"
    return Path(output_dir) / "objects" / filename


def write_object_table(frame: pd.DataFrame, destination: str | Path, *, strict: bool = False) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame = validate_object_table(frame, strict=strict)

    has_parquet_engine = (
        importlib.util.find_spec("pyarrow") is not None
        or importlib.util.find_spec("fastparquet") is not None
    )
    if has_parquet_engine:
        frame.to_parquet(destination, index=False)
        return destination

    fallback = destination.with_suffix(".csv")
    frame.to_csv(fallback, index=False)
    return fallback
