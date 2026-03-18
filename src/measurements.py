from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from skimage.measure import regionprops_table

from src.schema import OBJECT_TABLE_COLUMNS, OBJECT_TABLE_VERSION, order_columns, validate_object_table


def _local_contrast(gray_image: np.ndarray | None, labels: np.ndarray, object_id: int, bbox: tuple[int, int, int, int]) -> float | None:
    if gray_image is None:
        return None
    min_row, min_col, max_row, max_col = bbox
    pad = 3
    ymin = max(0, int(min_row) - pad)
    xmin = max(0, int(min_col) - pad)
    ymax = min(gray_image.shape[0], int(max_row) + pad)
    xmax = min(gray_image.shape[1], int(max_col) + pad)
    crop = gray_image[ymin:ymax, xmin:xmax]
    mask = labels[ymin:ymax, xmin:xmax] == int(object_id)
    if not np.any(mask):
        return None
    foreground = crop[mask]
    background = crop[~mask]
    if foreground.size == 0 or background.size == 0:
        return None
    return float(np.mean(foreground) - np.mean(background))


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
    if len(object_ids):
        properties = [
            "label",
            "area",
            "centroid",
            "bbox",
            "perimeter",
            "eccentricity",
            "solidity",
        ]
        if gray_image is not None:
            properties.extend(["intensity_mean", "intensity_max"])
        props = pd.DataFrame(regionprops_table(labels.astype(np.int32), intensity_image=gray_image, properties=properties))
        props = props.sort_values("label").reset_index(drop=True)

        for record in props.to_dict("records"):
            object_id = int(record["label"])
            ys, xs = np.where(labels == object_id)
            if len(ys) == 0:
                continue
            area_px = int(record["area"])
            focus_overlap_px = int(focus_mask[ys, xs].sum()) if focus_mask is not None else area_px
            perimeter = float(record.get("perimeter", 0.0) or 0.0)
            circularity = float(4.0 * np.pi * area_px / (perimeter * perimeter)) if perimeter > 1e-6 else 0.0
            bbox = (
                int(record["bbox-0"]),
                int(record["bbox-1"]),
                int(record["bbox-2"]),
                int(record["bbox-3"]),
            )

            row = {
                "object_table_version": OBJECT_TABLE_VERSION,
                "image_id": image_id,
                "source_path": source_path,
                "filename": filename,
                "reader": reader,
                "object_id": object_id,
                "kept": True,
                "area_px": area_px,
                "centroid_y_px": float(record["centroid-0"]),
                "centroid_x_px": float(record["centroid-1"]),
                "bbox_ymin_px": bbox[0],
                "bbox_xmin_px": bbox[1],
                "bbox_ymax_px": bbox[2],
                "bbox_xmax_px": bbox[3],
                "focus_overlap_px": focus_overlap_px,
                "focus_overlap_fraction": float(focus_overlap_px / area_px) if area_px else 0.0,
                "mean_intensity": float(record["intensity_mean"]) if gray_image is not None else None,
                "max_intensity": float(record["intensity_max"]) if gray_image is not None else None,
                "intensity.local_contrast": _local_contrast(gray_image, labels, object_id, bbox),
                "geometry.area_px": float(area_px),
                "geometry.perimeter_px": perimeter,
                "geometry.circularity": circularity,
                "geometry.eccentricity": float(record.get("eccentricity", 0.0) or 0.0),
                "geometry.solidity": float(record.get("solidity", 0.0) or 0.0),
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


def apply_object_filters(object_table: pd.DataFrame, filters: dict[str, Any] | None) -> pd.DataFrame:
    frame = object_table.copy()
    if frame.empty or not filters:
        return frame

    keep = np.ones(len(frame), dtype=bool)

    def numeric(column: str) -> pd.Series:
        if column not in frame.columns:
            return pd.Series([np.nan] * len(frame), index=frame.index, dtype=float)
        return pd.to_numeric(frame[column], errors="coerce")

    min_area = filters.get("min_area_px")
    if min_area is not None:
        keep &= numeric("area_px") >= float(min_area)
    max_area = filters.get("max_area_px")
    if max_area is not None:
        keep &= numeric("area_px") <= float(max_area)

    min_focus_overlap_fraction = filters.get("min_focus_overlap_fraction")
    if min_focus_overlap_fraction is not None:
        keep &= numeric("focus_overlap_fraction") >= float(min_focus_overlap_fraction)

    min_mean_intensity = filters.get("min_mean_intensity")
    if min_mean_intensity is not None:
        keep &= numeric("mean_intensity") >= float(min_mean_intensity)

    min_local_contrast = filters.get("min_local_contrast")
    if min_local_contrast is not None:
        keep &= numeric("intensity.local_contrast") >= float(min_local_contrast)

    max_eccentricity = filters.get("max_eccentricity")
    if max_eccentricity is not None:
        keep &= numeric("geometry.eccentricity") <= float(max_eccentricity)

    min_solidity = filters.get("min_solidity")
    if min_solidity is not None:
        keep &= numeric("geometry.solidity") >= float(min_solidity)

    min_circularity = filters.get("min_circularity")
    if min_circularity is not None:
        keep &= numeric("geometry.circularity") >= float(min_circularity)

    frame["kept"] = keep
    return order_columns(frame, OBJECT_TABLE_COLUMNS)


def relabel_kept_objects(labels: np.ndarray, object_table: pd.DataFrame) -> np.ndarray:
    kept = object_table.copy()
    if "kept" in kept.columns:
        kept = kept.loc[kept["kept"].fillna(True).astype(bool)]
    if kept.empty:
        return np.zeros_like(labels, dtype=np.uint16)

    relabeled = np.zeros_like(labels, dtype=np.uint32)
    next_id = 1
    for object_id in kept["object_id"].astype(int).tolist():
        relabeled[labels == object_id] = next_id
        next_id += 1
    dtype = np.uint16 if next_id - 1 <= np.iinfo(np.uint16).max else np.uint32
    return relabeled.astype(dtype, copy=False)


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
