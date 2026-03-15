from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


SCHEMA_VERSION = "1.0.0"
OBJECT_TABLE_VERSION = 2
PROVENANCE_VERSION = 2

OBJECT_REQUIRED = {
    "schema_version": "string",
    "table_kind": "string",
    "image_id": "string",
    "object_id": "int64",
    "centroid_x_px": "float64",
    "centroid_y_px": "float64",
    "area_px": "int64",
    "phenotype": "string",
    "kept": "boolean",
}

REGION_REQUIRED = {
    "schema_version": "string",
    "table_kind": "string",
    "image_id": "string",
    "region_schema": "string",
    "region_axis": "string",
    "region_label": "string",
    "area_mm2": "float64",
    "object_count": "int64",
    "density_cells_per_mm2": "float64",
}

STUDY_REQUIRED = {
    "schema_version": "string",
    "table_kind": "string",
    "sample_id": "string",
    "animal_id": "string",
    "eye": "string",
    "condition": "string",
    "timepoint_dpi": "float64",
    "cell_count": "float64",
    "density_cells_per_mm2": "float64",
}

OBJECT_TABLE_COLUMNS = [
    "schema_version",
    "table_kind",
    "object_table_version",
    "image_id",
    "source_path",
    "filename",
    "reader",
    "object_id",
    "kept",
    "area_px",
    "centroid_y_px",
    "centroid_x_px",
    "ret_x_um",
    "ret_y_um",
    "ecc_um",
    "theta_deg",
    "normalized_ecc",
    "ring",
    "quadrant",
    "sector",
    "peripapillary_bin",
    "retina_region_schema",
    "region_schema",
    "bbox_ymin_px",
    "bbox_xmin_px",
    "bbox_ymax_px",
    "bbox_xmax_px",
    "focus_overlap_px",
    "focus_overlap_fraction",
    "mean_intensity",
    "max_intensity",
    "geometry.area_px",
    "geometry.perimeter_px",
    "geometry.circularity",
    "geometry.eccentricity",
    "phenotype",
    "phenotype_priority",
    "phenotype_engine",
    "uncertainty_fg_prob_mean",
    "uncertainty_fg_prob_std",
    "uncertainty_fg_prob_p10",
    "interaction.nearest_any_px",
    "interaction.nearest_same_class_px",
]

REGION_TABLE_COLUMNS = [
    "schema_version",
    "table_kind",
    "image_id",
    "source_path",
    "retina_region_schema",
    "region_schema",
    "region_axis",
    "region_label",
    "area_px",
    "area_mm2",
    "object_count",
    "density_cells_per_mm2",
    "max_ecc_um",
]

STUDY_TABLE_COLUMNS = [
    "schema_version",
    "table_kind",
    "sample_id",
    "animal_id",
    "eye",
    "condition",
    "genotype",
    "timepoint_dpi",
    "modality",
    "cell_count",
    "density_cells_per_mm2",
]


def order_columns(frame: pd.DataFrame, preferred: list[str]) -> pd.DataFrame:
    present = [column for column in preferred if column in frame.columns]
    extra = [column for column in frame.columns if column not in present]
    return frame[present + extra]


def _image_id_from_frame(frame: pd.DataFrame) -> pd.Series:
    if "image_id" in frame.columns:
        return frame["image_id"].astype("string")
    if "filename" in frame.columns:
        return frame["filename"].map(lambda value: Path(str(value)).name.rsplit(".", 1)[0]).astype("string")
    if "source_path" in frame.columns:
        return frame["source_path"].map(lambda value: Path(str(value)).name.rsplit(".", 1)[0]).astype("string")
    if "sample_id" in frame.columns:
        return frame["sample_id"].astype("string")
    return pd.Series([""] * len(frame), index=frame.index, dtype="string")


def _fill_defaults(frame: pd.DataFrame, table_kind: str) -> pd.DataFrame:
    out = frame.copy()
    out["schema_version"] = SCHEMA_VERSION
    out["table_kind"] = table_kind

    if table_kind == "object":
        out["image_id"] = _image_id_from_frame(out)
        if "kept" not in out.columns:
            out["kept"] = True
        if "phenotype" not in out.columns:
            out["phenotype"] = "unclassified"
        if "object_table_version" not in out.columns:
            out["object_table_version"] = OBJECT_TABLE_VERSION
        if "region_schema" not in out.columns and "retina_region_schema" in out.columns:
            out["region_schema"] = out["retina_region_schema"]
    elif table_kind == "region":
        out["image_id"] = _image_id_from_frame(out)
        if "region_schema" not in out.columns and "retina_region_schema" in out.columns:
            out["region_schema"] = out["retina_region_schema"]
        if "retina_region_schema" not in out.columns and "region_schema" in out.columns:
            out["retina_region_schema"] = out["region_schema"]
    elif table_kind == "study":
        if "sample_id" in out.columns:
            out["sample_id"] = out["sample_id"].astype("string")
    return out


def _coerce_column(series: pd.Series, dtype: str) -> pd.Series:
    if dtype == "string":
        return series.astype("string")
    if dtype == "boolean":
        return series.astype("boolean")
    return series.astype(dtype)


def validate_table(
    frame: pd.DataFrame,
    *,
    required: dict[str, str],
    table_kind: str,
    strict: bool = False,
) -> pd.DataFrame:
    out = _fill_defaults(frame, table_kind)
    missing = [column for column in required if column not in out.columns]
    if missing and strict:
        raise ValueError(f"Missing required {table_kind} column(s): {missing}")

    for column, dtype in required.items():
        if column not in out.columns:
            out[column] = pd.Series([pd.NA] * len(out), index=out.index)
        try:
            out[column] = _coerce_column(out[column], dtype)
        except Exception:
            if strict:
                raise
    return out


def validate_object_table(frame: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
    return order_columns(validate_table(frame, required=OBJECT_REQUIRED, table_kind="object", strict=strict), OBJECT_TABLE_COLUMNS)


def validate_region_table(frame: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
    return order_columns(validate_table(frame, required=REGION_REQUIRED, table_kind="region", strict=strict), REGION_TABLE_COLUMNS)


def validate_study_table(frame: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
    return order_columns(validate_table(frame, required=STUDY_REQUIRED, table_kind="study", strict=strict), STUDY_TABLE_COLUMNS)
