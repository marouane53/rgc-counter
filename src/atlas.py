from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_ATLAS_COLUMNS = [
    "region_axis",
    "region_label",
    "expected_density_cells_per_mm2",
]


def load_atlas_reference(path: str | Path) -> pd.DataFrame:
    atlas_path = Path(path)
    frame = pd.read_csv(atlas_path)
    missing = [column for column in REQUIRED_ATLAS_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Atlas reference is missing required columns: {missing}")

    frame = frame.copy()
    if "atlas_name" not in frame.columns:
        frame["atlas_name"] = atlas_path.stem
    if "retina_region_schema" not in frame.columns:
        frame["retina_region_schema"] = pd.NA
    if "expected_sd" in frame.columns:
        frame["expected_sd"] = pd.to_numeric(frame["expected_sd"], errors="coerce")
    frame["expected_density_cells_per_mm2"] = pd.to_numeric(
        frame["expected_density_cells_per_mm2"],
        errors="coerce",
    )
    return frame


def compare_region_table_to_atlas(region_table: pd.DataFrame, atlas_reference: pd.DataFrame) -> pd.DataFrame:
    if region_table.empty:
        return pd.DataFrame()

    atlas = atlas_reference.copy()
    if "atlas_name" not in atlas.columns:
        atlas["atlas_name"] = "atlas"
    atlas["retina_region_schema"] = atlas["retina_region_schema"].astype("object")

    merge_columns = ["region_axis", "region_label"]
    if "retina_region_schema" in region_table.columns:
        schema_specific = atlas["retina_region_schema"].notna().any()
        if schema_specific:
            merge_columns.append("retina_region_schema")

    observed = region_table.copy()
    comparison = observed.merge(
        atlas,
        on=merge_columns,
        how="left",
        suffixes=("_observed", "_atlas"),
    )
    if comparison["expected_density_cells_per_mm2"].isna().all():
        return pd.DataFrame()

    comparison["observed_density_cells_per_mm2"] = comparison["density_cells_per_mm2"]
    comparison["delta_density_cells_per_mm2"] = (
        comparison["observed_density_cells_per_mm2"] - comparison["expected_density_cells_per_mm2"]
    )
    comparison["fold_change_vs_atlas"] = comparison["observed_density_cells_per_mm2"] / comparison[
        "expected_density_cells_per_mm2"
    ].replace({0: pd.NA})
    if "expected_sd" in comparison.columns:
        comparison["atlas_zscore"] = comparison["delta_density_cells_per_mm2"] / comparison["expected_sd"].replace(
            {0: pd.NA}
        )
    comparison = comparison[comparison["expected_density_cells_per_mm2"].notna()].reset_index(drop=True)
    return comparison


def summarize_atlas_comparison(
    comparison: pd.DataFrame,
    *,
    group_columns: list[str] | None = None,
) -> pd.DataFrame:
    if comparison.empty:
        return pd.DataFrame()

    group_columns = list(group_columns or [])
    for default_column in ("atlas_name", "condition"):
        if default_column in comparison.columns and default_column not in group_columns:
            group_columns.append(default_column)
    if not group_columns:
        group_columns = ["atlas_name"] if "atlas_name" in comparison.columns else []

    grouped = comparison.groupby(group_columns, dropna=False) if group_columns else [((), comparison)]
    rows: list[dict[str, object]] = []
    for key, frame in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        row = {column: value for column, value in zip(group_columns, key)}
        row["n_regions"] = int(len(frame))
        row["mean_abs_delta_density_cells_per_mm2"] = float(frame["delta_density_cells_per_mm2"].abs().mean())
        row["mean_fold_change_vs_atlas"] = float(frame["fold_change_vs_atlas"].dropna().mean())
        if "atlas_zscore" in frame.columns:
            row["mean_abs_atlas_zscore"] = float(frame["atlas_zscore"].abs().dropna().mean())
        rows.append(row)
    return pd.DataFrame(rows)
