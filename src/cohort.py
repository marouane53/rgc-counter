from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pandas as pd

from src.context import RunContext
from src.schema import validate_region_table, validate_study_table


def build_sample_table(manifest_df: pd.DataFrame, contexts: list[RunContext]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for manifest_row, ctx in zip(manifest_df.to_dict("records"), contexts):
        row = dict(manifest_row)
        row.update(ctx.summary_row)
        row["image_id"] = ctx.path.name.rsplit(".", 1)[0]
        for key, value in ctx.artifacts.items():
            row[f"artifact_{key}"] = str(value)
        row["warning_count"] = len(ctx.warnings)
        if "phenotype_counts" in ctx.metrics:
            for phenotype, count in ctx.metrics["phenotype_counts"].items():
                row[f"phenotype_count_{phenotype}"] = count
        if "atlas_subtype_top1_counts" in ctx.metrics:
            for subtype, count in ctx.metrics["atlas_subtype_top1_counts"].items():
                row[f"atlas_subtype_top1_count__{subtype}"] = count
        rows.append(row)
    return pd.DataFrame(rows)


def build_study_region_table(manifest_df: pd.DataFrame, contexts: list[RunContext]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for manifest_row, ctx in zip(manifest_df.to_dict("records"), contexts):
        if ctx.region_table is None or ctx.region_table.empty:
            continue
        frame = ctx.region_table.copy()
        for key, value in manifest_row.items():
            frame[key] = value
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def write_table_bundle(
    frame: pd.DataFrame,
    csv_path: str | Path,
    parquet_path: str | Path | None = None,
    *,
    table_kind: str = "study",
    strict: bool = False,
) -> tuple[Path, Path | None]:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if table_kind == "region":
        frame = validate_region_table(frame, strict=strict)
    else:
        frame = validate_study_table(frame, strict=strict)

    frame.to_csv(csv_path, index=False)

    written_parquet: Path | None = None
    if parquet_path is not None:
        parquet_path = Path(parquet_path)
        if importlib.util.find_spec("pyarrow") is not None or importlib.util.find_spec("fastparquet") is not None:
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(parquet_path, index=False)
            written_parquet = parquet_path
    return csv_path, written_parquet
