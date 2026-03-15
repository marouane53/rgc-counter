from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_MANIFEST_COLUMNS = [
    "sample_id",
    "animal_id",
    "eye",
    "condition",
    "genotype",
    "timepoint_dpi",
    "modality",
    "stain_panel",
    "path",
]


def _resolve_path(value: object, base_dir: Path) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


def load_manifest(path: str | Path) -> pd.DataFrame:
    manifest_path = Path(path)
    df = pd.read_csv(manifest_path)
    missing = [column for column in REQUIRED_MANIFEST_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")

    df = df.copy()
    base_dir = manifest_path.parent
    df["path"] = df["path"].map(lambda value: _resolve_path(value, base_dir))
    if "label_path" in df.columns:
        df["label_path"] = df["label_path"].map(lambda value: _resolve_path(value, base_dir))

    numeric_columns = [
        "timepoint_dpi",
        "onh_x_px",
        "onh_y_px",
        "dorsal_x_px",
        "dorsal_y_px",
        "expected_total_objects",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df
