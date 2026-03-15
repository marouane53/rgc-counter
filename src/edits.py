from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.schema import SCHEMA_VERSION


def default_edit_log_path(image_path: str | Path) -> Path:
    image_path = Path(image_path)
    return image_path.with_suffix(image_path.suffix + ".edits.json")


def load_edit_log(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_edit_log(document: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": SCHEMA_VERSION, **document}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def relabel_sequential(labels: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    labels = labels.copy()
    object_ids = sorted(int(value) for value in np.unique(labels) if int(value) != 0)
    mapping = {old: new for new, old in enumerate(object_ids, start=1)}
    relabeled = np.zeros_like(labels, dtype=np.uint32)
    for old_id, new_id in mapping.items():
        relabeled[labels == old_id] = new_id
    return relabeled, mapping


def apply_edit_log(
    labels: np.ndarray,
    object_table: pd.DataFrame,
    edit_doc: dict[str, Any],
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    out_labels = labels.copy()
    out_table = object_table.copy()
    run_meta: dict[str, Any] = {}
    phenotype_overrides: dict[int, str] = {}

    for edit in edit_doc.get("edits", []):
        op = str(edit["op"])
        if op == "delete_object":
            object_id = int(edit["object_id"])
            out_labels[out_labels == object_id] = 0
            if "object_id" in out_table.columns:
                out_table = out_table[out_table["object_id"].astype(int) != object_id].copy()
        elif op == "merge_objects":
            keep_object_id = int(edit["keep_object_id"])
            object_ids = [int(value) for value in edit["object_ids"]]
            for object_id in object_ids:
                if object_id == keep_object_id:
                    continue
                out_labels[out_labels == object_id] = keep_object_id
                if "object_id" in out_table.columns:
                    out_table = out_table[out_table["object_id"].astype(int) != object_id].copy()
        elif op == "relabel_phenotype":
            object_id = int(edit["object_id"])
            phenotype = str(edit["phenotype"])
            phenotype_overrides[object_id] = phenotype
            if "object_id" in out_table.columns and "phenotype" in out_table.columns:
                out_table.loc[out_table["object_id"].astype(int) == object_id, "phenotype"] = phenotype
        elif op == "set_landmarks":
            run_meta["onh_xy"] = tuple(edit["onh_xy"])
            run_meta["dorsal_xy"] = tuple(edit["dorsal_xy"])
        else:
            raise ValueError(f"Unsupported edit op: {op}")

    relabeled, mapping = relabel_sequential(out_labels)
    if "object_id" in out_table.columns:
        out_table["object_id"] = out_table["object_id"].map(mapping)
        out_table = out_table.dropna(subset=["object_id"]).copy()
        if not out_table.empty:
            out_table["object_id"] = out_table["object_id"].astype(int)

    run_meta["object_id_mapping"] = mapping
    run_meta["phenotype_overrides"] = {
        int(mapping[object_id]): phenotype
        for object_id, phenotype in phenotype_overrides.items()
        if object_id in mapping
    }
    return relabeled, out_table, run_meta
