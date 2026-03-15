from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def add_interaction_metrics(object_table: pd.DataFrame) -> pd.DataFrame:
    if object_table.empty or "phenotype" not in object_table.columns:
        return object_table.copy()

    out = object_table.copy()
    xy = out[["centroid_x_px", "centroid_y_px"]].to_numpy(dtype=float)
    phenotypes = out["phenotype"].fillna("unclassified").astype(str).to_numpy()

    if len(out) == 1:
        out["interaction.nearest_any_px"] = np.nan
        out["interaction.nearest_same_class_px"] = np.nan
        return out

    distances = cdist(xy, xy)
    np.fill_diagonal(distances, np.inf)

    out["interaction.nearest_any_px"] = distances.min(axis=1)

    same_class = np.full(len(out), np.nan, dtype=float)
    for idx, phenotype in enumerate(phenotypes):
        mask = phenotypes == phenotype
        mask[idx] = False
        if mask.any():
            same_class[idx] = distances[idx, mask].min()
    out["interaction.nearest_same_class_px"] = same_class

    for phenotype in sorted(set(phenotypes)):
        slug = _slug(phenotype)
        column = f"interaction.nearest_class.{slug}_px"
        values = np.full(len(out), np.nan, dtype=float)
        class_mask = phenotypes == phenotype
        for idx in range(len(out)):
            mask = class_mask.copy()
            if phenotypes[idx] == phenotype:
                mask[idx] = False
            if mask.any():
                values[idx] = distances[idx, mask].min()
        out[column] = values

    return out
