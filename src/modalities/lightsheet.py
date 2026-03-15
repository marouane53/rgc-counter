from __future__ import annotations

from typing import Any

import numpy as np

from src.modalities.common import project_volume


def adapt_lightsheet_image(
    image: np.ndarray,
    meta: dict[str, Any] | None = None,
    *,
    projection: str = "max",
    slab_start: int | None = None,
    slab_end: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    arr = np.asarray(image)
    adapted, projection_info = project_volume(
        arr,
        projection=projection,
        slab_start=slab_start,
        slab_end=slab_end,
    )
    out_meta = dict(meta or {})
    out_meta.update(
        {
            "modality_adapter": "lightsheet",
            "modality_projection": projection_info["projection"],
            "modality_depth_axis": projection_info["depth_axis"],
            "modality_source_ndim": int(arr.ndim),
        }
    )
    return adapted, out_meta
