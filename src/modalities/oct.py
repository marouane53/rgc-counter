from __future__ import annotations

from typing import Any

import numpy as np

from src.modalities.common import project_volume, reduce_channel_last


def adapt_oct_image(
    image: np.ndarray,
    meta: dict[str, Any] | None = None,
    *,
    projection: str = "max",
    channel_index: int | None = 0,
    slab_start: int | None = None,
    slab_end: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    adapted, info = reduce_channel_last(np.asarray(image), channel_index=channel_index)
    adapted, projection_info = project_volume(
        adapted,
        projection=projection,
        slab_start=slab_start,
        slab_end=slab_end,
    )
    out_meta = dict(meta or {})
    out_meta.update(
        {
            "modality_adapter": "oct",
            "modality_projection": projection_info["projection"],
            "modality_depth_axis": projection_info["depth_axis"],
            "modality_channel_index": info.get("selected_channel"),
            "modality_source_ndim": int(np.asarray(image).ndim),
        }
    )
    return adapted, out_meta
