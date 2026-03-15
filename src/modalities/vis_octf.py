from __future__ import annotations

from typing import Any

import numpy as np

from src.modalities.common import project_volume


def adapt_vis_octf_image(
    image: np.ndarray,
    meta: dict[str, Any] | None = None,
    *,
    projection: str = "max",
    slab_start: int | None = None,
    slab_end: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    arr = np.asarray(image)
    out_meta = dict(meta or {})

    if arr.ndim == 4 and arr.shape[-1] <= 4:
        projected_channels = []
        depth_axis = 0
        for channel_index in range(arr.shape[-1]):
            projected, projection_info = project_volume(
                arr[..., channel_index],
                projection=projection,
                depth_axis=depth_axis,
                slab_start=slab_start,
                slab_end=slab_end,
            )
            projected_channels.append(projected)
        adapted = np.stack(projected_channels, axis=-1)
        out_meta.update(
            {
                "modality_adapter": "vis_octf",
                "modality_projection": projection_info["projection"],
                "modality_depth_axis": projection_info["depth_axis"],
                "modality_source_ndim": int(arr.ndim),
            }
        )
        return adapted, out_meta

    adapted, projection_info = project_volume(
        arr,
        projection=projection,
        slab_start=slab_start,
        slab_end=slab_end,
    )
    out_meta.update(
        {
            "modality_adapter": "vis_octf",
            "modality_projection": projection_info["projection"],
            "modality_depth_axis": projection_info["depth_axis"],
            "modality_source_ndim": int(arr.ndim),
        }
    )
    return adapted, out_meta
