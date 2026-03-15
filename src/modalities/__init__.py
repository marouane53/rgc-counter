from __future__ import annotations

from typing import Any

import numpy as np

from src.modalities.lightsheet import adapt_lightsheet_image
from src.modalities.oct import adapt_oct_image
from src.modalities.vis_octf import adapt_vis_octf_image


def adapt_image_for_modality(
    image: np.ndarray,
    meta: dict[str, Any] | None = None,
    *,
    modality: str = "flatmount",
    projection: str = "max",
    channel_index: int | None = 0,
    slab_start: int | None = None,
    slab_end: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    normalized = (modality or "flatmount").lower()
    if normalized in ("flatmount", "fluorescence", "histology"):
        out_meta = dict(meta or {})
        out_meta["modality_adapter"] = "flatmount"
        out_meta["modality_source_ndim"] = int(np.asarray(image).ndim)
        return np.asarray(image), out_meta
    if normalized == "oct":
        return adapt_oct_image(
            image,
            meta,
            projection=projection,
            channel_index=channel_index,
            slab_start=slab_start,
            slab_end=slab_end,
        )
    if normalized in ("vis_octf", "visoctf", "vis-octf"):
        return adapt_vis_octf_image(
            image,
            meta,
            projection=projection,
            slab_start=slab_start,
            slab_end=slab_end,
        )
    if normalized in ("lightsheet", "light_sheet", "cleared_retina"):
        return adapt_lightsheet_image(
            image,
            meta,
            projection=projection,
            slab_start=slab_start,
            slab_end=slab_end,
        )
    raise ValueError(f"Unsupported modality: {modality}")
