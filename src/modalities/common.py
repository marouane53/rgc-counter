from __future__ import annotations

from typing import Any

import numpy as np


def _projection_fn(name: str):
    if name == "mean":
        return np.mean
    if name == "sum":
        return np.sum
    return np.max


def reduce_channel_last(image: np.ndarray, *, channel_index: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    info: dict[str, Any] = {}
    if image.ndim < 3 or image.shape[-1] > 4:
        return image, info
    info["channel_axis"] = image.ndim - 1
    info["n_channels"] = int(image.shape[-1])
    if channel_index is None:
        return image, info
    index = int(np.clip(channel_index, 0, image.shape[-1] - 1))
    info["selected_channel"] = index
    return image[..., index], info


def choose_depth_axis(image: np.ndarray) -> int | None:
    if image.ndim < 3:
        return None
    axes = list(range(image.ndim))
    if image.ndim >= 3 and image.shape[-1] <= 4:
        axes = axes[:-1]
    if not axes:
        return None
    candidate = min(axes, key=lambda axis: image.shape[axis])
    return int(candidate)


def project_volume(
    image: np.ndarray,
    *,
    projection: str = "max",
    depth_axis: int | None = None,
    slab_start: int | None = None,
    slab_end: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if image.ndim < 3:
        return image, {"projection": "identity", "depth_axis": None}

    axis = choose_depth_axis(image) if depth_axis is None else depth_axis
    if axis is None:
        return image, {"projection": "identity", "depth_axis": None}

    slicer = [slice(None)] * image.ndim
    if slab_start is not None or slab_end is not None:
        slicer[axis] = slice(slab_start, slab_end)
    slab = image[tuple(slicer)]
    reduced = _projection_fn(projection)(slab, axis=axis)
    return reduced, {
        "projection": projection,
        "depth_axis": axis,
        "slab_start": slab_start,
        "slab_end": slab_end,
    }
