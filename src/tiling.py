from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import label as connected_components

from src.uncertainty import segment_with_tta


@dataclass(frozen=True)
class TileWindow:
    y0: int
    y1: int
    x0: int
    x1: int
    core_y0: int
    core_y1: int
    core_x0: int
    core_x1: int


def generate_windows(shape: tuple[int, int], tile_size: int = 1024, overlap: int = 128):
    height, width = shape
    step = max(1, tile_size - overlap)
    for y0 in range(0, height, step):
        for x0 in range(0, width, step):
            y1 = min(height, y0 + tile_size)
            x1 = min(width, x0 + tile_size)
            core_y0 = y0 if y0 == 0 else min(y1, y0 + overlap // 2)
            core_x0 = x0 if x0 == 0 else min(x1, x0 + overlap // 2)
            core_y1 = y1 if y1 == height else max(core_y0, y1 - overlap // 2)
            core_x1 = x1 if x1 == width else max(core_x0, x1 - overlap // 2)
            yield TileWindow(y0, y1, x0, x1, core_y0, core_y1, core_x0, core_x1)


def _segment_tile(
    segmenter: Any,
    tile: np.ndarray,
    *,
    use_tta: bool,
    transforms: list[str] | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if use_tta:
        return segment_with_tta(segmenter, tile, transforms=transforms)
    return segmenter.segment(tile)


def segment_tiled(
    segmenter: Any,
    image: np.ndarray,
    *,
    tile_size: int = 1024,
    overlap: int = 128,
    use_tta: bool = False,
    transforms: list[str] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    canvas = np.zeros(image.shape[:2], dtype=np.uint32)
    fg_probability = np.zeros(image.shape[:2], dtype=np.float32) if use_tta else None
    next_id = 1
    first_info: dict[str, Any] = {}
    tile_count = 0

    for window in generate_windows(image.shape[:2], tile_size=tile_size, overlap=overlap):
        tile = image[window.y0:window.y1, window.x0:window.x1]
        labels, info = _segment_tile(segmenter, tile, use_tta=use_tta, transforms=transforms)
        if not first_info:
            first_info = dict(info)
        tile_count += 1

        core = labels[
            (window.core_y0 - window.y0):(window.core_y1 - window.y0),
            (window.core_x0 - window.x0):(window.core_x1 - window.x0),
        ]
        target = canvas[window.core_y0:window.core_y1, window.core_x0:window.core_x1]
        for label_id in np.unique(core):
            if int(label_id) == 0:
                continue
            target[core == label_id] = next_id
            next_id += 1

        if fg_probability is not None and info.get("foreground_probability") is not None:
            tile_prob = np.asarray(info["foreground_probability"], dtype=np.float32)
            fg_probability[window.core_y0:window.core_y1, window.core_x0:window.core_x1] = tile_prob[
                (window.core_y0 - window.y0):(window.core_y1 - window.y0),
                (window.core_x0 - window.x0):(window.core_x1 - window.x0),
            ]

    result_info = dict(first_info)
    merged_canvas, _ = connected_components(canvas > 0)
    result_info["tiling"] = True
    result_info["tile_count"] = tile_count
    result_info["tile_size"] = int(tile_size)
    result_info["tile_overlap"] = int(overlap)
    if fg_probability is not None:
        result_info["foreground_probability"] = fg_probability
    return merged_canvas.astype(np.uint32), result_info
