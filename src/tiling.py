from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.uncertainty import segment_with_tta


MIN_STITCH_OVERLAP_PX = 8
MIN_STITCH_IOU = 0.10
MIN_STITCH_OVERLAP_FRACTION = 0.25


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


@dataclass
class TileRecord:
    window: TileWindow
    labels: np.ndarray
    local_to_global: dict[int, int]


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[int, int] = {}

    def find(self, value: int) -> int:
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


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


def _next_global_mapping(labels: np.ndarray, next_id: int) -> tuple[dict[int, int], int]:
    mapping: dict[int, int] = {}
    for local_id in (int(value) for value in np.unique(labels) if int(value) != 0):
        mapping[local_id] = next_id
        next_id += 1
    return mapping, next_id


def _intersection_window(left: TileWindow, right: TileWindow) -> tuple[int, int, int, int] | None:
    y0 = max(left.y0, right.y0)
    y1 = min(left.y1, right.y1)
    x0 = max(left.x0, right.x0)
    x1 = min(left.x1, right.x1)
    if y1 <= y0 or x1 <= x0:
        return None
    return y0, y1, x0, x1


def _extract_overlap_pairs(
    left_crop: np.ndarray,
    right_crop: np.ndarray,
    left_mapping: dict[int, int],
    right_mapping: dict[int, int],
    *,
    min_overlap_px: int = MIN_STITCH_OVERLAP_PX,
    min_iou: float = MIN_STITCH_IOU,
    min_overlap_fraction: float = MIN_STITCH_OVERLAP_FRACTION,
) -> list[tuple[int, int]]:
    overlap_mask = (left_crop > 0) & (right_crop > 0)
    if not np.any(overlap_mask):
        return []

    left_positive = left_crop[left_crop > 0]
    right_positive = right_crop[right_crop > 0]
    left_ids, left_counts = np.unique(left_positive.astype(np.int64), return_counts=True)
    right_ids, right_counts = np.unique(right_positive.astype(np.int64), return_counts=True)
    left_area = {int(label_id): int(count) for label_id, count in zip(left_ids, left_counts)}
    right_area = {int(label_id): int(count) for label_id, count in zip(right_ids, right_counts)}

    stacked = np.stack(
        [
            left_crop[overlap_mask].astype(np.int64, copy=False),
            right_crop[overlap_mask].astype(np.int64, copy=False),
        ],
        axis=1,
    )
    pair_ids, pair_counts = np.unique(stacked, axis=0, return_counts=True)

    matches: list[tuple[int, int]] = []
    for (left_id, right_id), overlap_px in zip(pair_ids, pair_counts):
        left_id = int(left_id)
        right_id = int(right_id)
        overlap_px = int(overlap_px)
        if overlap_px < min_overlap_px:
            continue

        left_count = left_area.get(left_id, 0)
        right_count = right_area.get(right_id, 0)
        if left_count <= 0 or right_count <= 0:
            continue

        union = left_count + right_count - overlap_px
        iou = overlap_px / max(union, 1)
        overlap_fraction = overlap_px / max(min(left_count, right_count), 1)
        if iou >= min_iou or overlap_fraction >= min_overlap_fraction:
            matches.append((left_mapping[left_id], right_mapping[right_id]))
    return matches


def _paint_core_with_roots(canvas: np.ndarray, record: TileRecord, uf: UnionFind) -> None:
    window = record.window
    core = record.labels[
        (window.core_y0 - window.y0):(window.core_y1 - window.y0),
        (window.core_x0 - window.x0):(window.core_x1 - window.x0),
    ]
    target = canvas[window.core_y0:window.core_y1, window.core_x0:window.core_x1]
    for local_id in (int(value) for value in np.unique(core) if int(value) != 0):
        root = uf.find(record.local_to_global[local_id])
        target[core == local_id] = root


def _relabel_sequential(labels: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    out = np.zeros_like(labels, dtype=np.uint32)
    mapping: dict[int, int] = {}
    next_id = 1
    for old_id in (int(value) for value in np.unique(labels) if int(value) != 0):
        mapping[old_id] = next_id
        out[labels == old_id] = next_id
        next_id += 1
    return out, mapping


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
    first_info: dict[str, Any] = {}
    tile_records: list[TileRecord] = []
    next_global_id = 1
    uf = UnionFind()

    for window in generate_windows(image.shape[:2], tile_size=tile_size, overlap=overlap):
        tile = image[window.y0:window.y1, window.x0:window.x1]
        labels, info = _segment_tile(segmenter, tile, use_tta=use_tta, transforms=transforms)
        if not first_info:
            first_info = dict(info)

        local_to_global, next_global_id = _next_global_mapping(labels, next_global_id)
        for global_id in local_to_global.values():
            uf.find(global_id)
        tile_records.append(TileRecord(window=window, labels=np.asarray(labels, dtype=np.uint32), local_to_global=local_to_global))

        if fg_probability is not None and info.get("foreground_probability") is not None:
            tile_prob = np.asarray(info["foreground_probability"], dtype=np.float32)
            fg_probability[window.core_y0:window.core_y1, window.core_x0:window.core_x1] = tile_prob[
                (window.core_y0 - window.y0):(window.core_y1 - window.y0),
                (window.core_x0 - window.x0):(window.core_x1 - window.x0),
            ]

    matched_pair_count = 0
    for index, left_record in enumerate(tile_records):
        for right_record in tile_records[index + 1 :]:
            bounds = _intersection_window(left_record.window, right_record.window)
            if bounds is None:
                continue
            y0, y1, x0, x1 = bounds
            left_crop = left_record.labels[y0 - left_record.window.y0 : y1 - left_record.window.y0, x0 - left_record.window.x0 : x1 - left_record.window.x0]
            right_crop = right_record.labels[y0 - right_record.window.y0 : y1 - right_record.window.y0, x0 - right_record.window.x0 : x1 - right_record.window.x0]
            for left_id, right_id in _extract_overlap_pairs(
                left_crop,
                right_crop,
                left_record.local_to_global,
                right_record.local_to_global,
            ):
                uf.union(left_id, right_id)
                matched_pair_count += 1

    for record in tile_records:
        _paint_core_with_roots(canvas, record, uf)

    stitched_canvas, root_mapping = _relabel_sequential(canvas)
    result_info = dict(first_info)
    result_info["tiling"] = True
    result_info["tile_count"] = len(tile_records)
    result_info["tile_size"] = int(tile_size)
    result_info["tile_overlap"] = int(overlap)
    result_info["stitching"] = "overlap_unionfind"
    result_info["matched_overlap_pairs"] = int(matched_pair_count)
    result_info["stitch_overlap_min_px"] = int(MIN_STITCH_OVERLAP_PX)
    result_info["stitch_overlap_min_iou"] = float(MIN_STITCH_IOU)
    result_info["stitched_object_count"] = int(len(root_mapping))
    if fg_probability is not None:
        result_info["foreground_probability"] = fg_probability
    return stitched_canvas.astype(np.uint32), result_info
