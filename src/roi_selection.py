from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_SPLIT_TARGETS: "OrderedDict[str, int]" = OrderedDict(
    [("dev", 17), ("locked_eval", 6), ("qc_or_exclude", 5)]
)


def parse_split_targets(value: str | None) -> OrderedDict[str, int]:
    if value is None or not str(value).strip():
        return OrderedDict(DEFAULT_SPLIT_TARGETS)
    parsed: "OrderedDict[str, int]" = OrderedDict()
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"Invalid split target token: {token}")
        split, raw_count = token.split("=", 1)
        parsed[split.strip()] = int(raw_count)
    if not parsed:
        raise ValueError("Split target specification was empty.")
    return parsed


def remaining_split_targets(current_counts: dict[str, int], target_counts: dict[str, int]) -> OrderedDict[str, int]:
    remaining: "OrderedDict[str, int]" = OrderedDict()
    for split, target in target_counts.items():
        remaining[split] = max(int(target) - int(current_counts.get(split, 0)), 0)
    return remaining


def normalize_rectangle_to_fixed_roi(
    rectangle_yx: np.ndarray,
    *,
    image_shape: tuple[int, int],
    roi_size: int,
) -> tuple[int, int, int, int]:
    coords = np.asarray(rectangle_yx, dtype=float)
    ys = coords[:, 0]
    xs = coords[:, 1]
    center_y = float(ys.min() + ys.max()) / 2.0
    center_x = float(xs.min() + xs.max()) / 2.0
    half = int(roi_size // 2)
    y0 = int(round(center_y)) - half
    x0 = int(round(center_x)) - half
    max_y0 = max(int(image_shape[0]) - int(roi_size), 0)
    max_x0 = max(int(image_shape[1]) - int(roi_size), 0)
    y0 = max(0, min(y0, max_y0))
    x0 = max(0, min(x0, max_x0))
    return x0, y0, int(roi_size), int(roi_size)


def select_fixed_rois_napari(
    image: np.ndarray,
    *,
    roi_size: int,
    split_targets: dict[str, int] | None = None,
    title: str | None = None,
) -> list[dict[str, Any]]:
    import napari

    targets = OrderedDict(split_targets or DEFAULT_SPLIT_TARGETS)
    colors = {
        "dev": "lime",
        "locked_eval": "orange",
        "qc_or_exclude": "red",
    }
    collected: list[dict[str, Any]] = []
    with napari.gui_qt():
        viewer = napari.Viewer(title=title or "Fixed ROI selector")
        viewer.add_image(image, name="projected")
        viewer.text_overlay.visible = True
        viewer.text_overlay.text = (
            "Draw rough rectangles in the split-specific layers.\n"
            f"Each rectangle is converted to a clipped fixed {int(roi_size)}x{int(roi_size)} ROI.\n"
            "Press 's' to save and close or 'q' to close."
        )

        layers: dict[str, Any] = {}
        for split, target in targets.items():
            layers[split] = viewer.add_shapes(
                name=f"{split} (target {int(target)})",
                shape_type="rectangle",
                edge_color=colors.get(split, "yellow"),
                face_color=colors.get(split, "transparent"),
                opacity=0.35,
            )

        @viewer.bind_key("s")
        def _save_and_close(v):
            for split, layer in layers.items():
                for rectangle in layer.data:
                    x0, y0, width, height = normalize_rectangle_to_fixed_roi(
                        np.asarray(rectangle),
                        image_shape=(int(image.shape[0]), int(image.shape[1])),
                        roi_size=int(roi_size),
                    )
                    collected.append(
                        {
                            "split": split,
                            "x0": x0,
                            "y0": y0,
                            "width": width,
                            "height": height,
                        }
                    )
            v.close()

        @viewer.bind_key("q")
        def _close_without_collecting(v):
            v.close()

    return collected


def roi_row(
    *,
    roi_id: str,
    image_path: str | Path,
    split: str,
    x0: int,
    y0: int,
    width: int,
    height: int,
    annotator: str,
    marker: str = "RBPMS",
    modality: str = "flatmount",
) -> dict[str, Any]:
    manual_points_path = ""
    notes = ""
    if split in {"dev", "locked_eval"}:
        manual_points_path = str(Path("manual_points") / f"{roi_id}.csv")
    if split == "qc_or_exclude":
        notes = "qc_or_exclude"
    return {
        "roi_id": roi_id,
        "image_path": str(image_path),
        "marker": marker,
        "modality": modality,
        "x0": int(x0),
        "y0": int(y0),
        "width": int(width),
        "height": int(height),
        "annotator": annotator,
        "manual_points_path": manual_points_path,
        "split": split,
        "notes": notes,
    }
