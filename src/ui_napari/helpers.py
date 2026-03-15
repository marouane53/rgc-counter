from __future__ import annotations

from typing import Any

import numpy as np


def parse_optional_float(text: str) -> float | None:
    value = text.strip()
    if not value:
        return None
    return float(value)


def parse_optional_int(text: str) -> int | None:
    value = text.strip()
    if not value:
        return None
    return int(float(value))


def parse_xy_text(text: str) -> tuple[float, float] | None:
    value = text.strip()
    if not value:
        return None
    parts = [item.strip() for item in value.split(",")]
    if len(parts) != 2:
        raise ValueError("Expected coordinates in 'x, y' format.")
    return float(parts[0]), float(parts[1])


def format_xy_text(xy: tuple[float, float] | None) -> str:
    if xy is None:
        return ""
    return f"{xy[0]:.1f}, {xy[1]:.1f}"


def landmarks_from_points(points: Any) -> dict[str, tuple[float, float]]:
    coords = np.asarray(points, dtype=float)
    if coords.ndim != 2 or coords.shape[1] < 2 or coords.shape[0] < 2:
        raise ValueError("Need at least two points with (y, x) coordinates.")
    onh_y, onh_x = coords[0, :2]
    dorsal_y, dorsal_x = coords[1, :2]
    return {
        "onh_xy": (float(onh_x), float(onh_y)),
        "dorsal_xy": (float(dorsal_x), float(dorsal_y)),
    }
