from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes, binary_opening
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops


def build_tissue_mask(gray: np.ndarray) -> np.ndarray:
    image = np.asarray(gray, dtype=np.float32)
    if image.size == 0:
        return np.zeros_like(image, dtype=bool)
    threshold = threshold_otsu(image)
    mask = image > threshold
    mask = binary_opening(mask, structure=np.ones((5, 5)))
    mask = binary_closing(mask, structure=np.ones((9, 9)))
    return mask.astype(bool)


def detect_onh_hole(gray: np.ndarray) -> tuple[tuple[float, float] | None, dict[str, Any]]:
    tissue_mask = build_tissue_mask(gray)
    filled = binary_fill_holes(tissue_mask)
    holes = filled & ~tissue_mask

    props = regionprops(label(holes.astype(np.uint8)))
    if not props:
        return None, {"method": "auto_hole", "confidence": 0.0, "reason": "no_hole_found"}

    def score(region) -> float:
        perimeter = max(float(region.perimeter), 1.0)
        circularity = 4.0 * np.pi * float(region.area) / (perimeter * perimeter)
        return float(region.area) * float(circularity)

    best = max(props, key=score)
    y, x = best.centroid
    confidence = min(1.0, score(best) / max(float(gray.shape[0] * gray.shape[1]) * 0.01, 1.0))
    return (float(x), float(y)), {"method": "auto_hole", "confidence": float(confidence)}


def mask_coverage_fraction(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    return float(mask.sum() / mask.size)


def contours_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours
