# src/qc.py

from __future__ import annotations
import numpy as np
import cv2
from typing import Dict, Tuple


def _tenengrad(image: np.ndarray) -> float:
    """Tenengrad (Sobel) focus measure: mean gradient magnitude squared."""
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    g2 = gx * gx + gy * gy
    return float(np.mean(g2))


def _highfreq_energy(image: np.ndarray) -> float:
    """High-frequency energy via FFT magnitude beyond a low cutoff."""
    f = np.fft.rfft2(image.astype(np.float32))
    mag = np.abs(f)
    # Drop DC row/col and very low frequencies
    h, w = image.shape
    # Half of min dimension as naive band cutoff
    cutoff = max(4, min(h, w) // 32)
    # Build a mask that zeros out low frequencies on both axes
    hf = mag.copy()
    hf[:cutoff, :] = 0
    return float(np.mean(hf))


def focus_mask_multimetric(image: np.ndarray,
                           tile_size: int = 64,
                           brightness_min: float = 20,
                           brightness_max: float = 230,
                           weights: Dict[str, float] | None = None,
                           threshold_z: float = 0.0,
                           morph_kernel: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute an in-focus mask by combining multiple normalized focus metrics per tile.
    Returns (mask_bool, focus_score_map_float32).
    """
    if image.dtype != np.float32:
        img = image.astype(np.float32)
    else:
        img = image

    h, w = img.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    score_map = np.zeros((h, w), dtype=np.float32)

    weights = weights or {"lap": 1.0, "ten": 1.0, "hf": 1.0}
    lap_scores, ten_scores, hf_scores, means = [], [], [], []

    # First pass: compute raw metrics per tile
    tiles = []
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            y2, x2 = min(y + tile_size, h), min(x + tile_size, w)
            tile = img[y:y2, x:x2]
            m = float(tile.mean())
            means.append(m)
            lap = float(cv2.Laplacian(tile, cv2.CV_32F).var())
            ten = _tenengrad(tile)
            hf = _highfreq_energy(tile)
            lap_scores.append(lap)
            ten_scores.append(ten)
            hf_scores.append(hf)
            tiles.append((y, y2, x, x2))

    means = np.array(means, dtype=np.float32)
    lap_scores = np.array(lap_scores, dtype=np.float32)
    ten_scores = np.array(ten_scores, dtype=np.float32)
    hf_scores = np.array(hf_scores, dtype=np.float32)

    # Z-score normalization
    def z(x):
        std = x.std() if x.std() > 1e-12 else 1.0
        return (x - x.mean()) / std

    lap_z, ten_z, hf_z = z(lap_scores), z(ten_scores), z(hf_scores)

    # Combine with weights
    for i, (y, y2, x, x2) in enumerate(tiles):
        if means[i] < brightness_min or means[i] > brightness_max:
            # leave as out-of-focus
            continue
        combined = (weights.get("lap", 1.0) * lap_z[i] +
                    weights.get("ten", 1.0) * ten_z[i] +
                    weights.get("hf", 1.0) * hf_z[i])
        score_map[y:y2, x:x2] = combined
        if combined >= threshold_z:
            mask[y:y2, x:x2] = 1

    # Morphological cleanup
    if morph_kernel and morph_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask.astype(bool), score_map

