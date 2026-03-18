from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage as ndi
from skimage import exposure, feature, filters, measure, segmentation


def _normalize_float(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if arr.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.percentile(arr, [1.0, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32, copy=False)


def _apply_clahe_if_requested(image: np.ndarray, enabled: bool) -> np.ndarray:
    if not enabled:
        return image.astype(np.float32, copy=False)
    return exposure.equalize_adapthist(image.astype(np.float32, copy=False), clip_limit=0.02).astype(np.float32)


def segment_blob_watershed(
    image: np.ndarray,
    *,
    apply_clahe: bool = False,
    min_sigma: float = 2.0,
    max_sigma: float = 6.0,
    num_sigma: int = 5,
    threshold_rel: float = 0.15,
    min_distance: int = 6,
    min_size: int = 20,
    max_size: int = 400,
    min_mean_intensity: float | None = None,
    compactness: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    norm = _normalize_float(image)
    enhanced = _apply_clahe_if_requested(norm, apply_clahe)

    sigma_small = max(0.5, float(min_sigma))
    sigma_large = max(sigma_small + 0.5, float(max_sigma))
    response = filters.gaussian(enhanced, sigma=sigma_small, preserve_range=True) - filters.gaussian(
        enhanced,
        sigma=sigma_large,
        preserve_range=True,
    )
    response = np.clip(response, 0.0, None).astype(np.float32, copy=False)

    coords = feature.peak_local_max(
        response,
        min_distance=max(1, int(min_distance)),
        threshold_rel=float(threshold_rel),
        exclude_border=False,
    )
    markers = np.zeros_like(response, dtype=np.int32)
    if len(coords):
        markers[coords[:, 0], coords[:, 1]] = np.arange(1, len(coords) + 1, dtype=np.int32)
        markers = ndi.label(markers > 0)[0]

    if markers.max() == 0:
        return np.zeros_like(image, dtype=np.uint16), {
            "backend": "blob_watershed",
            "n_peaks": 0,
            "n_regions": 0,
            "foreground_probability": response.astype(np.float32, copy=False),
            "min_sigma": float(min_sigma),
            "max_sigma": float(max_sigma),
            "num_sigma": int(num_sigma),
            "threshold_rel": float(threshold_rel),
            "min_distance": int(min_distance),
        }

    finite_response = response[np.isfinite(response)]
    otsu_source = finite_response if finite_response.size else np.array([0.0], dtype=np.float32)
    threshold = float(filters.threshold_otsu(otsu_source))
    foreground = response > threshold
    labels = segmentation.watershed(-response, markers=markers, mask=foreground, compactness=float(compactness))

    filtered = np.zeros_like(labels, dtype=np.uint16)
    next_id = 1
    for region in measure.regionprops(labels, intensity_image=enhanced):
        area = int(region.area)
        if area < int(min_size) or area > int(max_size):
            continue
        if min_mean_intensity is not None and float(region.mean_intensity) < float(min_mean_intensity):
            continue
        filtered[labels == region.label] = next_id
        next_id += 1

    return filtered, {
        "backend": "blob_watershed",
        "n_peaks": int(len(coords)),
        "n_regions": int(next_id - 1),
        "foreground_probability": response.astype(np.float32, copy=False),
        "min_sigma": float(min_sigma),
        "max_sigma": float(max_sigma),
        "num_sigma": int(num_sigma),
        "threshold_rel": float(threshold_rel),
        "min_distance": int(min_distance),
    }
