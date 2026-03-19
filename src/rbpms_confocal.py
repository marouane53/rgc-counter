from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np
import pandas as pd
from skimage import morphology, restoration


def _as_float32(image: np.ndarray) -> np.ndarray:
    return np.asarray(image, dtype=np.float32)


def _robust_bounds(image: np.ndarray, *, low_q: float = 1.0, high_q: float = 99.5) -> tuple[float, float]:
    arr = _as_float32(image)
    if arr.size == 0:
        return 0.0, 1.0
    low = float(np.percentile(arr, low_q))
    high = float(np.percentile(arr, high_q))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(arr.min())
        high = float(arr.max()) if arr.size else low + 1.0
    if not np.isfinite(high) or high <= low:
        high = low + 1.0
    return low, high


def compute_slice_focus_scores(volume) -> pd.DataFrame:
    arr = np.asarray(volume)
    if arr.ndim != 3:
        raise ValueError(f"Expected a [Z, Y, X] volume, got shape={tuple(int(v) for v in arr.shape)}")
    rows: list[dict[str, float]] = []
    for z_index in range(int(arr.shape[0])):
        plane = _as_float32(arr[z_index])
        low, high = _robust_bounds(plane)
        robust = np.clip((plane - low) / max(high - low, 1e-6), 0.0, 1.0)
        gx = cv2.Sobel(robust, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(robust, cv2.CV_32F, 0, 1, ksize=3)
        tenengrad = float(np.mean((gx * gx) + (gy * gy)))
        rows.append(
            {
                "z_index": int(z_index),
                "focus_score": tenengrad,
                "mean_intensity": float(np.mean(plane)),
                "p99_intensity": float(np.percentile(plane, 99.0)),
            }
        )
    return pd.DataFrame(rows)


def enumerate_candidate_slabs(volume, window_sizes=(24, 32, 48, 64), stride=8) -> list[dict]:
    arr = np.asarray(volume)
    focus = compute_slice_focus_scores(arr)
    z_count = int(arr.shape[0])
    rows: list[dict[str, float | int | bool]] = []
    for raw_window in window_sizes:
        window = min(int(raw_window), z_count)
        if window <= 0:
            continue
        starts = [0] if window >= z_count else list(range(0, z_count - window + 1, int(stride)))
        if window < z_count and starts[-1] != z_count - window:
            starts.append(z_count - window)
        for start in starts:
            end = int(start + window)
            slab_scores = focus.loc[(focus["z_index"] >= start) & (focus["z_index"] < end), "focus_score"].to_numpy(dtype=float)
            rows.append(
                {
                    "window_size": int(window),
                    "start": int(start),
                    "end": int(end),
                    "mean_focus_score": float(np.mean(slab_scores)) if slab_scores.size else float("nan"),
                    "max_focus_score": float(np.max(slab_scores)) if slab_scores.size else float("nan"),
                }
            )
    catalog = pd.DataFrame(rows)
    if catalog.empty:
        return []
    catalog = catalog.sort_values(["window_size", "mean_focus_score", "max_focus_score", "start"], ascending=[True, False, False, True]).reset_index(drop=True)
    catalog["rank_within_window"] = (
        catalog.groupby("window_size", sort=False)["mean_focus_score"].rank(method="first", ascending=False).astype(int)
    )
    catalog["is_best"] = catalog["rank_within_window"] == 1
    return catalog.to_dict("records")


def project_slab(volume, start, end, mode="max") -> np.ndarray:
    arr = np.asarray(volume)
    if arr.ndim != 3:
        raise ValueError(f"Expected a [Z, Y, X] volume, got shape={tuple(int(v) for v in arr.shape)}")
    z0 = max(int(start), 0)
    z1 = min(int(end), int(arr.shape[0]))
    if z0 >= z1:
        raise ValueError(f"Invalid slab range [{z0}, {z1}) for shape={tuple(int(v) for v in arr.shape)}")
    slab = arr[z0:z1]
    if mode == "mean":
        return np.mean(slab, axis=0, dtype=np.float32)
    if mode == "sum":
        return np.sum(slab, axis=0, dtype=np.float32)
    if mode != "max":
        raise ValueError(f"Unsupported projection mode: {mode}")
    return np.max(slab, axis=0)


def project_topk_mean(volume, k: int) -> np.ndarray:
    arr = np.asarray(volume)
    if arr.ndim != 3:
        raise ValueError(f"Expected a [Z, Y, X] volume, got shape={tuple(int(v) for v in arr.shape)}")
    focus = compute_slice_focus_scores(arr)
    topk = max(1, min(int(k), int(arr.shape[0])))
    keep = focus.sort_values(["focus_score", "z_index"], ascending=[False, True]).head(topk)["z_index"].to_numpy(dtype=int)
    return np.mean(arr[np.sort(keep)], axis=0, dtype=np.float32)


def project_percentile(volume, q: float) -> np.ndarray:
    arr = np.asarray(volume)
    if arr.ndim != 3:
        raise ValueError(f"Expected a [Z, Y, X] volume, got shape={tuple(int(v) for v in arr.shape)}")
    return np.percentile(arr, float(q), axis=0).astype(np.float32)


def project_focus_weighted(volume, scores) -> np.ndarray:
    arr = np.asarray(volume)
    if arr.ndim != 3:
        raise ValueError(f"Expected a [Z, Y, X] volume, got shape={tuple(int(v) for v in arr.shape)}")
    if isinstance(scores, pd.DataFrame):
        weight_values = scores.sort_values("z_index")["focus_score"].to_numpy(dtype=np.float32)
    else:
        weight_values = np.asarray(list(scores), dtype=np.float32)
    if weight_values.shape[0] != arr.shape[0]:
        raise ValueError("Focus-weight array length must match the Z dimension.")
    weights = np.clip(weight_values, 0.0, None)
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
        weights = np.ones_like(weights, dtype=np.float32)
    weights = weights / float(weights.sum())
    return np.tensordot(weights, _as_float32(arr), axes=(0, 0)).astype(np.float32)


def subtract_background(image, method="white_tophat", radius_px=12) -> np.ndarray:
    arr = _as_float32(image)
    radius = max(int(radius_px), 1)
    if method == "white_tophat":
        footprint = morphology.disk(radius)
        return morphology.white_tophat(arr, footprint=footprint).astype(np.float32)
    if method == "rolling_ball":
        background = restoration.rolling_ball(arr, radius=radius)
        return np.clip(arr - _as_float32(background), a_min=0.0, a_max=None).astype(np.float32)
    raise ValueError(f"Unsupported background subtraction method: {method}")


def normalize_for_detection(image, mode="robust_float") -> np.ndarray:
    arr = _as_float32(image)
    low, high = _robust_bounds(arr)
    scaled = np.clip((arr - low) / max(high - low, 1e-6), 0.0, 1.0)
    if mode == "robust_float":
        return scaled.astype(np.float32)
    if mode == "percentile_uint16":
        return np.round(scaled * 65535.0).astype(np.uint16)
    if mode == "display_uint8":
        return np.round(scaled * 255.0).astype(np.uint8)
    raise ValueError(f"Unsupported normalization mode: {mode}")


def build_void_fold_mask(image) -> np.ndarray:
    arr = normalize_for_detection(image, mode="robust_float")
    threshold = max(0.05, float(np.percentile(arr, 8.0)))
    mask = arr <= threshold
    mask = morphology.binary_closing(mask, morphology.disk(3))
    mask = morphology.binary_opening(mask, morphology.disk(2))
    mask = morphology.remove_small_objects(mask.astype(bool), min_size=max(256, int(arr.size * 0.01)))
    mask = morphology.remove_small_holes(mask, area_threshold=max(256, int(arr.size * 0.01)))
    return np.asarray(mask, dtype=bool)
