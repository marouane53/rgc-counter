from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
import pandas as pd
from skimage import feature, morphology

from src.rbpms_confocal import normalize_for_detection


def _as_2d_float(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D detector input image, got shape={tuple(int(v) for v in arr.shape)}")
    return normalize_for_detection(arr, mode="robust_float")


def _coerce_points(points: Any) -> pd.DataFrame:
    if isinstance(points, pd.DataFrame):
        frame = points.copy()
    else:
        arr = np.asarray(points, dtype=float).reshape(-1, 2)
        frame = pd.DataFrame({"y_px": arr[:, 0] if len(arr) else [], "x_px": arr[:, 1] if len(arr) else []})
    if frame.empty:
        return pd.DataFrame(columns=["y_px", "x_px", "score", "radius_px", "detector"])
    if "y_px" not in frame.columns or "x_px" not in frame.columns:
        raise ValueError("Point table must contain y_px and x_px columns.")
    return frame


def _exclude_mask_filter(frame: pd.DataFrame, exclude_mask: np.ndarray | None) -> pd.DataFrame:
    if exclude_mask is None or frame.empty:
        return frame
    mask = np.asarray(exclude_mask, dtype=bool)
    keep: list[bool] = []
    for row in frame.itertuples(index=False):
        y = int(np.clip(round(float(row.y_px)), 0, mask.shape[0] - 1))
        x = int(np.clip(round(float(row.x_px)), 0, mask.shape[1] - 1))
        keep.append(not bool(mask[y, x]))
    return frame.loc[np.asarray(keep, dtype=bool)].reset_index(drop=True)


def _suppress_min_distance(frame: pd.DataFrame, min_distance: float) -> pd.DataFrame:
    if frame.empty:
        return frame
    ordered = frame.sort_values(["score", "radius_px", "y_px", "x_px"], ascending=[False, False, True, True]).reset_index(drop=True)
    kept: list[pd.Series] = []
    radius = float(min_distance)
    for _, candidate in ordered.iterrows():
        yx = np.asarray([float(candidate["y_px"]), float(candidate["x_px"])], dtype=float)
        if any(float(np.linalg.norm(yx - np.asarray([float(row["y_px"]), float(row["x_px"])], dtype=float))) < radius for row in kept):
            continue
        kept.append(candidate)
    if not kept:
        return frame.iloc[0:0].copy()
    return pd.DataFrame(kept).reset_index(drop=True)


def score_peak_candidates(image, points) -> pd.DataFrame:
    arr = _as_2d_float(image)
    frame = _coerce_points(points)
    if frame.empty:
        return frame
    smooth = cv2.GaussianBlur(arr, (0, 0), sigmaX=1.0)
    scores: list[float] = []
    for row in frame.itertuples(index=False):
        y = int(np.clip(round(float(row.y_px)), 0, smooth.shape[0] - 1))
        x = int(np.clip(round(float(row.x_px)), 0, smooth.shape[1] - 1))
        scores.append(float(smooth[y, x]))
    frame = frame.copy()
    frame["score"] = scores
    if "radius_px" not in frame.columns:
        frame["radius_px"] = np.nan
    if "detector" not in frame.columns:
        frame["detector"] = "unknown"
    return frame.sort_values(["score", "y_px", "x_px"], ascending=[False, True, True]).reset_index(drop=True)


def detect_log_peaks(image, sigma_min, sigma_max, num_sigma, threshold, min_distance, exclude_mask=None):
    arr = _as_2d_float(image)
    blobs = feature.blob_log(
        arr,
        min_sigma=float(sigma_min),
        max_sigma=float(sigma_max),
        num_sigma=int(num_sigma),
        threshold=float(threshold),
    )
    frame = pd.DataFrame(
        {
            "y_px": blobs[:, 0] if len(blobs) else [],
            "x_px": blobs[:, 1] if len(blobs) else [],
            "radius_px": blobs[:, 2] * math.sqrt(2.0) if len(blobs) else [],
            "detector": "log",
        }
    )
    frame = score_peak_candidates(arr, frame)
    frame = _exclude_mask_filter(frame, exclude_mask)
    return _suppress_min_distance(frame, float(min_distance))


def detect_dog_peaks(image, sigma_min, sigma_max, threshold, min_distance, exclude_mask=None):
    arr = _as_2d_float(image)
    blobs = feature.blob_dog(
        arr,
        min_sigma=float(sigma_min),
        max_sigma=float(sigma_max),
        threshold=float(threshold),
    )
    frame = pd.DataFrame(
        {
            "y_px": blobs[:, 0] if len(blobs) else [],
            "x_px": blobs[:, 1] if len(blobs) else [],
            "radius_px": blobs[:, 2] * math.sqrt(2.0) if len(blobs) else [],
            "detector": "dog",
        }
    )
    frame = score_peak_candidates(arr, frame)
    frame = _exclude_mask_filter(frame, exclude_mask)
    return _suppress_min_distance(frame, float(min_distance))


def detect_hmax_peaks(image, h, min_distance, exclude_mask=None):
    arr = _as_2d_float(image)
    maxima_mask = morphology.h_maxima(arr, float(h))
    coords = feature.peak_local_max(arr, min_distance=int(min_distance), labels=maxima_mask.astype(np.uint8))
    frame = pd.DataFrame(
        {
            "y_px": coords[:, 0] if len(coords) else [],
            "x_px": coords[:, 1] if len(coords) else [],
            "radius_px": [float(min_distance) / 2.0] * int(len(coords)),
            "detector": "hmax",
        }
    )
    frame = score_peak_candidates(arr, frame)
    frame = _exclude_mask_filter(frame, exclude_mask)
    return _suppress_min_distance(frame, float(min_distance))
