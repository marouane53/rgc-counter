from __future__ import annotations

import numpy as np

from src.point_detection import detect_dog_peaks, detect_hmax_peaks, detect_log_peaks, score_peak_candidates


def _synthetic_image() -> np.ndarray:
    yy, xx = np.mgrid[0:64, 0:64]
    image = (
        1.2 * np.exp(-((yy - 20.0) ** 2 + (xx - 20.0) ** 2) / (2.0 * 3.0**2))
        + 1.0 * np.exp(-((yy - 44.0) ** 2 + (xx - 42.0) ** 2) / (2.0 * 3.5**2))
    )
    return image.astype(np.float32)


def test_log_and_dog_detectors_find_soma_like_peaks():
    image = _synthetic_image()

    log_points = detect_log_peaks(image, sigma_min=2.5, sigma_max=5.0, num_sigma=4, threshold=0.08, min_distance=8)
    dog_points = detect_dog_peaks(image, sigma_min=2.5, sigma_max=5.0, threshold=0.06, min_distance=8)

    assert len(log_points) >= 2
    assert len(dog_points) >= 2
    assert {"y_px", "x_px", "score", "radius_px", "detector"} <= set(log_points.columns)


def test_hmax_detector_respects_exclude_mask():
    image = _synthetic_image()
    exclude_mask = np.zeros_like(image, dtype=bool)
    exclude_mask[15:28, 15:28] = True

    points = detect_hmax_peaks(image, h=0.08, min_distance=8, exclude_mask=exclude_mask)

    assert len(points) >= 1
    assert not ((points["x_px"].between(15, 27)) & (points["y_px"].between(15, 27))).any()


def test_score_peak_candidates_adds_score_column():
    image = _synthetic_image()
    scored = score_peak_candidates(image, [[20.0, 20.0], [44.0, 42.0]])

    assert list(scored.columns) == ["y_px", "x_px", "score", "radius_px", "detector"]
    assert scored["score"].max() > 0
