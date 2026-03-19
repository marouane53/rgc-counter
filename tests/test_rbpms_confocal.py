from __future__ import annotations

import numpy as np

from src.rbpms_confocal import (
    build_void_fold_mask,
    compute_slice_focus_scores,
    enumerate_candidate_slabs,
    normalize_for_detection,
    project_focus_weighted,
    project_percentile,
    project_slab,
    project_topk_mean,
    subtract_background,
)


def _synthetic_volume() -> np.ndarray:
    volume = np.zeros((5, 32, 32), dtype=np.uint16)
    volume[2, 8:24, 8:24] = 500
    volume[2, 12:20, 12:20] = 2000
    volume[1] = 100
    volume[3] = 100
    return volume


def test_compute_slice_focus_scores_prefers_sharp_slice():
    focus = compute_slice_focus_scores(_synthetic_volume())

    assert list(focus.columns) == ["z_index", "focus_score", "mean_intensity", "p99_intensity"]
    assert int(focus.sort_values("focus_score", ascending=False).iloc[0]["z_index"]) == 2


def test_enumerate_candidate_slabs_marks_best_per_window():
    catalog = enumerate_candidate_slabs(_synthetic_volume(), window_sizes=(2, 3), stride=1)

    assert any(int(row["window_size"]) == 2 and bool(row["is_best"]) for row in catalog)
    assert any(int(row["window_size"]) == 3 and bool(row["is_best"]) for row in catalog)


def test_projection_and_normalization_helpers_return_expected_shapes():
    volume = _synthetic_volume()
    focus = compute_slice_focus_scores(volume)

    full_max = project_slab(volume, 0, volume.shape[0], mode="max")
    topk = project_topk_mean(volume, 2)
    percentile = project_percentile(volume, 90)
    weighted = project_focus_weighted(volume, focus)

    assert full_max.shape == (32, 32)
    assert topk.shape == (32, 32)
    assert percentile.shape == (32, 32)
    assert weighted.shape == (32, 32)
    assert normalize_for_detection(full_max, mode="robust_float").dtype == np.float32
    assert normalize_for_detection(full_max, mode="percentile_uint16").dtype == np.uint16
    assert normalize_for_detection(full_max, mode="display_uint8").dtype == np.uint8


def test_background_subtraction_and_void_mask_behave_reasonably():
    image = np.full((64, 64), 400, dtype=np.uint16)
    image[20:40, 20:40] = 1200
    image[0:16, 0:16] = 0

    white_tophat = subtract_background(image, method="white_tophat", radius_px=6)
    rolling_ball = subtract_background(image, method="rolling_ball", radius_px=6)
    mask = build_void_fold_mask(image)

    assert white_tophat.shape == image.shape
    assert rolling_ball.shape == image.shape
    assert mask.shape == image.shape
    assert bool(mask[4, 4]) is True
