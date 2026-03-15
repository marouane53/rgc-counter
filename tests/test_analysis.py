import numpy as np
import pytest

from src.analysis import compute_cell_count_and_density
from src.config import MICRONS_PER_PIXEL


def test_compute_cell_count_and_density_uses_in_focus_area_only():
    masks = np.zeros((10, 10), dtype=np.uint16)
    masks[1:3, 1:3] = 1
    masks[6:8, 6:8] = 2

    in_focus_mask = np.zeros((10, 10), dtype=bool)
    in_focus_mask[:5, :] = True

    count, area_mm2, density = compute_cell_count_and_density(masks, in_focus_mask)

    expected_area = float(in_focus_mask.sum()) * (MICRONS_PER_PIXEL ** 2) / 1e6
    assert count == 2
    assert area_mm2 == pytest.approx(expected_area)
    assert density == pytest.approx(2 / expected_area)


def test_compute_cell_count_and_density_returns_zero_density_for_zero_area():
    masks = np.zeros((6, 6), dtype=np.uint16)
    masks[1:3, 1:3] = 1
    in_focus_mask = np.zeros((6, 6), dtype=bool)

    count, area_mm2, density = compute_cell_count_and_density(masks, in_focus_mask)

    assert count == 1
    assert area_mm2 == 0
    assert density == 0
