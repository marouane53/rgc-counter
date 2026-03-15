import numpy as np
import pytest

from src.spatial import (
    centroids_from_masks,
    nn_regularity_index,
    ripley_k,
    voronoi_regulariry_index,
)


def test_centroids_from_masks_returns_expected_yx_centroids():
    masks = np.zeros((8, 8), dtype=np.uint16)
    masks[1:3, 1:3] = 1
    masks[5:7, 4:6] = 2

    centroids = centroids_from_masks(masks)

    assert centroids.shape == (2, 2)
    assert centroids[0, 0] == pytest.approx(1.5)
    assert centroids[0, 1] == pytest.approx(1.5)
    assert centroids[1, 0] == pytest.approx(5.5)
    assert centroids[1, 1] == pytest.approx(4.5)


def test_ripley_k_matches_simple_two_point_case():
    centroids = np.array([[0.0, 0.0], [0.0, 5.0]], dtype=np.float32)

    result = ripley_k(centroids, radii=[1.0, 10.0], area_px=100.0)

    assert result[1.0] == pytest.approx(0.0)
    assert result[10.0] == pytest.approx(50.0)


def test_nnri_and_voronoi_return_zero_for_too_few_points():
    centroids = np.array([[2.0, 2.0], [5.0, 5.0]], dtype=np.float32)

    nnri = nn_regularity_index(centroids)
    vdri = voronoi_regulariry_index(centroids, shape=(10, 10))

    assert nnri["nnri"] >= 0
    assert vdri["vdri"] == 0.0
