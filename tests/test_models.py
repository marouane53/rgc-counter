import numpy as np

from src.blob_watershed import segment_blob_watershed


def _spot_image() -> np.ndarray:
    image = np.zeros((64, 64), dtype=np.float32)
    yy, xx = np.ogrid[:64, :64]
    image[(yy - 18) ** 2 + (xx - 18) ** 2 <= 16] = 1.0
    image[(yy - 42) ** 2 + (xx - 42) ** 2 <= 25] = 1.0
    return image


def test_blob_watershed_detects_separated_spots():
    labels, info = segment_blob_watershed(
        _spot_image(),
        threshold_rel=0.05,
        min_distance=8,
        min_size=10,
        max_size=400,
    )

    assert int(labels.max()) >= 2
    assert info["backend"] == "blob_watershed"


def test_blob_watershed_respects_size_filter():
    image = np.zeros((64, 64), dtype=np.float32)
    yy, xx = np.ogrid[:64, :64]
    image[(yy - 16) ** 2 + (xx - 16) ** 2 <= 2] = 1.0
    image[(yy - 40) ** 2 + (xx - 40) ** 2 <= 36] = 1.0

    labels, _ = segment_blob_watershed(
        image,
        threshold_rel=0.05,
        min_distance=6,
        min_size=20,
        max_size=400,
    )

    assert int(labels.max()) == 1
