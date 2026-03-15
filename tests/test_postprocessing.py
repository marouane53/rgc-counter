import numpy as np

from src.postprocessing import apply_clahe, filter_masks_by_size


def test_filter_masks_by_size_keeps_only_requested_objects_and_reindexes():
    masks = np.zeros((8, 8), dtype=np.uint16)
    masks[0, 0] = 1
    masks[2:4, 2:4] = 2
    masks[5:8, 5:8] = 3

    filtered = filter_masks_by_size(masks, min_size=2, max_size=5)

    assert set(np.unique(filtered)) == {0, 1}
    assert int((filtered == 1).sum()) == 4


def test_apply_clahe_returns_uint8_with_same_shape():
    image = (np.arange(64, dtype=np.uint16).reshape(8, 8) * 10).astype(np.uint16)

    enhanced = apply_clahe(image)

    assert enhanced.shape == image.shape
    assert enhanced.dtype == np.uint8
