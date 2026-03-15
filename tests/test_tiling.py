import numpy as np
from scipy.ndimage import label as connected_components

from src.tiling import segment_tiled


class ConnectedComponentSegmenter:
    def segment(self, image: np.ndarray):
        labels, _ = connected_components(image > 0)
        return labels.astype(np.uint16), {"backend": "fake"}


def test_segment_tiled_merges_object_crossing_tile_boundary():
    image = np.zeros((32, 32), dtype=np.uint8)
    image[8:14, 10:20] = 255
    image[20:26, 20:26] = 255

    labels, info = segment_tiled(
        ConnectedComponentSegmenter(),
        image,
        tile_size=16,
        overlap=8,
        use_tta=False,
    )

    object_ids = np.unique(labels)
    object_ids = object_ids[object_ids != 0]
    assert len(object_ids) == 2
    assert info["tiling"] is True
