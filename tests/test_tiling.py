from pathlib import Path

import numpy as np
from scipy.ndimage import label as connected_components
from tifffile import imread

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
    assert info["stitching"] == "overlap_unionfind"
    assert info["matched_overlap_pairs"] >= 1


def test_segment_tiled_merges_object_crossing_vertical_and_corner_boundaries():
    image = np.zeros((32, 32), dtype=np.uint8)
    image[10:22, 14:18] = 255
    image[14:18, 10:22] = 255

    labels, info = segment_tiled(
        ConnectedComponentSegmenter(),
        image,
        tile_size=16,
        overlap=8,
        use_tta=False,
    )

    object_ids = np.unique(labels)
    object_ids = object_ids[object_ids != 0]
    assert len(object_ids) == 1
    assert info["matched_overlap_pairs"] >= 2


def test_segment_tiled_preserves_tta_foreground_probability_map():
    image = np.zeros((32, 32), dtype=np.uint8)
    image[8:14, 10:20] = 255
    image[20:26, 20:26] = 255

    labels, info = segment_tiled(
        ConnectedComponentSegmenter(),
        image,
        tile_size=16,
        overlap=8,
        use_tta=True,
        transforms=["flip_h", "flip_v"],
    )

    fg_probability = info.get("foreground_probability")
    assert fg_probability is not None
    assert fg_probability.shape == image.shape
    assert fg_probability.dtype == np.float32
    assert float(fg_probability[labels > 0].mean()) >= 0.5


def test_segment_tiled_matches_full_frame_count_on_tracked_smoke_fixture():
    image = imread(Path("/Users/marouane/Documents/code/rgc-counter/examples/smoke_data/example_retina_a.tif"))
    segmenter = ConnectedComponentSegmenter()

    full_frame_labels, _ = segmenter.segment((image > 100).astype(np.uint8))
    tiled_labels, info = segment_tiled(
        segmenter,
        (image > 100).astype(np.uint8),
        tile_size=24,
        overlap=8,
        use_tta=False,
    )

    full_count = int(np.max(full_frame_labels))
    tiled_count = int(np.max(tiled_labels))
    assert full_count == tiled_count
    assert info["stitched_object_count"] == tiled_count
