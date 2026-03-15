from pathlib import Path

import numpy as np

from src.context import RunContext
from src.pipeline import build_default_pipeline


class FakeSegmenter:
    def __init__(self, labels: np.ndarray):
        self._labels = labels

    def segment(self, image: np.ndarray):
        return self._labels.copy(), {"backend": "fake"}


def test_pipeline_supports_auto_hole_registration_and_review_replay(tmp_path: Path):
    image = np.zeros((64, 64), dtype=np.uint8)
    yy, xx = np.mgrid[:64, :64]
    dist = np.sqrt((xx - 32) ** 2 + (yy - 32) ** 2)
    image[(dist <= 24) & (dist >= 6)] = 220

    labels = np.zeros((64, 64), dtype=np.uint16)
    labels[20:24, 40:44] = 1
    labels[40:44, 20:24] = 2

    edit_path = tmp_path / "sample.tif.edits.json"
    edit_path.write_text(
        '{"edits":[{"op":"delete_object","object_id":2},{"op":"set_landmarks","onh_xy":[32,32],"dorsal_xy":[32,8]}]}',
        encoding="utf-8",
    )

    ctx = RunContext(path=Path(tmp_path / "sample.tif"), image=image, meta={"reader": "test"})
    pipeline = build_default_pipeline(FakeSegmenter(labels))
    cfg = {
        "apply_clahe": False,
        "focus_mode": "none",
        "tta": False,
        "tta_transforms": None,
        "tiling": False,
        "tile_size": 32,
        "tile_overlap": 8,
        "min_size": 1,
        "max_size": 1000,
        "qc_config": {},
        "spatial_stats": False,
        "backend": "fake",
        "use_gpu": False,
        "register_retina": True,
        "region_schema": "mouse_flatmount_v1",
        "onh_mode": "auto_combined",
        "onh_xy": None,
        "dorsal_xy": None,
        "retina_frame_path": None,
        "apply_edits": str(edit_path),
        "phenotype_engine": "legacy",
        "marker_metrics": False,
        "interaction_metrics": False,
    }

    out = pipeline.run(ctx, cfg)

    assert out.object_table is not None
    assert len(out.object_table) == 1
    assert out.region_table is not None
    assert out.metrics["retina_registration"]["onh_confidence"] > 0.0
