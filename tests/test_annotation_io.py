from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.annotate_roi_points import (
    build_points_metadata,
    read_existing_points,
    write_points_csv,
    write_points_metadata,
)


def test_point_csv_read_write_round_trip(tmp_path: Path):
    path = tmp_path / "manual_points.csv"
    points = np.asarray([[10, 20], [30, 40]], dtype=float)

    write_points_csv(path, points)
    loaded = read_existing_points(path)

    assert loaded.tolist() == points.tolist()
    frame = pd.read_csv(path)
    assert list(frame.columns) == ["x_px", "y_px"]


def test_sidecar_json_contents(tmp_path: Path):
    path = tmp_path / "manual_points.meta.json"
    payload = build_points_metadata(
        roi_id="ROI_001",
        annotator="tester",
        image_path="/tmp/image.tif",
        marker="RBPMS",
        modality="flatmount",
        roi_xywh=(1, 2, 512, 512),
        n_points=7,
    )

    write_points_metadata(path, payload)
    loaded = json.loads(path.read_text(encoding="utf-8"))

    assert loaded["roi_id"] == "ROI_001"
    assert loaded["annotator"] == "tester"
    assert loaded["marker"] == "RBPMS"
    assert loaded["n_points"] == 7
    assert loaded["roi_xywh"] == [1, 2, 512, 512]
