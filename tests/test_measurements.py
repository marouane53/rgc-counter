from pathlib import Path

import numpy as np

from src.measurements import build_object_table, object_table_path_for, write_object_table


def test_build_object_table_collects_geometry_and_focus_metrics():
    labels = np.zeros((8, 8), dtype=np.uint16)
    labels[1:3, 1:3] = 1
    labels[4:7, 4:7] = 2
    focus = np.ones((8, 8), dtype=bool)
    focus[5:, 5:] = False
    image = np.arange(64, dtype=np.uint16).reshape(8, 8)

    frame = build_object_table("sample.tif", labels, focus, gray_image=image, meta={"reader": "test"})

    assert list(frame["object_id"]) == [1, 2]
    assert list(frame["area_px"]) == [4, 9]
    assert frame.loc[0, "focus_overlap_fraction"] == 1.0
    assert frame.loc[1, "focus_overlap_px"] == 5
    assert frame.loc[0, "reader"] == "test"
    assert "geometry.circularity" in frame.columns
    assert "geometry.solidity" in frame.columns
    assert "intensity.local_contrast" in frame.columns


def test_write_object_table_creates_a_file(tmp_path: Path):
    labels = np.zeros((4, 4), dtype=np.uint16)
    labels[1:3, 1:3] = 1
    focus = np.ones((4, 4), dtype=bool)
    frame = build_object_table("sample.tif", labels, focus)

    destination = object_table_path_for(tmp_path, "sample.tif")
    written = write_object_table(frame, destination)

    assert written.exists()
    assert written.suffix in {".parquet", ".csv"}
