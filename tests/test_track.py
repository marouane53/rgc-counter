from pathlib import Path

import numpy as np
import pandas as pd

from src.context import RunContext
from src.track import build_longitudinal_track_table, summarize_tracks, track_points


def test_track_points_matches_nearest_neighbors():
    prev_xy = np.array([[10.0, 10.0], [30.0, 30.0]])
    next_xy = np.array([[12.0, 11.0], [31.0, 32.0]])

    matches = track_points(prev_xy, next_xy, max_disp_px=5.0)

    assert matches.shape == (2, 3)
    assert list(matches[:, 0].astype(int)) == [0, 1]
    assert list(matches[:, 1].astype(int)) == [0, 1]


def test_build_longitudinal_track_table_propagates_track_ids():
    manifest = pd.DataFrame(
        {
            "sample_id": ["A0", "A7"],
            "animal_id": ["M1", "M1"],
            "eye": ["OD", "OD"],
            "timepoint_dpi": [0, 7],
            "modality": ["flatmount", "flatmount"],
            "condition": ["control", "control"],
            "genotype": ["WT", "WT"],
            "stain_panel": ["RBPMS", "RBPMS"],
            "path": ["a0.tif", "a7.tif"],
        }
    )

    ctx0 = RunContext(path=Path("a0.tif"), image=np.zeros((8, 8), dtype=np.uint16), meta={})
    ctx0.object_table = pd.DataFrame(
        {
            "filename": ["a0.tif", "a0.tif"],
            "object_id": [1, 2],
            "centroid_x_px": [10.0, 40.0],
            "centroid_y_px": [10.0, 40.0],
        }
    )
    ctx1 = RunContext(path=Path("a7.tif"), image=np.zeros((8, 8), dtype=np.uint16), meta={})
    ctx1.object_table = pd.DataFrame(
        {
            "filename": ["a7.tif", "a7.tif"],
            "object_id": [1, 2],
            "centroid_x_px": [12.0, 42.0],
            "centroid_y_px": [11.0, 39.0],
        }
    )

    track_table = build_longitudinal_track_table(manifest, [ctx0, ctx1], max_disp_px=5.0)
    summary = summarize_tracks(track_table)

    assert track_table["track_id"].nunique() == 2
    assert len(track_table) == 4
    assert summary.loc[0, "n_tracks"] == 2
