from __future__ import annotations

import numpy as np
import pandas as pd

from src.point_curation import merge_candidate_point_frames, summarize_curation_edits


def test_merge_candidate_point_frames_deduplicates_nearby_points():
    frame_a = pd.DataFrame(
        [
            {"y_px": 10.0, "x_px": 10.0, "score": 0.9, "radius_px": 4.0, "detector": "log"},
            {"y_px": 30.0, "x_px": 30.0, "score": 0.8, "radius_px": 4.0, "detector": "log"},
        ]
    )
    frame_b = pd.DataFrame(
        [
            {"y_px": 10.8, "x_px": 10.5, "score": 0.7, "radius_px": 4.0, "detector": "dog"},
            {"y_px": 50.0, "x_px": 50.0, "score": 0.6, "radius_px": 4.0, "detector": "dog"},
        ]
    )

    merged = merge_candidate_point_frames([frame_a, frame_b], tolerance_px=2.0)

    assert len(merged) == 3


def test_summarize_curation_edits_counts_added_and_deleted_points():
    initial = np.asarray([[10.0, 10.0], [20.0, 20.0]], dtype=float)
    final = np.asarray([[10.0, 10.0], [25.0, 25.0]], dtype=float)

    summary = summarize_curation_edits(initial, final, tolerance_px=1.5)

    assert summary["initial_candidate_count"] == 2
    assert summary["deleted_count"] == 1
    assert summary["added_count"] == 1
    assert summary["final_count"] == 2
