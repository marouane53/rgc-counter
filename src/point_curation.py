from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from src.validation import match_points


def merge_candidate_point_frames(frames: Iterable[pd.DataFrame], *, tolerance_px: float = 3.0) -> pd.DataFrame:
    merged_rows: list[dict[str, object]] = []
    merged_points = np.empty((0, 2), dtype=float)
    for frame in frames:
        if frame is None or frame.empty:
            continue
        working = frame.copy()
        if "y_px" not in working.columns or "x_px" not in working.columns:
            raise ValueError("Candidate point frames must contain y_px and x_px columns.")
        if "detector" not in working.columns:
            working["detector"] = "unknown"
        if "score" not in working.columns:
            working["score"] = np.nan
        if "radius_px" not in working.columns:
            working["radius_px"] = np.nan
        for row in working.to_dict("records"):
            point = np.asarray([[float(row["y_px"]), float(row["x_px"])]], dtype=float)
            if len(merged_points):
                matches = match_points(merged_points, point, tolerance_px=float(tolerance_px))
                if len(matches.matched_pred_indices):
                    continue
            merged_points = np.vstack([merged_points, point])
            merged_rows.append(
                {
                    "y_px": float(row["y_px"]),
                    "x_px": float(row["x_px"]),
                    "score": float(row["score"]) if pd.notna(row["score"]) else float("nan"),
                    "radius_px": float(row["radius_px"]) if pd.notna(row["radius_px"]) else float("nan"),
                    "detector": str(row["detector"]),
                }
            )
    if not merged_rows:
        return pd.DataFrame(columns=["y_px", "x_px", "score", "radius_px", "detector"])
    return pd.DataFrame(merged_rows).sort_values(["score", "detector", "y_px", "x_px"], ascending=[False, True, True, True]).reset_index(drop=True)


def summarize_curation_edits(initial_points_yx: np.ndarray, final_points_yx: np.ndarray, *, tolerance_px: float = 1.5) -> dict[str, int]:
    initial = np.asarray(initial_points_yx, dtype=float).reshape(-1, 2)
    final = np.asarray(final_points_yx, dtype=float).reshape(-1, 2)
    matches = match_points(initial, final, tolerance_px=float(tolerance_px))
    deleted = int(len(matches.unmatched_manual_indices))
    added = int(len(matches.unmatched_pred_indices))
    return {
        "initial_candidate_count": int(len(initial)),
        "added_count": added,
        "deleted_count": deleted,
        "final_count": int(len(final)),
    }
