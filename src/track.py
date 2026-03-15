from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from src.context import RunContext


def track_points(prev_xy: np.ndarray, next_xy: np.ndarray, max_disp_px: float = 10.0) -> np.ndarray:
    if len(prev_xy) == 0 or len(next_xy) == 0:
        return np.empty((0, 3), dtype=float)

    cost = cdist(prev_xy, next_xy)
    cost[cost > max_disp_px] = 1e6
    rows, cols = linear_sum_assignment(cost)
    keep = cost[rows, cols] < 1e6
    return np.c_[rows[keep], cols[keep], cost[rows[keep], cols[keep]]]


def build_longitudinal_track_table(
    manifest_df: pd.DataFrame,
    contexts: list[RunContext],
    *,
    max_disp_px: float = 20.0,
) -> pd.DataFrame:
    sample_to_ctx = {
        str(sample_id): ctx
        for sample_id, ctx in zip(manifest_df["sample_id"].astype(str).tolist(), contexts)
        if ctx.object_table is not None and not ctx.object_table.empty
    }

    rows: list[dict[str, object]] = []
    next_track_id = 1

    group_columns = [column for column in ("animal_id", "eye") if column in manifest_df.columns]
    if not group_columns:
        return pd.DataFrame()

    for _, group in manifest_df.sort_values("timepoint_dpi").groupby(group_columns, dropna=False):
        ordered = group.sort_values("timepoint_dpi").to_dict("records")
        if not ordered:
            continue

        first = ordered[0]
        first_ctx = sample_to_ctx.get(str(first["sample_id"]))
        if first_ctx is None or first_ctx.object_table is None or first_ctx.object_table.empty:
            continue

        prev_table = first_ctx.object_table.reset_index(drop=True).copy()
        prev_track_ids = {}
        for idx, row in prev_table.iterrows():
            track_id = next_track_id
            next_track_id += 1
            prev_track_ids[idx] = track_id
            rows.append(
                {
                    "track_id": track_id,
                    "animal_id": first.get("animal_id"),
                    "eye": first.get("eye"),
                    "sample_id": first["sample_id"],
                    "timepoint_dpi": first.get("timepoint_dpi"),
                    "object_id": int(row["object_id"]),
                    "centroid_x_px": float(row["centroid_x_px"]),
                    "centroid_y_px": float(row["centroid_y_px"]),
                    "matched_from_sample_id": None,
                    "matched_from_object_id": None,
                    "displacement_px": 0.0,
                }
            )

        previous_sample_id = first["sample_id"]
        for current in ordered[1:]:
            current_ctx = sample_to_ctx.get(str(current["sample_id"]))
            if current_ctx is None or current_ctx.object_table is None or current_ctx.object_table.empty:
                previous_sample_id = current["sample_id"]
                continue

            current_table = current_ctx.object_table.reset_index(drop=True).copy()
            prev_xy = prev_table[["centroid_x_px", "centroid_y_px"]].to_numpy(dtype=float)
            current_xy = current_table[["centroid_x_px", "centroid_y_px"]].to_numpy(dtype=float)
            matches = track_points(prev_xy, current_xy, max_disp_px=max_disp_px)

            matched_next: dict[int, dict[str, object]] = {}
            for prev_idx, next_idx, distance in matches:
                prev_idx = int(prev_idx)
                next_idx = int(next_idx)
                prev_row = prev_table.iloc[prev_idx]
                matched_next[next_idx] = {
                    "track_id": prev_track_ids[prev_idx],
                    "matched_from_object_id": int(prev_row["object_id"]),
                    "displacement_px": float(distance),
                }

            current_track_ids: dict[int, int] = {}
            for idx, row in current_table.iterrows():
                if idx in matched_next:
                    payload = matched_next[idx]
                    track_id = int(payload["track_id"])
                    matched_from_object_id = payload["matched_from_object_id"]
                    displacement_px = float(payload["displacement_px"])
                else:
                    track_id = next_track_id
                    next_track_id += 1
                    matched_from_object_id = None
                    displacement_px = np.nan

                current_track_ids[idx] = track_id
                rows.append(
                    {
                        "track_id": track_id,
                        "animal_id": current.get("animal_id"),
                        "eye": current.get("eye"),
                        "sample_id": current["sample_id"],
                        "timepoint_dpi": current.get("timepoint_dpi"),
                        "object_id": int(row["object_id"]),
                        "centroid_x_px": float(row["centroid_x_px"]),
                        "centroid_y_px": float(row["centroid_y_px"]),
                        "matched_from_sample_id": previous_sample_id,
                        "matched_from_object_id": matched_from_object_id,
                        "displacement_px": displacement_px,
                    }
                )

            prev_table = current_table
            prev_track_ids = current_track_ids
            previous_sample_id = current["sample_id"]

    return pd.DataFrame(rows)


def summarize_tracks(track_table: pd.DataFrame) -> pd.DataFrame:
    if track_table.empty:
        return pd.DataFrame()

    frame = track_table.copy()
    grouped = frame.groupby(["animal_id", "eye"], dropna=False)
    rows: list[dict[str, object]] = []
    for key, group in grouped:
        animal_id, eye = key
        rows.append(
            {
                "animal_id": animal_id,
                "eye": eye,
                "n_tracks": int(group["track_id"].nunique()),
                "n_observations": int(len(group)),
                "mean_displacement_px": float(group["displacement_px"].dropna().mean()),
            }
        )
    return pd.DataFrame(rows)
