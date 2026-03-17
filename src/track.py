from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage.registration import phase_cross_correlation

from src.context import RunContext

TRACKING_MODE_CHOICES = ("centroid", "registered")
ALIGNMENT_METHOD = "phase_cross_correlation"
FALLBACK_POLICY = "centroid_fallback_with_qc_flag"
TRACK_PASS_THROUGH_COLUMNS = ("ret_x_um", "ret_y_um", "ecc_um", "theta_deg")


@dataclass
class PairTrackingResult:
    tracking_mode_requested: str
    tracking_mode_used: str
    registration_status: str
    registration_shift_dx_px: float
    registration_shift_dy_px: float
    registration_error: float
    registration_phase_diff: float
    raw_matches: np.ndarray
    registered_matches: np.ndarray
    actual_matches: np.ndarray


def track_points(prev_xy: np.ndarray, next_xy: np.ndarray, max_disp_px: float = 10.0) -> np.ndarray:
    if len(prev_xy) == 0 or len(next_xy) == 0:
        return np.empty((0, 3), dtype=float)

    cost = cdist(prev_xy, next_xy)
    cost[cost > max_disp_px] = 1e6
    rows, cols = linear_sum_assignment(cost)
    keep = cost[rows, cols] < 1e6
    return np.c_[rows[keep], cols[keep], cost[rows[keep], cols[keep]]]


def _safe_float(value: object) -> float:
    if value is None:
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _match_mean(matches: np.ndarray) -> float:
    return float(matches[:, 2].mean()) if len(matches) else np.nan


def _match_median(matches: np.ndarray) -> float:
    return float(np.median(matches[:, 2])) if len(matches) else np.nan


def _row_matches(matches: np.ndarray) -> dict[int, dict[str, float]]:
    payload: dict[int, dict[str, float]] = {}
    for prev_idx, next_idx, distance in matches:
        payload[int(next_idx)] = {"prev_idx": int(prev_idx), "distance": float(distance)}
    return payload


def _estimate_pair_registration(prev_ctx: RunContext, current_ctx: RunContext) -> tuple[float, float, float, float, str]:
    prev_gray = prev_ctx.gray
    current_gray = current_ctx.gray
    if prev_gray is None or current_gray is None:
        return np.nan, np.nan, np.nan, np.nan, "fallback_missing_gray"
    if prev_gray.shape != current_gray.shape:
        return np.nan, np.nan, np.nan, np.nan, "fallback_shape_mismatch"

    try:
        shift, error, phase_diff = phase_cross_correlation(
            prev_gray.astype(np.float32),
            current_gray.astype(np.float32),
            upsample_factor=10,
        )
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, "fallback_registration_error"

    if shift is None or len(shift) < 2 or not np.isfinite(shift[:2]).all():
        return np.nan, np.nan, np.nan, np.nan, "fallback_invalid_shift"
    return float(shift[1]), float(shift[0]), float(error), float(phase_diff), "estimated"


def _pairwise_tracking_result(
    *,
    prev_ctx: RunContext,
    current_ctx: RunContext,
    prev_xy: np.ndarray,
    current_xy: np.ndarray,
    max_disp_px: float,
    tracking_mode: str,
) -> PairTrackingResult:
    raw_matches = track_points(prev_xy, current_xy, max_disp_px=max_disp_px)
    if tracking_mode == "centroid":
        return PairTrackingResult(
            tracking_mode_requested="centroid",
            tracking_mode_used="centroid",
            registration_status="centroid_only",
            registration_shift_dx_px=np.nan,
            registration_shift_dy_px=np.nan,
            registration_error=np.nan,
            registration_phase_diff=np.nan,
            raw_matches=raw_matches,
            registered_matches=np.empty((0, 3), dtype=float),
            actual_matches=raw_matches,
        )

    dx, dy, error, phase_diff, registration_status = _estimate_pair_registration(prev_ctx, current_ctx)
    if not np.isfinite(dx) or not np.isfinite(dy):
        return PairTrackingResult(
            tracking_mode_requested="registered",
            tracking_mode_used="centroid",
            registration_status=registration_status,
            registration_shift_dx_px=dx,
            registration_shift_dy_px=dy,
            registration_error=error,
            registration_phase_diff=phase_diff,
            raw_matches=raw_matches,
            registered_matches=np.empty((0, 3), dtype=float),
            actual_matches=raw_matches,
        )

    shifted_xy = current_xy + np.array([dx, dy], dtype=float)
    registered_matches = track_points(prev_xy, shifted_xy, max_disp_px=max_disp_px)

    use_registered = len(registered_matches) >= len(raw_matches)
    if use_registered and len(raw_matches) >= 3 and len(registered_matches) >= 3:
        raw_median = _match_median(raw_matches)
        registered_median = _match_median(registered_matches)
        if np.isfinite(raw_median) and np.isfinite(registered_median) and registered_median > raw_median:
            use_registered = False

    if use_registered:
        return PairTrackingResult(
            tracking_mode_requested="registered",
            tracking_mode_used="registered",
            registration_status="registered",
            registration_shift_dx_px=dx,
            registration_shift_dy_px=dy,
            registration_error=error,
            registration_phase_diff=phase_diff,
            raw_matches=raw_matches,
            registered_matches=registered_matches,
            actual_matches=registered_matches,
        )

    return PairTrackingResult(
        tracking_mode_requested="registered",
        tracking_mode_used="centroid",
        registration_status="fallback_no_improvement",
        registration_shift_dx_px=dx,
        registration_shift_dy_px=dy,
        registration_error=error,
        registration_phase_diff=phase_diff,
        raw_matches=raw_matches,
        registered_matches=registered_matches,
        actual_matches=raw_matches,
    )


def _observation_row(
    *,
    table_row: pd.Series,
    animal_id: object,
    eye: object,
    sample_id: object,
    timepoint_dpi: object,
    track_id: int,
    matched_from_sample_id: object,
    matched_from_object_id: object,
    displacement_px: float,
    raw_displacement_px: float,
    registered_displacement_px: float,
    tracking_mode_requested: str,
    tracking_mode_used: str,
    common_frame_segment_id: int,
    common_centroid_x_px: float,
    common_centroid_y_px: float,
    registration_status: str,
    registration_shift_dx_px: float,
    registration_shift_dy_px: float,
    registration_error: float,
    registration_phase_diff: float,
) -> dict[str, object]:
    row = {
        "track_id": int(track_id),
        "animal_id": animal_id,
        "eye": eye,
        "sample_id": sample_id,
        "timepoint_dpi": timepoint_dpi,
        "object_id": int(table_row["object_id"]),
        "centroid_x_px": float(table_row["centroid_x_px"]),
        "centroid_y_px": float(table_row["centroid_y_px"]),
        "matched_from_sample_id": matched_from_sample_id,
        "matched_from_object_id": matched_from_object_id,
        "displacement_px": displacement_px,
        "tracking_mode_requested": tracking_mode_requested,
        "tracking_mode_used": tracking_mode_used,
        "common_frame_segment_id": int(common_frame_segment_id),
        "common_centroid_x_px": float(common_centroid_x_px),
        "common_centroid_y_px": float(common_centroid_y_px),
        "raw_displacement_px": raw_displacement_px,
        "registered_displacement_px": registered_displacement_px,
        "registration_shift_dx_px": registration_shift_dx_px,
        "registration_shift_dy_px": registration_shift_dy_px,
        "registration_error": registration_error,
        "registration_phase_diff": registration_phase_diff,
        "registration_status": registration_status,
    }
    for column in TRACK_PASS_THROUGH_COLUMNS:
        if column in table_row.index:
            row[column] = _safe_float(table_row[column])
    return row


def _pair_qc_row(
    *,
    animal_id: object,
    eye: object,
    prev_sample_id: object,
    sample_id: object,
    prev_timepoint_dpi: object,
    timepoint_dpi: object,
    result: PairTrackingResult,
    n_prev_objects: int,
    n_current_objects: int,
) -> dict[str, object]:
    n_matches = int(len(result.actual_matches))
    denominator = min(int(n_prev_objects), int(n_current_objects))
    matched_fraction = float(n_matches / denominator) if denominator > 0 else np.nan
    return {
        "animal_id": animal_id,
        "eye": eye,
        "prev_sample_id": prev_sample_id,
        "sample_id": sample_id,
        "prev_timepoint_dpi": prev_timepoint_dpi,
        "timepoint_dpi": timepoint_dpi,
        "tracking_mode_requested": result.tracking_mode_requested,
        "tracking_mode_used": result.tracking_mode_used,
        "registration_status": result.registration_status,
        "registration_shift_dx_px": result.registration_shift_dx_px,
        "registration_shift_dy_px": result.registration_shift_dy_px,
        "registration_error": result.registration_error,
        "registration_phase_diff": result.registration_phase_diff,
        "n_prev_objects": int(n_prev_objects),
        "n_current_objects": int(n_current_objects),
        "n_matches": n_matches,
        "matched_fraction": matched_fraction,
        "unmatched_prev_count": int(max(n_prev_objects - n_matches, 0)),
        "unmatched_current_count": int(max(n_current_objects - n_matches, 0)),
        "mean_raw_displacement_px": _match_mean(result.raw_matches),
        "median_raw_displacement_px": _match_median(result.raw_matches),
        "mean_registered_displacement_px": _match_mean(result.registered_matches),
        "median_registered_displacement_px": _match_median(result.registered_matches),
    }


def build_longitudinal_tracking_outputs(
    manifest_df: pd.DataFrame,
    contexts: list[RunContext],
    *,
    max_disp_px: float = 20.0,
    tracking_mode: str = "centroid",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if tracking_mode not in TRACKING_MODE_CHOICES:
        raise ValueError(f"Unsupported tracking mode: {tracking_mode}")

    sample_to_ctx = {
        str(sample_id): ctx
        for sample_id, ctx in zip(manifest_df["sample_id"].astype(str).tolist(), contexts)
    }

    rows: list[dict[str, object]] = []
    pair_qc_rows: list[dict[str, object]] = []
    next_track_id = 1

    group_columns = [column for column in ("animal_id", "eye") if column in manifest_df.columns]
    if not group_columns:
        empty = pd.DataFrame()
        return empty, empty, empty

    sort_columns = [column for column in ("timepoint_dpi", "sample_id") if column in manifest_df.columns]

    for _, group in manifest_df.sort_values(sort_columns).groupby(group_columns, dropna=False):
        ordered = group.sort_values(sort_columns).to_dict("records")
        if not ordered:
            continue

        prev_table: pd.DataFrame | None = None
        prev_ctx: RunContext | None = None
        prev_track_ids: dict[int, int] = {}
        prev_sample_id: object = None
        prev_timepoint_dpi: object = None
        prev_segment_id: int | None = None
        prev_common_offset = np.zeros(2, dtype=float)
        next_segment_id = 1

        for current in ordered:
            sample_id = current["sample_id"]
            current_ctx = sample_to_ctx.get(str(sample_id))
            current_table = None
            if current_ctx is not None and current_ctx.object_table is not None:
                current_table = current_ctx.object_table.reset_index(drop=True).copy()

            if current_table is None:
                if prev_table is not None and prev_ctx is not None:
                    pair_qc_rows.append(
                        _pair_qc_row(
                            animal_id=current.get("animal_id"),
                            eye=current.get("eye"),
                            prev_sample_id=prev_sample_id,
                            sample_id=sample_id,
                            prev_timepoint_dpi=prev_timepoint_dpi,
                            timepoint_dpi=current.get("timepoint_dpi"),
                            result=PairTrackingResult(
                                tracking_mode_requested=tracking_mode,
                                tracking_mode_used="centroid",
                                registration_status="fallback_missing_objects",
                                registration_shift_dx_px=np.nan,
                                registration_shift_dy_px=np.nan,
                                registration_error=np.nan,
                                registration_phase_diff=np.nan,
                                raw_matches=np.empty((0, 3), dtype=float),
                                registered_matches=np.empty((0, 3), dtype=float),
                                actual_matches=np.empty((0, 3), dtype=float),
                            ),
                            n_prev_objects=len(prev_table),
                            n_current_objects=0,
                        )
                    )
                prev_table = None
                prev_ctx = None
                prev_track_ids = {}
                prev_sample_id = sample_id
                prev_timepoint_dpi = current.get("timepoint_dpi")
                prev_segment_id = None
                prev_common_offset = np.zeros(2, dtype=float)
                continue

            animal_id = current.get("animal_id")
            eye = current.get("eye")
            timepoint_dpi = current.get("timepoint_dpi")
            current_xy = current_table[["centroid_x_px", "centroid_y_px"]].to_numpy(dtype=float)

            if prev_table is None or prev_ctx is None or prev_segment_id is None:
                current_segment_id = next_segment_id
                next_segment_id += 1
                current_common_offset = np.zeros(2, dtype=float)
                current_track_ids: dict[int, int] = {}
                for idx, row in current_table.iterrows():
                    track_id = next_track_id
                    next_track_id += 1
                    current_track_ids[idx] = track_id
                    rows.append(
                        _observation_row(
                            table_row=row,
                            animal_id=animal_id,
                            eye=eye,
                            sample_id=sample_id,
                            timepoint_dpi=timepoint_dpi,
                            track_id=track_id,
                            matched_from_sample_id=None,
                            matched_from_object_id=None,
                            displacement_px=0.0,
                            raw_displacement_px=0.0,
                            registered_displacement_px=0.0 if tracking_mode == "registered" else np.nan,
                            tracking_mode_requested=tracking_mode,
                            tracking_mode_used=tracking_mode,
                            common_frame_segment_id=current_segment_id,
                            common_centroid_x_px=float(row["centroid_x_px"] + current_common_offset[0]),
                            common_centroid_y_px=float(row["centroid_y_px"] + current_common_offset[1]),
                            registration_status="seed",
                            registration_shift_dx_px=0.0,
                            registration_shift_dy_px=0.0,
                            registration_error=np.nan,
                            registration_phase_diff=np.nan,
                        )
                    )
                prev_table = current_table
                prev_ctx = current_ctx
                prev_track_ids = current_track_ids
                prev_sample_id = sample_id
                prev_timepoint_dpi = timepoint_dpi
                prev_segment_id = current_segment_id
                prev_common_offset = current_common_offset
                continue

            prev_xy = prev_table[["centroid_x_px", "centroid_y_px"]].to_numpy(dtype=float)
            result = _pairwise_tracking_result(
                prev_ctx=prev_ctx,
                current_ctx=current_ctx,
                prev_xy=prev_xy,
                current_xy=current_xy,
                max_disp_px=max_disp_px,
                tracking_mode=tracking_mode,
            )

            pair_qc_rows.append(
                _pair_qc_row(
                    animal_id=animal_id,
                    eye=eye,
                    prev_sample_id=prev_sample_id,
                    sample_id=sample_id,
                    prev_timepoint_dpi=prev_timepoint_dpi,
                    timepoint_dpi=timepoint_dpi,
                    result=result,
                    n_prev_objects=len(prev_table),
                    n_current_objects=len(current_table),
                )
            )

            if result.tracking_mode_used == "registered":
                current_segment_id = prev_segment_id
                current_common_offset = prev_common_offset + np.array(
                    [result.registration_shift_dx_px, result.registration_shift_dy_px],
                    dtype=float,
                )
            else:
                current_segment_id = next_segment_id
                next_segment_id += 1
                current_common_offset = np.zeros(2, dtype=float)

            raw_match_map = _row_matches(result.raw_matches)
            registered_match_map = _row_matches(result.registered_matches)
            actual_match_map = _row_matches(result.actual_matches)
            current_track_ids = {}

            for idx, row in current_table.iterrows():
                raw_payload = raw_match_map.get(idx)
                registered_payload = registered_match_map.get(idx)
                actual_payload = actual_match_map.get(idx)

                if actual_payload is not None:
                    prev_idx = int(actual_payload["prev_idx"])
                    prev_row = prev_table.iloc[prev_idx]
                    track_id = prev_track_ids[prev_idx]
                    matched_from_object_id = int(prev_row["object_id"])
                    displacement_px = float(actual_payload["distance"])
                else:
                    track_id = next_track_id
                    next_track_id += 1
                    matched_from_object_id = None
                    displacement_px = np.nan

                current_track_ids[idx] = track_id
                rows.append(
                    _observation_row(
                        table_row=row,
                        animal_id=animal_id,
                        eye=eye,
                        sample_id=sample_id,
                        timepoint_dpi=timepoint_dpi,
                        track_id=track_id,
                        matched_from_sample_id=prev_sample_id,
                        matched_from_object_id=matched_from_object_id,
                        displacement_px=displacement_px,
                        raw_displacement_px=float(raw_payload["distance"]) if raw_payload is not None else np.nan,
                        registered_displacement_px=float(registered_payload["distance"]) if registered_payload is not None else np.nan,
                        tracking_mode_requested=tracking_mode,
                        tracking_mode_used=result.tracking_mode_used,
                        common_frame_segment_id=current_segment_id,
                        common_centroid_x_px=float(row["centroid_x_px"] + current_common_offset[0]),
                        common_centroid_y_px=float(row["centroid_y_px"] + current_common_offset[1]),
                        registration_status=result.registration_status,
                        registration_shift_dx_px=result.registration_shift_dx_px,
                        registration_shift_dy_px=result.registration_shift_dy_px,
                        registration_error=result.registration_error,
                        registration_phase_diff=result.registration_phase_diff,
                    )
                )

            prev_table = current_table
            prev_ctx = current_ctx
            prev_track_ids = current_track_ids
            prev_sample_id = sample_id
            prev_timepoint_dpi = timepoint_dpi
            prev_segment_id = current_segment_id
            prev_common_offset = current_common_offset

    track_table = pd.DataFrame(rows)
    pair_qc = pd.DataFrame(pair_qc_rows)
    summary = summarize_tracks(track_table, pair_qc=pair_qc, tracking_mode_requested=tracking_mode)
    return track_table, pair_qc, summary


def build_longitudinal_track_table(
    manifest_df: pd.DataFrame,
    contexts: list[RunContext],
    *,
    max_disp_px: float = 20.0,
    tracking_mode: str = "centroid",
) -> pd.DataFrame:
    track_table, _, _ = build_longitudinal_tracking_outputs(
        manifest_df,
        contexts,
        max_disp_px=max_disp_px,
        tracking_mode=tracking_mode,
    )
    return track_table


def summarize_tracks(
    track_table: pd.DataFrame,
    *,
    pair_qc: pd.DataFrame | None = None,
    tracking_mode_requested: str = "centroid",
) -> pd.DataFrame:
    if track_table.empty:
        if pair_qc is None or pair_qc.empty:
            return pd.DataFrame()
        rows: list[dict[str, object]] = []
        for key, qc_group in pair_qc.groupby(["animal_id", "eye"], dropna=False):
            animal_id, eye = key
            rows.append(
                {
                    "animal_id": animal_id,
                    "eye": eye,
                    "tracking_mode_requested": tracking_mode_requested,
                    "n_tracks": 0,
                    "n_observations": 0,
                    "n_pairs": int(len(qc_group)),
                    "n_pairs_registered": int((qc_group.get("tracking_mode_used", pd.Series(dtype=object)) == "registered").sum()),
                    "n_pairs_fallback": int(
                        (
                            (qc_group.get("tracking_mode_requested", pd.Series(dtype=object)) == "registered")
                            & qc_group.get("registration_status", pd.Series(dtype=object)).astype(str).str.startswith("fallback")
                        ).sum()
                    ),
                    "matched_fraction_mean": float(qc_group["matched_fraction"].dropna().mean()) if not qc_group.empty else np.nan,
                    "mean_displacement_px": np.nan,
                    "mean_raw_displacement_px": np.nan,
                    "mean_registered_displacement_px": np.nan,
                }
            )
        return pd.DataFrame(rows)

    frame = track_table.copy()
    grouped = frame.groupby(["animal_id", "eye"], dropna=False)
    rows: list[dict[str, object]] = []
    for key, group in grouped:
        animal_id, eye = key
        matched_group = group[group["matched_from_sample_id"].notna()]
        qc_group = pd.DataFrame()
        if pair_qc is not None and not pair_qc.empty:
            qc_group = pair_qc[
                (pair_qc["animal_id"] == animal_id)
                & (pair_qc["eye"] == eye)
            ]
        rows.append(
            {
                "animal_id": animal_id,
                "eye": eye,
                "tracking_mode_requested": tracking_mode_requested,
                "n_tracks": int(group["track_id"].nunique()),
                "n_observations": int(len(group)),
                "n_pairs": int(len(qc_group)),
                "n_pairs_registered": int((qc_group.get("tracking_mode_used", pd.Series(dtype=object)) == "registered").sum()),
                "n_pairs_fallback": int(
                    (
                        (qc_group.get("tracking_mode_requested", pd.Series(dtype=object)) == "registered")
                        & qc_group.get("registration_status", pd.Series(dtype=object)).astype(str).str.startswith("fallback")
                    ).sum()
                ),
                "matched_fraction_mean": float(qc_group["matched_fraction"].dropna().mean()) if not qc_group.empty else np.nan,
                "mean_displacement_px": float(matched_group["displacement_px"].dropna().mean()) if not matched_group.empty else np.nan,
                "mean_raw_displacement_px": float(matched_group["raw_displacement_px"].dropna().mean()) if not matched_group.empty else np.nan,
                "mean_registered_displacement_px": float(matched_group["registered_displacement_px"].dropna().mean()) if not matched_group.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)
