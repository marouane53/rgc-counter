from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from main import _write_tracking_outputs
from src.context import RunContext
from src.track import (
    build_longitudinal_track_table,
    build_longitudinal_tracking_outputs,
    summarize_tracks,
    track_points,
)


def _make_gray(points: list[tuple[float, float]], shape: tuple[int, int] = (64, 64)) -> np.ndarray:
    image = np.zeros(shape, dtype=np.float32)
    for x, y in points:
        xi = int(round(x))
        yi = int(round(y))
        for oy in range(-1, 2):
            for ox in range(-1, 2):
                yy = yi + oy
                xx = xi + ox
                if 0 <= yy < shape[0] and 0 <= xx < shape[1]:
                    image[yy, xx] = 1.0
    return image


def _make_context(
    path: str,
    points: list[tuple[float, float]],
    *,
    gray_points: list[tuple[float, float]] | None = None,
    gray_shape: tuple[int, int] = (64, 64),
    include_retina_coords: bool = False,
) -> RunContext:
    ctx = RunContext(path=Path(path), image=np.zeros(gray_shape, dtype=np.uint16), meta={})
    ctx.gray = _make_gray(gray_points or points, shape=gray_shape)
    table = {
        "filename": [path] * len(points),
        "object_id": list(range(1, len(points) + 1)),
        "centroid_x_px": [float(x) for x, _ in points],
        "centroid_y_px": [float(y) for _, y in points],
    }
    if include_retina_coords:
        table["ret_x_um"] = [float(x * 2.0) for x, _ in points]
        table["ret_y_um"] = [float(y * 2.0) for _, y in points]
        table["ecc_um"] = [float(np.hypot(x * 2.0, y * 2.0)) for x, y in points]
        table["theta_deg"] = [float((np.degrees(np.arctan2(y, x)) + 360.0) % 360.0) for x, y in points]
    ctx.object_table = pd.DataFrame(table)
    return ctx


def _make_manifest(sample_ids: list[str], timepoints: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": sample_ids,
            "animal_id": ["M1"] * len(sample_ids),
            "eye": ["OD"] * len(sample_ids),
            "timepoint_dpi": timepoints,
            "modality": ["flatmount"] * len(sample_ids),
            "condition": ["control"] * len(sample_ids),
            "genotype": ["WT"] * len(sample_ids),
            "stain_panel": ["RBPMS"] * len(sample_ids),
            "path": [f"{sample_id}.tif" for sample_id in sample_ids],
        }
    )


def test_track_points_matches_nearest_neighbors():
    prev_xy = np.array([[10.0, 10.0], [30.0, 30.0]])
    next_xy = np.array([[12.0, 11.0], [31.0, 32.0]])

    matches = track_points(prev_xy, next_xy, max_disp_px=5.0)

    assert matches.shape == (2, 3)
    assert list(matches[:, 0].astype(int)) == [0, 1]
    assert list(matches[:, 1].astype(int)) == [0, 1]


def test_build_longitudinal_track_table_propagates_track_ids_in_centroid_mode():
    manifest = _make_manifest(["A0", "A7"], [0, 7])
    ctx0 = _make_context("a0.tif", [(10.0, 10.0), (40.0, 40.0)])
    ctx1 = _make_context("a7.tif", [(12.0, 11.0), (42.0, 39.0)])

    track_table = build_longitudinal_track_table(manifest, [ctx0, ctx1], max_disp_px=5.0)
    summary = summarize_tracks(track_table)

    assert track_table["track_id"].nunique() == 2
    assert len(track_table) == 4
    assert summary.loc[0, "n_tracks"] == 2
    assert summary.loc[0, "tracking_mode_requested"] == "centroid"


def test_registered_tracking_reduces_displacement_on_translated_pair():
    base_points = [(12.0, 14.0), (28.0, 20.0), (44.0, 36.0)]
    shifted_points = [(x + 4.0, y + 3.0) for x, y in base_points]
    manifest = _make_manifest(["A0", "A7"], [0, 7])
    ctx0 = _make_context("a0.tif", base_points)
    ctx1 = _make_context("a7.tif", shifted_points)

    track_table, pair_qc, summary = build_longitudinal_tracking_outputs(
        manifest,
        [ctx0, ctx1],
        max_disp_px=10.0,
        tracking_mode="registered",
    )

    current_rows = track_table[track_table["sample_id"] == "A7"].sort_values("object_id")
    seed_rows = track_table[track_table["sample_id"] == "A0"].sort_values("object_id")

    assert pair_qc.loc[0, "tracking_mode_used"] == "registered"
    assert pair_qc.loc[0, "registration_status"] == "registered"
    assert pair_qc.loc[0, "median_registered_displacement_px"] < pair_qc.loc[0, "median_raw_displacement_px"]
    assert summary.loc[0, "n_pairs_registered"] == 1
    assert np.allclose(current_rows["common_centroid_x_px"], seed_rows["centroid_x_px"], atol=0.25)
    assert np.allclose(current_rows["common_centroid_y_px"], seed_rows["centroid_y_px"], atol=0.25)


def test_registered_tracking_does_not_make_no_shift_pair_worse():
    points = [(10.0, 10.0), (24.0, 18.0), (40.0, 35.0)]
    manifest = _make_manifest(["A0", "A7"], [0, 7])
    ctx0 = _make_context("a0.tif", points)
    ctx1 = _make_context("a7.tif", points)

    track_table, pair_qc, _ = build_longitudinal_tracking_outputs(
        manifest,
        [ctx0, ctx1],
        max_disp_px=8.0,
        tracking_mode="registered",
    )

    current_rows = track_table[track_table["sample_id"] == "A7"]
    assert pair_qc.loc[0, "tracking_mode_used"] == "registered"
    assert float(current_rows["displacement_px"].dropna().mean()) <= float(current_rows["raw_displacement_px"].dropna().mean())


def test_registered_tracking_falls_back_on_shape_mismatch_and_starts_new_segment():
    base_points = [(12.0, 14.0), (28.0, 20.0), (44.0, 36.0)]
    shifted_points = [(x + 2.0, y + 1.0) for x, y in base_points]
    manifest = _make_manifest(["A0", "A7"], [0, 7])
    ctx0 = _make_context("a0.tif", base_points, gray_shape=(64, 64))
    ctx1 = _make_context("a7.tif", shifted_points, gray_shape=(48, 48))

    track_table, pair_qc, summary = build_longitudinal_tracking_outputs(
        manifest,
        [ctx0, ctx1],
        max_disp_px=6.0,
        tracking_mode="registered",
    )

    current_rows = track_table[track_table["sample_id"] == "A7"]
    assert pair_qc.loc[0, "registration_status"] == "fallback_shape_mismatch"
    assert pair_qc.loc[0, "tracking_mode_used"] == "centroid"
    assert current_rows["common_frame_segment_id"].nunique() == 1
    assert int(current_rows["common_frame_segment_id"].iloc[0]) == 2
    assert summary.loc[0, "n_pairs_fallback"] == 1


def test_registered_tracking_falls_back_when_alignment_provides_no_improvement():
    points = [(12.0, 14.0), (28.0, 20.0), (44.0, 36.0)]
    shifted_gray_points = [(x + 5.0, y + 4.0) for x, y in points]
    manifest = _make_manifest(["A0", "A7"], [0, 7])
    ctx0 = _make_context("a0.tif", points)
    ctx1 = _make_context("a7.tif", points, gray_points=shifted_gray_points)

    track_table, pair_qc, _ = build_longitudinal_tracking_outputs(
        manifest,
        [ctx0, ctx1],
        max_disp_px=8.0,
        tracking_mode="registered",
    )

    current_rows = track_table[track_table["sample_id"] == "A7"]
    assert pair_qc.loc[0, "registration_status"] == "fallback_no_improvement"
    assert pair_qc.loc[0, "tracking_mode_used"] == "centroid"
    assert int(current_rows["common_frame_segment_id"].iloc[0]) == 2


def test_registered_tracking_accumulates_common_frame_coordinates_across_successful_pairs():
    base_points = [(12.0, 14.0), (28.0, 20.0), (44.0, 36.0)]
    sample1 = base_points
    sample2 = [(x + 4.0, y + 3.0) for x, y in base_points]
    sample3 = [(x + 8.0, y + 6.0) for x, y in base_points]
    manifest = _make_manifest(["A0", "A7", "A14"], [0, 7, 14])
    ctx0 = _make_context("a0.tif", sample1)
    ctx1 = _make_context("a7.tif", sample2)
    ctx2 = _make_context("a14.tif", sample3)

    track_table, pair_qc, summary = build_longitudinal_tracking_outputs(
        manifest,
        [ctx0, ctx1, ctx2],
        max_disp_px=12.0,
        tracking_mode="registered",
    )

    first_rows = track_table[track_table["sample_id"] == "A0"].sort_values("object_id")
    third_rows = track_table[track_table["sample_id"] == "A14"].sort_values("object_id")
    assert pair_qc["tracking_mode_used"].tolist() == ["registered", "registered"]
    assert summary.loc[0, "n_pairs_registered"] == 2
    assert np.allclose(third_rows["common_centroid_x_px"], first_rows["centroid_x_px"], atol=0.35)
    assert np.allclose(third_rows["common_centroid_y_px"], first_rows["centroid_y_px"], atol=0.35)


def test_tracking_outputs_pass_through_retina_registered_coordinates():
    base_points = [(12.0, 14.0), (28.0, 20.0), (44.0, 36.0)]
    shifted_points = [(x + 4.0, y + 3.0) for x, y in base_points]
    manifest = _make_manifest(["A0", "A7"], [0, 7])
    ctx0 = _make_context("a0.tif", base_points, include_retina_coords=True)
    ctx1 = _make_context("a7.tif", shifted_points, include_retina_coords=True)

    track_table, _, _ = build_longitudinal_tracking_outputs(
        manifest,
        [ctx0, ctx1],
        max_disp_px=10.0,
        tracking_mode="registered",
    )

    assert {"ret_x_um", "ret_y_um", "ecc_um", "theta_deg"}.issubset(track_table.columns)


def test_write_tracking_outputs_writes_pair_qc_and_summary(tmp_path: Path):
    base_points = [(12.0, 14.0), (28.0, 20.0), (44.0, 36.0)]
    shifted_points = [(x + 4.0, y + 3.0) for x, y in base_points]
    manifest = _make_manifest(["A0", "A7"], [0, 7])
    ctx0 = _make_context("a0.tif", base_points)
    ctx1 = _make_context("a7.tif", shifted_points)

    track_table, pair_qc, summary, assets = _write_tracking_outputs(
        manifest_df=manifest,
        contexts=[ctx0, ctx1],
        output_dir=tmp_path,
        max_disp_px=10.0,
        tracking_mode="registered",
    )

    assert not track_table.empty
    assert not pair_qc.empty
    assert not summary.empty
    assert len(assets) == 3
    assert (tmp_path / "tracking" / "track_observations.csv").exists()
    assert (tmp_path / "tracking" / "track_pair_qc.csv").exists()
    assert (tmp_path / "tracking" / "track_summary.csv").exists()
