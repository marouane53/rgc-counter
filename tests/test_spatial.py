import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon

from src.regions import assign_regions
from src.retina_coords import register_cells, register_focus_mask_pixels, retina_frame_from_points
from src.spatial import (
    choose_valid_radii_px,
    compute_csr_envelopes,
    compute_rigorous_spatial_bundle,
    exact_voronoi_clip_area,
    centroids_from_masks,
    nn_regularity_index,
    ripley_k,
    voronoi_regulariry_index,
)


def _grid_points() -> np.ndarray:
    values = [20.0, 60.0, 100.0]
    return np.asarray([[y, x] for y in values for x in values], dtype=np.float32)


def _object_table_from_points(points_yx: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_id": ["sample"] * len(points_yx),
            "object_id": np.arange(1, len(points_yx) + 1),
            "centroid_y_px": points_yx[:, 0],
            "centroid_x_px": points_yx[:, 1],
            "area_px": [25] * len(points_yx),
            "phenotype": ["unclassified"] * len(points_yx),
            "kept": [True] * len(points_yx),
        }
    )


def test_centroids_from_masks_returns_expected_yx_centroids():
    masks = np.zeros((8, 8), dtype=np.uint16)
    masks[1:3, 1:3] = 1
    masks[5:7, 4:6] = 2

    centroids = centroids_from_masks(masks)

    assert centroids.shape == (2, 2)
    assert centroids[0, 0] == pytest.approx(1.5)
    assert centroids[0, 1] == pytest.approx(1.5)
    assert centroids[1, 0] == pytest.approx(5.5)
    assert centroids[1, 1] == pytest.approx(4.5)


def test_ripley_k_matches_simple_two_point_case():
    centroids = np.array([[0.0, 0.0], [0.0, 5.0]], dtype=np.float32)

    result = ripley_k(centroids, radii=[1.0, 10.0], area_px=100.0)

    assert result[1.0] == pytest.approx(0.0)
    assert result[10.0] == pytest.approx(50.0)


def test_nnri_and_voronoi_return_zero_for_too_few_points():
    centroids = np.array([[2.0, 2.0], [5.0, 5.0]], dtype=np.float32)

    nnri = nn_regularity_index(centroids)
    vdri = voronoi_regulariry_index(centroids, shape=(10, 10))

    assert nnri["nnri"] >= 0
    assert vdri["vdri"] == 0.0


def test_exact_voronoi_clip_area_uses_exact_domain_intersection():
    poly = np.array([[-5.0, -5.0], [15.0, -5.0], [15.0, 15.0], [-5.0, 15.0]], dtype=float)

    rect_area = exact_voronoi_clip_area(poly, width=10, height=10)
    wedge_area = exact_voronoi_clip_area(
        poly,
        width=10,
        height=10,
        domain_polygon=Polygon([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]),
    )

    assert rect_area == pytest.approx(100.0)
    assert wedge_area == pytest.approx(50.0)


def test_compute_csr_envelopes_is_deterministic_for_fixed_seed():
    mask = np.ones((128, 128), dtype=bool)
    points = _grid_points()

    first = compute_csr_envelopes(points, mask, radii_px=[25.0, 50.0, 75.0], simulation_count=8, seed=11)
    second = compute_csr_envelopes(points, mask, radii_px=[25.0, 50.0, 75.0], simulation_count=8, seed=11)

    assert np.allclose(first["l_env_low"], second["l_env_low"], equal_nan=True)
    assert np.allclose(first["l_env_high"], second["l_env_high"], equal_nan=True)
    assert np.allclose(first["g_env_low"], second["g_env_low"], equal_nan=True)
    assert np.allclose(first["g_env_high"], second["g_env_high"], equal_nan=True)
    assert first["l_outside_envelope_any"] == second["l_outside_envelope_any"]


def test_compute_rigorous_bundle_uses_tissue_mask_for_global_domain():
    tissue_mask = np.zeros((128, 128), dtype=bool)
    tissue_mask[8:120, 8:120] = True
    tissue_mask[:40, :40] = False
    object_table = _object_table_from_points(_grid_points())

    bundle = compute_rigorous_spatial_bundle(
        image_id="sample",
        object_table=object_table,
        image_shape=tissue_mask.shape,
        tissue_mask=tissue_mask,
        um_per_px=1.0,
        radii_px=[5.0, 10.0],
        simulation_count=8,
        base_seed=7,
    )

    global_row = bundle["summary"].query("analysis_level == 'global'").iloc[0]
    assert global_row["domain_area_px"] == pytest.approx(float(tissue_mask.sum()))
    assert int(global_row["n_points"]) == len(object_table)
    assert global_row["status"] == "ok"
    assert "l_obs" in bundle["curves"].columns


def test_compute_rigorous_bundle_global_point_count_matches_kept_object_table_count():
    tissue_mask = np.ones((128, 128), dtype=bool)
    object_table = _object_table_from_points(_grid_points())
    object_table.loc[object_table["object_id"] == 1, "kept"] = False

    bundle = compute_rigorous_spatial_bundle(
        image_id="sample",
        object_table=object_table,
        image_shape=tissue_mask.shape,
        tissue_mask=tissue_mask,
        um_per_px=1.0,
        simulation_count=8,
        base_seed=11,
    )

    global_row = bundle["summary"].query("analysis_level == 'global'").iloc[0]
    assert int(global_row["n_points"]) == int(object_table["kept"].fillna(True).astype(bool).sum())


def test_compute_rigorous_bundle_includes_registered_region_domains():
    tissue_mask = np.ones((128, 128), dtype=bool)
    frame = retina_frame_from_points(
        onh_xy_px=(64.0, 64.0),
        dorsal_xy_px=(64.0, 8.0),
        um_per_px=1.0,
        source="cli",
    )
    tissue_pixels = register_focus_mask_pixels(tissue_mask, frame)
    points = np.asarray(
        [
            [24.0, 24.0],
            [24.0, 64.0],
            [24.0, 104.0],
            [64.0, 24.0],
            [64.0, 104.0],
            [104.0, 24.0],
            [104.0, 64.0],
            [104.0, 104.0],
        ],
        dtype=float,
    )
    object_table = _object_table_from_points(points)

    bundle = compute_rigorous_spatial_bundle(
        image_id="sample",
        object_table=object_table,
        image_shape=tissue_mask.shape,
        tissue_mask=tissue_mask,
        um_per_px=1.0,
        registered_tissue_pixels=tissue_pixels,
        schema_name="mouse_flatmount_v1",
        max_ecc_um=float(tissue_pixels["ecc_um"].max()),
        simulation_count=8,
        base_seed=17,
    )

    summary = bundle["summary"]
    assert {"global", "ring", "quadrant", "sector", "peripapillary_bin"}.issubset(set(summary["region_axis"]))
    assert bundle["spatial_analysis"]["regionwise_analysis_run"] is True


def test_region_point_counts_match_assign_regions_membership():
    tissue_mask = np.ones((128, 128), dtype=bool)
    frame = retina_frame_from_points(
        onh_xy_px=(64.0, 64.0),
        dorsal_xy_px=(64.0, 8.0),
        um_per_px=1.0,
        source="cli",
    )
    tissue_pixels = register_focus_mask_pixels(tissue_mask, frame)
    points = np.asarray(
        [
            [24.0, 24.0],
            [24.0, 64.0],
            [24.0, 104.0],
            [64.0, 24.0],
            [64.0, 104.0],
            [104.0, 24.0],
            [104.0, 64.0],
            [104.0, 104.0],
        ],
        dtype=float,
    )
    registered = register_cells(_object_table_from_points(points), frame)
    assigned = assign_regions(
        registered,
        schema_name="mouse_flatmount_v1",
        max_ecc_um=float(tissue_pixels["ecc_um"].max()),
    )

    bundle = compute_rigorous_spatial_bundle(
        image_id="sample",
        object_table=assigned,
        image_shape=tissue_mask.shape,
        tissue_mask=tissue_mask,
        um_per_px=1.0,
        registered_tissue_pixels=tissue_pixels,
        schema_name="mouse_flatmount_v1",
        max_ecc_um=float(tissue_pixels["ecc_um"].max()),
        simulation_count=8,
        base_seed=23,
    )

    region_summary = bundle["summary"].query("analysis_level == 'region'")
    for axis in ("ring", "quadrant", "sector", "peripapillary_bin"):
        expected = assigned.groupby(axis).size().to_dict()
        actual = (
            region_summary[region_summary["region_axis"] == axis]
            .set_index("region_label")["n_points"]
            .to_dict()
        )
        for label, count in expected.items():
            assert int(actual[str(label)]) == int(count)


def test_compute_rigorous_bundle_marks_insufficient_point_domains():
    tissue_mask = np.ones((64, 64), dtype=bool)
    points = np.asarray([[20.0, 20.0], [40.0, 40.0]], dtype=float)
    object_table = _object_table_from_points(points)

    bundle = compute_rigorous_spatial_bundle(
        image_id="sample",
        object_table=object_table,
        image_shape=tissue_mask.shape,
        tissue_mask=tissue_mask,
        um_per_px=1.0,
        simulation_count=8,
        base_seed=5,
    )

    global_row = bundle["summary"].query("analysis_level == 'global'").iloc[0]
    assert global_row["status"] == "insufficient_points"
    assert np.isnan(global_row["l_global_p_value"])


def test_choose_valid_radii_drops_impossible_radii():
    tissue_mask = np.zeros((64, 64), dtype=bool)
    tissue_mask[18:32, 8:56] = True
    points = np.asarray([[24.0, 16.0], [24.0, 24.0], [24.0, 32.0], [24.0, 40.0], [24.0, 48.0]], dtype=float)

    payload = choose_valid_radii_px(points, tissue_mask, [3.0, 5.0, 10.0, 25.0])

    assert payload["used_radii_px"] == [3.0, 5.0]
    assert payload["status_reason"] == "ok"


def test_domain_status_not_ok_when_all_curves_nan():
    tissue_mask = np.zeros((64, 64), dtype=bool)
    tissue_mask[28:34, 8:56] = True
    points = np.asarray([[31.0, 12.0], [31.0, 20.0], [31.0, 28.0], [31.0, 36.0], [31.0, 44.0]], dtype=float)
    object_table = _object_table_from_points(points)

    bundle = compute_rigorous_spatial_bundle(
        image_id="sample",
        object_table=object_table,
        image_shape=tissue_mask.shape,
        tissue_mask=tissue_mask,
        um_per_px=1.0,
        radii_px=[25.0, 50.0, 75.0],
        simulation_count=8,
        base_seed=13,
    )

    global_row = bundle["summary"].query("analysis_level == 'global'").iloc[0]
    assert global_row["status"] != "ok"
    assert bool(global_row["spatial_curve_valid"]) is False
    assert int(global_row["n_finite_l"]) == 0
    assert int(global_row["n_finite_g"]) == 0
