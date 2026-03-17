from pathlib import Path

import numpy as np

from src.context import RunContext
from src.pipeline import build_default_pipeline


class FakeSegmenter:
    def __init__(self, labels: np.ndarray):
        self._labels = labels

    def segment(self, image: np.ndarray):
        return self._labels.copy(), {"backend": "fake"}


def test_default_pipeline_populates_summary_and_object_table():
    image = np.zeros((16, 16), dtype=np.uint16)
    labels = np.zeros((16, 16), dtype=np.uint16)
    labels[2:5, 2:5] = 1
    labels[9:13, 9:13] = 2

    ctx = RunContext(path=Path("sample.tif"), image=image, meta={"reader": "test"})
    pipeline = build_default_pipeline(FakeSegmenter(labels))
    cfg = {
        "apply_clahe": False,
        "focus_mode": "none",
        "tta": False,
        "tta_transforms": None,
        "min_size": 1,
        "max_size": 1000,
        "spatial_stats": False,
        "backend": "fake",
        "use_gpu": False,
    }

    out = pipeline.run(ctx, cfg)

    assert out.summary_row["cell_count"] == 2
    assert out.summary_row["backend"] == "fake"
    assert out.object_table is not None
    assert len(out.object_table) == 2


def test_pipeline_populates_region_outputs_when_registration_enabled():
    image = np.zeros((16, 16), dtype=np.uint16)
    labels = np.zeros((16, 16), dtype=np.uint16)
    labels[2:5, 2:5] = 1
    labels[9:13, 9:13] = 2

    ctx = RunContext(path=Path("sample.tif"), image=image, meta={"reader": "test"})
    pipeline = build_default_pipeline(FakeSegmenter(labels))
    cfg = {
        "apply_clahe": False,
        "focus_mode": "none",
        "tta": False,
        "tta_transforms": None,
        "min_size": 1,
        "max_size": 1000,
        "spatial_stats": False,
        "backend": "fake",
        "use_gpu": False,
        "register_retina": True,
        "region_schema": "mouse_flatmount_v1",
        "onh_mode": "cli",
        "onh_xy": (8.0, 8.0),
        "dorsal_xy": (8.0, 0.0),
        "retina_frame_path": None,
    }

    out = pipeline.run(ctx, cfg)

    assert out.object_table is not None
    assert out.region_table is not None
    assert "ret_x_um" in out.object_table.columns
    assert "ring" in out.object_table.columns
    assert out.summary_row["retina_registered"] is True
    assert "retina_frame" in out.state


def test_pipeline_runs_v2_phenotype_engine_and_interaction_metrics():
    image = np.zeros((32, 32, 2), dtype=np.uint8)
    image[4:10, 4:10, 0] = 220
    image[18:24, 18:24, 0] = 220
    image[18:24, 18:24, 1] = 230

    labels = np.zeros((32, 32), dtype=np.uint16)
    labels[4:10, 4:10] = 1
    labels[18:24, 18:24] = 2

    ctx = RunContext(path=Path("sample.tif"), image=image, meta={"reader": "test"})
    config = {
        "schema_version": 2,
        "channels": {"RBPMS": 0, "MICROGLIA": 1},
        "masks": {
            "rgc_positive": {"channel": "RBPMS", "min_intensity": 180},
            "microglia": {"channel": "MICROGLIA", "min_intensity": 180},
        },
        "classes": {
            "rgc": {
                "priority": 100,
                "include": [
                    {"feature": "relation.overlap_fraction", "target": "rgc_positive", "op": "gt", "value": 0.0}
                ],
                "exclude": [
                    {"feature": "relation.overlap_fraction", "target": "microglia", "op": "gt", "value": 0.0}
                ],
            },
            "microglia": {
                "priority": 90,
                "include": [
                    {"feature": "relation.overlap_fraction", "target": "microglia", "op": "gt", "value": 0.0}
                ],
                "exclude": [],
            },
        },
    }
    pipeline = build_default_pipeline(FakeSegmenter(labels), phenotype_engine_config=config)
    cfg = {
        "apply_clahe": False,
        "focus_mode": "none",
        "tta": False,
        "tta_transforms": None,
        "min_size": 1,
        "max_size": 1000,
        "spatial_stats": False,
        "backend": "fake",
        "use_gpu": False,
        "phenotype_engine": "v2",
        "marker_metrics": True,
        "interaction_metrics": True,
        "register_retina": False,
        "region_schema": "mouse_flatmount_v1",
        "onh_mode": "cli",
        "onh_xy": None,
        "dorsal_xy": None,
        "retina_frame_path": None,
    }

    out = pipeline.run(ctx, cfg)

    assert list(out.object_table["phenotype"]) == ["rgc", "microglia"]
    assert "interaction.nearest_any_px" in out.object_table.columns
    assert out.metrics["phenotype_counts"]["rgc"] == 1


def test_pipeline_populates_rigorous_spatial_outputs_when_enabled():
    image = np.zeros((128, 128), dtype=np.uint16)
    image[8:120, 8:120] = 200
    labels = np.zeros((128, 128), dtype=np.uint16)
    labels[18:28, 18:28] = 1
    labels[18:28, 56:66] = 2
    labels[18:28, 96:106] = 3
    labels[56:66, 18:28] = 4
    labels[56:66, 96:106] = 5
    labels[96:106, 18:28] = 6
    labels[96:106, 56:66] = 7
    labels[96:106, 96:106] = 8

    ctx = RunContext(path=Path("sample.tif"), image=image, meta={"reader": "test"})
    pipeline = build_default_pipeline(FakeSegmenter(labels))
    cfg = {
        "apply_clahe": False,
        "focus_mode": "none",
        "tta": False,
        "tta_transforms": None,
        "min_size": 1,
        "max_size": 1000,
        "spatial_stats": True,
        "spatial_mode": "rigorous",
        "spatial_envelope_sims": 8,
        "spatial_random_seed": 19,
        "backend": "fake",
        "use_gpu": False,
        "register_retina": True,
        "region_schema": "mouse_flatmount_v1",
        "onh_mode": "cli",
        "onh_xy": (64.0, 64.0),
        "dorsal_xy": (64.0, 8.0),
        "retina_frame_path": None,
    }

    out = pipeline.run(ctx, cfg)

    assert out.summary_row["spatial_mode"] == "rigorous"
    assert out.summary_row["rigorous_global_point_count"] == 8
    assert out.metrics["object_flow"]["n_objects_kept"] == 8
    assert out.metrics["object_flow"]["n_points_global_domain"] == 8
    assert "rigorous_spatial" in out.state
    summary = out.state["rigorous_spatial"]["summary"]
    assert {"global", "ring", "quadrant", "sector", "peripapillary_bin"}.issubset(set(summary["region_axis"]))


def test_pipeline_warns_when_rigorous_spatial_has_too_few_points():
    image = np.zeros((64, 64), dtype=np.uint16)
    image[4:60, 4:60] = 200
    labels = np.zeros((64, 64), dtype=np.uint16)
    labels[8:14, 8:14] = 1
    labels[40:46, 40:46] = 2

    ctx = RunContext(path=Path("sample.tif"), image=image, meta={"reader": "test"})
    pipeline = build_default_pipeline(FakeSegmenter(labels))
    cfg = {
        "apply_clahe": False,
        "focus_mode": "none",
        "tta": False,
        "tta_transforms": None,
        "min_size": 1,
        "max_size": 1000,
        "spatial_stats": True,
        "spatial_mode": "rigorous",
        "spatial_envelope_sims": 8,
        "spatial_random_seed": 19,
        "backend": "fake",
        "use_gpu": False,
        "register_retina": False,
    }

    out = pipeline.run(ctx, cfg)

    assert any("too few points" in warning for warning in out.warnings)


def test_pipeline_runs_atlas_subtype_priors_with_registration():
    image = np.zeros((32, 32, 2), dtype=np.uint8)
    image[4:10, 4:10, 0] = 220
    image[18:24, 18:24, 1] = 220
    labels = np.zeros((32, 32), dtype=np.uint16)
    labels[4:10, 4:10] = 1
    labels[18:24, 18:24] = 2

    priors = {
        "schema_version": 1,
        "config_path": "demo.yaml",
        "atlas_name": "demo_subtypes",
        "retina_region_schema": "mouse_flatmount_v1",
        "location_weight": 0.7,
        "marker_weight": 0.3,
        "channels": {"RBPMS": 0, "MELANOPSIN": 1},
        "compose": {},
        "subtypes": {
            "alpha_rgc": {
                "slug": "alpha_rgc",
                "location_priors": {"quadrant": {"weight": 1.0, "priors": {"dorsal_temporal": 0.9, "ventral_nasal": 0.2}}},
                "markers": [{"feature": "channel.mean_bgsub.RBPMS", "direction": "high", "center": 1.0, "scale": 0.5, "weight": 1.0}],
            },
            "iprgc": {
                "slug": "iprgc",
                "location_priors": {"quadrant": {"weight": 1.0, "priors": {"dorsal_temporal": 0.2, "ventral_nasal": 0.9}}},
                "markers": [{"feature": "channel.mean_bgsub.MELANOPSIN", "direction": "high", "center": 1.0, "scale": 0.5, "weight": 1.0}],
            },
        },
    }

    ctx = RunContext(path=Path("sample.tif"), image=image, meta={"reader": "test"})
    pipeline = build_default_pipeline(FakeSegmenter(labels))
    cfg = {
        "apply_clahe": False,
        "focus_mode": "none",
        "tta": False,
        "tta_transforms": None,
        "min_size": 1,
        "max_size": 1000,
        "spatial_stats": False,
        "backend": "fake",
        "use_gpu": False,
        "register_retina": True,
        "region_schema": "mouse_flatmount_v1",
        "onh_mode": "cli",
        "onh_xy": (16.0, 16.0),
        "dorsal_xy": (16.0, 0.0),
        "retina_frame_path": None,
        "atlas_subtype_priors_config": priors,
    }

    out = pipeline.run(ctx, cfg)

    assert "atlas_subtype_top1" in out.object_table.columns
    assert "atlas_subtypes" in out.state
    assert out.metrics["atlas_subtype_top1_counts"]["alpha_rgc"] == 1
    assert not out.state["atlas_subtypes"]["summary"].empty
    assert not out.state["atlas_subtypes"]["region_summary"].empty


def test_pipeline_runs_atlas_subtype_priors_without_registration_with_warning():
    image = np.zeros((32, 32, 2), dtype=np.uint8)
    image[4:10, 4:10, 0] = 220
    labels = np.zeros((32, 32), dtype=np.uint16)
    labels[4:10, 4:10] = 1

    priors = {
        "schema_version": 1,
        "config_path": "demo.yaml",
        "atlas_name": "demo_subtypes",
        "retina_region_schema": "mouse_flatmount_v1",
        "location_weight": 0.7,
        "marker_weight": 0.3,
        "channels": {"RBPMS": 0},
        "compose": {},
        "subtypes": {
            "alpha_rgc": {
                "slug": "alpha_rgc",
                "location_priors": {"quadrant": {"weight": 1.0, "priors": {"dorsal_temporal": 0.9}}},
                "markers": [{"feature": "channel.mean_bgsub.RBPMS", "direction": "high", "center": 1.0, "scale": 0.5, "weight": 1.0}],
            }
        },
    }

    ctx = RunContext(path=Path("sample.tif"), image=image, meta={"reader": "test"})
    pipeline = build_default_pipeline(FakeSegmenter(labels))
    cfg = {
        "apply_clahe": False,
        "focus_mode": "none",
        "tta": False,
        "tta_transforms": None,
        "min_size": 1,
        "max_size": 1000,
        "spatial_stats": False,
        "backend": "fake",
        "use_gpu": False,
        "register_retina": False,
        "atlas_subtype_priors_config": priors,
    }

    out = pipeline.run(ctx, cfg)

    assert "atlas_subtype_top1" in out.object_table.columns
    assert out.state["atlas_subtypes"]["region_summary"].empty
    assert any("marker evidence only" in warning for warning in out.warnings)
