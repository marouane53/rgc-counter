from pathlib import Path

import numpy as np
import pandas as pd

from src.run_service import RuntimeOptions, build_runtime, export_context, run_array


class FakeSegmenter:
    def __init__(self, labels: np.ndarray):
        self._labels = labels

    def segment(self, image: np.ndarray):
        return self._labels.copy(), {"backend": "fake"}


def test_run_service_exports_single_image_bundle(tmp_path: Path):
    image = np.zeros((16, 16), dtype=np.uint16)
    labels = np.zeros((16, 16), dtype=np.uint16)
    labels[2:5, 2:5] = 1
    labels[10:14, 9:13] = 2

    runtime = build_runtime(
        RuntimeOptions(
            backend="fake",
            focus_mode="none",
            save_debug=True,
            write_html_report=True,
            write_object_table=True,
            write_provenance=True,
        ),
        segmenter_override=FakeSegmenter(labels),
    )
    ctx = run_array(runtime, image=image, source_path="sample.tif", meta={"reader": "test"})
    artifacts = export_context(runtime, ctx, tmp_path)

    assert (tmp_path / "results.csv").exists()
    assert artifacts["debug_overlay"].exists()
    assert artifacts["object_table"].exists()
    assert artifacts["html_report"].exists()
    assert artifacts["provenance"].exists()
    results = pd.read_csv(tmp_path / "results.csv")
    assert "model_label" in results.columns
    assert results.loc[0, "model_source"] == "builtin"


def test_run_service_exports_retina_registration_artifacts(tmp_path: Path):
    image = np.zeros((16, 16), dtype=np.uint16)
    labels = np.zeros((16, 16), dtype=np.uint16)
    labels[2:5, 2:5] = 1
    labels[10:14, 9:13] = 2

    runtime = build_runtime(
        RuntimeOptions(
            backend="fake",
            focus_mode="none",
            register_retina=True,
            onh_mode="cli",
            onh_xy=(8.0, 8.0),
            dorsal_xy=(8.0, 0.0),
            write_object_table=True,
            write_provenance=True,
            write_html_report=False,
            save_debug=False,
        ),
        segmenter_override=FakeSegmenter(labels),
    )
    ctx = run_array(runtime, image=image, source_path="registered.tif", meta={"reader": "test"})
    artifacts = export_context(runtime, ctx, tmp_path)

    assert artifacts["retina_frame"].exists()
    assert artifacts["region_table"].exists()
    assert artifacts["registered_density_map_png"].exists()
    assert artifacts["registered_density_map_svg"].exists()


def test_run_service_marks_legacy_custom_model_type_in_provenance(tmp_path: Path):
    image = np.zeros((16, 16), dtype=np.uint16)
    labels = np.zeros((16, 16), dtype=np.uint16)
    labels[2:5, 2:5] = 1
    legacy_model = tmp_path / "legacy_model"
    legacy_model.write_text("legacy", encoding="utf-8")

    runtime = build_runtime(
        RuntimeOptions(
            backend="cellpose",
            model_type=str(legacy_model),
            focus_mode="none",
            write_provenance=True,
            write_object_table=False,
            write_html_report=False,
            save_debug=False,
        ),
        segmenter_override=FakeSegmenter(labels),
    )
    ctx = run_array(runtime, image=image, source_path="legacy.tif", meta={"reader": "test"})
    artifacts = export_context(runtime, ctx, tmp_path / "legacy_bundle")

    provenance = (tmp_path / "legacy_bundle" / "provenance.json").read_text(encoding="utf-8")
    assert "legacy_custom" in provenance
    assert artifacts["provenance"].exists()


def test_run_service_exports_rigorous_spatial_artifacts(tmp_path: Path):
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

    runtime = build_runtime(
        RuntimeOptions(
            backend="fake",
            focus_mode="none",
            spatial_stats=True,
            spatial_mode="rigorous",
            spatial_envelope_sims=8,
            spatial_random_seed=23,
            register_retina=True,
            onh_mode="cli",
            onh_xy=(64.0, 64.0),
            dorsal_xy=(64.0, 8.0),
            write_provenance=True,
            write_html_report=True,
            write_object_table=False,
            save_debug=False,
        ),
        segmenter_override=FakeSegmenter(labels),
    )
    ctx = run_array(runtime, image=image, source_path="rigorous.tif", meta={"reader": "test"})
    artifacts = export_context(runtime, ctx, tmp_path / "rigorous_bundle")

    assert artifacts["spatial_summary"].exists()
    assert artifacts["spatial_curves"].exists()
    assert artifacts["ripley_l_global_plot"].exists()
    assert artifacts["pair_correlation_global_plot"].exists()
    results = pd.read_csv(tmp_path / "rigorous_bundle" / "results.csv")
    assert results.loc[0, "spatial_mode"] == "rigorous"
    assert int(results.loc[0, "rigorous_global_point_count"]) == 8
    provenance = (tmp_path / "rigorous_bundle" / "provenance.json").read_text(encoding="utf-8")
    assert "spatial_analysis" in provenance


def test_run_service_exports_atlas_subtype_artifacts(tmp_path: Path):
    image = np.zeros((32, 32, 2), dtype=np.uint8)
    image[4:10, 4:10, 0] = 220
    image[18:24, 18:24, 1] = 220
    labels = np.zeros((32, 32), dtype=np.uint16)
    labels[4:10, 4:10] = 1
    labels[18:24, 18:24] = 2

    priors_path = tmp_path / "atlas_subtypes.yaml"
    priors_path.write_text(
        """
schema_version: 1
atlas_name: demo_subtypes
retina_region_schema: mouse_flatmount_v1
location_weight: 0.7
marker_weight: 0.3
channels:
  RBPMS: 0
  MELANOPSIN: 1
subtypes:
  alpha_rgc:
    location_priors:
      quadrant:
        weight: 1.0
        priors:
          dorsal_temporal: 0.9
          ventral_nasal: 0.2
    markers:
      - feature: channel.mean_bgsub.RBPMS
        direction: high
        center: 1.0
        scale: 0.5
        weight: 1.0
  iprgc:
    location_priors:
      quadrant:
        weight: 1.0
        priors:
          dorsal_temporal: 0.2
          ventral_nasal: 0.9
    markers:
      - feature: channel.mean_bgsub.MELANOPSIN
        direction: high
        center: 1.0
        scale: 0.5
        weight: 1.0
        """.strip(),
        encoding="utf-8",
    )

    runtime = build_runtime(
        RuntimeOptions(
            backend="fake",
            focus_mode="none",
            atlas_subtype_priors=str(priors_path),
            register_retina=True,
            onh_mode="cli",
            onh_xy=(16.0, 16.0),
            dorsal_xy=(16.0, 0.0),
            write_provenance=True,
            write_html_report=True,
            write_object_table=True,
            save_debug=False,
        ),
        segmenter_override=FakeSegmenter(labels),
    )
    ctx = run_array(runtime, image=image, source_path="atlas_subtypes.tif", meta={"reader": "test"})
    artifacts = export_context(runtime, ctx, tmp_path / "atlas_bundle")

    assert artifacts["atlas_subtype_summary"].exists()
    assert artifacts["atlas_subtype_region_summary"].exists()
    results = pd.read_csv(tmp_path / "atlas_bundle" / "results.csv")
    assert "cell_count" in results.columns
    object_table = pd.read_parquet(artifacts["object_table"])
    assert "atlas_subtype_top1" in object_table.columns
    provenance = (tmp_path / "atlas_bundle" / "provenance.json").read_text(encoding="utf-8")
    assert "atlas_subtypes" in provenance
