from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import tifffile

from scripts.run_roi_benchmark_suite import run_benchmark_suite


class FakeContext:
    def __init__(self, points_yx: list[list[float]]):
        self.object_table = pd.DataFrame(
            [{"centroid_y_px": y, "centroid_x_px": x, "kept": True} for y, x in points_yx]
        )


def _fake_runtime_builder(options):
    backend = options.backend or "cellpose"
    if backend == "blob_watershed" and (options.segmenter_config or {}).get("threshold_rel") == 0.18:
        label = "rbpms_blob_tight"
    elif backend == "blob_watershed":
        label = "blob_watershed_rbpms_preset"
    else:
        label = "cellpose_default"
    return SimpleNamespace(
        backend=backend,
        model_spec=SimpleNamespace(model_label=label),
    )


def _fake_runtime_runner(runtime, image, source_path, meta):
    if runtime.model_spec.model_label == "blob_watershed_rbpms_preset":
        return FakeContext([[10.0, 10.0], [20.0, 20.0]])
    if runtime.model_spec.model_label == "rbpms_blob_tight":
        return FakeContext([[10.0, 10.0]])
    return FakeContext([[10.0, 10.0]])


def _write_manifest(root: Path, *, n_rois: int = 24) -> Path:
    image_path = root / "roi_source.tif"
    tifffile.imwrite(image_path, np.zeros((32, 32), dtype=np.uint8))
    rows = []
    for index in range(n_rois):
        manual_path = root / f"manual_{index:03d}.csv"
        manual_path.write_text("x_px,y_px\n10,10\n20,20\n", encoding="utf-8")
        rows.append(
            {
                "roi_id": f"ROI_{index:03d}",
                "image_path": str(image_path),
                "marker": "RBPMS",
                "modality": "flatmount",
                "x0": 0,
                "y0": 0,
                "width": 32,
                "height": 32,
                "annotator": "tester",
                "manual_points_path": str(manual_path),
                "split": "benchmark",
                "notes": "",
            }
        )
    manifest = root / "roi_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest


def _write_config_manifest(root: Path) -> Path:
    path = root / "config_manifest.csv"
    pd.DataFrame(
        [
            {"config_id": "cellpose_default", "backend": "cellpose", "segmentation_preset": None, "diameter": 30, "min_size": 20, "max_size": 400, "apply_clahe": False, "modality": "flatmount", "modality_channel_index": 0, "segmenter_config_json": "{}", "object_filters_json": "{}", "notes": "baseline"},
            {"config_id": "blob_watershed_rbpms_preset", "backend": "blob_watershed", "segmentation_preset": "flatmount_rgc_rbpms_demo", "diameter": None, "min_size": 20, "max_size": 400, "apply_clahe": True, "modality": "flatmount", "modality_channel_index": 0, "segmenter_config_json": "{}", "object_filters_json": "{}", "notes": "winner"},
            {"config_id": "rbpms_blob_tight", "backend": "blob_watershed", "segmentation_preset": "flatmount_rgc_rbpms_demo", "diameter": None, "min_size": 20, "max_size": 300, "apply_clahe": True, "modality": "flatmount", "modality_channel_index": 0, "segmenter_config_json": "{\"threshold_rel\": 0.18}", "object_filters_json": "{\"min_mean_intensity\": 0.05}", "notes": "tight"},
        ]
    ).to_csv(path, index=False)
    return path


def test_suite_runner_writes_best_config_and_comparison(tmp_path: Path):
    manifest = _write_manifest(tmp_path)
    config_manifest = _write_config_manifest(tmp_path)
    output_dir = tmp_path / "suite_output"

    result = run_benchmark_suite(
        roi_manifest=manifest,
        config_manifest=config_manifest,
        output_dir=output_dir,
        runtime_builder=_fake_runtime_builder,
        runtime_runner=_fake_runtime_runner,
    )

    comparison = pd.read_csv(output_dir / "results" / "config_comparison.csv")
    best = json.loads((output_dir / "results" / "best_config.json").read_text(encoding="utf-8"))

    assert {"cellpose_default", "blob_watershed_rbpms_preset", "rbpms_blob_tight"} <= set(comparison["config_id"])
    assert "beats_baseline" in comparison.columns
    assert best["config_id"] == "blob_watershed_rbpms_preset"
    assert result["exit_code"] == 0


def test_suite_runner_exits_2_when_best_config_fails_threshold(tmp_path: Path):
    manifest = _write_manifest(tmp_path, n_rois=5)
    config_manifest = _write_config_manifest(tmp_path)
    output_dir = tmp_path / "suite_output"

    result = run_benchmark_suite(
        roi_manifest=manifest,
        config_manifest=config_manifest,
        output_dir=output_dir,
        runtime_builder=_fake_runtime_builder,
        runtime_runner=_fake_runtime_runner,
    )

    assert result["exit_code"] == 2


def test_suite_runner_handles_json_columns_in_config_manifest(tmp_path: Path):
    manifest = _write_manifest(tmp_path)
    config_manifest = _write_config_manifest(tmp_path)
    output_dir = tmp_path / "suite_output"

    run_benchmark_suite(
        roi_manifest=manifest,
        config_manifest=config_manifest,
        output_dir=output_dir,
        runtime_builder=_fake_runtime_builder,
        runtime_runner=_fake_runtime_runner,
    )

    comparison = pd.read_csv(output_dir / "results" / "config_comparison.csv")
    assert not comparison.empty
    assert (output_dir / "report" / "benchmark_quality.csv").exists()


def test_suite_runner_filters_out_qc_split_rows(tmp_path: Path):
    image_path = tmp_path / "roi_source.tif"
    tifffile.imwrite(image_path, np.zeros((32, 32), dtype=np.uint8))
    rows = []
    for index in range(22):
        manual_path = tmp_path / f"manual_{index:03d}.csv"
        manual_path.write_text("x_px,y_px\n10,10\n20,20\n", encoding="utf-8")
        rows.append(
            {
                "roi_id": f"DEV_{index:03d}",
                "image_path": str(image_path),
                "marker": "RBPMS",
                "modality": "flatmount",
                "x0": 0,
                "y0": 0,
                "width": 32,
                "height": 32,
                "annotator": "tester",
                "manual_points_path": str(manual_path),
                "split": "dev",
                "notes": "",
            }
        )
    for index in range(3):
        rows.append(
            {
                "roi_id": f"QC_{index:03d}",
                "image_path": str(image_path),
                "marker": "RBPMS",
                "modality": "flatmount",
                "x0": 0,
                "y0": 0,
                "width": 32,
                "height": 32,
                "annotator": "tester",
                "manual_points_path": "",
                "split": "qc_or_exclude",
                "notes": "exclude",
            }
        )
    manifest = tmp_path / "roi_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    config_manifest = _write_config_manifest(tmp_path)
    output_dir = tmp_path / "suite_output"

    result = run_benchmark_suite(
        roi_manifest=manifest,
        config_manifest=config_manifest,
        output_dir=output_dir,
        runtime_builder=_fake_runtime_builder,
        runtime_runner=_fake_runtime_runner,
        include_splits=["dev"],
    )

    comparison = pd.read_csv(output_dir / "results" / "config_comparison.csv")
    assert result["exit_code"] == 0
    assert set(comparison["n_rois"]) == {22}
