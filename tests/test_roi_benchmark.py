from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import tifffile

from scripts.run_roi_benchmark import run_roi_benchmark_config
from src.roi_benchmark import (
    RoiBenchmarkConfig,
    build_roi_benchmark_report,
    run_single_roi_case,
    summarize_config_results,
)
from src.roi_data import iter_roi_records, load_roi_manifest


class FakeContext:
    def __init__(self, points_yx: list[list[float]]):
        self.object_table = pd.DataFrame(
            [{"centroid_y_px": y, "centroid_x_px": x, "kept": True} for y, x in points_yx]
        )


def _fake_runtime_builder(options):
    backend = options.backend or "cellpose"
    return SimpleNamespace(
        backend=backend,
        model_spec=SimpleNamespace(model_label=backend),
    )


def _fake_runtime_runner(runtime, image, source_path, meta):
    if runtime.backend == "blob_watershed":
        return FakeContext([[10.0, 10.0], [20.0, 20.0]])
    return FakeContext([[10.0, 10.0]])


def _write_manifest(root: Path) -> Path:
    image_path = root / "roi_source.tif"
    manual_points = root / "manual_points.csv"
    tifffile.imwrite(image_path, np.zeros((32, 32), dtype=np.uint8))
    manual_points.write_text("x_px,y_px\n10,10\n20,20\n", encoding="utf-8")
    manifest = root / "roi_manifest.csv"
    pd.DataFrame(
        [
            {
                "roi_id": "ROI_001",
                "image_path": str(image_path),
                "marker": "RBPMS",
                "modality": "flatmount",
                "x0": 0,
                "y0": 0,
                "width": 32,
                "height": 32,
                "annotator": "tester",
                "manual_points_path": str(manual_points),
                "split": "benchmark",
                "notes": "",
            }
        ]
    ).to_csv(manifest, index=False)
    return manifest


def test_run_single_roi_case_on_synthetic_spots(tmp_path: Path):
    manifest = load_roi_manifest(_write_manifest(tmp_path))
    record = iter_roi_records(manifest, manifest_path=tmp_path / "roi_manifest.csv")[0]
    config = RoiBenchmarkConfig(config_id="blob", backend="blob_watershed")

    primary, per_tol = run_single_roi_case(
        record,
        config=config,
        runtime_builder=_fake_runtime_builder,
        runtime_runner=_fake_runtime_runner,
    )

    assert primary["config_id"] == "blob"
    assert primary["true_positive"] == 2
    assert len(per_tol) == 3


def test_summarize_config_results_ranks_by_f1_then_recall_then_mae():
    frame = pd.DataFrame(
        [
            {"config_id": "a", "roi_id": "R1", "backend": "blob_watershed", "model_label": "a", "segmentation_preset": None, "marker": "RBPMS", "modality": "flatmount", "precision": 0.9, "recall": 0.8, "f1": 0.85, "count_mae": 1.0, "count_bias": 0.0, "manual_count": 10, "match_tolerance_px": 8.0, "runtime_seconds": 0.2},
            {"config_id": "b", "roi_id": "R1", "backend": "blob_watershed", "model_label": "b", "segmentation_preset": None, "marker": "RBPMS", "modality": "flatmount", "precision": 0.85, "recall": 0.82, "f1": 0.83, "count_mae": 0.5, "count_bias": 0.0, "manual_count": 10, "match_tolerance_px": 8.0, "runtime_seconds": 0.1},
            {"config_id": "a", "roi_id": "R1", "backend": "blob_watershed", "model_label": "a", "segmentation_preset": None, "marker": "RBPMS", "modality": "flatmount", "precision": 0.9, "recall": 0.8, "f1": 0.85, "count_mae": 1.0, "count_bias": 0.0, "manual_count": 10, "match_tolerance_px": 6.0, "runtime_seconds": 0.2},
            {"config_id": "b", "roi_id": "R1", "backend": "blob_watershed", "model_label": "b", "segmentation_preset": None, "marker": "RBPMS", "modality": "flatmount", "precision": 0.85, "recall": 0.82, "f1": 0.83, "count_mae": 0.5, "count_bias": 0.0, "manual_count": 10, "match_tolerance_px": 6.0, "runtime_seconds": 0.1},
        ]
    )

    summary = summarize_config_results(frame)

    assert summary.iloc[0]["config_id"] == "a"
    assert summary.iloc[0]["rank"] == 1


def test_benchmark_report_calls_out_failed_threshold():
    manifest = pd.DataFrame(
        [{"roi_id": "ROI_001", "image_path": "x.tif", "marker": "RBPMS", "modality": "flatmount", "x0": 0, "y0": 0, "width": 32, "height": 32, "annotator": "tester", "manual_points_path": "x.csv", "split": "benchmark", "notes": ""}]
    )
    config_summary = pd.DataFrame(
        [
            {
                "config_id": "cellpose_default",
                "backend": "cellpose",
                "segmentation_preset": None,
                "precision_mean_8px": 0.6,
                "recall_mean_8px": 0.6,
                "f1_mean_8px": 0.6,
                "count_mae_mean_8px": 4.0,
                "pass_threshold": False,
            }
        ]
    )
    per_roi_primary = pd.DataFrame([{"roi_id": "ROI_001", "f1": 0.6, "count_mae": 4.0}])

    report = build_roi_benchmark_report(
        roi_manifest=manifest,
        config_summary=config_summary,
        per_roi_primary=per_roi_primary,
    )

    assert "did not pass the project acceptance threshold" in report


def test_run_roi_benchmark_writes_expected_artifacts(tmp_path: Path):
    manifest = _write_manifest(tmp_path)
    output_dir = tmp_path / "output"

    result = run_roi_benchmark_config(
        roi_manifest=manifest,
        output_dir=output_dir,
        config_id="blob_config",
        backend="blob_watershed",
        save_overlays=True,
        runtime_builder=_fake_runtime_builder,
        runtime_runner=_fake_runtime_runner,
    )

    assert (output_dir / "results" / "per_roi_metrics.csv").exists()
    assert (output_dir / "results" / "per_roi_tolerance_metrics.csv").exists()
    assert (output_dir / "results" / "config_comparison.csv").exists()
    assert (output_dir / "results" / "best_config.json").exists()
    assert (output_dir / "report" / "benchmark_report.md").exists()
    assert (output_dir / "report" / "benchmark_quality.csv").exists()
    assert any((output_dir / "results" / "overlays").glob("*.png"))

    best = json.loads((output_dir / "results" / "best_config.json").read_text(encoding="utf-8"))
    assert best["config_id"] == "blob_config"
    assert result["summary_row"]["config_id"] == "blob_config"
