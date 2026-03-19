from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from scripts.audit_micro_roi_benchmark import audit_micro_roi_benchmark_dir
from scripts.run_micro_roi_benchmark_suite import main as run_micro_benchmark_main
from src.micro_roi_benchmark import build_default_projection_lab_config_manifest, moderate_locked_eval_gate


def _synthetic_image() -> np.ndarray:
    yy, xx = np.mgrid[0:64, 0:64]
    image = (
        1.2 * np.exp(-((yy - 20.0) ** 2 + (xx - 20.0) ** 2) / (2.0 * 3.0**2))
        + 1.0 * np.exp(-((yy - 44.0) ** 2 + (xx - 42.0) ** 2) / (2.0 * 3.5**2))
    )
    return image.astype(np.float32)


def test_default_projection_lab_config_manifest_has_required_columns():
    frame = build_default_projection_lab_config_manifest()

    assert {"config_id", "projection_recipe_json", "preprocess_json", "predictor_backend", "predictor_config_json"} <= set(frame.columns)
    assert not frame.empty


def test_moderate_locked_eval_gate_accepts_and_rejects_expected_cases():
    assert moderate_locked_eval_gate(
        {
            "dev_n_rois": 4,
            "locked_eval_n_rois": 2,
            "dev_f1_8px": 0.7,
            "dev_recall_8px": 0.7,
            "locked_eval_f1_8px": 0.6,
            "locked_eval_recall_8px": 0.6,
        }
    )
    assert not moderate_locked_eval_gate(
        {
            "dev_n_rois": 4,
            "locked_eval_n_rois": 2,
            "dev_f1_8px": 0.7,
            "dev_recall_8px": 0.7,
            "locked_eval_f1_8px": 0.4,
            "locked_eval_recall_8px": 0.6,
        }
    )


def test_run_micro_roi_benchmark_suite_writes_quality_summary(tmp_path: Path):
    image = _synthetic_image()
    detector_input_path = tmp_path / "detector_input.tif"
    tifffile.imwrite(detector_input_path, image.astype(np.float32))
    source_image_path = tmp_path / "source_image.tif"
    tifffile.imwrite(source_image_path, image.astype(np.float32))

    manifest_rows = []
    view_rows = []
    for index, split in enumerate(["dev", "dev", "dev", "dev", "locked_eval", "locked_eval"]):
        manual_path = tmp_path / f"manual_{index}.csv"
        pd.DataFrame([{"x_px": 20.0, "y_px": 20.0}, {"x_px": 42.0, "y_px": 44.0}]).to_csv(manual_path, index=False)
        manifest_rows.append(
            {
                "roi_id": f"ROI_{index}",
                "image_path": str(source_image_path),
                "marker": "RBPMS",
                "modality": "flatmount",
                "x0": 0,
                "y0": 0,
                "width": 64,
                "height": 64,
                "annotator": "tester",
                "manual_points_path": str(manual_path),
                "split": split,
                "notes": "",
                "image_marker": "RBPMS",
                "image_source_channel": 1,
                "truth_marker": "RBPMS",
                "truth_source_channel": 1,
                "truth_derivation": "assisted_curated_point_truth",
                "truth_source_path": str(source_image_path),
            }
        )
        view_rows.append(
            {
                "roi_id": f"ROI_{index}",
                "split": split,
                "projection_id": "full_max",
                "preprocess_id": "none",
                "projection_recipe_json": json.dumps({"projection_id": "full_max", "projection_kind": "full_max"}, sort_keys=True, separators=(",", ":")),
                "preprocess_json": json.dumps({"background_subtraction": "none", "normalization": "robust_float", "preprocess_id": "none"}, sort_keys=True, separators=(",", ":")),
                "detector_input_path": str(detector_input_path),
                "exclude_mask_path": "",
            }
        )

    manifest_path = tmp_path / "micro_roi_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    view_manifest_path = tmp_path / "view_manifest.csv"
    pd.DataFrame(view_rows).to_csv(view_manifest_path, index=False)
    config_manifest_path = tmp_path / "config_manifest.csv"
    pd.DataFrame(
        [
            {
                "config_id": "hmax_smoke",
                "projection_recipe_json": view_rows[0]["projection_recipe_json"],
                "preprocess_json": view_rows[0]["preprocess_json"],
                "predictor_backend": "hmax",
                "predictor_config_json": json.dumps({"h": 0.08, "min_distance": 8}),
            }
        ]
    ).to_csv(config_manifest_path, index=False)

    output_dir = tmp_path / "micro_benchmark"
    exit_code = run_micro_benchmark_main(
        [
            "--manifest",
            str(manifest_path),
            "--view-manifest",
            str(view_manifest_path),
            "--config-manifest",
            str(config_manifest_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    quality = pd.read_csv(output_dir / "report" / "benchmark_quality.csv")
    assert quality.loc[0, "benchmark_kind"] == "micro_roi_projection_lab"
    assert int(quality.loc[0, "dev_n_rois"]) == 4
    assert int(quality.loc[0, "locked_eval_n_rois"]) == 2


def test_audit_micro_roi_benchmark_fails_invalid_truth_provenance(tmp_path: Path):
    benchmark_dir = tmp_path / "benchmark"
    (benchmark_dir / "results").mkdir(parents=True)
    (benchmark_dir / "report").mkdir(parents=True)
    pd.DataFrame([{"config_id": "winner"}]).to_csv(benchmark_dir / "results" / "config_comparison.csv", index=False)
    (benchmark_dir / "results" / "best_config.json").write_text('{"config_id":"winner"}\n', encoding="utf-8")
    pd.DataFrame(
        [
            {
                "benchmark_kind": "micro_roi_projection_lab",
                "truth_provenance_valid": False,
                "truth_provenance_status": "invalid",
                "dev_n_rois": 4,
                "locked_eval_n_rois": 2,
                "dev_f1_8px": 0.8,
                "dev_recall_8px": 0.8,
                "locked_eval_f1_8px": 0.7,
                "locked_eval_recall_8px": 0.7,
                "pass_threshold": False,
            }
        ]
    ).to_csv(benchmark_dir / "report" / "benchmark_quality.csv", index=False)
    (benchmark_dir / "report" / "benchmark_report.md").write_text("# report\n", encoding="utf-8")

    audit = audit_micro_roi_benchmark_dir(benchmark_dir)

    assert audit["passed"] is False
    assert audit["reason"] == "invalid_truth_provenance"
