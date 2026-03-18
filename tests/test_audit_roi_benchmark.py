from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.audit_roi_benchmark import audit_roi_benchmark_dir


def _write_benchmark_dir(root: Path, *, n_rois: int = 24, matched_modality: bool = True, pass_threshold: bool = True) -> Path:
    benchmark_dir = root / "benchmark"
    (benchmark_dir / "results").mkdir(parents=True)
    (benchmark_dir / "report").mkdir(parents=True)
    pd.DataFrame([{"config_id": "winner", "f1_mean_8px": 0.8}]).to_csv(
        benchmark_dir / "results" / "config_comparison.csv",
        index=False,
    )
    (benchmark_dir / "results" / "best_config.json").write_text('{"config_id":"winner"}\n', encoding="utf-8")
    pd.DataFrame(
        [
            {
                "benchmark_kind": "roi_point_matching",
                "matched_modality": matched_modality,
                "n_rois": n_rois,
                "precision": 0.8,
                "recall": 0.8,
                "f1": 0.8,
                "mae": 1.0,
                "pass_threshold": pass_threshold,
            }
        ]
    ).to_csv(benchmark_dir / "report" / "benchmark_quality.csv", index=False)
    (benchmark_dir / "report" / "benchmark_report.md").write_text("# report\n", encoding="utf-8")
    return benchmark_dir


def test_audit_roi_benchmark_fails_when_required_files_missing(tmp_path: Path):
    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir()

    audit = audit_roi_benchmark_dir(benchmark_dir)

    assert audit["passed"] is False
    assert audit["reason"] == "missing_required_files"


def test_audit_roi_benchmark_fails_when_too_few_rois(tmp_path: Path):
    benchmark_dir = _write_benchmark_dir(tmp_path, n_rois=5)

    audit = audit_roi_benchmark_dir(benchmark_dir)

    assert audit["passed"] is False
    assert audit["reason"] == "too_few_rois"


def test_audit_roi_benchmark_fails_when_threshold_not_passed(tmp_path: Path):
    benchmark_dir = _write_benchmark_dir(tmp_path, pass_threshold=False)

    audit = audit_roi_benchmark_dir(benchmark_dir)

    assert audit["passed"] is False
    assert audit["reason"] == "benchmark_failed_threshold"


def test_audit_roi_benchmark_passes_for_valid_outputs(tmp_path: Path):
    benchmark_dir = _write_benchmark_dir(tmp_path)

    audit = audit_roi_benchmark_dir(benchmark_dir)

    assert audit["passed"] is True
    assert audit["reason"] == "benchmark_passed"
