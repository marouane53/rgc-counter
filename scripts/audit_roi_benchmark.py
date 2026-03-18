from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def audit_roi_benchmark_dir(benchmark_dir: str | Path) -> dict[str, object]:
    benchmark_dir = Path(benchmark_dir)
    required = [
        benchmark_dir / "results" / "config_comparison.csv",
        benchmark_dir / "results" / "best_config.json",
        benchmark_dir / "report" / "benchmark_quality.csv",
        benchmark_dir / "report" / "benchmark_report.md",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        return {"passed": False, "reason": "missing_required_files", "missing": missing}

    quality = pd.read_csv(benchmark_dir / "report" / "benchmark_quality.csv")
    if quality.empty:
        return {"passed": False, "reason": "empty_benchmark_quality"}

    row = quality.iloc[0]
    n_rois = int(row.get("n_rois", 0))
    matched_modality = bool(row.get("matched_modality", False))
    pass_threshold = bool(row.get("pass_threshold", False))

    if n_rois < 20:
        return {"passed": False, "reason": "too_few_rois", "n_rois": n_rois}
    if not matched_modality:
        return {"passed": False, "reason": "unmatched_modality"}
    if not pass_threshold:
        return {"passed": False, "reason": "benchmark_failed_threshold"}
    return {"passed": True, "reason": "benchmark_passed", "n_rois": n_rois}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit real ROI benchmark outputs.")
    parser.add_argument("benchmark_dir", type=Path)
    args = parser.parse_args(argv)

    audit = audit_roi_benchmark_dir(args.benchmark_dir)
    print(audit["reason"])
    return 0 if bool(audit["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
