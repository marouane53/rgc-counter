from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def audit_micro_roi_benchmark_dir(benchmark_dir: str | Path) -> dict[str, object]:
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
    if not bool(row.get("truth_provenance_valid", False)) or str(row.get("truth_provenance_status", "unknown")) == "invalid":
        return {"passed": False, "reason": "invalid_truth_provenance"}
    if int(row.get("dev_n_rois", 0)) < 4 or int(row.get("locked_eval_n_rois", 0)) < 2:
        return {"passed": False, "reason": "incomplete_dev_locked_eval"}
    if not bool(row.get("pass_threshold", False)):
        return {"passed": False, "reason": "micro_benchmark_failed_threshold"}
    return {"passed": True, "reason": "micro_benchmark_passed"}


def write_audit_artifacts(benchmark_dir: str | Path, audit: dict[str, object]) -> dict[str, Path]:
    benchmark_dir = Path(benchmark_dir)
    report_dir = benchmark_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "micro_benchmark_audit.json"
    markdown_path = report_dir / "micro_benchmark_audit.md"
    json_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(
        "# Micro ROI Benchmark Audit\n\n"
        f"- Passed: `{bool(audit.get('passed'))}`\n"
        f"- Reason: `{audit.get('reason')}`\n",
        encoding="utf-8",
    )
    return {"json_path": json_path, "markdown_path": markdown_path}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit the private micro-ROI projection-lab benchmark.")
    parser.add_argument("benchmark_dir", type=Path)
    args = parser.parse_args(argv)

    audit = audit_micro_roi_benchmark_dir(args.benchmark_dir)
    if args.benchmark_dir.exists():
        write_audit_artifacts(args.benchmark_dir, audit)
    print(audit["reason"])
    return 0 if bool(audit["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
