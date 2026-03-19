from __future__ import annotations

import argparse
import json
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
    truth_provenance_valid = bool(row.get("truth_provenance_valid", True))
    truth_provenance_status = str(row.get("truth_provenance_status", "unknown")).strip() or "unknown"

    if n_rois < 20:
        return {"passed": False, "reason": "too_few_rois", "n_rois": n_rois}
    if not truth_provenance_valid or truth_provenance_status == "invalid":
        return {
            "passed": False,
            "reason": "invalid_truth_provenance",
            "n_rois": n_rois,
            "truth_provenance_status": truth_provenance_status,
        }
    if not matched_modality:
        return {"passed": False, "reason": "unmatched_modality"}
    if not pass_threshold:
        return {"passed": False, "reason": "benchmark_failed_threshold"}
    return {
        "passed": True,
        "reason": "benchmark_passed",
        "n_rois": n_rois,
        "truth_provenance_status": truth_provenance_status,
    }


def write_audit_artifacts(benchmark_dir: str | Path, audit: dict[str, object]) -> dict[str, Path]:
    benchmark_dir = Path(benchmark_dir)
    report_dir = benchmark_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "benchmark_audit.json"
    markdown_path = report_dir / "benchmark_audit.md"
    json_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(
        "# ROI Benchmark Audit\n\n"
        f"- Passed: `{bool(audit.get('passed'))}`\n"
        f"- Reason: `{audit.get('reason')}`\n"
        f"- ROIs: `{audit.get('n_rois', 'n/a')}`\n"
        f"- Truth provenance: `{audit.get('truth_provenance_status', 'unknown')}`\n",
        encoding="utf-8",
    )
    return {"json_path": json_path, "markdown_path": markdown_path}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit real ROI benchmark outputs.")
    parser.add_argument("benchmark_dir_positional", nargs="?", type=Path)
    parser.add_argument("--benchmark-dir", dest="benchmark_dir_option", type=Path, default=None)
    args = parser.parse_args(argv)

    benchmark_dir = args.benchmark_dir_option or args.benchmark_dir_positional
    if benchmark_dir is None:
        parser.error("benchmark_dir is required.")
    audit = audit_roi_benchmark_dir(benchmark_dir)
    if Path(benchmark_dir).exists():
        write_audit_artifacts(benchmark_dir, audit)
    print(audit["reason"])
    return 0 if bool(audit["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
