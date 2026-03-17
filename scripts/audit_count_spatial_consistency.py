from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


def _resolve_summary_path(root: Path) -> Path:
    candidates = [
        root / "study_summary.csv",
        root / "results.csv",
        root / "01_tables" / "tracked_example_study_summary.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find a summary CSV under {root}. "
        "Expected study_summary.csv, results.csv, or 01_tables/tracked_example_study_summary.csv."
    )


def audit_summary_frame(frame: pd.DataFrame) -> dict[str, Any]:
    issues: list[str] = []
    mismatches: list[dict[str, Any]] = []

    required = {"cell_count", "rigorous_global_point_count"}
    missing = sorted(required - set(frame.columns))
    if missing:
        issues.append(f"Missing required columns: {missing}")
        return {"passed": False, "issues": issues, "mismatches": mismatches}

    for row in frame.to_dict("records"):
        cell_count = float(row.get("cell_count") or 0)
        global_points = float(row.get("rigorous_global_point_count") or 0)
        if cell_count > 0 and global_points <= 0:
            mismatches.append(
                {
                    "sample_id": row.get("sample_id") or row.get("image_id") or row.get("filename"),
                    "filename": row.get("filename"),
                    "cell_count": cell_count,
                    "rigorous_global_point_count": global_points,
                }
            )

    if mismatches:
        issues.append("Detected samples with nonzero cell_count but zero rigorous_global_point_count.")

    return {
        "passed": not issues,
        "issues": issues,
        "mismatches": mismatches,
    }


def audit_path(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    summary_path = _resolve_summary_path(root)
    frame = pd.read_csv(summary_path)
    result = audit_summary_frame(frame)
    result["root"] = str(root)
    result["summary_path"] = str(summary_path)
    return result


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("Usage: python scripts/audit_count_spatial_consistency.py <run_or_packet_root>", file=sys.stderr)
        return 2
    try:
        payload = audit_path(argv[1])
    except Exception as exc:
        print(json.dumps({"passed": False, "issues": [str(exc)]}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(payload, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
