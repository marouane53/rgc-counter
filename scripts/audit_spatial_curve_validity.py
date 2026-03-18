from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


def _resolve_curve_paths(root: Path) -> tuple[Path, list[Path]]:
    packet_curves = list((root / "03_reports").glob("**/*_spatial_curves.csv"))
    if packet_curves:
        return root, packet_curves
    run_curves = list(root.glob("**/*_spatial_curves.csv"))
    if run_curves:
        return root, run_curves
    raise FileNotFoundError(f"Could not find any *_spatial_curves.csv files under {root}")


def _status_lookup_for_curve(curve_path: Path) -> dict[tuple[str, str, str], str]:
    summary_path = curve_path.with_name(curve_path.name.replace("_spatial_curves.csv", "_spatial_summary.csv"))
    if not summary_path.exists():
        return {}
    summary = pd.read_csv(summary_path)
    required = {"analysis_level", "region_axis", "region_label", "status"}
    if not required.issubset(summary.columns):
        return {}
    return {
        (str(row["analysis_level"]), str(row["region_axis"]), str(row["region_label"])): str(row["status"])
        for row in summary.to_dict("records")
    }


def audit_curve_frames(curve_paths: list[Path]) -> dict[str, Any]:
    issues: list[str] = []
    invalid_domains: list[dict[str, Any]] = []
    nonfinite_domains: list[dict[str, Any]] = []

    for curve_path in curve_paths:
        frame = pd.read_csv(curve_path)
        status_lookup = _status_lookup_for_curve(curve_path)
        required = {"analysis_level", "region_axis", "region_label", "l_obs", "g_obs"}
        missing = sorted(required - set(frame.columns))
        if missing:
            issues.append(f"{curve_path}: missing required columns {missing}")
            continue

        for keys, group in frame.groupby(["analysis_level", "region_axis", "region_label"], dropna=False):
            finite_l = int(pd.to_numeric(group["l_obs"], errors="coerce").notna().sum())
            finite_g = int(pd.to_numeric(group["g_obs"], errors="coerce").notna().sum())
            if finite_l == 0 and finite_g == 0:
                status = status_lookup.get((str(keys[0]), str(keys[1]), str(keys[2])))
                payload = {
                    "curve_path": str(curve_path),
                    "analysis_level": keys[0],
                    "region_axis": keys[1],
                    "region_label": keys[2],
                    "status": status,
                    "issue": "all_curves_nonfinite",
                }
                nonfinite_domains.append(payload)
                if status == "ok" or status is None:
                    invalid_domains.append(payload)

    if invalid_domains:
        issues.append("Detected rigorous spatial domains marked ok (or missing summary status) despite having no finite L(r) or g(r) values.")

    return {
        "passed": not issues,
        "issues": issues,
        "invalid_domains": invalid_domains,
        "nonfinite_domains": nonfinite_domains,
        "n_curve_files": int(len(curve_paths)),
    }


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("Usage: python scripts/audit_spatial_curve_validity.py <run_or_packet_root>", file=sys.stderr)
        return 2
    try:
        root = Path(argv[1])
        _, curve_paths = _resolve_curve_paths(root)
        payload = audit_curve_frames(curve_paths)
        payload["root"] = str(root)
    except Exception as exc:
        print(json.dumps({"passed": False, "issues": [str(exc)]}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(payload, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
