#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.advisor_packet import audit_advisor_packet


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check advisor packet internal consistency.")
    parser.add_argument("packet_root", type=Path, help="Path to paper_evidence/10_ai_advisor_packet")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    audit = audit_advisor_packet(args.packet_root)
    if args.json:
        print(json.dumps(audit, indent=2))
    else:
        print(f"Auditing: {args.packet_root}")
        for item in audit["report_references"]:
            print(f"{item['report']}: refs missing = {len(item['missing'])}")
            for missing in item["missing"]:
                print(f"  MISSING: {missing}")
        print(f"Tracked study match: {audit['tracked_study']['matches']}")
        print(f"Manual benchmark match: {audit['manual_benchmark']['matches']}")
        for row in audit["figure_hashes"]:
            print(f"{row['target']} == {row['source']} ? {row['matches']}")
        print(f"Pytest counts: {audit['pytest_counts']}")
        print(f"Repo snapshot status: {audit['repo_snapshot_status']}")
        if audit["issues"]:
            print("\nIssues:")
            for issue in audit["issues"]:
                print(f"- {issue}")
    return 0 if audit["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
