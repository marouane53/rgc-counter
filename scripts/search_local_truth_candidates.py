from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


KEYWORDS = (
    "sana",
    "el hajji",
    "mouse1",
    "il",
    "oht",
    "rbpms",
    "insulin",
    "saline",
    "density",
    "survival",
    "manual",
    "count",
    "point",
    "multipoint",
    "roi",
    "imagej",
    "alexa647",
    "cy5",
)
ALLOWED_SUFFIXES = {".csv", ".tsv", ".txt", ".xls", ".xlsx", ".ods", ".numbers", ".roi", ".zip", ".json", ".xml"}
SPOTLIGHT_ALLOWED_SUFFIXES = ALLOWED_SUFFIXES | {".ims"}
IGNORED_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "node_modules",
    "site-packages",
    "dist",
    "build",
}
TOKEN_RE = re.compile(r"[a-z0-9]+")


def default_search_roots() -> list[Path]:
    home = Path.home()
    candidates = [
        ROOT / "data",
        home / "Documents",
        home / "Downloads",
        home / "Desktop",
        home / "Library" / "Containers" / "com.apple.mail" / "Data" / "Library" / "Mail Downloads",
        home / "Library" / "Mail",
    ]
    return [path for path in candidates if path.exists()]


def _keyword_hits(text: str) -> list[str]:
    haystack = text.lower()
    tokens = TOKEN_RE.findall(haystack)
    token_set = set(tokens)
    hits: list[str] = []
    for keyword in KEYWORDS:
        if " " in keyword:
            if keyword in haystack:
                hits.append(keyword)
            continue
        if len(keyword) <= 3:
            if keyword in token_set:
                hits.append(keyword)
            continue
        if any(token == keyword or token.startswith(keyword) for token in token_set):
            hits.append(keyword)
    return hits


def _score_path(path: Path) -> tuple[int, list[str]]:
    text = f"{path.name} {' '.join(part for part in path.parts[-4:])}"
    hits = _keyword_hits(text)
    if not hits:
        return 0, []
    score = len(hits) * 10
    reasons = [f"keyword:{hit}" for hit in hits]
    if path.suffix.lower() in {".csv", ".tsv", ".json", ".xml", ".roi"}:
        score += 10
        reasons.append("point_friendly_extension")
    if path.suffix.lower() in {".xls", ".xlsx", ".ods", ".numbers"}:
        score += 5
        reasons.append("spreadsheet_extension")
    return score, reasons


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _skip_path(path: Path) -> bool:
    if any(part in IGNORED_DIR_NAMES for part in path.parts):
        return True
    private_root = ROOT / "test_subjects" / "private"
    if _is_relative_to(path, private_root):
        return True
    if _is_relative_to(path, ROOT) and not _is_relative_to(path, ROOT / "data"):
        return True
    return False


def classify_candidate(path: Path, reasons: list[str]) -> str:
    lower = path.name.lower()
    if any(token in lower for token in ("point", "multipoint", "roi", "manual")) and path.suffix.lower() in {".csv", ".tsv", ".json", ".xml", ".roi"}:
        return "external_same_image_point_truth"
    if path.suffix.lower() in {".xls", ".xlsx", ".ods", ".numbers"} and any(token in lower for token in ("count", "density", "survival")):
        return "weak_sanity_counts"
    if any("paper" in reason for reason in reasons):
        return "paper_sanity_only"
    return "candidate_review_needed"


def scan_zip_names(path: Path) -> list[str]:
    try:
        with zipfile.ZipFile(path) as archive:
            return archive.namelist()
    except Exception:
        return []


def scan_roots(roots: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            current_dir = Path(dirpath)
            dirnames[:] = [name for name in dirnames if not _skip_path(current_dir / name)]
            if _skip_path(current_dir):
                dirnames[:] = []
                continue
            for filename in filenames:
                path = current_dir / filename
                if _skip_path(path) or path.suffix.lower() not in ALLOWED_SUFFIXES:
                    continue
                score, reasons = _score_path(path)
                if score <= 0:
                    continue
                zip_names = scan_zip_names(path) if path.suffix.lower() == ".zip" else []
                if zip_names:
                    zip_hits = _keyword_hits(" ".join(zip_names))
                    score += len(zip_hits) * 3
                    reasons.extend(f"zip_member_keyword:{hit}" for hit in zip_hits)
                rows.append(
                    {
                        "path": str(path.resolve()),
                        "suffix": path.suffix.lower(),
                        "score": int(score),
                        "classification": classify_candidate(path, reasons),
                        "reasons": ";".join(sorted(dict.fromkeys(reasons))),
                    }
                )
    return rows


def spotlight_rows(roots: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    mdfind = Path("/usr/bin/mdfind")
    if not mdfind.exists():
        return rows
    query = " OR ".join(f'"{keyword}"' for keyword in KEYWORDS if " " not in keyword)
    for root in roots:
        completed = subprocess.run(
            [str(mdfind), "-onlyin", str(root), query],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode not in (0, 1):
            continue
        for line in completed.stdout.splitlines():
            path = Path(line.strip())
            if not path.is_file() or _skip_path(path) or path.suffix.lower() not in SPOTLIGHT_ALLOWED_SUFFIXES:
                continue
            score, reasons = _score_path(path)
            if score <= 0:
                continue
            rows.append(
                {
                    "path": str(path.resolve()),
                    "suffix": path.suffix.lower(),
                    "score": int(score + 2),
                    "classification": classify_candidate(path, reasons),
                    "reasons": ";".join(sorted(dict.fromkeys(reasons + ["spotlight_hit"]))),
                }
            )
    return rows


def scene_rows(metadata_dir: Path | None) -> list[dict[str, Any]]:
    if metadata_dir is None or not metadata_dir.exists():
        return []
    rows: list[dict[str, Any]] = []
    for json_path in sorted(metadata_dir.glob("*.json")):
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        scene = dict(payload.get("scene", {}))
        for candidate in scene.get("annotation_candidates", []):
            rows.append(
                {
                    "path": f"{payload.get('path')}::{candidate.get('path')}",
                    "suffix": ".ims",
                    "score": 100,
                    "classification": "embedded_same_image_point_truth",
                    "reasons": "embedded_scene_annotation_candidate",
                }
            )
    return rows


def build_report(frame: pd.DataFrame) -> str:
    def _markdown_table(local_frame: pd.DataFrame) -> str:
        if local_frame.empty:
            return "_No rows._"
        columns = list(local_frame.columns)
        lines = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join(["---"] * len(columns)) + " |",
        ]
        for row in local_frame.to_dict("records"):
            lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
        return "\n".join(lines)

    lines = [
        "# Truth Search Report",
        "",
        f"- Candidates: `{len(frame)}`",
        "",
    ]
    if frame.empty:
        lines.extend(["No candidate files or embedded scene annotations were found.", ""])
        return "\n".join(lines)
    lines.extend(["## Ranked Candidates", "", _markdown_table(frame), ""])
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search local roots for private truth candidates related to the Sana RBPMS `.ims` lane.")
    parser.add_argument("--roots", nargs="*", type=Path, default=None)
    parser.add_argument("--metadata-dir", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    roots = [path.expanduser().resolve() for path in (args.roots or default_search_roots()) if path.expanduser().exists()]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = scan_roots(roots)
    rows.extend(spotlight_rows(roots))
    rows.extend(scene_rows(args.metadata_dir.resolve() if args.metadata_dir is not None else None))
    frame = pd.DataFrame(rows).drop_duplicates(subset=["path"]).sort_values(["score", "path"], ascending=[False, True]).reset_index(drop=True)
    frame.to_csv(output_dir / "truth_search_candidates.csv", index=False)
    (output_dir / "truth_search_report.md").write_text(build_report(frame) + "\n", encoding="utf-8")
    (output_dir / "paper_sanity_only.md").write_text(
        "# Paper Sanity Only\n\nPublished figures or group-level summaries are context only and are not benchmark truth for this lane.\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
