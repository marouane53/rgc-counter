from __future__ import annotations

import hashlib
import json
import re
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pandas as pd

from src.report import find_missing_report_references


TRACKED_STUDY_COMPARE_COLUMNS = ["sample_id", "filename", "cell_count", "warning_count"]
MANUAL_BENCHMARK_COMPARE_COLUMNS = ["sample_id", "filename", "manual_count", "cell_count"]
TRACKED_FIGURE_MAPPINGS = [
    ("02_images/tracked_example_summary.png", "03_reports/tracked_example/figures/cell_count_by_condition.png"),
    (
        "02_images/tracked_example_agreement_scatter.png",
        "03_reports/tracked_example_manual_validation/validation/agreement_scatter.png",
    ),
    (
        "02_images/tracked_example_bland_altman.png",
        "03_reports/tracked_example_manual_validation/validation/bland_altman.png",
    ),
]
REAL_ROI_BENCHMARK_FILES = [
    "01_tables/real_roi_config_comparison.csv",
    "01_tables/real_roi_benchmark_quality.csv",
    "03_reports/real_roi_benchmark/benchmark_report.md",
]
PYTEST_PATTERNS = [
    re.compile(r"(\d+)\s+passed"),
    re.compile(r"passed with\s+(\d+)\s+tests"),
    re.compile(r"(\d+)\s+tests"),
]
ROOT = Path(__file__).resolve().parents[1]
PRIVATE_TEST_SUBJECTS_ROOT = ROOT / "test_subjects" / "private"


class _SimpleHTMLTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self._current_table: list[list[str]] | None = None
        self._current_row: list[str] | None = None
        self._current_cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self._current_table = []
        elif tag == "tr" and self._current_table is not None:
            self._current_row = []
        elif tag in {"th", "td"} and self._current_row is not None:
            self._current_cell = []

    def handle_data(self, data: str) -> None:
        if self._current_cell is not None:
            self._current_cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"th", "td"} and self._current_row is not None and self._current_cell is not None:
            self._current_row.append(unescape("".join(self._current_cell).strip()))
            self._current_cell = None
        elif tag == "tr" and self._current_table is not None and self._current_row is not None:
            if any(cell for cell in self._current_row):
                self._current_table.append(self._current_row)
            self._current_row = None
        elif tag == "table" and self._current_table is not None:
            if self._current_table:
                self.tables.append(self._current_table)
            self._current_table = None


def is_private_roi_benchmark_dir(path: str | Path | None) -> bool:
    if path is None:
        return False
    resolved = Path(path).expanduser().resolve()
    if resolved == PRIVATE_TEST_SUBJECTS_ROOT or PRIVATE_TEST_SUBJECTS_ROOT in resolved.parents:
        return True
    parts = [part.lower() for part in resolved.parts]
    if "test_subjects" in parts:
        index = parts.index("test_subjects")
        return "private" in parts[index + 1 :]
    return False


def assert_public_roi_benchmark_export_allowed(
    roi_benchmark_dir: str | Path | None,
    *,
    allow_private_roi_benchmark_export: bool = False,
) -> None:
    if roi_benchmark_dir is None:
        return
    if is_private_roi_benchmark_dir(roi_benchmark_dir) and not allow_private_roi_benchmark_export:
        raise ValueError(
            "Refusing to export a private ROI benchmark directory into the advisor packet without "
            "--allow-private-roi-benchmark-export."
        )


def file_sha256(path: str | Path) -> str:
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_hash_rows(packet_root: str | Path, subdirs: tuple[str, ...] = ("01_tables", "02_images", "03_reports")) -> list[dict[str, Any]]:
    packet_root = Path(packet_root)
    rows: list[dict[str, Any]] = []
    for subdir in subdirs:
        base = packet_root / subdir
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            rows.append(
                {
                    "relative_path": str(path.relative_to(packet_root)),
                    "sha256": file_sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    return rows


def extract_pytest_count(text: str) -> int | None:
    for pattern in PYTEST_PATTERNS:
        match = pattern.search(text)
        if match:
            return int(match.group(1))
    return None


def read_html_tables(report_html: str | Path) -> list[pd.DataFrame]:
    parser = _SimpleHTMLTableParser()
    parser.feed(Path(report_html).read_text(encoding="utf-8"))
    tables: list[pd.DataFrame] = []
    for rows in parser.tables:
        if not rows:
            continue
        header = rows[0]
        body = rows[1:] if len(rows) > 1 else []
        if body:
            tables.append(pd.DataFrame(body, columns=header))
        else:
            tables.append(pd.DataFrame(columns=header))
    return tables


def _normalize_scalar(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.12g}"
    return str(value)


def normalize_records(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, str]]:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    normalized = frame[columns].copy()
    for column in columns:
        normalized[column] = normalized[column].map(_normalize_scalar)
    return normalized.to_dict("records")


def compare_csv_to_report(
    csv_path: str | Path,
    report_html: str | Path,
    columns: list[str],
    *,
    report_table_index: int = 0,
) -> dict[str, Any]:
    csv_path = Path(csv_path)
    report_html = Path(report_html)
    csv_frame = pd.read_csv(csv_path)
    report_tables = read_html_tables(report_html)
    if report_table_index >= len(report_tables):
        raise IndexError(f"Report table index {report_table_index} missing for {report_html}")
    report_frame = report_tables[report_table_index]
    csv_records = normalize_records(csv_frame, columns)
    report_records = normalize_records(report_frame, columns)
    return {
        "matches": csv_records == report_records,
        "csv_records": csv_records,
        "report_records": report_records,
    }


def _compare_figure_hashes(packet_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target_rel, source_rel in TRACKED_FIGURE_MAPPINGS:
        target = packet_root / target_rel
        source = packet_root / source_rel
        rows.append(
            {
                "target": target_rel,
                "source": source_rel,
                "target_exists": target.exists(),
                "source_exists": source.exists(),
                "matches": target.exists() and source.exists() and file_sha256(target) == file_sha256(source),
            }
        )
    return rows


def _load_run_manifest(packet_root: Path) -> dict[str, Any]:
    manifest_path = packet_root / "run_manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _validate_export_hashes(packet_root: Path, run_manifest: dict[str, Any]) -> tuple[bool, list[str]]:
    rows = run_manifest.get("export_hashes")
    if not rows:
        return False, ["run_manifest.json is missing export_hashes."]
    expected = {row["relative_path"]: row["sha256"] for row in rows}
    actual_rows = export_hash_rows(packet_root)
    actual = {row["relative_path"]: row["sha256"] for row in actual_rows}
    issues: list[str] = []
    if set(expected) != set(actual):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        if missing:
            issues.append(f"run_manifest.json references missing exported files: {missing}")
        if extra:
            issues.append(f"run_manifest.json is missing exported file hashes for: {extra}")
    for relative_path, expected_hash in expected.items():
        actual_hash = actual.get(relative_path)
        if actual_hash is not None and actual_hash != expected_hash:
            issues.append(f"Hash mismatch for {relative_path}")
    return not issues, issues


def _validate_repo_snapshot(packet_root: Path, run_manifest: dict[str, Any]) -> tuple[bool, list[str], str]:
    snapshot_dir = packet_root / "04_repo_snapshot"
    if not snapshot_dir.exists():
        return True, [], "omitted"
    snapshot_path = snapshot_dir / "retinal-phenotyper.txt"
    if not snapshot_path.exists():
        return False, ["04_repo_snapshot exists but retinal-phenotyper.txt is missing."], "invalid"
    snapshot_meta = run_manifest.get("repo_snapshot")
    if not snapshot_meta:
        return False, ["run_manifest.json is missing repo_snapshot metadata."], "invalid"
    expected_rel = snapshot_meta.get("relative_path")
    expected_hash = snapshot_meta.get("sha256")
    actual_rel = str(snapshot_path.relative_to(packet_root))
    actual_hash = file_sha256(snapshot_path)
    issues: list[str] = []
    if expected_rel != actual_rel:
        issues.append(f"run_manifest repo_snapshot relative_path mismatch: expected {expected_rel}, got {actual_rel}")
    if expected_hash != actual_hash:
        issues.append("run_manifest repo_snapshot sha256 does not match the bundled snapshot.")
    return not issues, issues, "present"


def _validate_real_roi_benchmark(packet_root: Path) -> tuple[bool, list[str]]:
    present = [(packet_root / rel).exists() for rel in REAL_ROI_BENCHMARK_FILES]
    if not any(present):
        return True, []
    issues: list[str] = []
    for relative_path, exists in zip(REAL_ROI_BENCHMARK_FILES, present):
        if not exists:
            issues.append(f"Real ROI benchmark export is incomplete: missing {relative_path}")

    quality_path = packet_root / "01_tables" / "real_roi_benchmark_quality.csv"
    report_path = packet_root / "03_reports" / "real_roi_benchmark" / "benchmark_report.md"
    if quality_path.exists() and report_path.exists():
        try:
            quality = pd.read_csv(quality_path)
            if not quality.empty and "pass_threshold" in quality.columns:
                pass_value = bool(quality.iloc[0]["pass_threshold"])
                report_text = report_path.read_text(encoding="utf-8").lower()
                expected_phrase = "passed the project acceptance threshold" if pass_value else "did not pass the project acceptance threshold"
                if expected_phrase not in report_text:
                    issues.append("Real ROI benchmark report/pass-threshold wording does not match the exported quality table.")
        except Exception as exc:
            issues.append(f"Real ROI benchmark validation failed: {exc}")
    return not issues, issues


def audit_advisor_packet(packet_root: str | Path) -> dict[str, Any]:
    packet_root = Path(packet_root)
    report_root = packet_root / "03_reports"
    report_refs: list[dict[str, Any]] = []
    issues: list[str] = []

    if report_root.exists():
        for report_html in sorted(report_root.glob("*/report.html")):
            missing = find_missing_report_references(report_html)
            report_refs.append({"report": str(report_html.relative_to(packet_root)), "missing": missing})
            if missing:
                issues.append(f"{report_html.relative_to(packet_root)} has missing relative refs: {missing}")

    try:
        tracked = compare_csv_to_report(
            packet_root / "01_tables" / "tracked_example_study_summary.csv",
            packet_root / "03_reports" / "tracked_example" / "report.html",
            TRACKED_STUDY_COMPARE_COLUMNS,
        )
        if not tracked["matches"]:
            issues.append("Tracked study table does not match the tracked study report.")
    except Exception as exc:
        tracked = {"matches": False, "csv_records": [], "report_records": []}
        issues.append(f"Tracked study comparison failed: {exc}")

    try:
        manual = compare_csv_to_report(
            packet_root / "01_tables" / "count_error_metrics.csv",
            packet_root / "03_reports" / "tracked_example_manual_validation" / "report.html",
            MANUAL_BENCHMARK_COMPARE_COLUMNS,
        )
        if not manual["matches"]:
            issues.append("Manual benchmark table does not match the manual validation report.")
    except Exception as exc:
        manual = {"matches": False, "csv_records": [], "report_records": []}
        issues.append(f"Manual benchmark comparison failed: {exc}")

    figure_checks = _compare_figure_hashes(packet_root)
    for row in figure_checks:
        if not row["matches"]:
            issues.append(f"Tracked figure mapping mismatch: {row['target']} != {row['source']}")

    codex_text = (packet_root / "00_summary" / "codex_report.md").read_text(encoding="utf-8")
    executive_text = (packet_root / "00_summary" / "executive_summary.md").read_text(encoding="utf-8")
    run_manifest = _load_run_manifest(packet_root)
    pytest_counts = {
        "codex_report": extract_pytest_count(codex_text),
        "executive_summary": extract_pytest_count(executive_text),
        "run_manifest": run_manifest.get("pytest", {}).get("passed_count"),
    }
    pytest_values = {value for value in pytest_counts.values() if value is not None}
    pytest_consistent = len(pytest_values) <= 1
    if not pytest_consistent:
        issues.append(f"Pytest counts disagree across packet summaries: {pytest_counts}")

    export_hashes_valid, export_hash_issues = _validate_export_hashes(packet_root, run_manifest)
    issues.extend(export_hash_issues)

    repo_snapshot_valid, repo_snapshot_issues, repo_snapshot_status = _validate_repo_snapshot(packet_root, run_manifest)
    issues.extend(repo_snapshot_issues)
    real_roi_valid, real_roi_issues = _validate_real_roi_benchmark(packet_root)
    issues.extend(real_roi_issues)

    return {
        "packet_root": str(packet_root),
        "report_references": report_refs,
        "tracked_study": tracked,
        "manual_benchmark": manual,
        "figure_hashes": figure_checks,
        "pytest_counts": pytest_counts,
        "pytest_consistent": pytest_consistent,
        "export_hashes_valid": export_hashes_valid,
        "repo_snapshot_valid": repo_snapshot_valid,
        "repo_snapshot_status": repo_snapshot_status,
        "real_roi_benchmark_valid": real_roi_valid,
        "issues": issues,
        "passed": not issues,
    }


def build_tracked_lane_comparison_md(
    study_provenance_path: str | Path,
    single_provenance_path: str | Path,
    study_summary_path: str | Path,
    single_report_path: str | Path,
) -> str:
    study_provenance = json.loads(Path(study_provenance_path).read_text(encoding="utf-8"))
    single_provenance = json.loads(Path(single_provenance_path).read_text(encoding="utf-8"))
    study_summary = pd.read_csv(study_summary_path)
    single_summary = read_html_tables(single_report_path)[0]

    compare_keys = [
        "focus_mode",
        "tta",
        "register_retina",
        "onh_mode",
        "spatial_mode",
        "write_uncertainty_maps",
        "write_qc_maps",
        "strict_schemas",
        "manifest",
        "input_dir",
    ]
    def render_markdown_table(frame: pd.DataFrame) -> list[str]:
        columns = list(frame.columns)
        header = "| " + " | ".join(columns) + " |"
        separator = "| " + " | ".join(["---"] * len(columns)) + " |"
        rows = []
        for row in frame.to_dict("records"):
            rows.append("| " + " | ".join(_normalize_scalar(row[column]) for column in columns) + " |")
        return [header, separator, *rows]

    lines = [
        "# Tracked Lane Comparison",
        "",
        "The single-image tracked lane is a QC/demo lane and is not count-comparable to the tracked study lane.",
        "",
        "## Count Snapshot",
        "",
    ]
    study_subset = study_summary[["sample_id", "filename", "cell_count"]].copy()
    lines.append("### Study Mode")
    lines.extend(render_markdown_table(study_subset))
    lines.append("")
    lines.append("### Single-Image QC")
    single_subset = single_summary[["filename", "cell_count"]].copy()
    lines.extend(render_markdown_table(single_subset))
    lines.append("")
    lines.append("## Configuration Differences")
    lines.append("")
    lines.append("| key | study_mode | single_image_qc |")
    lines.append("| --- | --- | --- |")
    study_args = study_provenance.get("args", {})
    single_args = single_provenance.get("args", {})
    for key in compare_keys:
        lines.append(
            f"| {key} | `{study_args.get(key, study_provenance.get('resolved_config', {}).get(key, ''))}` | "
            f"`{single_args.get(key, single_provenance.get('resolved_config', {}).get(key, ''))}` |"
        )
    return "\n".join(lines) + "\n"
