from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.advisor_packet import audit_advisor_packet, build_tracked_lane_comparison_md, export_hash_rows, file_sha256


def _write_report(path: Path, frame: pd.DataFrame, refs: list[str] | None = None) -> None:
    refs = refs or []
    body = [frame.to_html(index=False)]
    for ref in refs:
        body.append(f'<img src="{ref}" alt="{ref}">')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("<html><body>" + "\n".join(body) + "</body></html>", encoding="utf-8")


def _build_minimal_packet(root: Path) -> Path:
    packet_root = root / "10_ai_advisor_packet"
    (packet_root / "00_summary").mkdir(parents=True)
    (packet_root / "01_tables").mkdir(parents=True)
    (packet_root / "02_images").mkdir(parents=True)
    (packet_root / "03_reports" / "tracked_example" / "figures").mkdir(parents=True)
    (packet_root / "03_reports" / "tracked_example_manual_validation" / "validation").mkdir(parents=True)
    (packet_root / "03_reports" / "tracked_example_single_image").mkdir(parents=True)

    tracked = pd.DataFrame(
        [
            {"sample_id": "EX01_OD", "filename": "example_retina_a.tif", "cell_count": 1, "warning_count": 1},
            {"sample_id": "EX01_OS", "filename": "example_retina_b.tif", "cell_count": 0, "warning_count": 1},
        ]
    )
    manual = pd.DataFrame(
        [
            {"sample_id": "EX01_OD", "filename": "example_retina_a.tif", "manual_count": 4, "cell_count": 1},
            {"sample_id": "EX01_OS", "filename": "example_retina_b.tif", "manual_count": 3, "cell_count": 0},
        ]
    )
    single = pd.DataFrame(
        [
            {"filename": "example_retina_a.tif", "cell_count": 0},
            {"filename": "example_retina_b.tif", "cell_count": 1},
        ]
    )
    tracked.to_csv(packet_root / "01_tables" / "tracked_example_study_summary.csv", index=False)
    manual.to_csv(packet_root / "01_tables" / "count_error_metrics.csv", index=False)
    pd.DataFrame([{"stage": "pytest", "status": "ok"}]).to_csv(packet_root / "01_tables" / "runtime_summary.csv", index=False)

    tracked_png = packet_root / "03_reports" / "tracked_example" / "figures" / "cell_count_by_condition.png"
    tracked_png.write_bytes(b"tracked-figure")
    agreement_png = packet_root / "03_reports" / "tracked_example_manual_validation" / "validation" / "agreement_scatter.png"
    agreement_png.write_bytes(b"agreement-figure")
    bland_png = packet_root / "03_reports" / "tracked_example_manual_validation" / "validation" / "bland_altman.png"
    bland_png.write_bytes(b"bland-figure")

    _write_report(
        packet_root / "03_reports" / "tracked_example" / "report.html",
        tracked,
        refs=["figures/cell_count_by_condition.png"],
    )
    _write_report(
        packet_root / "03_reports" / "tracked_example_manual_validation" / "report.html",
        manual,
        refs=["validation/agreement_scatter.png", "validation/bland_altman.png"],
    )
    _write_report(packet_root / "03_reports" / "tracked_example_single_image" / "report.html", single)

    (packet_root / "02_images" / "tracked_example_summary.png").write_bytes(tracked_png.read_bytes())
    (packet_root / "02_images" / "tracked_example_agreement_scatter.png").write_bytes(agreement_png.read_bytes())
    (packet_root / "02_images" / "tracked_example_bland_altman.png").write_bytes(bland_png.read_bytes())

    (packet_root / "00_summary" / "codex_report.md").write_text("128 passed\n", encoding="utf-8")
    (packet_root / "00_summary" / "executive_summary.md").write_text("The fresh evidence environment pytest run passed with 128 tests.\n", encoding="utf-8")

    snapshot_path = packet_root / "04_repo_snapshot" / "retinal-phenotyper.txt"
    snapshot_path.parent.mkdir(parents=True)
    snapshot_path.write_text("snapshot", encoding="utf-8")

    run_manifest = {
        "pytest": {"passed_count": 128},
        "export_hashes": export_hash_rows(packet_root),
        "repo_snapshot": {
            "relative_path": "04_repo_snapshot/retinal-phenotyper.txt",
            "sha256": file_sha256(snapshot_path),
        },
    }
    (packet_root / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    return packet_root


def test_audit_advisor_packet_passes_for_minimal_packet(tmp_path: Path):
    packet_root = _build_minimal_packet(tmp_path)

    audit = audit_advisor_packet(packet_root)

    assert audit["passed"] is True
    assert audit["issues"] == []


def test_audit_advisor_packet_detects_table_mismatch(tmp_path: Path):
    packet_root = _build_minimal_packet(tmp_path)
    tracked = pd.read_csv(packet_root / "01_tables" / "tracked_example_study_summary.csv")
    tracked.loc[0, "cell_count"] = 9
    tracked.to_csv(packet_root / "01_tables" / "tracked_example_study_summary.csv", index=False)

    audit = audit_advisor_packet(packet_root)

    assert audit["passed"] is False
    assert any("Tracked study table does not match" in issue for issue in audit["issues"])


def test_audit_advisor_packet_detects_missing_report_ref(tmp_path: Path):
    packet_root = _build_minimal_packet(tmp_path)
    (packet_root / "03_reports" / "tracked_example" / "figures" / "cell_count_by_condition.png").unlink()

    audit = audit_advisor_packet(packet_root)

    assert audit["passed"] is False
    assert any("missing relative refs" in issue for issue in audit["issues"])


def test_build_tracked_lane_comparison_md_marks_single_image_as_qc_only(tmp_path: Path):
    study_provenance = {
        "args": {"focus_mode": "focus_none", "tta": False, "register_retina": True, "manifest": "examples/manifests/example_study_manifest.csv"},
        "resolved_config": {},
    }
    single_provenance = {
        "args": {"focus_mode": "focus_qc", "tta": True, "register_retina": True, "input_dir": "examples/smoke_data"},
        "resolved_config": {},
    }
    study_path = tmp_path / "study.json"
    single_path = tmp_path / "single.json"
    study_path.write_text(json.dumps(study_provenance), encoding="utf-8")
    single_path.write_text(json.dumps(single_provenance), encoding="utf-8")
    study_summary = tmp_path / "study_summary.csv"
    pd.DataFrame(
        [
            {"sample_id": "EX01_OD", "filename": "example_retina_a.tif", "cell_count": 1},
            {"sample_id": "EX01_OS", "filename": "example_retina_b.tif", "cell_count": 0},
        ]
    ).to_csv(study_summary, index=False)
    single_report = tmp_path / "single_report.html"
    _write_report(
        single_report,
        pd.DataFrame(
            [
                {"filename": "example_retina_a.tif", "cell_count": 0},
                {"filename": "example_retina_b.tif", "cell_count": 1},
            ]
        ),
    )

    content = build_tracked_lane_comparison_md(study_path, single_path, study_summary, single_report)

    assert "not count-comparable" in content
    assert "focus_mode" in content
    assert "focus_qc" in content


def test_audit_advisor_packet_detects_incomplete_real_roi_export(tmp_path: Path):
    packet_root = _build_minimal_packet(tmp_path)
    pd.DataFrame([{"config_id": "winner"}]).to_csv(packet_root / "01_tables" / "real_roi_config_comparison.csv", index=False)

    audit = audit_advisor_packet(packet_root)

    assert audit["passed"] is False
    assert any("Real ROI benchmark export is incomplete" in issue for issue in audit["issues"])


def test_audit_advisor_packet_accepts_complete_real_roi_export(tmp_path: Path):
    packet_root = _build_minimal_packet(tmp_path)
    pd.DataFrame([{"config_id": "winner"}]).to_csv(packet_root / "01_tables" / "real_roi_config_comparison.csv", index=False)
    pd.DataFrame(
        [{"benchmark_kind": "roi_point_matching", "matched_modality": True, "n_rois": 24, "precision": 0.8, "recall": 0.8, "f1": 0.8, "mae": 1.0, "pass_threshold": True}]
    ).to_csv(packet_root / "01_tables" / "real_roi_benchmark_quality.csv", index=False)
    report_dir = packet_root / "03_reports" / "real_roi_benchmark"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "benchmark_report.md").write_text(
        "This benchmark passed the project acceptance threshold for one narrow use-case.\n",
        encoding="utf-8",
    )
    run_manifest = json.loads((packet_root / "run_manifest.json").read_text(encoding="utf-8"))
    run_manifest["export_hashes"] = export_hash_rows(packet_root)
    (packet_root / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

    audit = audit_advisor_packet(packet_root)

    assert audit["passed"] is True
    assert audit["real_roi_benchmark_valid"] is True
