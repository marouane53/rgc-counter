from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_main_version_flag_reports_current_package_version():
    completed = subprocess.run(
        [sys.executable, "main.py", "--version"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "1.0.0" in completed.stdout


def test_readme_links_to_researcher_docs():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "docs/researcher-guide.md" in readme
    assert "docs/paper-workflow.md" in readme
    assert "docs/model-training.md" in readme
    assert "TESTING.md" in readme


def test_canonical_tracked_example_workflow_runs(tmp_path: Path):
    output_dir = tmp_path / "Outputs_example"
    completed = subprocess.run(
        [
            sys.executable,
            "main.py",
            "--manifest",
            "examples/manifests/example_study_manifest.csv",
            "--study_output_dir",
            str(output_dir),
            "--focus_none",
            "--register_retina",
            "--region_schema",
            "mouse_flatmount_v1",
            "--spatial_stats",
            "--spatial_mode",
            "rigorous",
            "--spatial_envelope_sims",
            "8",
            "--write_object_table",
            "--write_provenance",
            "--write_html_report",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert completed.returncode == 0
    assert (output_dir / "study_summary.csv").exists()
    assert (output_dir / "study_regions.csv").exists()
    assert (output_dir / "stats").exists()
    assert (output_dir / "figures").exists()
    assert (output_dir / "methods_appendix.md").exists()
    assert (output_dir / "report.html").exists()
    assert (output_dir / "samples" / "EX01_OD" / "spatial").exists()
    assert (output_dir / "samples" / "EX01_OD" / "objects").exists()
    assert (output_dir / "samples" / "EX01_OD" / "regions").exists()
    assert (output_dir / "samples" / "EX01_OD" / "retina_frames").exists()


def test_registered_tracking_study_smoke_writes_pair_qc_and_report_sections(tmp_path: Path):
    manifest_path = tmp_path / "tracking_manifest.csv"
    output_dir = tmp_path / "Outputs_tracking"
    example_image = (ROOT / "examples" / "smoke_data" / "example_retina_a.tif").resolve()
    manifest = pd.DataFrame(
        [
            {
                "sample_id": "EX01_OD_D0",
                "animal_id": "M01",
                "eye": "OD",
                "condition": "treated",
                "genotype": "WT",
                "timepoint_dpi": 0,
                "modality": "flatmount",
                "stain_panel": "RBPMS",
                "path": str(example_image),
                "onh_x_px": 48,
                "onh_y_px": 48,
                "dorsal_x_px": 48,
                "dorsal_y_px": 14,
            },
            {
                "sample_id": "EX01_OD_D7",
                "animal_id": "M01",
                "eye": "OD",
                "condition": "treated",
                "genotype": "WT",
                "timepoint_dpi": 7,
                "modality": "flatmount",
                "stain_panel": "RBPMS",
                "path": str(example_image),
                "onh_x_px": 48,
                "onh_y_px": 48,
                "dorsal_x_px": 48,
                "dorsal_y_px": 14,
            },
        ]
    )
    manifest.to_csv(manifest_path, index=False)

    completed = subprocess.run(
        [
            sys.executable,
            "main.py",
            "--manifest",
            str(manifest_path),
            "--study_output_dir",
            str(output_dir),
            "--focus_none",
            "--track_longitudinal",
            "--tracking_mode",
            "registered",
            "--write_object_table",
            "--write_provenance",
            "--write_html_report",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    report_html = (output_dir / "report.html").read_text(encoding="utf-8")
    provenance_json = (output_dir / "provenance.json").read_text(encoding="utf-8")

    assert completed.returncode == 0
    assert (output_dir / "tracking" / "track_observations.csv").exists()
    assert (output_dir / "tracking" / "track_pair_qc.csv").exists()
    assert (output_dir / "tracking" / "track_summary.csv").exists()
    assert "Tracking Summary" in report_html
    assert "Tracking Pair QC" in report_html
    assert '"tracking_mode": "registered"' in provenance_json
