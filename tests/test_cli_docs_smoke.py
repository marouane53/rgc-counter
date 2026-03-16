from __future__ import annotations

import subprocess
import sys
from pathlib import Path


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
