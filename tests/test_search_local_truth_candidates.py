from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import scripts.search_local_truth_candidates as truth_search


def test_truth_search_main_ranks_local_candidates_and_scene_hits(tmp_path: Path, monkeypatch):
    data_root = tmp_path / "data"
    data_root.mkdir()
    manual_csv = data_root / "Sana_Mouse1_RBPMS_manual_points.csv"
    manual_csv.write_text("x_px,y_px\n1,2\n", encoding="utf-8")
    spreadsheet = data_root / "IL_OHT_survival_density_counts.xlsx"
    spreadsheet.write_text("placeholder", encoding="utf-8")

    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    metadata_payload = {
        "path": str(tmp_path / "fixture.ims"),
        "scene": {
            "annotation_candidates": [
                {"path": "Scene8/Spots/Position", "attrs": {"Name": "Manual Spots"}}
            ]
        },
    }
    (metadata_dir / "fixture.json").write_text(json.dumps(metadata_payload), encoding="utf-8")

    monkeypatch.setattr(truth_search, "spotlight_rows", lambda roots: [])

    output_dir = tmp_path / "truth_search"
    exit_code = truth_search.main(
        [
            "--roots",
            str(data_root),
            "--metadata-dir",
            str(metadata_dir),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    frame = pd.read_csv(output_dir / "truth_search_candidates.csv")
    assert any(frame["classification"] == "embedded_same_image_point_truth")
    assert any(frame["classification"] == "external_same_image_point_truth")
    assert (output_dir / "truth_search_report.md").exists()
    assert (output_dir / "paper_sanity_only.md").exists()


def test_scan_roots_skips_ignored_directories(tmp_path: Path):
    root = tmp_path / "scan-root"
    data_dir = root / "data"
    data_dir.mkdir(parents=True)
    ignored_dir = root / "node_modules"
    ignored_dir.mkdir()

    kept = data_dir / "Sana_Mouse1_RBPMS_manual_points.csv"
    kept.write_text("x_px,y_px\n1,2\n", encoding="utf-8")
    ignored = ignored_dir / "Sana_Mouse1_RBPMS_manual_points.csv"
    ignored.write_text("x_px,y_px\n3,4\n", encoding="utf-8")
    unrelated = data_dir / "summary.json"
    unrelated.write_text("{}", encoding="utf-8")

    rows = truth_search.scan_roots([root])

    assert [row["path"] for row in rows] == [str(kept.resolve())]


def test_keyword_hits_are_token_aware():
    hits = truth_search._keyword_hits("endpoint-profile.json profile manual_points IL-OHT Mouse1")

    assert "manual" in hits
    assert "point" in hits
    assert "il" in hits
    assert "mouse1" in hits
    assert "count" not in hits
