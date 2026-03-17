from pathlib import Path

from src.report import copy_report_bundle, find_missing_report_references, write_html_report


def test_copy_report_bundle_preserves_relative_report_assets(tmp_path: Path):
    report_root = tmp_path / "report_source"
    (report_root / "figures").mkdir(parents=True)
    (report_root / "stats").mkdir(parents=True)
    (report_root / "figures" / "plot.png").write_bytes(b"png")
    (report_root / "stats" / "summary.csv").write_text("metric,value\ncount,1\n", encoding="utf-8")

    report_path = write_html_report(
        str(report_root),
        {"mode": "test"},
        [{"sample_id": "S1", "cell_count": 1}],
        images=[("Plot", "figures/plot.png")],
        assets=[("Summary", "stats/summary.csv")],
    )

    bundled_report = copy_report_bundle(report_path, tmp_path / "portable_bundle")

    assert bundled_report.exists()
    assert (bundled_report.parent / "figures" / "plot.png").exists()
    assert (bundled_report.parent / "stats" / "summary.csv").exists()
    assert find_missing_report_references(bundled_report) == []
