from pathlib import Path

import pandas as pd

from src.methods import build_methods_appendix
from src.report import write_html_report
from src.stats import (
    build_design_audit,
    compute_outcome_stats,
    compute_region_stats,
    run_study_statistics,
    write_study_statistics_artifacts,
)


def test_compute_outcome_stats_uses_paired_design_when_animals_have_both_conditions():
    sample_table = pd.DataFrame(
        [
            {"animal_id": "A1", "condition": "control", "cell_count": 50},
            {"animal_id": "A1", "condition": "treated", "cell_count": 40},
            {"animal_id": "A2", "condition": "control", "cell_count": 48},
            {"animal_id": "A2", "condition": "treated", "cell_count": 38},
        ]
    )

    out = compute_outcome_stats(sample_table, outcome="cell_count")

    assert not out.empty
    assert set(out["design"]) == {"paired"}
    assert set(out["test"]) >= {"paired_t"}


def test_compute_region_stats_returns_rows_per_region():
    region_table = pd.DataFrame(
        [
            {"animal_id": "A1", "condition": "control", "region_axis": "ring", "region_label": "central", "density_cells_per_mm2": 100},
            {"animal_id": "A1", "condition": "treated", "region_axis": "ring", "region_label": "central", "density_cells_per_mm2": 80},
            {"animal_id": "A2", "condition": "control", "region_axis": "ring", "region_label": "central", "density_cells_per_mm2": 90},
            {"animal_id": "A2", "condition": "treated", "region_axis": "ring", "region_label": "central", "density_cells_per_mm2": 70},
        ]
    )

    out = compute_region_stats(region_table)

    assert not out.empty
    assert out.loc[0, "region_axis"] == "ring"
    assert out.loc[0, "region_label"] == "central"


def test_run_study_statistics_auto_uses_simple_for_plain_paired_design():
    sample_table = pd.DataFrame(
        [
            {"sample_id": "S1", "animal_id": "A1", "condition": "control", "timepoint_dpi": 7, "cell_count": 50},
            {"sample_id": "S2", "animal_id": "A1", "condition": "treated", "timepoint_dpi": 7, "cell_count": 40},
            {"sample_id": "S3", "animal_id": "A2", "condition": "control", "timepoint_dpi": 7, "cell_count": 48},
            {"sample_id": "S4", "animal_id": "A2", "condition": "treated", "timepoint_dpi": 7, "cell_count": 38},
        ]
    )

    out = run_study_statistics(sample_table, pd.DataFrame(), requested_mode="auto")

    assert out.decision["sample_outcome"]["selected_mode"] == "simple"
    assert not out.study_stats.empty
    assert out.sample_mixed.empty


def test_run_study_statistics_auto_uses_mixed_for_nested_sample_design():
    sample_table = pd.DataFrame(
        [
            {"sample_id": "S1", "animal_id": "A1", "eye": "OD", "condition": "control", "timepoint_dpi": 7, "cell_count": 50},
            {"sample_id": "S2", "animal_id": "A1", "eye": "OD", "condition": "control", "timepoint_dpi": 14, "cell_count": 47},
            {"sample_id": "S3", "animal_id": "A1", "eye": "OS", "condition": "treated", "timepoint_dpi": 7, "cell_count": 40},
            {"sample_id": "S4", "animal_id": "A1", "eye": "OS", "condition": "treated", "timepoint_dpi": 14, "cell_count": 36},
            {"sample_id": "S5", "animal_id": "A2", "eye": "OD", "condition": "control", "timepoint_dpi": 7, "cell_count": 52},
            {"sample_id": "S6", "animal_id": "A2", "eye": "OD", "condition": "control", "timepoint_dpi": 14, "cell_count": 49},
            {"sample_id": "S7", "animal_id": "A2", "eye": "OS", "condition": "treated", "timepoint_dpi": 7, "cell_count": 42},
            {"sample_id": "S8", "animal_id": "A2", "eye": "OS", "condition": "treated", "timepoint_dpi": 14, "cell_count": 37},
        ]
    )

    out = run_study_statistics(sample_table, pd.DataFrame(), requested_mode="auto")

    assert out.decision["sample_outcome"]["selected_mode"] == "mixed"
    assert not out.sample_mixed.empty
    assert out.study_stats.empty


def test_run_study_statistics_mixed_outputs_region_coefficients():
    sample_table = pd.DataFrame(
        [
            {"sample_id": "S1", "animal_id": "A1", "eye": "OD", "condition": "control", "timepoint_dpi": 7, "cell_count": 50},
            {"sample_id": "S2", "animal_id": "A1", "eye": "OS", "condition": "treated", "timepoint_dpi": 7, "cell_count": 40},
            {"sample_id": "S3", "animal_id": "A2", "eye": "OD", "condition": "control", "timepoint_dpi": 7, "cell_count": 48},
            {"sample_id": "S4", "animal_id": "A2", "eye": "OS", "condition": "treated", "timepoint_dpi": 7, "cell_count": 38},
        ]
    )
    region_table = pd.DataFrame(
        [
            {"sample_id": "S1", "animal_id": "A1", "eye": "OD", "condition": "control", "timepoint_dpi": 7, "region_axis": "ring", "region_label": "central", "density_cells_per_mm2": 100},
            {"sample_id": "S1", "animal_id": "A1", "eye": "OD", "condition": "control", "timepoint_dpi": 7, "region_axis": "ring", "region_label": "peripheral", "density_cells_per_mm2": 80},
            {"sample_id": "S2", "animal_id": "A1", "eye": "OS", "condition": "treated", "timepoint_dpi": 7, "region_axis": "ring", "region_label": "central", "density_cells_per_mm2": 70},
            {"sample_id": "S2", "animal_id": "A1", "eye": "OS", "condition": "treated", "timepoint_dpi": 7, "region_axis": "ring", "region_label": "peripheral", "density_cells_per_mm2": 55},
            {"sample_id": "S3", "animal_id": "A2", "eye": "OD", "condition": "control", "timepoint_dpi": 7, "region_axis": "ring", "region_label": "central", "density_cells_per_mm2": 98},
            {"sample_id": "S3", "animal_id": "A2", "eye": "OD", "condition": "control", "timepoint_dpi": 7, "region_axis": "ring", "region_label": "peripheral", "density_cells_per_mm2": 79},
            {"sample_id": "S4", "animal_id": "A2", "eye": "OS", "condition": "treated", "timepoint_dpi": 7, "region_axis": "ring", "region_label": "central", "density_cells_per_mm2": 68},
            {"sample_id": "S4", "animal_id": "A2", "eye": "OS", "condition": "treated", "timepoint_dpi": 7, "region_axis": "ring", "region_label": "peripheral", "density_cells_per_mm2": 53},
        ]
    )

    out = run_study_statistics(sample_table, region_table, requested_mode="auto")

    assert out.decision["region_outcome"]["selected_mode"] == "mixed"
    assert not out.region_mixed.empty
    assert set(out.region_mixed["analysis_level"]) == {"region"}


def test_run_study_statistics_auto_falls_back_to_simple_when_mixed_fails():
    sample_table = pd.DataFrame(
        [
            {"sample_id": "S1", "animal_id": "A1", "eye": "OD", "condition": "control", "timepoint_dpi": 7, "cell_count": 50},
            {"sample_id": "S2", "animal_id": "A1", "eye": "OD", "condition": "control", "timepoint_dpi": 14, "cell_count": 50},
            {"sample_id": "S3", "animal_id": "A1", "eye": "OS", "condition": "treated", "timepoint_dpi": 7, "cell_count": 50},
            {"sample_id": "S4", "animal_id": "A1", "eye": "OS", "condition": "treated", "timepoint_dpi": 14, "cell_count": 50},
            {"sample_id": "S5", "animal_id": "A2", "eye": "OD", "condition": "control", "timepoint_dpi": 7, "cell_count": 50},
            {"sample_id": "S6", "animal_id": "A2", "eye": "OD", "condition": "control", "timepoint_dpi": 14, "cell_count": 50},
            {"sample_id": "S7", "animal_id": "A2", "eye": "OS", "condition": "treated", "timepoint_dpi": 7, "cell_count": 50},
            {"sample_id": "S8", "animal_id": "A2", "eye": "OS", "condition": "treated", "timepoint_dpi": 14, "cell_count": 50},
        ]
    )

    out = run_study_statistics(sample_table, pd.DataFrame(), requested_mode="auto")

    assert out.decision["sample_outcome"]["selected_mode"] in {"simple", "none"}
    assert out.decision["sample_outcome"]["fallback_reason"] is not None
    assert out.sample_mixed.empty


def test_run_study_statistics_explicit_mixed_raises_when_design_is_underspecified():
    sample_table = pd.DataFrame(
        [
            {"sample_id": "S1", "animal_id": "A1", "condition": "control", "cell_count": 50},
            {"sample_id": "S2", "animal_id": "A2", "condition": "treated", "cell_count": 40},
        ]
    )

    try:
        run_study_statistics(sample_table, pd.DataFrame(), requested_mode="mixed")
    except RuntimeError as exc:
        assert "Sample mixed-effects" in str(exc)
    else:
        raise AssertionError("Expected explicit mixed mode to raise for an underspecified design.")


def test_build_design_audit_summarizes_missingness_and_repeated_structure():
    sample_table = pd.DataFrame(
        [
            {"sample_id": "S1", "animal_id": "A1", "eye": "OD", "condition": "control", "timepoint_dpi": 7, "cell_count": 10, "density_cells_per_mm2": 100},
            {"sample_id": "S2", "animal_id": "A1", "eye": "OD", "condition": "control", "timepoint_dpi": 14, "cell_count": 9, "density_cells_per_mm2": 95},
            {"sample_id": "S3", "animal_id": "A2", "eye": "OS", "condition": "treated", "timepoint_dpi": 7, "cell_count": None, "density_cells_per_mm2": 88},
        ]
    )
    region_table = pd.DataFrame(
        [
            {"sample_id": "S1", "animal_id": "A1", "condition": "control", "region_axis": "ring", "region_label": "central", "density_cells_per_mm2": 100},
            {"sample_id": "S1", "animal_id": "A1", "condition": "control", "region_axis": "ring", "region_label": "peripheral", "density_cells_per_mm2": 80},
        ]
    )

    out = build_design_audit(sample_table, region_table)

    assert not out.empty
    assert ((out["category"] == "missingness_sample") & (out["key"] == "cell_count") & (out["value"] == 1)).any()
    assert ((out["category"] == "regions_per_sample") & (out["key"] == "S1") & (out["value"] == 2)).any()


def test_write_study_statistics_artifacts_writes_design_and_mixed_outputs(tmp_path):
    sample_table = pd.DataFrame(
        [
            {"sample_id": "S1", "animal_id": "A1", "eye": "OD", "condition": "control", "timepoint_dpi": 7, "cell_count": 50},
            {"sample_id": "S2", "animal_id": "A1", "eye": "OD", "condition": "control", "timepoint_dpi": 14, "cell_count": 47},
            {"sample_id": "S3", "animal_id": "A1", "eye": "OS", "condition": "treated", "timepoint_dpi": 7, "cell_count": 40},
            {"sample_id": "S4", "animal_id": "A1", "eye": "OS", "condition": "treated", "timepoint_dpi": 14, "cell_count": 36},
            {"sample_id": "S5", "animal_id": "A2", "eye": "OD", "condition": "control", "timepoint_dpi": 7, "cell_count": 52},
            {"sample_id": "S6", "animal_id": "A2", "eye": "OD", "condition": "control", "timepoint_dpi": 14, "cell_count": 49},
            {"sample_id": "S7", "animal_id": "A2", "eye": "OS", "condition": "treated", "timepoint_dpi": 7, "cell_count": 42},
            {"sample_id": "S8", "animal_id": "A2", "eye": "OS", "condition": "treated", "timepoint_dpi": 14, "cell_count": 37},
        ]
    )
    region_table = pd.DataFrame(
        [
            {"sample_id": "S1", "animal_id": "A1", "eye": "OD", "condition": "control", "timepoint_dpi": 7, "region_axis": "ring", "region_label": "central", "density_cells_per_mm2": 100},
            {"sample_id": "S1", "animal_id": "A1", "eye": "OD", "condition": "control", "timepoint_dpi": 7, "region_axis": "ring", "region_label": "peripheral", "density_cells_per_mm2": 80},
            {"sample_id": "S2", "animal_id": "A1", "eye": "OS", "condition": "treated", "timepoint_dpi": 7, "region_axis": "ring", "region_label": "central", "density_cells_per_mm2": 70},
            {"sample_id": "S2", "animal_id": "A1", "eye": "OS", "condition": "treated", "timepoint_dpi": 7, "region_axis": "ring", "region_label": "peripheral", "density_cells_per_mm2": 55},
            {"sample_id": "S3", "animal_id": "A2", "eye": "OD", "condition": "control", "timepoint_dpi": 7, "region_axis": "ring", "region_label": "central", "density_cells_per_mm2": 98},
            {"sample_id": "S3", "animal_id": "A2", "eye": "OD", "condition": "control", "timepoint_dpi": 7, "region_axis": "ring", "region_label": "peripheral", "density_cells_per_mm2": 79},
            {"sample_id": "S4", "animal_id": "A2", "eye": "OS", "condition": "treated", "timepoint_dpi": 7, "region_axis": "ring", "region_label": "central", "density_cells_per_mm2": 68},
            {"sample_id": "S4", "animal_id": "A2", "eye": "OS", "condition": "treated", "timepoint_dpi": 7, "region_axis": "ring", "region_label": "peripheral", "density_cells_per_mm2": 53},
        ]
    )

    result = run_study_statistics(sample_table, region_table, requested_mode="auto")
    written = write_study_statistics_artifacts(result, stats_dir=tmp_path / "stats", stats_mixed_dir=tmp_path / "stats_mixed")

    assert written["design_audit_csv"].exists()
    assert written["design_audit_md"].exists()
    assert written["statistics_decision"].exists()
    assert written["study_stats"].exists()
    assert written["region_stats"].exists()
    assert "sample_mixed" in written
    assert "region_mixed" in written


def test_study_statistics_smoke_writes_methods_and_report_sections(tmp_path):
    sample_table = pd.DataFrame(
        [
            {"sample_id": "S1", "animal_id": "A1", "eye": "OD", "condition": "control", "timepoint_dpi": 7, "cell_count": 50},
            {"sample_id": "S2", "animal_id": "A1", "eye": "OD", "condition": "control", "timepoint_dpi": 14, "cell_count": 47},
            {"sample_id": "S3", "animal_id": "A1", "eye": "OS", "condition": "treated", "timepoint_dpi": 7, "cell_count": 40},
            {"sample_id": "S4", "animal_id": "A1", "eye": "OS", "condition": "treated", "timepoint_dpi": 14, "cell_count": 36},
            {"sample_id": "S5", "animal_id": "A2", "eye": "OD", "condition": "control", "timepoint_dpi": 7, "cell_count": 52},
            {"sample_id": "S6", "animal_id": "A2", "eye": "OD", "condition": "control", "timepoint_dpi": 14, "cell_count": 49},
            {"sample_id": "S7", "animal_id": "A2", "eye": "OS", "condition": "treated", "timepoint_dpi": 7, "cell_count": 42},
            {"sample_id": "S8", "animal_id": "A2", "eye": "OS", "condition": "treated", "timepoint_dpi": 14, "cell_count": 37},
        ]
    )
    region_table = pd.DataFrame(
        [
            {"sample_id": "S1", "animal_id": "A1", "eye": "OD", "condition": "control", "timepoint_dpi": 7, "region_axis": "ring", "region_label": "central", "density_cells_per_mm2": 100},
            {"sample_id": "S1", "animal_id": "A1", "eye": "OD", "condition": "control", "timepoint_dpi": 7, "region_axis": "ring", "region_label": "peripheral", "density_cells_per_mm2": 80},
            {"sample_id": "S2", "animal_id": "A1", "eye": "OS", "condition": "treated", "timepoint_dpi": 7, "region_axis": "ring", "region_label": "central", "density_cells_per_mm2": 70},
            {"sample_id": "S2", "animal_id": "A1", "eye": "OS", "condition": "treated", "timepoint_dpi": 7, "region_axis": "ring", "region_label": "peripheral", "density_cells_per_mm2": 55},
            {"sample_id": "S3", "animal_id": "A2", "eye": "OD", "condition": "control", "timepoint_dpi": 7, "region_axis": "ring", "region_label": "central", "density_cells_per_mm2": 98},
            {"sample_id": "S3", "animal_id": "A2", "eye": "OD", "condition": "control", "timepoint_dpi": 7, "region_axis": "ring", "region_label": "peripheral", "density_cells_per_mm2": 79},
            {"sample_id": "S4", "animal_id": "A2", "eye": "OS", "condition": "treated", "timepoint_dpi": 7, "region_axis": "ring", "region_label": "central", "density_cells_per_mm2": 68},
            {"sample_id": "S4", "animal_id": "A2", "eye": "OS", "condition": "treated", "timepoint_dpi": 7, "region_axis": "ring", "region_label": "peripheral", "density_cells_per_mm2": 53},
        ]
    )

    result = run_study_statistics(sample_table, region_table, requested_mode="auto")
    write_study_statistics_artifacts(result, stats_dir=tmp_path / "stats", stats_mixed_dir=tmp_path / "stats_mixed")
    methods = build_methods_appendix(
        resolved_config={
            "backend": "cellpose",
            "modality": "flatmount",
            "modality_projection": "max",
            "focus_mode": "none",
            "use_gpu": False,
            "diameter": None,
            "min_size": 5,
            "max_size": 1000,
            "tta": False,
            "spatial_stats": False,
            "phenotype_engine": "legacy",
            "marker_metrics": False,
            "interaction_metrics": False,
            "register_retina": False,
            "atlas_reference": None,
            "track_longitudinal": False,
        },
        sample_table=sample_table,
        region_table=region_table,
        study_statistics=result.decision,
    )
    report_path = write_html_report(
        str(tmp_path),
        {"backend": "cellpose", "stats_mode": "auto"},
        sample_table.to_dict("records"),
        tables=[
            {"title": "Statistics Decision", "html": pd.DataFrame([result.decision["sample_outcome"]]).to_html(index=False)},
            {"title": "Study Design Audit", "html": result.design_audit.to_html(index=False)},
            {"title": "Sample Mixed Effects", "html": result.sample_mixed.to_html(index=False)},
            {"title": "Region Mixed Effects", "html": result.region_mixed.to_html(index=False)},
        ],
        methods_appendix=methods,
    )

    html = Path(report_path).read_text(encoding="utf-8")
    assert "Study Design Audit" in html
    assert "Sample Mixed Effects" in html
    assert "Region Mixed Effects" in html
    assert "Statistical Analysis" in html
