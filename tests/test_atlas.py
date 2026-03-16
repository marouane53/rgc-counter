import pandas as pd

from src.atlas import compare_region_table_to_atlas, summarize_atlas_comparison
from src.context import RunContext
from main import _write_atlas_subtype_outputs


def test_compare_region_table_to_atlas_computes_density_deltas():
    region_table = pd.DataFrame(
        {
            "sample_id": ["S1", "S1", "S1"],
            "condition": ["treated", "treated", "treated"],
            "region_axis": ["ring", "ring", "ring"],
            "region_label": ["central", "pericentral", "peripheral"],
            "density_cells_per_mm2": [100.0, 75.0, 50.0],
            "retina_region_schema": ["mouse_flatmount_v1"] * 3,
        }
    )
    atlas_reference = pd.DataFrame(
        {
            "atlas_name": ["demo"] * 3,
            "region_axis": ["ring", "ring", "ring"],
            "region_label": ["central", "pericentral", "peripheral"],
            "expected_density_cells_per_mm2": [80.0, 80.0, 60.0],
            "expected_sd": [10.0, 20.0, 10.0],
            "retina_region_schema": ["mouse_flatmount_v1"] * 3,
        }
    )

    comparison = compare_region_table_to_atlas(region_table, atlas_reference)

    assert list(comparison["delta_density_cells_per_mm2"]) == [20.0, -5.0, -10.0]
    assert "fold_change_vs_atlas" in comparison.columns
    assert "atlas_zscore" in comparison.columns


def test_summarize_atlas_comparison_groups_by_condition():
    comparison = pd.DataFrame(
        {
            "atlas_name": ["demo", "demo"],
            "condition": ["treated", "treated"],
            "delta_density_cells_per_mm2": [10.0, -20.0],
            "fold_change_vs_atlas": [1.2, 0.8],
            "atlas_zscore": [1.0, -2.0],
        }
    )

    summary = summarize_atlas_comparison(comparison)

    assert len(summary) == 1
    assert summary.loc[0, "n_regions"] == 2
    assert summary.loc[0, "mean_abs_delta_density_cells_per_mm2"] == 15.0


def test_write_atlas_subtype_outputs_aggregates_study_tables(tmp_path):
    manifest = pd.DataFrame(
        [
            {
                "sample_id": "S1",
                "animal_id": "A1",
                "eye": "OD",
                "condition": "treated",
                "timepoint_dpi": 7,
                "genotype": "WT",
            }
        ]
    )
    ctx = RunContext(path=None, image=None, meta={})  # type: ignore[arg-type]
    ctx.state["atlas_subtypes"] = {
        "summary": pd.DataFrame(
            [{"image_id": "img1", "subtype": "alpha_rgc", "top1_count": 3, "top1_fraction": 0.6, "mean_probability": 0.7, "atlas_name": "demo"}]
        ),
        "region_summary": pd.DataFrame(
            [{"image_id": "img1", "region_axis": "quadrant", "region_label": "dorsal_temporal", "subtype": "alpha_rgc", "top1_count": 2, "top1_fraction": 0.5, "mean_probability": 0.75, "atlas_name": "demo"}]
        ),
    }

    summary, region_summary, assets = _write_atlas_subtype_outputs(
        manifest_df=manifest,
        contexts=[ctx],
        output_dir=tmp_path,
    )

    assert not summary.empty
    assert not region_summary.empty
    assert summary.loc[0, "sample_id"] == "S1"
    assert region_summary.loc[0, "region_axis"] == "quadrant"
    assert len(assets) == 2
    assert (tmp_path / "atlas_subtypes" / "study_atlas_subtype_summary.csv").exists()
