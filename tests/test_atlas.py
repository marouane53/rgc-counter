import pandas as pd

from src.atlas import compare_region_table_to_atlas, summarize_atlas_comparison


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
