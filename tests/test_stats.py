import pandas as pd

from src.stats import compute_outcome_stats, compute_region_stats


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
