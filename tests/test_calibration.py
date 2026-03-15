import pandas as pd

from src.calibration import apply_dotted_overrides, evaluate_count_agreement, rank_grid_results


def test_apply_dotted_overrides_updates_nested_dicts():
    base = {"qc_config": {"threshold_z": 0.0}, "phenotype_engine_config": {"classes": {"rgc": {"priority": 100}}}}

    out = apply_dotted_overrides(base, {"qc_config.threshold_z": 0.5, "phenotype_engine_config.classes.rgc.priority": 120})

    assert out["qc_config"]["threshold_z"] == 0.5
    assert out["phenotype_engine_config"]["classes"]["rgc"]["priority"] == 120


def test_evaluate_count_agreement_and_ranking():
    validation_table = pd.DataFrame(
        [
            {"cell_count": 10, "manual_count": 12},
            {"cell_count": 20, "manual_count": 19},
        ]
    )
    metrics = evaluate_count_agreement(validation_table)
    ranked = rank_grid_results(
        pd.DataFrame(
            [
                {"params_json": "a", "mae": 3.0, "bias": 1.0, "mape": 10.0, "corr": 0.8},
                {"params_json": "b", "mae": 1.0, "bias": 0.1, "mape": 5.0, "corr": 0.9},
            ]
        ),
        "mae",
    )

    assert metrics["mae"] == 1.5
    assert ranked.iloc[0]["params_json"] == "b"
