from pathlib import Path

import pandas as pd
import tifffile

from src.validation import build_validation_table, summarize_validation


def test_validation_uses_label_paths_when_present(tmp_path: Path):
    labels = [[0, 1, 1], [0, 0, 2], [0, 0, 2]]
    label_path = tmp_path / "labels.tif"
    tifffile.imwrite(label_path, labels)
    sample_table = pd.DataFrame(
        [
            {
                "sample_id": "S1",
                "cell_count": 3,
                "label_path": str(label_path),
                "expected_total_objects": 5,
            }
        ]
    )

    validation = build_validation_table(sample_table)

    assert validation.loc[0, "manual_count"] == 2
    assert validation.loc[0, "validation_source"] == "label_path"


def test_summarize_validation_returns_core_metrics():
    validation_table = pd.DataFrame(
        [
            {"cell_count": 10, "manual_count": 12},
            {"cell_count": 8, "manual_count": 9},
            {"cell_count": 11, "manual_count": 11},
        ]
    )

    summary = summarize_validation(validation_table)

    assert summary.loc[0, "n_samples"] == 3
    assert "rmse" in summary.columns


def test_validation_uses_manual_count_column_when_present():
    sample_table = pd.DataFrame(
        [
            {
                "sample_id": "S1",
                "cell_count": 7,
                "manual_count": 9,
            }
        ]
    )

    validation = build_validation_table(sample_table)

    assert validation.loc[0, "manual_count"] == 9
    assert validation.loc[0, "validation_source"] == "manual_count"
