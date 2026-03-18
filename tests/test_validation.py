from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

from src.manifest import load_manifest
from src.validation import (
    build_validation_table,
    match_points,
    point_matching_metrics,
    summarize_roi_benchmark,
    summarize_validation,
    validate_roi_benchmark_manifest,
)


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


def test_validation_raises_on_conflicting_label_and_manual_count(tmp_path: Path):
    labels = [[0, 1, 1], [0, 0, 2], [0, 0, 2]]
    label_path = tmp_path / "labels.tif"
    tifffile.imwrite(label_path, labels)
    sample_table = pd.DataFrame(
        [
            {
                "sample_id": "S1",
                "cell_count": 3,
                "label_path": str(label_path),
                "manual_count": 3,
            }
        ]
    )

    with pytest.raises(ValueError, match="Conflicting references"):
        build_validation_table(sample_table)


def test_tracked_example_fixture_uses_manual_counts_for_study_benchmark():
    root = Path(__file__).resolve().parents[1]
    manifest = load_manifest(root / "examples" / "manifests" / "example_study_manifest.csv")
    manual = pd.read_csv(root / "examples" / "manual_annotations" / "example_manual_annotations.csv")
    sample_table = manifest.merge(manual, on="sample_id", how="left")
    sample_table["cell_count"] = [1, 0]

    validation = build_validation_table(sample_table)

    assert "label_path" not in sample_table.columns
    assert set(validation["validation_source"]) == {"manual_count"}


def test_point_matching_metrics_on_toy_case():
    manual = [[10, 10], [20, 20], [30, 30]]
    predicted = [[10, 11], [20, 21], [45, 45]]

    metrics = point_matching_metrics(manual, predicted, tolerance_px=3.0)

    assert metrics["true_positive"] == 2
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(2 / 3)


def test_match_points_returns_pair_indices():
    manual = np.asarray([[10, 10], [20, 20], [40, 40]], dtype=float)
    predicted = np.asarray([[10, 11], [19, 21], [80, 80]], dtype=float)

    result = match_points(manual, predicted, tolerance_px=3.0)

    assert result.matched_pred_indices.tolist() == [0, 1]
    assert result.matched_manual_indices.tolist() == [0, 1]
    assert result.unmatched_pred_indices.tolist() == [2]
    assert result.unmatched_manual_indices.tolist() == [2]
    assert result.matched_distances_px.shape == (2,)


def test_point_matching_metrics_uses_match_points_consistently():
    manual = np.asarray([[10, 10], [20, 20], [40, 40]], dtype=float)
    predicted = np.asarray([[10, 11], [19, 21], [80, 80]], dtype=float)

    matches = match_points(manual, predicted, tolerance_px=3.0)
    metrics = point_matching_metrics(manual, predicted, tolerance_px=3.0)

    assert metrics["true_positive"] == len(matches.matched_pred_indices)
    assert metrics["false_positive"] == len(matches.unmatched_pred_indices)
    assert metrics["false_negative"] == len(matches.unmatched_manual_indices)
    assert metrics["mean_match_distance_px"] == pytest.approx(float(matches.matched_distances_px.mean()))


def test_roi_benchmark_summary_contains_precision_recall_f1():
    frame = pd.DataFrame(
        [
            {"manual_count": 4, "predicted_count": 4, "precision": 1.0, "recall": 1.0, "f1": 1.0, "count_mae": 0.0, "runtime_seconds": 0.1},
            {"manual_count": 5, "predicted_count": 4, "precision": 0.8, "recall": 0.8, "f1": 0.8, "count_mae": 1.0, "runtime_seconds": 0.2},
        ]
    )

    summary = summarize_roi_benchmark(frame)

    assert "precision" in summary.columns
    assert "recall" in summary.columns
    assert "f1" in summary.columns


def test_manual_benchmark_rejects_mixed_marker_manifest():
    manifest = pd.DataFrame(
        [
            {
                "roi_id": "R1",
                "image_path": "a.tif",
                "marker": "RBPMS",
                "modality": "flatmount",
                "x0": 0,
                "y0": 0,
                "width": 32,
                "height": 32,
                "annotator": "A",
                "manual_points_path": "a.csv",
            },
            {
                "roi_id": "R2",
                "image_path": "b.tif",
                "marker": "BRN3A",
                "modality": "flatmount",
                "x0": 0,
                "y0": 0,
                "width": 32,
                "height": 32,
                "annotator": "A",
                "manual_points_path": "b.csv",
            },
        ]
    )

    with pytest.raises(ValueError, match="mixes markers"):
        validate_roi_benchmark_manifest(manifest)
