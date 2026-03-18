import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from src.model_evaluation import (
    evaluate_model_manifest,
    load_model_manifest,
    write_evaluation_outputs,
)
from src.model_registry import ModelSpec


@dataclass
class FakeRuntime:
    model_spec: ModelSpec


class FakeContext:
    def __init__(self, labels: np.ndarray):
        self.labels = labels
        self.metrics = {"cell_count": int(len(np.unique(labels[labels > 0])))}


def test_load_model_manifest_fills_required_columns(tmp_path: Path):
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame([{"run_id": "r1", "image_path": "img.tif", "label_path": "lab.tif", "backend": "cellpose"}]).to_csv(
        manifest_path, index=False
    )

    frame = load_model_manifest(manifest_path)

    assert "model_alias" in frame.columns
    assert "diameter" in frame.columns


def test_evaluate_model_manifest_ranks_by_dice_then_mae(tmp_path: Path):
    image_path = tmp_path / "image.tif"
    label_path = tmp_path / "labels.tif"
    tifffile.imwrite(image_path, np.zeros((8, 8), dtype=np.uint8))
    labels = np.zeros((8, 8), dtype=np.uint16)
    labels[1:4, 1:4] = 1
    tifffile.imwrite(label_path, labels)

    manifest = pd.DataFrame(
        [
            {
                "run_id": "builtin",
                "image_path": str(image_path),
                "label_path": str(label_path),
                "backend": "cellpose",
                "model_type": "cyto",
                "cellpose_model": None,
                "stardist_weights": None,
                "sam_checkpoint": None,
                "model_alias": None,
                "diameter": None,
                "channel_index": 0,
                "notes": "",
            },
            {
                "run_id": "custom",
                "image_path": str(image_path),
                "label_path": str(label_path),
                "backend": "cellpose",
                "model_type": None,
                "cellpose_model": str(tmp_path / "custom_model"),
                "stardist_weights": None,
                "sam_checkpoint": None,
                "model_alias": "custom",
                "diameter": None,
                "channel_index": 0,
                "notes": "",
            },
        ]
    )
    (tmp_path / "custom_model").write_text("custom", encoding="utf-8")

    def fake_runtime_builder(options):
        spec = ModelSpec(
            backend="cellpose",
            source="custom" if options.cellpose_model else "builtin",
            model_label="custom" if options.cellpose_model else "builtin",
            display_label="custom" if options.cellpose_model else "builtin",
            builtin_name=None if options.cellpose_model else "cyto",
            asset_path=options.cellpose_model,
            model_type=options.cellpose_model or options.model_type or "cyto",
            alias=options.model_alias,
            trust_mode="trusted_local_only",
        )
        return FakeRuntime(model_spec=spec)

    def fake_runtime_runner(runtime, image, source_path, meta):
        if runtime.model_spec.model_label == "custom":
            pred = labels.copy()
        else:
            pred = np.zeros_like(labels)
            pred[1:3, 1:3] = 1
        return FakeContext(pred)

    per_run_frame, ranked_summary, metadata = evaluate_model_manifest(
        manifest,
        use_gpu=False,
        runtime_builder=fake_runtime_builder,
        runtime_runner=fake_runtime_runner,
    )

    assert len(per_run_frame) == 2
    assert metadata["ranking_rule"] == "dice_then_mae"
    assert ranked_summary.iloc[0]["model_label"] == "custom"


def test_count_only_rows_do_not_override_overlap_ranking(tmp_path: Path):
    image_path = tmp_path / "image.tif"
    label_path = tmp_path / "labels.tif"
    tifffile.imwrite(image_path, np.zeros((8, 8), dtype=np.uint8))
    labels = np.zeros((8, 8), dtype=np.uint16)
    labels[1:4, 1:4] = 1
    tifffile.imwrite(label_path, labels)
    custom_model = tmp_path / "custom_model"
    custom_model.write_text("custom", encoding="utf-8")

    manifest = pd.DataFrame(
        [
            {
                "run_id": "builtin-overlap",
                "image_path": str(image_path),
                "label_path": str(label_path),
                "backend": "cellpose",
                "model_type": "cyto",
                "cellpose_model": None,
                "stardist_weights": None,
                "sam_checkpoint": None,
                "model_alias": None,
                "diameter": None,
                "channel_index": 0,
                "notes": "",
            },
            {
                "run_id": "custom-count",
                "image_path": str(image_path),
                "label_path": None,
                "manual_count": 1,
                "backend": "cellpose",
                "model_type": None,
                "cellpose_model": str(custom_model),
                "stardist_weights": None,
                "sam_checkpoint": None,
                "model_alias": "custom",
                "diameter": None,
                "channel_index": 0,
                "notes": "",
            },
        ]
    )

    def fake_runtime_builder(options):
        is_custom = bool(options.cellpose_model)
        return FakeRuntime(
            model_spec=ModelSpec(
                backend="cellpose",
                source="custom" if is_custom else "builtin",
                model_label="custom" if is_custom else "builtin",
                display_label="custom" if is_custom else "builtin",
                builtin_name=None if is_custom else "cyto",
                asset_path=options.cellpose_model,
                model_type=options.cellpose_model or options.model_type or "cyto",
                alias=options.model_alias,
                trust_mode="trusted_local_only",
            )
        )

    def fake_runtime_runner(runtime, image, source_path, meta):
        return FakeContext(labels)

    _, ranked_summary, metadata = evaluate_model_manifest(
        manifest,
        runtime_builder=fake_runtime_builder,
        runtime_runner=fake_runtime_runner,
    )

    assert metadata["ranking_rule"] == "dice_then_mae"
    assert ranked_summary.iloc[0]["model_label"] == "builtin"


def test_backend_comparison_prefers_lower_mae_when_only_counts_exist(tmp_path: Path):
    image_path = tmp_path / "image.tif"
    tifffile.imwrite(image_path, np.zeros((8, 8), dtype=np.uint8))

    manifest = pd.DataFrame(
        [
            {
                "run_id": "blob",
                "image_path": str(image_path),
                "label_path": None,
                "manual_count": 2,
                "backend": "blob_watershed",
                "segmentation_preset": "flatmount_rgc_rbpms_demo",
                "model_type": None,
                "cellpose_model": None,
                "stardist_weights": None,
                "sam_checkpoint": None,
                "model_alias": "blob",
                "diameter": None,
                "channel_index": 0,
                "notes": "",
            },
            {
                "run_id": "cellpose",
                "image_path": str(image_path),
                "label_path": None,
                "manual_count": 2,
                "backend": "cellpose",
                "segmentation_preset": None,
                "model_type": "cyto",
                "cellpose_model": None,
                "stardist_weights": None,
                "sam_checkpoint": None,
                "model_alias": "cellpose",
                "diameter": None,
                "channel_index": 0,
                "notes": "",
            },
        ]
    )

    def fake_runtime_builder(options):
        return FakeRuntime(
            model_spec=ModelSpec(
                backend=options.backend or "cellpose",
                source="builtin",
                model_label=str(options.model_alias or options.backend),
                display_label=str(options.model_alias or options.backend),
                builtin_name=options.model_type,
                asset_path=None,
                model_type=options.model_type,
                alias=options.model_alias,
                trust_mode="builtin",
            )
        )

    def fake_runtime_runner(runtime, image, source_path, meta):
        labels = np.zeros((8, 8), dtype=np.uint16)
        if runtime.model_spec.model_label == "blob":
            labels[1:3, 1:3] = 1
            labels[5:7, 5:7] = 2
        else:
            labels[1:3, 1:3] = 1
        return FakeContext(labels)

    _, ranked_summary, metadata = evaluate_model_manifest(
        manifest,
        runtime_builder=fake_runtime_builder,
        runtime_runner=fake_runtime_runner,
    )

    assert metadata["ranking_rule"] == "mae_only"
    assert ranked_summary.iloc[0]["model_label"] == "blob"


def test_write_evaluation_outputs(tmp_path: Path):
    per_run = pd.DataFrame([{"run_id": "r1", "dice_score": 1.0}])
    summary = pd.DataFrame([{"model_label": "builtin", "rank": 1}])
    metadata = {
        "best_model_json": {"model_label": "builtin", "ranking_rule": "dice_then_mae"},
        "report_markdown": "# Report\n",
    }

    outputs = write_evaluation_outputs(
        output_dir=tmp_path,
        per_run_frame=per_run,
        ranked_summary=summary,
        metadata=metadata,
    )

    assert outputs["per_run_metrics"].exists()
    assert outputs["model_summary"].exists()
    assert json.loads(outputs["best_model"].read_text(encoding="utf-8"))["model_label"] == "builtin"
