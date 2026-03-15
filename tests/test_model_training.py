from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from src.model_training import load_train_manifest, train_cellpose_from_manifest, validate_train_manifest


def test_load_train_manifest_fills_required_columns(tmp_path: Path):
    manifest_path = tmp_path / "train.csv"
    pd.DataFrame([{"image_path": "img.tif", "label_path": "lab.tif", "split": "train"}]).to_csv(manifest_path, index=False)

    frame = load_train_manifest(manifest_path)

    assert "channel_index" in frame.columns
    assert "notes" in frame.columns


def test_validate_train_manifest_requires_train_rows(tmp_path: Path):
    image_path = tmp_path / "image.tif"
    label_path = tmp_path / "label.tif"
    tifffile.imwrite(image_path, np.zeros((8, 8), dtype=np.uint8))
    tifffile.imwrite(label_path, np.zeros((8, 8), dtype=np.uint16))
    frame = pd.DataFrame([{"image_path": str(image_path), "label_path": str(label_path), "split": "val", "channel_index": 0, "notes": ""}])

    try:
        validate_train_manifest(frame)
        raise AssertionError("Expected validate_train_manifest to raise")
    except ValueError as exc:
        assert "at least one 'train'" in str(exc)


def test_train_cellpose_from_manifest_writes_bundle(tmp_path: Path):
    image_path = tmp_path / "image.tif"
    label_path = tmp_path / "label.tif"
    checkpoint_path = tmp_path / "trained_model"
    tifffile.imwrite(image_path, np.zeros((8, 8), dtype=np.uint8))
    tifffile.imwrite(label_path, np.zeros((8, 8), dtype=np.uint16))
    manifest = pd.DataFrame(
        [
            {"image_path": str(image_path), "label_path": str(label_path), "split": "train", "channel_index": 0, "notes": ""},
            {"image_path": str(image_path), "label_path": str(label_path), "split": "val", "channel_index": 0, "notes": ""},
        ]
    )

    def fake_trainer(**kwargs):
        checkpoint_path.write_text("checkpoint", encoding="utf-8")
        return {
            "checkpoint_path": str(checkpoint_path),
            "final_train_loss": 0.25,
            "final_val_loss": 0.5,
        }

    result, artifacts = train_cellpose_from_manifest(
        manifest,
        output_dir=tmp_path / "out",
        pretrained_model="cyto",
        diameter=18.0,
        use_gpu=False,
        n_epochs=5,
        trainer=fake_trainer,
    )

    assert result["checkpoint_path"] == str(checkpoint_path)
    assert artifacts["training_config"].exists()
    assert artifacts["train_manifest"].exists()
    assert artifacts["training_report"].exists()
