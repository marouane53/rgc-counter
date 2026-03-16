from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd


REQUIRED_TRAIN_MANIFEST_COLUMNS = [
    "image_path",
    "label_path",
    "split",
    "channel_index",
    "notes",
]


def load_train_manifest(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in REQUIRED_TRAIN_MANIFEST_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[REQUIRED_TRAIN_MANIFEST_COLUMNS + [c for c in frame.columns if c not in REQUIRED_TRAIN_MANIFEST_COLUMNS]]


def validate_train_manifest(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        raise ValueError("Training manifest is empty.")

    normalized = frame.copy()
    normalized["split"] = normalized["split"].astype(str).str.lower()
    invalid = normalized.loc[~normalized["split"].isin(["train", "val"])]
    if not invalid.empty:
        raise ValueError("Training manifest split must be 'train' or 'val'.")

    for column in ("image_path", "label_path"):
        missing = normalized.loc[normalized[column].isna() | (normalized[column].astype(str).str.strip() == "")]
        if not missing.empty:
            raise ValueError(f"Training manifest column '{column}' contains missing values.")
        bad_paths = [str(value) for value in normalized[column].tolist() if not Path(str(value)).exists()]
        if bad_paths:
            raise FileNotFoundError(f"Training manifest references missing paths in '{column}': {bad_paths[:3]}")

    train_rows = normalized.loc[normalized["split"] == "train"].copy()
    val_rows = normalized.loc[normalized["split"] == "val"].copy()
    if train_rows.empty:
        raise ValueError("Training manifest must include at least one 'train' row.")
    return train_rows, val_rows


def default_cellpose_trainer(
    *,
    train_rows: pd.DataFrame,
    val_rows: pd.DataFrame,
    output_dir: Path,
    pretrained_model: str,
    use_gpu: bool,
    n_epochs: int | None,
) -> dict[str, Any]:
    from cellpose import models, train

    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = f"retinal_phenotyper_{pretrained_model}"
    model = models.CellposeModel(gpu=use_gpu, model_type=pretrained_model)
    filename, train_losses, test_losses = train.train_seg(
        model.net,
        train_files=train_rows["image_path"].astype(str).tolist(),
        train_labels_files=train_rows["label_path"].astype(str).tolist(),
        test_files=val_rows["image_path"].astype(str).tolist() if not val_rows.empty else None,
        test_labels_files=val_rows["label_path"].astype(str).tolist() if not val_rows.empty else None,
        load_files=True,
        channels=[0, 0],
        save_path=output_dir,
        model_name=model_name,
        n_epochs=int(n_epochs) if n_epochs is not None else 2000,
    )
    return {
        "checkpoint_path": str(filename),
        "final_train_loss": float(train_losses[-1]) if len(train_losses) else None,
        "final_val_loss": float(test_losses[-1]) if len(test_losses) else None,
    }


def train_cellpose_from_manifest(
    manifest_df: pd.DataFrame,
    *,
    output_dir: str | Path,
    pretrained_model: str = "cyto",
    diameter: float | None = None,
    use_gpu: bool = False,
    n_epochs: int | None = None,
    trainer: Callable[..., dict[str, Any]] = default_cellpose_trainer,
) -> tuple[dict[str, Any], dict[str, Path]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows, val_rows = validate_train_manifest(manifest_df)

    result = trainer(
        train_rows=train_rows,
        val_rows=val_rows,
        output_dir=output_dir,
        pretrained_model=pretrained_model,
        use_gpu=use_gpu,
        n_epochs=n_epochs,
    )

    config = {
        "pretrained_model": pretrained_model,
        "diameter": diameter,
        "use_gpu": use_gpu,
        "n_epochs": n_epochs,
        "n_train": int(len(train_rows)),
        "n_val": int(len(val_rows)),
        "checkpoint_path": result.get("checkpoint_path"),
    }
    config_path = output_dir / "training_config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    manifest_copy_path = output_dir / "train_manifest.csv"
    manifest_df.to_csv(manifest_copy_path, index=False)

    report_lines = [
        "# Cellpose Training Report",
        "",
        f"- Pretrained model: `{pretrained_model}`",
        f"- Diameter recorded: `{diameter}`",
        f"- GPU enabled: `{use_gpu}`",
        f"- Epochs: `{n_epochs}`",
        f"- Train rows: `{len(train_rows)}`",
        f"- Validation rows: `{len(val_rows)}`",
        f"- Checkpoint path: `{result.get('checkpoint_path')}`",
    ]
    if result.get("final_train_loss") is not None:
        report_lines.append(f"- Final train loss: `{result['final_train_loss']}`")
    if result.get("final_val_loss") is not None:
        report_lines.append(f"- Final validation loss: `{result['final_val_loss']}`")
    report_path = output_dir / "training_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    artifacts = {
        "training_config": config_path,
        "train_manifest": manifest_copy_path,
        "training_report": report_path,
    }
    checkpoint = result.get("checkpoint_path")
    if checkpoint:
        artifacts["checkpoint"] = Path(str(checkpoint))
    return result, artifacts
