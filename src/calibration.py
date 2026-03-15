from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


SUPPORTED_SELECTION_METRICS = {"mae", "bias", "mape", "corr"}


def load_calibration_grid(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    grid = payload.get("grid")
    if not isinstance(grid, list) or not grid:
        raise ValueError("Calibration YAML must contain a non-empty 'grid' list.")
    selection_metric = str(payload.get("selection_metric", "mae")).lower()
    if selection_metric not in SUPPORTED_SELECTION_METRICS:
        raise ValueError(f"Unsupported selection metric: {selection_metric}")
    return {"grid": grid, "selection_metric": selection_metric}


def apply_dotted_overrides(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for dotted_key, value in overrides.items():
        cursor = out
        parts = str(dotted_key).split(".")
        for part in parts[:-1]:
            if part not in cursor or not isinstance(cursor[part], dict):
                cursor[part] = {}
            cursor = cursor[part]
        cursor[parts[-1]] = value
    return out


def evaluate_count_agreement(sample_table: pd.DataFrame) -> dict[str, float | int]:
    if sample_table.empty or "manual_count" not in sample_table.columns:
        return {"n": 0, "mae": np.nan, "bias": np.nan, "mape": np.nan, "corr": np.nan}

    eval_df = sample_table.dropna(subset=["manual_count"]).copy()
    if eval_df.empty:
        return {"n": 0, "mae": np.nan, "bias": np.nan, "mape": np.nan, "corr": np.nan}

    eval_df["count_error"] = eval_df["cell_count"] - eval_df["manual_count"]
    eval_df["ape"] = (eval_df["count_error"].abs() / eval_df["manual_count"].clip(lower=1)) * 100.0
    corr = float(eval_df[["cell_count", "manual_count"]].corr().iloc[0, 1]) if len(eval_df) > 1 else np.nan
    return {
        "n": int(len(eval_df)),
        "mae": float(eval_df["count_error"].abs().mean()),
        "bias": float(eval_df["count_error"].mean()),
        "mape": float(eval_df["ape"].mean()),
        "corr": corr,
    }


def rank_grid_results(frame: pd.DataFrame, selection_metric: str) -> pd.DataFrame:
    ascending = selection_metric != "corr"
    return frame.sort_values([selection_metric, "mae"], ascending=[ascending, True], na_position="last").reset_index(drop=True)


def write_best_params(path: str | Path, params: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(params, handle, indent=2)
    return path


def write_calibration_report(
    path: str | Path,
    *,
    selection_metric: str,
    best_row: dict[str, Any],
    grid_size: int,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Calibration Report",
        "",
        f"- Grid points evaluated: {grid_size}",
        f"- Selection metric: {selection_metric}",
        f"- Best MAE: {best_row.get('mae')}",
        f"- Best bias: {best_row.get('bias')}",
        f"- Best MAPE: {best_row.get('mape')}",
        f"- Best correlation: {best_row.get('corr')}",
        "",
        "## Best Parameters",
        "",
        "```json",
        json.dumps(best_row.get("params", {}), indent=2),
        "```",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
