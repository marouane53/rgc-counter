from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd


def canonical_recipe_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def default_projection_recipes() -> list[dict[str, Any]]:
    return [
        {"projection_id": "full_max", "projection_kind": "full_max"},
        {"projection_id": "best_slab_max_24", "projection_kind": "best_slab_max", "window_size": 24},
        {"projection_id": "best_slab_max_32", "projection_kind": "best_slab_max", "window_size": 32},
        {"projection_id": "best_slab_max_48", "projection_kind": "best_slab_max", "window_size": 48},
        {"projection_id": "best_slab_topkmean_8_of_32", "projection_kind": "best_slab_topkmean", "window_size": 32, "topk": 8},
        {"projection_id": "best_slab_topkmean_16_of_48", "projection_kind": "best_slab_topkmean", "window_size": 48, "topk": 16},
        {"projection_id": "full_focus_weighted", "projection_kind": "full_focus_weighted"},
    ]


def default_preprocess_variants() -> list[dict[str, Any]]:
    return [
        {"preprocess_id": "none", "background_subtraction": "none", "normalization": "robust_float"},
        {"preprocess_id": "white_tophat_r10", "background_subtraction": "white_tophat", "radius_px": 10, "normalization": "robust_float"},
        {"preprocess_id": "white_tophat_r14", "background_subtraction": "white_tophat", "radius_px": 14, "normalization": "robust_float"},
        {"preprocess_id": "rolling_ball_r12", "background_subtraction": "rolling_ball", "radius_px": 12, "normalization": "robust_float"},
        {"preprocess_id": "rolling_ball_r16", "background_subtraction": "rolling_ball", "radius_px": 16, "normalization": "robust_float"},
    ]


def default_point_detector_configs() -> list[dict[str, Any]]:
    return [
        {"predictor_backend": "log", "predictor_config": {"sigma_min": 2.5, "sigma_max": 4.5, "num_sigma": 4, "threshold": 0.10, "min_distance": 8}},
        {"predictor_backend": "log", "predictor_config": {"sigma_min": 2.5, "sigma_max": 4.5, "num_sigma": 4, "threshold": 0.12, "min_distance": 10}},
        {"predictor_backend": "log", "predictor_config": {"sigma_min": 3.0, "sigma_max": 5.0, "num_sigma": 4, "threshold": 0.10, "min_distance": 10}},
        {"predictor_backend": "log", "predictor_config": {"sigma_min": 3.5, "sigma_max": 5.5, "num_sigma": 4, "threshold": 0.12, "min_distance": 12}},
        {"predictor_backend": "dog", "predictor_config": {"sigma_min": 2.5, "sigma_max": 4.5, "threshold": 0.08, "min_distance": 8}},
        {"predictor_backend": "dog", "predictor_config": {"sigma_min": 2.5, "sigma_max": 4.5, "threshold": 0.10, "min_distance": 10}},
        {"predictor_backend": "dog", "predictor_config": {"sigma_min": 3.0, "sigma_max": 5.0, "threshold": 0.08, "min_distance": 10}},
        {"predictor_backend": "dog", "predictor_config": {"sigma_min": 3.5, "sigma_max": 5.5, "threshold": 0.10, "min_distance": 12}},
        {"predictor_backend": "hmax", "predictor_config": {"h": 0.08, "min_distance": 8}},
        {"predictor_backend": "hmax", "predictor_config": {"h": 0.10, "min_distance": 10}},
        {"predictor_backend": "hmax", "predictor_config": {"h": 0.12, "min_distance": 12}},
    ]


def build_default_projection_lab_config_manifest() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for projection in default_projection_recipes():
        for preprocess in default_preprocess_variants():
            projection_json = canonical_recipe_json(projection)
            preprocess_json = canonical_recipe_json(preprocess)
            for detector in default_point_detector_configs():
                predictor_backend = str(detector["predictor_backend"])
                predictor_config = dict(detector["predictor_config"])
                predictor_slug = (
                    canonical_recipe_json(predictor_config)
                    .replace("{", "")
                    .replace("}", "")
                    .replace(":", "-")
                    .replace(",", "__")
                    .replace('"', "")
                )
                config_id = (
                    f"{projection['projection_id']}__{preprocess['preprocess_id']}__"
                    f"{predictor_backend}__{predictor_slug}"
                )
                rows.append(
                    {
                        "config_id": config_id,
                        "projection_label": projection["projection_id"],
                        "projection_recipe_json": projection_json,
                        "preprocess_label": preprocess["preprocess_id"],
                        "preprocess_json": preprocess_json,
                        "predictor_backend": predictor_backend,
                        "predictor_config_json": canonical_recipe_json(predictor_config),
                        "notes": "private_projection_lab_default",
                    }
                )
    return pd.DataFrame(rows)


def load_or_build_projection_lab_config_manifest(path: str | None) -> pd.DataFrame:
    if path is None:
        return build_default_projection_lab_config_manifest()
    return pd.read_csv(path)


def summarize_split_metrics(frame: pd.DataFrame, split: str) -> dict[str, Any]:
    subset = frame.loc[frame["split"] == split].copy()
    if subset.empty:
        return {
            f"{split}_n_rois": 0,
            f"{split}_precision_8px": float("nan"),
            f"{split}_recall_8px": float("nan"),
            f"{split}_f1_8px": float("nan"),
            f"{split}_count_mae_8px": float("nan"),
        }
    primary = subset.loc[np.isclose(subset["match_tolerance_px"].astype(float), 8.0)].copy()
    if primary.empty:
        primary = subset
    return {
        f"{split}_n_rois": int(primary["roi_id"].nunique()),
        f"{split}_precision_8px": float(primary["precision"].mean()),
        f"{split}_recall_8px": float(primary["recall"].mean()),
        f"{split}_f1_8px": float(primary["f1"].mean()),
        f"{split}_count_mae_8px": float(primary["count_mae"].mean()),
    }


def moderate_locked_eval_gate(row: dict[str, Any]) -> bool:
    dev_f1 = float(row.get("dev_f1_8px", float("nan")))
    dev_recall = float(row.get("dev_recall_8px", float("nan")))
    eval_f1 = float(row.get("locked_eval_f1_8px", float("nan")))
    eval_recall = float(row.get("locked_eval_recall_8px", float("nan")))
    dev_n = int(row.get("dev_n_rois", 0))
    eval_n = int(row.get("locked_eval_n_rois", 0))
    if dev_n < 4 or eval_n < 2:
        return False
    if not np.isfinite([dev_f1, dev_recall, eval_f1, eval_recall]).all():
        return False
    if dev_f1 < 0.60 or dev_recall < 0.60:
        return False
    if eval_f1 < 0.50 or eval_recall < 0.50:
        return False
    if abs(dev_f1 - eval_f1) > 0.15 or abs(dev_recall - eval_recall) > 0.15:
        return False
    return True
