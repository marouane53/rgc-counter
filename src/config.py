from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG: dict[str, Any] = {
    "microns_per_pixel": 0.227,
    "cell_detection": {
        "diameter": None,
        "model_type": "cyto",
        "use_gpu": True,
        "backend": "cellpose",
    },
    "analysis": {
        "min_cell_size": 10,
        "max_cell_size": 10000,
        "compute_spatial_stats": False,
        "optic_nerve_center": None,
        "ripley_radii": [25, 50, 75, 100, 150, 200],
    },
    "visualization": {
        "overlay_alpha": 0.5,
        "save_debug": False,
    },
    "io": {
        "save_ome_zarr": False,
        "ome_zarr_chunk": 256,
        "write_html_report": True,
    },
    "qc": {
        "tile_size": 64,
        "brightness_min": 20,
        "brightness_max": 230,
        "laplacian_z": 1.0,
        "tenengrad_z": 1.0,
        "highfreq_z": 1.0,
        "threshold_z": 0.0,
        "morph_kernel": 5,
    },
    "phenotype_rules": None,
    "tta": {
        "enabled": False,
        "transforms": ["flip_h", "flip_v", "rot90", "rot270"],
        "combine": "pixel_vote",
    },
}


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def candidate_config_paths() -> list[Path]:
    env_path = os.environ.get("RETINAL_PHENOTYPER_CONFIG") or os.environ.get("RGC_COUNTER_CONFIG")
    module_dir = Path(__file__).resolve().parent

    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.append(module_dir.parent / "config.yaml")
    candidates.append(module_dir / "config.yaml")
    return candidates


def load_config() -> tuple[dict[str, Any], Path | None]:
    for candidate in candidate_config_paths():
        if candidate.exists():
            with candidate.open("r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle) or {}
            return _deep_update(DEFAULT_CONFIG, loaded), candidate.resolve()
    return deepcopy(DEFAULT_CONFIG), None


data, _resolved_path = load_config()
CONFIG_PATH = str(_resolved_path) if _resolved_path is not None else None
CONFIG_SOURCE = CONFIG_PATH or "builtin_defaults"

# Basic settings
MICRONS_PER_PIXEL = data.get("microns_per_pixel", 0.227)

# Cell detection settings
cell_detection = data.get("cell_detection", {})
CELL_DIAMETER = cell_detection.get("diameter", None)
MODEL_TYPE = cell_detection.get("model_type", "cyto")
USE_GPU = cell_detection.get("use_gpu", True)

# Analysis settings
analysis = data.get("analysis", {})
MIN_CELL_SIZE = analysis.get("min_cell_size", 5)
MAX_CELL_SIZE = analysis.get("max_cell_size", 1000000)

# Visualization settings
visualization = data.get("visualization", {})
OVERLAY_ALPHA = visualization.get("overlay_alpha", 0.5)
