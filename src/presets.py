from __future__ import annotations

from copy import deepcopy
from typing import Any


SEGMENTATION_PRESETS: dict[str, dict[str, Any]] = {
    "flatmount_rgc_rbpms_demo": {
        "backend_priority": ["blob_watershed", "cellpose"],
        "apply_clahe": True,
        "diameter": 12.0,
        "min_size": 20,
        "max_size": 400,
        "marker_metrics": True,
        "segmenter_config": {
            "apply_clahe": True,
            "min_sigma": 2.0,
            "max_sigma": 6.0,
            "num_sigma": 5,
            "threshold_rel": 0.15,
            "min_distance": 6,
            "min_size": 20,
            "max_size": 400,
            "min_mean_intensity": 0.05,
            "compactness": 0.0,
        },
        "object_filters": {
            "min_area_px": 20,
            "max_area_px": 400,
            "min_focus_overlap_fraction": 0.0,
            "min_mean_intensity": 10.0,
            "min_local_contrast": 2.0,
            "max_eccentricity": 0.98,
            "min_solidity": 0.45,
            "min_circularity": 0.05,
        },
    },
    "flatmount_rgc_brn3a_demo": {
        "backend_priority": ["blob_watershed", "cellpose"],
        "apply_clahe": True,
        "diameter": 10.0,
        "min_size": 15,
        "max_size": 300,
        "marker_metrics": True,
        "segmenter_config": {
            "apply_clahe": True,
            "min_sigma": 1.5,
            "max_sigma": 5.0,
            "num_sigma": 5,
            "threshold_rel": 0.12,
            "min_distance": 5,
            "min_size": 15,
            "max_size": 300,
            "min_mean_intensity": 0.04,
            "compactness": 0.0,
        },
        "object_filters": {
            "min_area_px": 15,
            "max_area_px": 300,
            "min_focus_overlap_fraction": 0.0,
            "min_mean_intensity": 8.0,
            "min_local_contrast": 1.5,
            "max_eccentricity": 0.98,
            "min_solidity": 0.4,
            "min_circularity": 0.04,
        },
    },
    "vascular_demo": {
        "backend_priority": ["cellpose"],
        "apply_clahe": False,
        "diameter": 18.0,
        "min_size": 20,
        "max_size": 5000,
        "marker_metrics": False,
        "segmenter_config": {},
        "object_filters": {
            "min_area_px": 20,
            "max_area_px": 5000,
        },
    },
}


def segmentation_preset_names() -> list[str]:
    return sorted(SEGMENTATION_PRESETS)


def resolve_segmentation_preset(name: str | None) -> dict[str, Any] | None:
    if name is None:
        return None
    key = str(name).strip()
    if not key:
        return None
    if key not in SEGMENTATION_PRESETS:
        raise ValueError(f"Unknown segmentation preset: {key}")
    return deepcopy(SEGMENTATION_PRESETS[key])


def _merge_missing_dict_values(base: dict[str, Any] | None, defaults: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base or {})
    for key, value in defaults.items():
        if key not in merged or merged[key] is None:
            merged[key] = deepcopy(value)
    return merged


def apply_segmentation_preset(base: dict[str, Any]) -> dict[str, Any]:
    resolved = deepcopy(base)
    preset = resolve_segmentation_preset(resolved.get("segmentation_preset"))
    if preset is None:
        return resolved

    backend_priority = preset.get("backend_priority") or []
    if not resolved.get("backend") and backend_priority:
        resolved["backend"] = str(backend_priority[0])

    for key in ("diameter", "min_size", "max_size"):
        if resolved.get(key) is None and preset.get(key) is not None:
            resolved[key] = preset[key]

    if not bool(resolved.get("apply_clahe")) and bool(preset.get("apply_clahe")):
        resolved["apply_clahe"] = True

    if not bool(resolved.get("marker_metrics")) and bool(preset.get("marker_metrics")):
        resolved["marker_metrics"] = True

    resolved["segmenter_config"] = _merge_missing_dict_values(
        resolved.get("segmenter_config"),
        preset.get("segmenter_config", {}),
    )
    resolved["object_filters"] = _merge_missing_dict_values(
        resolved.get("object_filters"),
        preset.get("object_filters", {}),
    )
    return resolved
