# src/phenotype.py

from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np
import yaml
import cv2

def load_rules(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _binarize_channel(img: np.ndarray, min_intensity: int) -> np.ndarray:
    """Simple threshold after contrast normalization to 0..255 if needed."""
    if img.dtype != np.uint8:
        norm = (img.astype(np.float32) / max(1.0, float(img.max()))) * 255.0
        img8 = norm.astype(np.uint8)
    else:
        img8 = img
    _, mask = cv2.threshold(img8, int(min_intensity), 255, cv2.THRESH_BINARY)
    return mask > 0

def _circularity(mask: np.ndarray) -> float:
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    cnt = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    if peri <= 1e-6:
        return 0.0
    return float(4.0 * np.pi * area / (peri * peri))

def apply_marker_rules(image_multi: np.ndarray,
                       masks: np.ndarray,
                       rules: Dict[str, Any]) -> Tuple[np.ndarray, Dict[int, Dict[str, Any]]]:
    """
    Apply marker-aware inclusion/exclusion based on phenotype rules.
    Returns (filtered_masks, annotations_per_object)
    """
    # Parse rules
    channels = rules.get("channels", {})
    thr = rules.get("thresholds", {})
    logic = rules.get("logic", {})
    morph = rules.get("morphology_priors", {})

    rgc_ch = channels.get("rgc_channel", None)
    mg_ch = channels.get("microglia_channel", None)

    # Validate channels
    if image_multi.ndim == 2:
        raise ValueError("Phenotype rules require a multi-channel image.")
    if rgc_ch is None:
        raise ValueError("rules.channels.rgc_channel must be set.")

    rgc_min = int(thr.get("rgc_min_intensity", 120))
    mg_min = int(thr.get("microglia_min_intensity", 120))

    require_rgc = bool(logic.get("require_rgc_positive", True))
    exclude_mg = bool(logic.get("exclude_microglia_overlap", True))

    min_area = int(morph.get("min_area_px", 0))
    max_area = int(morph.get("max_area_px", 1_000_000))
    min_circ = float(morph.get("min_circularity", 0.0))

    # Extract channels
    # Assume image_multi is (Y, X, C)
    yx = image_multi.shape[:2]
    def get_ch(idx):
        if idx is None:
            return None
        return image_multi[..., int(idx)]

    rgc_img = get_ch(rgc_ch)
    mg_img = get_ch(mg_ch)

    rgc_pos = _binarize_channel(rgc_img, rgc_min) if rgc_img is not None else None
    mg_pos = _binarize_channel(mg_img, mg_min) if mg_img is not None else None

    filtered = np.zeros_like(masks, dtype=np.uint16)
    annotations: Dict[int, Dict[str, Any]] = {}

    ids = np.unique(masks)
    next_id = 1
    for oid in ids:
        if oid == 0:
            continue
        cmask = masks == oid
        area = int(cmask.sum())
        # Morphology prior
        if area < min_area or area > max_area:
            annotations[oid] = {"kept": False, "reason": "area_out_of_range", "area": area}
            continue

        # Circularity on the object
        circ = _circularity(cmask)
        if circ < min_circ:
            annotations[oid] = {"kept": False, "reason": "circularity_low", "circularity": circ}
            continue

        # Marker logic
        keep = True
        if require_rgc and rgc_pos is not None:
            if not np.any(rgc_pos & cmask):
                keep = False
                annotations[oid] = {"kept": False, "reason": "rgc_negative", "circularity": circ, "area": area}
                continue

        if exclude_mg and mg_pos is not None:
            if np.any(mg_pos & cmask):
                keep = False
                annotations[oid] = {"kept": False, "reason": "microglia_overlap", "circularity": circ, "area": area}
                continue

        if keep:
            filtered[cmask] = next_id
            annotations[oid] = {"kept": True, "area": area, "circularity": circ}
            next_id += 1

    return filtered, annotations

