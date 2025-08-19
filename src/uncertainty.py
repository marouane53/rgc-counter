# src/uncertainty.py

from __future__ import annotations
from typing import List, Tuple, Dict, Any, Callable
import numpy as np
from scipy.ndimage import rotate


Transform = Callable[[np.ndarray], np.ndarray]
InverseTransform = Callable[[np.ndarray], np.ndarray]

def _flip_h(img: np.ndarray) -> np.ndarray:
    return np.flip(img, axis=1)

def _flip_v(img: np.ndarray) -> np.ndarray:
    return np.flip(img, axis=0)

def _rot90(img: np.ndarray) -> np.ndarray:
    return np.rot90(img, k=1)

def _rot270(img: np.ndarray) -> np.ndarray:
    return np.rot90(img, k=3)

def _inv_flip_h(img: np.ndarray) -> np.ndarray:
    return np.flip(img, axis=1)

def _inv_flip_v(img: np.ndarray) -> np.ndarray:
    return np.flip(img, axis=0)

def _inv_rot90(img: np.ndarray) -> np.ndarray:
    return np.rot90(img, k=3)

def _inv_rot270(img: np.ndarray) -> np.ndarray:
    return np.rot90(img, k=1)

TRANSFORMS = {
    "flip_h": (_flip_h, _inv_flip_h),
    "flip_v": (_flip_v, _inv_flip_v),
    "rot90": (_rot90, _inv_rot90),
    "rot270": (_rot270, _inv_rot270),
}

def _label_to_binary(label: np.ndarray) -> np.ndarray:
    """Foreground/background binary image."""
    return (label > 0).astype(np.uint8)

def _pixel_vote(aggregated_bins: List[np.ndarray], threshold: float = 0.5) -> np.ndarray:
    """Combine foreground votes and return a clean binary mask."""
    stack = np.stack(aggregated_bins, axis=0).astype(np.float32)
    prob = stack.mean(axis=0)
    bin_mask = (prob >= threshold).astype(np.uint8)
    return bin_mask, prob

def _binary_to_instances(bin_mask: np.ndarray) -> np.ndarray:
    """Connected components to instances."""
    from scipy.ndimage import label
    lbl, _ = label(bin_mask)
    return lbl.astype(np.uint16)

def segment_with_tta(segmenter,
                     image: np.ndarray,
                     transforms: List[str] | None = None,
                     combine: str = "pixel_vote") -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Run segmentation with test-time augmentations and combine results.
    combine = 'pixel_vote' uses pixel-level majority voting across transforms.
    """
    transforms = transforms or []
    fwd_inv = [(lambda x: x, lambda x: x)]  # identity
    for tname in transforms:
        if tname not in TRANSFORMS:
            continue
        fwd_inv.append(TRANSFORMS[tname])

    # Segment original
    masks0, info0 = segmenter.segment(image)
    agg_bin = [_label_to_binary(masks0)]

    # Apply TTA
    for fwd, inv in fwd_inv[1:]:
        img_t = fwd(image)
        masks_t, _ = segmenter.segment(img_t)
        masks_back = inv(masks_t)
        agg_bin.append(_label_to_binary(masks_back))

    # Combine
    if combine == "pixel_vote":
        bin_mask, prob = _pixel_vote(agg_bin, threshold=0.5)
        inst = _binary_to_instances(bin_mask)
        info = dict(info0)
        info["tta"] = True
        info["tta_transforms"] = list(transforms)
        info["foreground_probability"] = prob  # float32 map
        info["combiner"] = "pixel_vote"
        return inst, info
    else:
        # fallback: choose the mask most similar to the average
        # Here we just return the original for simplicity
        info = dict(info0)
        info["tta"] = False
        info["combiner"] = "none"
        return masks0, info

