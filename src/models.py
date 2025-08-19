# src/models.py

from __future__ import annotations
import warnings
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any

import numpy as np

# We keep Cellpose as a dependable default
from src.cell_segmentation import segment_cells_cellpose


class Segmenter(ABC):
    """Abstract segmenter interface. All segmenters must return integer masks."""
    @abstractmethod
    def segment(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        :param image: 2D numpy array (grayscale) to segment.
        :return: (masks, info)
                 masks: 2D uint16 labeled mask, 0 = background.
                 info: dict with model metadata (e.g., 'backend', 'diams', 'extras').
        """
        raise NotImplementedError


class CellposeSegmenter(Segmenter):
    """Adapter for Cellpose with your existing wrapper."""
    def __init__(self, diameter: Optional[float], model_type: str, use_gpu: bool):
        self.diameter = diameter
        self.model_type = model_type
        self.use_gpu = use_gpu

    def segment(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        masks, flows, styles, diams = segment_cells_cellpose(
            image,
            diameter=self.diameter,
            model_type=self.model_type,
            channels=[0, 0],
            use_gpu=self.use_gpu
        )
        info = {
            "backend": "cellpose",
            "diams": diams,
            "flows_shape": tuple(f.shape for f in flows) if isinstance(flows, (list, tuple)) else None,
            "model_type": self.model_type,
            "use_gpu": self.use_gpu
        }
        # Ensure uint16 labels
        masks = masks.astype(np.uint16, copy=False)
        return masks, info


class StarDistSegmenter(Segmenter):
    """Optional StarDist segmenter. Requires stardist + csbdeep installed."""
    def __init__(self, pretrained: str = "2D_versatile_fluo"):
        try:
            from stardist.models import StarDist2D
            self._StarDist2D = StarDist2D
        except Exception as e:
            raise ImportError(
                "StarDistSegmenter requires 'stardist' and 'csbdeep' to be installed. "
                "Install extras or switch backend to 'cellpose'."
            ) from e
        self.model = self._StarDist2D.from_pretrained(pretrained)
        self.pretrained = pretrained

    def segment(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        # StarDist expects float in [0,1]
        img = image.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        labels, _ = self.model.predict_instances(img)
        labels = labels.astype(np.uint16, copy=False)
        info = {"backend": "stardist", "pretrained": self.pretrained}
        return labels, info


class SAMSegmenter(Segmenter):
    """
    Promptless SAM auto-mask generator as a fallback for tough images.
    This is experimental for cell somas. Requires 'segment-anything' or a SAM2 lib and weights.
    """
    def __init__(self, model_checkpoint: str, model_type: str = "vit_h", device: str = "cpu"):
        try:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor  # type: ignore
            self._sam_model_registry = sam_model_registry
            self._SamAutomaticMaskGenerator = SamAutomaticMaskGenerator
            self._SamPredictor = SamPredictor
        except Exception as e:
            raise ImportError(
                "SAMSegmenter requires the 'segment-anything' package and model weights. "
                "Install it and provide a valid checkpoint."
            ) from e

        sam = self._sam_model_registry[model_type](checkpoint=model_checkpoint)
        sam.to(device)
        self.mask_generator = self._SamAutomaticMaskGenerator(
            sam,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            min_mask_region_area=64
        )
        self.device = device
        self.model_type = model_type

    @staticmethod
    def _masks_to_label(mask_list, shape) -> np.ndarray:
        """Convert list of dict masks from SAM to a single label image."""
        label = np.zeros(shape, dtype=np.uint16)
        next_id = 1
        for m in mask_list:
            msk = m.get("segmentation", None)
            if msk is None:
                continue
            label[msk] = next_id
            next_id += 1
        return label

    def segment(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        # SAM expects 3-channel RGB in uint8
        img = image
        if img.ndim == 2:
            # stack to 3 channels
            if img.dtype != np.uint8:
                norm = (img.astype(np.float32) / max(1.0, float(img.max()))) * 255.0
                img_rgb = np.stack([norm, norm, norm], axis=-1).astype(np.uint8)
            else:
                img_rgb = np.stack([img, img, img], axis=-1)
        else:
            img_rgb = image

        masks = self.mask_generator.generate(img_rgb)
        labels = self._masks_to_label(masks, shape=img_rgb.shape[:2])
        info = {
            "backend": "sam",
            "device": self.device,
            "model_type": self.model_type,
            "n_regions": int(labels.max())
        }
        return labels, info


def build_segmenter(backend: str,
                    diameter: Optional[float],
                    model_type: str,
                    use_gpu: bool,
                    sam_checkpoint: Optional[str] = None,
                    stardist_pretrained: str = "2D_versatile_fluo") -> Segmenter:
    """
    Factory for segmenters.
    """
    backend = (backend or "cellpose").lower()
    if backend == "cellpose":
        return CellposeSegmenter(diameter, model_type, use_gpu)
    elif backend == "stardist":
        return StarDistSegmenter(pretrained=stardist_pretrained)
    elif backend == "sam":
        if not sam_checkpoint:
            raise ValueError("SAM backend selected but no model checkpoint was provided.")
        return SAMSegmenter(model_checkpoint=sam_checkpoint, model_type="vit_h", device="cuda" if use_gpu else "cpu")
    else:
        warnings.warn(f"Unknown backend '{backend}'. Falling back to Cellpose.")
        return CellposeSegmenter(diameter, model_type, use_gpu)

