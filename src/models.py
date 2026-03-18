# src/models.py

from __future__ import annotations
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import numpy as np

from src.blob_watershed import segment_blob_watershed
# We keep Cellpose as a dependable default
from src.cell_segmentation import segment_cells_cellpose
from src.model_registry import (
    DEFAULT_STARDIST_MODEL,
    DEFAULT_SAM_MODEL_TYPE,
    ModelSpec,
    model_summary_fields,
)


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
    def __init__(self, diameter: Optional[float], model_spec: ModelSpec, use_gpu: bool):
        self.diameter = diameter
        self.model_spec = model_spec
        self.use_gpu = use_gpu

    def segment(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        effective_model_type = self.model_spec.asset_path or self.model_spec.builtin_name or self.model_spec.model_type
        masks, flows, styles, diams = segment_cells_cellpose(
            image,
            diameter=self.diameter,
            model_type=effective_model_type,
            channels=[0, 0],
            use_gpu=self.use_gpu
        )
        info = {
            "backend": "cellpose",
            "diams": diams,
            "flows_shape": tuple(f.shape for f in flows) if isinstance(flows, (list, tuple)) else None,
            "model_type": effective_model_type,
            "use_gpu": self.use_gpu
        }
        info.update(model_summary_fields(self.model_spec))
        # Ensure uint16 labels
        masks = masks.astype(np.uint16, copy=False)
        return masks, info


class StarDistSegmenter(Segmenter):
    """Optional StarDist segmenter. Requires stardist + csbdeep installed."""
    def __init__(self, model_spec: ModelSpec):
        try:
            from stardist.models import StarDist2D
            self._StarDist2D = StarDist2D
        except Exception as e:
            raise ImportError(
                "StarDistSegmenter requires 'stardist' and 'csbdeep' to be installed. "
                "Install extras or switch backend to 'cellpose'."
            ) from e
        self.model_spec = model_spec
        self.pretrained = model_spec.builtin_name or DEFAULT_STARDIST_MODEL
        self.model = self._load_model()

    def _load_model(self):
        if self.model_spec.asset_path is None:
            return self._StarDist2D.from_pretrained(self.pretrained)

        asset_path = Path(self.model_spec.asset_path)
        try:
            if asset_path.is_dir():
                return self._StarDist2D(None, name=asset_path.name, basedir=str(asset_path.parent))

            model_dir = asset_path.parent.parent if asset_path.parent.name == "weights" else asset_path.parent
            model_name = model_dir.name
            model = self._StarDist2D(None, name=model_name, basedir=str(model_dir.parent))
            if hasattr(model, "load_weights"):
                model.load_weights(str(asset_path))
            return model
        except Exception as exc:  # pragma: no cover - depends on optional runtime package
            raise RuntimeError(
                f"Could not load custom StarDist weights from '{asset_path}'. "
                "Provide a StarDist model directory or weights file compatible with StarDist2D."
            ) from exc

    def segment(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        # StarDist expects float in [0,1]
        img = image.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        labels, _ = self.model.predict_instances(img)
        labels = labels.astype(np.uint16, copy=False)
        info = {"backend": "stardist", "pretrained": self.pretrained}
        info.update(model_summary_fields(self.model_spec))
        return labels, info


class SAMSegmenter(Segmenter):
    """
    Promptless SAM auto-mask generator as a fallback for tough images.
    This is experimental for cell somas. Requires 'segment-anything' or a SAM2 lib and weights.
    """
    def __init__(self, model_spec: ModelSpec, device: str = "cpu"):
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

        model_type = model_spec.model_type or DEFAULT_SAM_MODEL_TYPE
        model_checkpoint = model_spec.asset_path
        if not model_checkpoint:
            raise ValueError("SAMSegmenter requires a resolved checkpoint path.")

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
        self.model_spec = model_spec
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
        info.update(model_summary_fields(self.model_spec))
        return labels, info


class BlobWatershedSegmenter(Segmenter):
    """Marker-based blob detector for sparse, bright soma-like objects."""

    def __init__(self, model_spec: ModelSpec, config: dict[str, Any] | None = None):
        self.model_spec = model_spec
        self.config = dict(config or {})

    def segment(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        labels, info = segment_blob_watershed(
            image,
            apply_clahe=bool(self.config.get("apply_clahe", False)),
            min_sigma=float(self.config.get("min_sigma", 2.0)),
            max_sigma=float(self.config.get("max_sigma", 6.0)),
            num_sigma=int(self.config.get("num_sigma", 5)),
            threshold_rel=float(self.config.get("threshold_rel", 0.15)),
            min_distance=int(self.config.get("min_distance", 6)),
            min_size=int(self.config.get("min_size", 20)),
            max_size=int(self.config.get("max_size", 400)),
            min_mean_intensity=float(self.config["min_mean_intensity"]) if self.config.get("min_mean_intensity") is not None else None,
            compactness=float(self.config.get("compactness", 0.0)),
        )
        info.update(model_summary_fields(self.model_spec))
        info["model_type"] = self.model_spec.model_type or "blob_watershed"
        return labels.astype(np.uint16, copy=False), info


def build_segmenter(model_spec: ModelSpec,
                    diameter: Optional[float],
                    use_gpu: bool,
                    segmenter_config: dict[str, Any] | None = None) -> Segmenter:
    """
    Factory for segmenters.
    """
    backend = (model_spec.backend or "cellpose").lower()
    if backend == "cellpose":
        return CellposeSegmenter(diameter, model_spec, use_gpu)
    elif backend == "stardist":
        return StarDistSegmenter(model_spec=model_spec)
    elif backend == "sam":
        return SAMSegmenter(model_spec=model_spec, device="cuda" if use_gpu else "cpu")
    elif backend == "blob_watershed":
        return BlobWatershedSegmenter(model_spec=model_spec, config=segmenter_config)
    else:
        warnings.warn(f"Unknown backend '{backend}'. Falling back to Cellpose.")
        fallback_spec = ModelSpec(
            backend="cellpose",
            source="builtin",
            model_label="cellpose_builtin:cyto",
            display_label="cellpose_builtin:cyto",
            builtin_name="cyto",
            asset_path=None,
            model_type="cyto",
            alias=None,
            trust_mode=model_spec.trust_mode,
        )
        return CellposeSegmenter(diameter, fallback_spec, use_gpu)
