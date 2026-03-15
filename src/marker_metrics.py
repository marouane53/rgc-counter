from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt


def _is_channel_last(image: np.ndarray) -> bool:
    return image.ndim == 3 and image.shape[-1] < 32


def _channel_arrays(image: np.ndarray, config: dict[str, Any] | None) -> dict[str, np.ndarray]:
    channels_cfg = (config or {}).get("channels", {})
    arrays: dict[str, np.ndarray] = {}

    if image.ndim == 2:
        arrays["GRAY"] = image.astype(np.float32)
    elif _is_channel_last(image):
        for idx in range(image.shape[-1]):
            arrays[f"C{idx}"] = image[..., idx].astype(np.float32)
    else:
        arrays["GRAY"] = image.astype(np.float32)

    for name, index in channels_cfg.items():
        if isinstance(index, int):
            if image.ndim == 2:
                if index != 0:
                    raise ValueError(f"Channel index {index} requested for single-channel image.")
                arrays[str(name)] = image.astype(np.float32)
            elif _is_channel_last(image):
                arrays[str(name)] = image[..., int(index)].astype(np.float32)
            else:
                arrays[str(name)] = image[int(index), ...].astype(np.float32)

    for name, compose_cfg in (config or {}).get("compose", {}).items():
        mode = compose_cfg.get("mode", "max")
        sources = compose_cfg.get("sources", [])
        if not sources:
            continue
        built = []
        for source in sources:
            source_name = str(source["channel"])
            if source_name not in arrays:
                raise ValueError(f"Composed channel '{name}' references unknown channel '{source_name}'.")
            weight = float(source.get("weight", 1.0))
            built.append(arrays[source_name] * weight)
        if mode == "max":
            arrays[str(name)] = np.maximum.reduce(built)
        elif mode == "sum":
            arrays[str(name)] = np.sum(built, axis=0)
        else:
            raise ValueError(f"Unsupported compose mode: {mode}")

    return arrays


def _circularity(mask: np.ndarray) -> tuple[float, float]:
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0, 0.0
    cnt = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    peri = float(cv2.arcLength(cnt, True))
    if peri <= 1e-6:
        return peri, 0.0
    circ = float(4.0 * np.pi * area / (peri * peri))
    return peri, circ


def _eccentricity(ys: np.ndarray, xs: np.ndarray) -> float:
    if len(xs) <= 2:
        return 0.0
    centered = np.column_stack([xs.astype(float), ys.astype(float)])
    centered -= centered.mean(axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    vals = np.linalg.eigvalsh(cov)
    major = float(vals.max())
    minor = float(vals.min())
    if major <= 1e-12:
        return 0.0
    ratio = max(0.0, min(1.0, minor / major))
    return float(np.sqrt(1.0 - ratio))


def _local_background(channel: np.ndarray, ys: np.ndarray, xs: np.ndarray, pad: int = 3) -> float:
    ymin = max(0, int(ys.min()) - pad)
    ymax = min(channel.shape[0], int(ys.max()) + pad + 1)
    xmin = max(0, int(xs.min()) - pad)
    xmax = min(channel.shape[1], int(xs.max()) + pad + 1)
    crop = channel[ymin:ymax, xmin:xmax]
    mask = np.zeros_like(crop, dtype=bool)
    mask[ys - ymin, xs - xmin] = True
    bg = crop[~mask]
    if bg.size == 0:
        return 0.0
    return float(bg.mean())


def _build_mask_specs(config: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    config = config or {}
    masks = dict(config.get("masks", {}))
    if "microglia" not in masks:
        channels_cfg = config.get("channels", {})
        if "MICROGLIA" in channels_cfg or "microglia" in channels_cfg:
            channel_name = "MICROGLIA" if "MICROGLIA" in channels_cfg else "microglia"
            masks["microglia"] = {"channel": channel_name, "min_intensity": 180}
    return masks


def _build_masks(channel_arrays: dict[str, np.ndarray], config: dict[str, Any] | None) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    for name, spec in _build_mask_specs(config).items():
        channel_name = str(spec["channel"])
        if channel_name not in channel_arrays:
            raise ValueError(f"Mask '{name}' references unknown channel '{channel_name}'.")
        threshold = float(spec.get("min_intensity", 0))
        masks[str(name)] = channel_arrays[channel_name] >= threshold
    return masks


def add_marker_metrics(
    object_table: pd.DataFrame,
    image: np.ndarray,
    labels: np.ndarray,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    if object_table.empty:
        return object_table.copy()

    out = object_table.copy()
    channel_arrays = _channel_arrays(image, config)
    mask_arrays = _build_masks(channel_arrays, config)
    distance_arrays = {
        name: distance_transform_edt(~mask.astype(bool)).astype(np.float32)
        for name, mask in mask_arrays.items()
    }

    out["geometry.area_px"] = out["area_px"].astype(float)

    perimeters: list[float] = []
    circularities: list[float] = []
    eccentricities: list[float] = []

    channel_metric_values: dict[str, list[float]] = {}
    for channel_name in channel_arrays:
        for prefix in ("channel.mean", "channel.max", "channel.integrated", "channel.mean_bgsub"):
            channel_metric_values[f"{prefix}.{channel_name}"] = []

    relation_values: dict[str, list[float]] = {}
    for mask_name in mask_arrays:
        relation_values[f"relation.overlap_fraction.{mask_name}"] = []
        relation_values[f"relation.distance_to_mask_px.{mask_name}"] = []

    for object_id in out["object_id"].astype(int):
        mask = labels == object_id
        ys, xs = np.where(mask)
        peri, circ = _circularity(mask)
        ecc = _eccentricity(ys, xs)
        perimeters.append(peri)
        circularities.append(circ)
        eccentricities.append(ecc)

        cy = int(round(float(ys.mean())))
        cx = int(round(float(xs.mean())))

        for channel_name, channel in channel_arrays.items():
            pixels = channel[ys, xs]
            bg = _local_background(channel, ys, xs)
            channel_metric_values[f"channel.mean.{channel_name}"].append(float(pixels.mean()))
            channel_metric_values[f"channel.max.{channel_name}"].append(float(pixels.max()))
            channel_metric_values[f"channel.integrated.{channel_name}"].append(float(pixels.sum()))
            channel_metric_values[f"channel.mean_bgsub.{channel_name}"].append(float(pixels.mean() - bg))

        for mask_name, mask_array in mask_arrays.items():
            overlap = float(mask_array[ys, xs].mean()) if len(ys) else 0.0
            dist = float(distance_arrays[mask_name][cy, cx])
            relation_values[f"relation.overlap_fraction.{mask_name}"].append(overlap)
            relation_values[f"relation.distance_to_mask_px.{mask_name}"].append(dist)

    out["geometry.perimeter_px"] = perimeters
    out["geometry.circularity"] = circularities
    out["geometry.eccentricity"] = eccentricities

    for column, values in channel_metric_values.items():
        out[column] = values
        base = np.asarray(values, dtype=float)
        std = float(base.std())
        z = np.zeros_like(base) if std <= 1e-12 else (base - float(base.mean())) / std
        stat_name, channel_name = column.rsplit(".", 1)
        if stat_name in {"channel.mean", "channel.max", "channel.integrated", "channel.mean_bgsub"}:
            out[f"{stat_name}_z.{channel_name}"] = z

    for column, values in relation_values.items():
        out[column] = values

    return out
