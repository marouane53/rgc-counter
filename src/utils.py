# src/utils.py

import os
import glob
import tifffile
import numpy as np
from typing import List, Tuple, Dict, Any, Iterable, Optional

def load_tiff_images(folder_path: str) -> List[Tuple[str, np.ndarray]]:
    """
    Legacy loader for .tif/.tiff. Returns a list of (image_path, image_array).
    """
    images_list = []
    for ext in ("*.tif", "*.tiff"):
        for filepath in glob.glob(os.path.join(folder_path, ext)):
            img = tifffile.imread(filepath)
            images_list.append((filepath, img))
    return images_list

def load_images_any(folder_path: str) -> List[Tuple[str, np.ndarray, Dict[str, Any]]]:
    """
    Universal image loader. Reads TIFF/OME-TIFF/OME-Zarr using aicsimageio when available.
    Returns list of (path, array, metadata).
    """
    from src.io_ome import load_any_image
    images_list = []
    for root, _, files in os.walk(folder_path):
        for fname in files:
            low = fname.lower()
            if low.endswith((".tif", ".tiff", ".ome.tif", ".ome.tiff", ".png", ".jpg", ".jpeg", ".zarr")):
                path = os.path.join(root, fname)
                try:
                    arr, meta = load_any_image(path)
                    images_list.append((path, arr, meta))
                except Exception:
                    # Ignore unreadable files silently to keep batch robust
                    continue
    # If no images found, fall back to tif loader in the root folder
    if not images_list:
        for ext in ("*.tif", "*.tiff"):
            for filepath in glob.glob(os.path.join(folder_path, ext)):
                try:
                    img = tifffile.imread(filepath)
                    images_list.append((filepath, img, {"reader": "tifffile", "path": filepath}))
                except Exception:
                    pass
    return images_list


def save_results_to_csv(results: List[Dict[str, Any]], output_csv_path: str) -> None:
    """
    Save the cell counts/density results to a CSV file.
    """
    import csv
    fieldnames = sorted({k for r in results for k in r.keys()})
    with open(output_csv_path, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def _select_channel_last(image: np.ndarray, channel_index: int) -> np.ndarray:
    n_channels = int(image.shape[-1])
    if channel_index < 0 or channel_index >= n_channels:
        raise IndexError(f"channel_index {channel_index} is out of range for trailing axis with {n_channels} channels")
    return np.asarray(image[..., channel_index])


def _select_channel_first(image: np.ndarray, channel_index: int) -> np.ndarray:
    n_channels = int(image.shape[0])
    if channel_index < 0 or channel_index >= n_channels:
        raise IndexError(f"channel_index {channel_index} is out of range for leading axis with {n_channels} channels")
    return np.asarray(image[channel_index, ...])


def ensure_grayscale(image: np.ndarray, channel_index: int = 0) -> np.ndarray:
    """
    Canonicalize common microscopy layouts into a 2D grayscale array.

    Supported layouts:
    - YX
    - YXC
    - CYX
    - ZYX
    - ZYXC
    - CZYX

    Singleton dimensions are squeezed when they originate from 3D/4D layouts.
    """
    raw = np.asarray(image)
    if raw.ndim == 2:
        gray = raw
    elif raw.ndim == 3:
        squeezed = np.squeeze(raw)
        if squeezed.ndim == 2:
            gray = squeezed
        elif raw.shape[-1] <= 4:
            gray = _select_channel_last(raw, channel_index)
        elif raw.shape[0] <= 4:
            gray = _select_channel_first(raw, channel_index)
        else:
            gray = np.max(raw, axis=0)
    elif raw.ndim == 4:
        squeezed = np.squeeze(raw)
        if squeezed.ndim == 2:
            gray = squeezed
        elif raw.shape[-1] <= 4:
            if raw.shape[0] == 1:
                gray = _select_channel_last(np.squeeze(raw, axis=0), channel_index)
            else:
                gray = _select_channel_last(np.max(raw, axis=0), channel_index)
        elif raw.shape[0] <= 4:
            channel_view = _select_channel_first(raw, channel_index)
            gray = np.max(channel_view, axis=0) if channel_view.ndim == 3 else channel_view
        else:
            raise ValueError(f"Unexpected image dimensions: {raw.shape}")
    else:
        raise ValueError(f"Unexpected image dimensions: {raw.shape}")

    gray = np.asarray(gray)
    if gray.ndim != 2:
        raise ValueError(f"Unexpected grayscale shape {gray.shape} derived from input {raw.shape}")
    return gray

def image_is_multichannel(image: np.ndarray) -> bool:
    return image.ndim >= 3 and (image.shape[-1] < 32 and image.shape[-1] > 1)

def safe_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    f = image.astype(np.float32)
    m = f.max()
    if m <= 0:
        return np.zeros_like(image, dtype=np.uint8)
    return (f / m * 255.0).astype(np.uint8)
