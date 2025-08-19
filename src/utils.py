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


def ensure_grayscale(image: np.ndarray, channel_index: int = 0) -> np.ndarray:
    """
    If the image has multiple channels, extract the one we need.
    If the image is already single-channel, just return it.
    """
    if image.ndim == 2:
        return image
    elif image.ndim == 3:
        # Assume last dim is channels
        if image.shape[-1] < 10:
            return image[..., channel_index]
        else:
            # Channels-first or Z dimension present; pick best guess
            return image[channel_index, ...]
    elif image.ndim == 4:
        # Z, Y, X, C or Y, X, Z, C. We try to take a max projection on Z then channel.
        # We assume last dimension is channels when small (<10)
        if image.shape[-1] < 10:
            # max project Z
            proj = image.max(axis=-2) if image.shape[-2] > 1 else image[..., 0, :]
            return proj[..., channel_index]
        else:
            # fallback: take first slice/channel
            return image[0, ..., 0]
    else:
        raise ValueError(f"Unexpected image dimensions: {image.shape}")

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
