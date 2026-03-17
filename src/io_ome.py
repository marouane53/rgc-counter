# src/io_ome.py

from __future__ import annotations
import os
from typing import Tuple, Dict, Any, Optional

import numpy as np


def _shape_list(shape: tuple[int, ...] | list[int]) -> list[int]:
    return [int(dim) for dim in shape]


def _normalize_loaded_array(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    arr = np.squeeze(arr)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return arr

def is_ome_tiff(path: str) -> bool:
    lower = path.lower()
    return lower.endswith(".ome.tif") or lower.endswith(".ome.tiff")

def is_zarr_like(path: str) -> bool:
    return path.lower().endswith(".zarr") or os.path.isdir(path) and path.lower().endswith(".zarr")

def load_any_image(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load OME-TIFF, OME-Zarr, or standard TIFF. Returns (array, metadata).
    If multi-dimensional, it tries to squeeze singleton dims and leave (Y, X, C) or (Z, Y, X, C).
    """
    meta: Dict[str, Any] = {"path": path}
    lower = path.lower()
    if lower.endswith((".jpg", ".jpeg", ".png")):
        from PIL import Image

        with Image.open(path) as img:
            arr = np.asarray(img)
        meta["reader"] = "pillow"
        meta["reader_raw_shape"] = _shape_list(arr.shape)
        meta["canonical_loaded_shape"] = _shape_list(_normalize_loaded_array(arr).shape)
        return arr, meta

    try:
        from aicsimageio import AICSImage
        img = AICSImage(path)
        meta["reader"] = "aicsimageio"
        # Get data in standard order (TCZYX)
        data = img.get_image_data("CZYX")  # channels-first for 2D/3D
        meta["aics_requested_dims"] = "CZYX"
        meta["aics_raw_shape"] = _shape_list(np.asarray(data).shape)
        # Move channels last if more convenient
        if data.ndim == 4:
            # C, Z, Y, X -> Z, Y, X, C
            data = np.moveaxis(data, 0, -1)
        elif data.ndim == 3:
            # C, Y, X -> Y, X, C
            data = np.moveaxis(data, 0, -1)

        data = _normalize_loaded_array(data)
        meta["canonical_loaded_shape"] = _shape_list(data.shape)

        # Pull pixel size if present
        try:
            pixel_sizes = getattr(img, "physical_pixel_sizes", None)
            if pixel_sizes is not None:
                if getattr(pixel_sizes, "X", None) is not None:
                    meta["microns_per_pixel_x"] = float(pixel_sizes.X)
                if getattr(pixel_sizes, "Y", None) is not None:
                    meta["microns_per_pixel_y"] = float(pixel_sizes.Y)
                if getattr(pixel_sizes, "Z", None) is not None:
                    meta["microns_per_pixel_z"] = float(pixel_sizes.Z)
        except Exception:
            pass

        return data, meta
    except Exception:
        # Fallback to tifffile for standard microscopy formats first
        try:
            import tifffile

            raw = tifffile.imread(path)
            arr = _normalize_loaded_array(raw)
            meta["reader"] = "tifffile"
            meta["reader_raw_shape"] = _shape_list(np.asarray(raw).shape)
            meta["canonical_loaded_shape"] = _shape_list(arr.shape)
            return arr, meta
        except Exception:
            # Final fallback for common non-TIFF images used in smoke tests and reports
            try:
                from PIL import Image

                with Image.open(path) as img:
                    raw = np.asarray(img)
                    arr = _normalize_loaded_array(raw)
                meta["reader"] = "pillow"
                meta["reader_raw_shape"] = _shape_list(raw.shape)
                meta["canonical_loaded_shape"] = _shape_list(arr.shape)
                return arr, meta
            except Exception as e:
                raise RuntimeError(f"Could not read image: {path}. Error: {e}") from e


def save_labels_to_ome_zarr(image: np.ndarray,
                            labels: np.ndarray,
                            out_dir: str,
                            metadata: Optional[Dict[str, Any]] = None,
                            chunk: int = 256) -> str:
    """
    Save an image and label mask to an OME-Zarr store.
    This is a minimal writer. For full NGFF metadata, use ome_zarr.writer.
    """
    try:
        import zarr
        from ome_zarr.io import parse_url
        from ome_zarr.writer import write_image
        store = parse_url(out_dir, mode="w").store
        root = zarr.group(store=store)

        # Ensure 3D array (C, Y, X) for image
        if image.ndim == 2:
            img = image[np.newaxis, ...]  # 1, Y, X
        elif image.ndim == 3 and image.shape[-1] in (1, 3, 4):
            # Y, X, C -> C, Y, X
            img = np.moveaxis(image, -1, 0)
        else:
            # attempt best effort
            img = image

        chunks = (1, chunk, chunk) if img.ndim == 3 else None
        write_image(image=img, group=root, axes="CYX", storage_options={"chunks": chunks})

        # Labels
        labels_grp = root.create_group("labels")
        lg = labels_grp.create_group("masks")
        # Store as (1, Y, X)
        if labels.ndim == 2:
            lab = labels[np.newaxis, ...]
        else:
            lab = labels
        lg.array("0", data=lab, chunks=(1, chunk, chunk), dtype=labels.dtype)
        if metadata:
            for k, v in metadata.items():
                lg.attrs[k] = v

        return out_dir
    except Exception as e:
        raise RuntimeError(
            f"Failed to write OME-Zarr to {out_dir}. "
            f"Install 'ome-zarr' and 'zarr' packages. Error: {e}"
        ) from e
