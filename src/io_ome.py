# src/io_ome.py

from __future__ import annotations
import os
from typing import Tuple, Dict, Any, Optional

import numpy as np

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
    try:
        from aicsimageio import AICSImage
        img = AICSImage(path)
        meta["reader"] = "aicsimageio"
        # Get data in standard order (TCZYX)
        data = img.get_image_data("CZYX")  # channels-first for 2D/3D
        # Move channels last if more convenient
        if data.ndim == 4:
            # C, Z, Y, X -> Z, Y, X, C
            data = np.moveaxis(data, 0, -1)
        elif data.ndim == 3:
            # C, Y, X -> Y, X, C
            data = np.moveaxis(data, 0, -1)

        # If single-channel, return 2D
        if data.ndim == 3 and data.shape[-1] == 1:
            data = data[..., 0]

        # Pull pixel size if present
        try:
            mpp_x = img.get_physical_pixel_size_x()
            mpp_y = img.get_physical_pixel_size_y()
            if mpp_x and mpp_y:
                meta["microns_per_pixel_x"] = float(mpp_x)
                meta["microns_per_pixel_y"] = float(mpp_y)
        except Exception:
            pass

        return data, meta
    except Exception:
        # Fallback to tifffile
        try:
            import tifffile
            arr = tifffile.imread(path)
            meta["reader"] = "tifffile"
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

