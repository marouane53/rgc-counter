from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import tifffile


def save_float_map(
    array: np.ndarray,
    path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        str(path),
        np.asarray(array, dtype=np.float32),
        metadata=metadata or {"axes": "YX"},
    )
    return path


def read_float_map(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    with tifffile.TiffFile(str(path)) as handle:
        array = handle.asarray().astype(np.float32, copy=False)
        metadata: dict[str, Any] = {}
        if handle.imagej_metadata:
            metadata.update(dict(handle.imagej_metadata))
        description = getattr(handle.pages[0], "description", None)
        if description:
            metadata["ImageDescription"] = description
    return array, metadata
