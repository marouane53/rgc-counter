from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.io_ome import load_any_image
from src.validation import validate_roi_benchmark_manifest


REQUIRED_ROI_COLUMNS = [
    "roi_id",
    "image_path",
    "marker",
    "modality",
    "x0",
    "y0",
    "width",
    "height",
    "annotator",
    "manual_points_path",
    "split",
    "notes",
]


@dataclass(frozen=True)
class RoiRecord:
    roi_id: str
    image_path: Path
    marker: str
    modality: str
    x0: int
    y0: int
    width: int
    height: int
    annotator: str
    manual_points_path: Path | None
    split: str = "benchmark"
    notes: str = ""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def crop_2d_or_yxc(image: np.ndarray, *, x0: int, y0: int, width: int, height: int) -> np.ndarray:
    x1 = int(x0) + int(width)
    y1 = int(y0) + int(height)
    if image.ndim == 2:
        return image[y0:y1, x0:x1]
    if image.ndim == 3 and image.shape[-1] <= 8:
        return image[y0:y1, x0:x1, :]
    raise ValueError(
        "ROI benchmark currently supports 2D or channels-last 2D images only. "
        f"Got shape={tuple(int(v) for v in image.shape)}."
    )


def crop_sha256(image: np.ndarray, *, x0: int, y0: int, width: int, height: int) -> str:
    crop = crop_2d_or_yxc(image, x0=x0, y0=y0, width=width, height=height)
    return _sha256_bytes(np.ascontiguousarray(crop).tobytes())


def load_roi_manifest(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in REQUIRED_ROI_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame = frame[REQUIRED_ROI_COLUMNS].copy()
    return validate_roi_benchmark_manifest(frame)


def iter_roi_records(frame: pd.DataFrame, *, manifest_path: str | Path | None = None) -> list[RoiRecord]:
    base = Path(manifest_path).resolve().parent if manifest_path is not None else Path.cwd()
    rows: list[RoiRecord] = []
    seen_ids: set[str] = set()
    for row in validate_roi_benchmark_manifest(frame).to_dict("records"):
        roi_id = str(row["roi_id"]).strip()
        if not roi_id:
            raise ValueError("ROI benchmark manifest contains blank roi_id.")
        if roi_id in seen_ids:
            raise ValueError(f"ROI benchmark manifest contains duplicate roi_id: {roi_id}")
        seen_ids.add(roi_id)

        image_path = Path(str(row["image_path"]))
        if not image_path.is_absolute():
            image_path = (base / image_path).resolve()

        manual_points_path: Path | None = None
        raw_manual_points = row.get("manual_points_path")
        if pd.notna(raw_manual_points) and str(raw_manual_points).strip():
            manual_points_path = Path(str(raw_manual_points))
            if not manual_points_path.is_absolute():
                manual_points_path = (base / manual_points_path).resolve()

        x0 = int(row["x0"])
        y0 = int(row["y0"])
        width = int(row["width"])
        height = int(row["height"])
        if width <= 0 or height <= 0:
            raise ValueError(f"ROI {roi_id} has non-positive width/height.")

        rows.append(
            RoiRecord(
                roi_id=roi_id,
                image_path=image_path,
                marker=str(row["marker"]).strip(),
                modality=str(row["modality"]).strip(),
                x0=x0,
                y0=y0,
                width=width,
                height=height,
                annotator=str(row.get("annotator") or "").strip(),
                manual_points_path=manual_points_path,
                split=str(row.get("split") or "benchmark").strip() or "benchmark",
                notes=str(row.get("notes") or "").strip(),
            )
        )
    return rows


def qc_roi_manifest(frame: pd.DataFrame, *, manifest_path: str | Path | None = None) -> pd.DataFrame:
    validated = validate_roi_benchmark_manifest(frame)
    records = iter_roi_records(validated, manifest_path=manifest_path)

    rows: list[dict[str, Any]] = []
    for record in records:
        image_exists = record.image_path.exists()
        image_sha = None
        crop_sha = None
        shape = None
        bounds_ok = False
        crop_nonempty = False
        error = None

        try:
            if image_exists:
                image_sha = file_sha256(record.image_path)
                image, _ = load_any_image(str(record.image_path))
                shape = tuple(int(v) for v in image.shape)
                height_px = int(image.shape[0])
                width_px = int(image.shape[1])
                bounds_ok = (
                    record.x0 >= 0
                    and record.y0 >= 0
                    and record.x0 + record.width <= width_px
                    and record.y0 + record.height <= height_px
                )
                if bounds_ok:
                    crop = crop_2d_or_yxc(image, x0=record.x0, y0=record.y0, width=record.width, height=record.height)
                    crop_nonempty = crop.size > 0
                    crop_sha = _sha256_bytes(np.ascontiguousarray(crop).tobytes())
        except Exception as exc:
            error = str(exc)

        rows.append(
            {
                "roi_id": record.roi_id,
                "image_path": str(record.image_path),
                "marker": record.marker,
                "modality": record.modality,
                "split": record.split,
                "image_exists": bool(image_exists),
                "manual_points_exists": bool(record.manual_points_path and record.manual_points_path.exists()),
                "image_sha256": image_sha,
                "crop_sha256": crop_sha,
                "image_shape": shape,
                "bounds_ok": bool(bounds_ok),
                "crop_nonempty": bool(crop_nonempty),
                "error": error,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["duplicate_image"] = out["image_sha256"].duplicated(keep=False) & out["image_sha256"].notna()
    out["duplicate_crop"] = out["crop_sha256"].duplicated(keep=False) & out["crop_sha256"].notna()
    out["marker_consistent"] = len({record.marker for record in records if record.marker}) <= 1
    out["modality_consistent"] = len({record.modality for record in records if record.modality}) <= 1
    return out
