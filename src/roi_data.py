from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

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

OPTIONAL_PROVENANCE_COLUMNS = [
    "image_marker",
    "image_source_channel",
    "truth_marker",
    "truth_source_channel",
    "truth_derivation",
    "truth_source_path",
]

TRUTH_PROVENANCE_STATUS_MATCHED = "matched"
TRUTH_PROVENANCE_STATUS_UNKNOWN = "unknown"
TRUTH_PROVENANCE_STATUS_INVALID = "invalid"


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
    image_marker: str | None = None
    image_source_channel: int | None = None
    truth_marker: str | None = None
    truth_source_channel: int | None = None
    truth_derivation: str | None = None
    truth_source_path: Path | None = None
    truth_provenance_status: str = TRUTH_PROVENANCE_STATUS_UNKNOWN
    truth_provenance_valid: bool = True
    truth_provenance_issues: tuple[str, ...] = ()


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


def _clean_optional_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _clean_optional_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _candidate_sidecar_paths(path: Path, *, meta: bool = False) -> list[Path]:
    if meta:
        candidates = [path.with_suffix(".meta.json"), Path(str(path) + ".meta.json")]
    else:
        candidates = [path.with_suffix(".json"), Path(str(path) + ".json")]
    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _load_existing_json(paths: list[Path]) -> tuple[Path | None, dict[str, Any]]:
    for candidate in paths:
        if not candidate.exists():
            continue
        try:
            return candidate, json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            return candidate, {}
    return None, {}


def _coerce_optional_path(value: Any, *, base: Path) -> Path | None:
    text = _clean_optional_text(value)
    if text is None:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def _normalize_compare_value(field: str, value: Any) -> Any:
    if value is None:
        return None
    if field.endswith("_channel"):
        return int(value)
    if field.endswith("_path"):
        return str(Path(str(value)).resolve())
    return str(value).strip().lower()


def _truth_derivation_from_payload(payload: Mapping[str, Any]) -> str | None:
    explicit = _clean_optional_text(payload.get("truth_derivation"))
    if explicit is not None:
        return explicit
    tool = _clean_optional_text(payload.get("tool"))
    if tool == "scripts/annotate_roi_points.py":
        return "manual_point_truth"
    derivation = _clean_optional_text(payload.get("derivation"))
    if derivation is not None:
        return derivation
    if _clean_optional_text(payload.get("scene_dataset_path")) is not None:
        return "embedded_imaris_scene_spots"
    return None


def _truth_meta_fields(payload: Mapping[str, Any], *, base: Path) -> dict[str, Any]:
    return {
        "truth_marker": _clean_optional_text(payload.get("truth_marker") or payload.get("marker")),
        "truth_source_channel": _clean_optional_int(
            payload.get("truth_source_channel")
            if payload.get("truth_source_channel") is not None
            else payload.get("source_channel_index")
        ),
        "truth_derivation": _truth_derivation_from_payload(payload),
        "truth_source_path": _coerce_optional_path(
            payload.get("truth_source_path")
            or payload.get("source_path")
            or payload.get("ims_path")
            or payload.get("image_path"),
            base=base,
        ),
    }


def _image_sidecar_fields(payload: Mapping[str, Any], *, base: Path) -> dict[str, Any]:
    return {
        "image_marker": _clean_optional_text(payload.get("image_marker") or payload.get("marker")),
        "image_source_channel": _clean_optional_int(
            payload.get("image_source_channel")
            if payload.get("image_source_channel") is not None
            else payload.get("channel_index")
        ),
        "truth_source_path": _coerce_optional_path(payload.get("source_ims_path"), base=base),
    }


def resolve_roi_truth_provenance(
    row: Mapping[str, Any],
    *,
    image_path: Path,
    manual_points_path: Path | None,
    manifest_base: Path,
) -> dict[str, Any]:
    image_sidecar_path, image_sidecar_payload = _load_existing_json(_candidate_sidecar_paths(image_path))
    truth_sidecar_path, truth_sidecar_payload = (
        _load_existing_json(_candidate_sidecar_paths(manual_points_path, meta=True)) if manual_points_path is not None else (None, {})
    )

    manifest_values = {
        "image_marker": _clean_optional_text(row.get("image_marker")) or _clean_optional_text(row.get("marker")),
        "image_source_channel": _clean_optional_int(row.get("image_source_channel")),
        "truth_marker": _clean_optional_text(row.get("truth_marker")),
        "truth_source_channel": _clean_optional_int(row.get("truth_source_channel")),
        "truth_derivation": _clean_optional_text(row.get("truth_derivation")),
        "truth_source_path": _coerce_optional_path(row.get("truth_source_path"), base=manifest_base),
    }
    truth_meta_values = _truth_meta_fields(truth_sidecar_payload, base=manifest_base)
    image_sidecar_values = _image_sidecar_fields(image_sidecar_payload, base=manifest_base)

    field_sources: dict[str, list[tuple[str, Any]]] = {
        "image_marker": [
            ("manifest", manifest_values["image_marker"]),
            ("image_sidecar", image_sidecar_values["image_marker"]),
        ],
        "image_source_channel": [
            ("manifest", manifest_values["image_source_channel"]),
            ("image_sidecar", image_sidecar_values["image_source_channel"]),
        ],
        "truth_marker": [
            ("manifest", manifest_values["truth_marker"]),
            ("truth_meta", truth_meta_values["truth_marker"]),
        ],
        "truth_source_channel": [
            ("manifest", manifest_values["truth_source_channel"]),
            ("truth_meta", truth_meta_values["truth_source_channel"]),
        ],
        "truth_derivation": [
            ("manifest", manifest_values["truth_derivation"]),
            ("truth_meta", truth_meta_values["truth_derivation"]),
        ],
        "truth_source_path": [
            ("manifest", manifest_values["truth_source_path"]),
            ("truth_meta", truth_meta_values["truth_source_path"]),
            ("image_sidecar", image_sidecar_values["truth_source_path"]),
        ],
    }

    issues: list[str] = []
    resolved: dict[str, Any] = {}
    for field, candidates in field_sources.items():
        populated = [(source, value) for source, value in candidates if value is not None]
        if populated:
            reference = _normalize_compare_value(field, populated[0][1])
            for source, value in populated[1:]:
                if _normalize_compare_value(field, value) != reference:
                    issues.append(
                        f"{field} conflict between {populated[0][0]}={populated[0][1]} and {source}={value}"
                    )
        resolved[field] = populated[0][1] if populated else None

    image_marker = _clean_optional_text(resolved.get("image_marker")) or _clean_optional_text(row.get("marker"))
    truth_marker = _clean_optional_text(resolved.get("truth_marker"))
    image_source_channel = _clean_optional_int(resolved.get("image_source_channel"))
    truth_source_channel = _clean_optional_int(resolved.get("truth_source_channel"))

    roi_marker = _clean_optional_text(row.get("marker"))
    if truth_source_channel is not None and image_source_channel is not None and truth_source_channel != image_source_channel:
        issues.append(
            f"truth_source_channel {truth_source_channel} does not match image_source_channel {image_source_channel}"
        )
    if truth_marker is not None and roi_marker is not None and truth_marker.lower() != roi_marker.lower():
        issues.append(f"truth_marker {truth_marker} does not match roi marker {roi_marker}")
    if truth_marker is not None and image_marker is not None and truth_marker.lower() != image_marker.lower():
        issues.append(f"truth_marker {truth_marker} does not match image_marker {image_marker}")

    valid = not issues
    if not valid:
        status = TRUTH_PROVENANCE_STATUS_INVALID
    elif (
        image_marker is not None
        and truth_marker is not None
        and image_source_channel is not None
        and truth_source_channel is not None
    ):
        status = TRUTH_PROVENANCE_STATUS_MATCHED
    else:
        status = TRUTH_PROVENANCE_STATUS_UNKNOWN

    truth_source_path = resolved.get("truth_source_path")
    if isinstance(truth_source_path, Path):
        resolved["truth_source_path"] = truth_source_path
    elif truth_source_path is not None:
        resolved["truth_source_path"] = _coerce_optional_path(truth_source_path, base=manifest_base)

    return {
        "image_marker": image_marker,
        "image_source_channel": image_source_channel,
        "truth_marker": truth_marker,
        "truth_source_channel": truth_source_channel,
        "truth_derivation": _clean_optional_text(resolved.get("truth_derivation")),
        "truth_source_path": resolved.get("truth_source_path"),
        "truth_provenance_status": status,
        "truth_provenance_valid": valid,
        "truth_provenance_issues": tuple(issues),
        "truth_meta_path": truth_sidecar_path,
        "image_sidecar_path": image_sidecar_path,
    }


def load_roi_manifest(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in REQUIRED_ROI_COLUMNS + OPTIONAL_PROVENANCE_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return validate_roi_benchmark_manifest(frame)


def normalize_split_filters(values: Iterable[str] | str | None) -> set[str] | None:
    if values is None:
        return None
    raw_values = [values] if isinstance(values, str) else list(values)
    normalized: set[str] = set()
    for raw in raw_values:
        for token in str(raw).split(","):
            candidate = token.strip()
            if candidate:
                normalized.add(candidate)
    return normalized or None


def filter_roi_manifest_by_split(
    frame: pd.DataFrame,
    *,
    include_splits: Iterable[str] | str | None = None,
    exclude_splits: Iterable[str] | str | None = None,
) -> pd.DataFrame:
    filtered = validate_roi_benchmark_manifest(frame).copy()
    include = normalize_split_filters(include_splits)
    exclude = normalize_split_filters(exclude_splits)
    filtered["split"] = filtered["split"].fillna("benchmark").astype(str).str.strip().replace("", "benchmark")
    if include is not None:
        filtered = filtered.loc[filtered["split"].isin(include)].copy()
    if exclude is not None:
        filtered = filtered.loc[~filtered["split"].isin(exclude)].copy()
    if filtered.empty:
        raise ValueError("ROI benchmark manifest has no rows after split filtering.")
    return filtered.reset_index(drop=True)


def iter_roi_records(
    frame: pd.DataFrame,
    *,
    manifest_path: str | Path | None = None,
    include_splits: Iterable[str] | str | None = None,
    exclude_splits: Iterable[str] | str | None = None,
) -> list[RoiRecord]:
    base = Path(manifest_path).resolve().parent if manifest_path is not None else Path.cwd()
    rows: list[RoiRecord] = []
    seen_ids: set[str] = set()
    filtered = filter_roi_manifest_by_split(frame, include_splits=include_splits, exclude_splits=exclude_splits)
    for row in filtered.to_dict("records"):
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

        provenance = resolve_roi_truth_provenance(
            row,
            image_path=image_path,
            manual_points_path=manual_points_path,
            manifest_base=base,
        )

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
                image_marker=provenance["image_marker"],
                image_source_channel=provenance["image_source_channel"],
                truth_marker=provenance["truth_marker"],
                truth_source_channel=provenance["truth_source_channel"],
                truth_derivation=provenance["truth_derivation"],
                truth_source_path=provenance["truth_source_path"],
                truth_provenance_status=provenance["truth_provenance_status"],
                truth_provenance_valid=bool(provenance["truth_provenance_valid"]),
                truth_provenance_issues=tuple(provenance["truth_provenance_issues"]),
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
                "truth_provenance_status": record.truth_provenance_status,
                "truth_provenance_valid": bool(record.truth_provenance_valid),
                "truth_provenance_issues": "; ".join(record.truth_provenance_issues),
                "image_source_channel": record.image_source_channel,
                "truth_source_channel": record.truth_source_channel,
                "image_marker": record.image_marker,
                "truth_marker": record.truth_marker,
                "error": error,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["reused_source_image"] = out["image_path"].duplicated(keep=False)
    out["duplicate_image"] = out["image_sha256"].duplicated(keep=False) & out["image_sha256"].notna()
    out["duplicate_crop"] = out["crop_sha256"].duplicated(keep=False) & out["crop_sha256"].notna()
    out["marker_consistent"] = len({record.marker for record in records if record.marker}) <= 1
    out["modality_consistent"] = len({record.modality for record in records if record.modality}) <= 1
    overlap_flags = [False] * len(records)
    overlap_details = [""] * len(records)
    for left_index, left in enumerate(records):
        left_x1 = left.x0 + left.width
        left_y1 = left.y0 + left.height
        for right_index in range(left_index + 1, len(records)):
            right = records[right_index]
            if left.image_path != right.image_path:
                continue
            right_x1 = right.x0 + right.width
            right_y1 = right.y0 + right.height
            overlap_width = min(left_x1, right_x1) - max(left.x0, right.x0)
            overlap_height = min(left_y1, right_y1) - max(left.y0, right.y0)
            if overlap_width <= 0 or overlap_height <= 0:
                continue
            overlap_area = int(overlap_width * overlap_height)
            overlap_flags[left_index] = True
            overlap_flags[right_index] = True
            overlap_details[left_index] = ", ".join(filter(None, [overlap_details[left_index], f"{right.roi_id}:{overlap_area}px"]))
            overlap_details[right_index] = ", ".join(filter(None, [overlap_details[right_index], f"{left.roi_id}:{overlap_area}px"]))
    out["overlaps_with_other_roi"] = overlap_flags
    out["overlap_details"] = overlap_details
    return out
