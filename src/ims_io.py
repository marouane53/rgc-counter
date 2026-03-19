"""Private benchmark preprocessing bridge for Imaris `.ims` files.

This module is intentionally narrow in scope. It exists to support local-only
private benchmark preprocessing and metadata extraction. It does not make
`.ims` a supported general input format for the default `retinal-phenotyper`
CLI or public workflows.
"""

from __future__ import annotations

import hashlib
import io
import math
import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np


CHANNEL_GROUP_RE = re.compile(r"^Channel\s+(\d+)$")
ANNOTATION_HINTS = (
    "spot",
    "point",
    "vertex",
    "position",
    "coord",
    "center",
    "radius",
    "annotation",
)
FAR_RED_HINTS = (
    "alexa 647",
    "af647",
    "cy5",
    "far red",
    "far-red",
    "647",
)
EXPLICIT_RBPMS_HINTS = ("rbpms",)
TEXTUAL_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _decode_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _decode_scalar(value[()])
        if value.dtype.kind in {"S", "U"}:
            flat = value.reshape(-1).tolist()
            return "".join(str(_decode_scalar(item)) for item in flat)
        return [_decode_scalar(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_decode_scalar(item) for item in value]
    return value


def _stringify(value: Any) -> str:
    decoded = _decode_scalar(value)
    if decoded is None:
        return ""
    if isinstance(decoded, list):
        return ", ".join(str(item) for item in decoded)
    return str(decoded)


def _maybe_float(value: Any) -> float | None:
    text = _stringify(value).strip()
    if not text:
        return None
    match = TEXTUAL_NUM_RE.search(text)
    if match is None:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _maybe_int(value: Any) -> int | None:
    parsed = _maybe_float(value)
    if parsed is None or not math.isfinite(parsed):
        return None
    return int(parsed)


def _normalized_unit(unit: str | None) -> str | None:
    if unit is None:
        return None
    normalized = str(unit).strip().lower().replace("µ", "u")
    return normalized or None


def _unit_scale_to_um(unit: str | None) -> float | None:
    normalized = _normalized_unit(unit)
    if normalized is None:
        return None
    mapping = {
        "m": 1_000_000.0,
        "meter": 1_000_000.0,
        "metre": 1_000_000.0,
        "mm": 1_000.0,
        "millimeter": 1_000.0,
        "millimetre": 1_000.0,
        "um": 1.0,
        "micrometer": 1.0,
        "micrometre": 1.0,
        "nm": 0.001,
        "nanometer": 0.001,
        "nanometre": 0.001,
    }
    return mapping.get(normalized)


def read_hdf5_attrs(group_or_dataset: h5py.Group | h5py.Dataset) -> dict[str, str]:
    return {str(key): _stringify(value) for key, value in group_or_dataset.attrs.items()}


def compute_voxel_sizes_um(image_info: dict[str, str]) -> dict[str, float | None]:
    unit_scale = _unit_scale_to_um(image_info.get("Unit"))
    sizes: dict[str, float | None] = {"voxel_x_um": None, "voxel_y_um": None, "voxel_z_um": None}
    if unit_scale is None:
        return sizes

    axis_map = {
        "x": ("ExtMin0", "ExtMax0", "X"),
        "y": ("ExtMin1", "ExtMax1", "Y"),
        "z": ("ExtMin2", "ExtMax2", "Z"),
    }
    for axis, (minimum_key, maximum_key, count_key) in axis_map.items():
        minimum = _maybe_float(image_info.get(minimum_key))
        maximum = _maybe_float(image_info.get(maximum_key))
        count = _maybe_int(image_info.get(count_key))
        if minimum is None or maximum is None or count is None or count <= 0:
            continue
        sizes[f"voxel_{axis}_um"] = abs(maximum - minimum) * unit_scale / float(count)
    return sizes


def _image_info_group(h5: h5py.File) -> h5py.Group | None:
    dataset_info = h5.get("DataSetInfo")
    if not isinstance(dataset_info, h5py.Group):
        return None
    image_group = dataset_info.get("Image")
    return image_group if isinstance(image_group, h5py.Group) else None


def _channel_info_groups(h5: h5py.File) -> list[tuple[int, h5py.Group]]:
    dataset_info = h5.get("DataSetInfo")
    if not isinstance(dataset_info, h5py.Group):
        return []
    rows: list[tuple[int, h5py.Group]] = []
    for key in dataset_info.keys():
        match = CHANNEL_GROUP_RE.match(str(key))
        value = dataset_info.get(key)
        if match is None or not isinstance(value, h5py.Group):
            continue
        rows.append((int(match.group(1)), value))
    return sorted(rows, key=lambda item: item[0])


def _channel_dataset_group(
    h5: h5py.File,
    channel_index: int,
    *,
    resolution_level: int = 0,
    timepoint: int = 0,
) -> h5py.Group | None:
    path = f"DataSet/ResolutionLevel {int(resolution_level)}/TimePoint {int(timepoint)}/Channel {int(channel_index)}"
    value = h5.get(path)
    return value if isinstance(value, h5py.Group) else None


def _channel_dataset(
    h5: h5py.File,
    channel_index: int,
    *,
    resolution_level: int = 0,
    timepoint: int = 0,
) -> h5py.Dataset | None:
    dataset_group = _channel_dataset_group(
        h5,
        channel_index=channel_index,
        resolution_level=resolution_level,
        timepoint=timepoint,
    )
    if dataset_group is None:
        return None
    data = dataset_group.get("Data")
    return data if isinstance(data, h5py.Dataset) else None


def _dataset_details(dataset_group: h5py.Group | None) -> dict[str, Any]:
    if dataset_group is None:
        return {}
    data = dataset_group.get("Data")
    histogram = dataset_group.get("Histogram")
    details: dict[str, Any] = {"dataset_group_keys": sorted(str(key) for key in dataset_group.keys())}
    if isinstance(data, h5py.Dataset):
        details.update(
            {
                "data_dataset_path": data.name,
                "data_shape": [int(value) for value in data.shape],
                "dtype": str(data.dtype),
                "chunks": [int(value) for value in data.chunks] if data.chunks is not None else None,
                "compression": data.compression,
                "data_attrs": read_hdf5_attrs(data),
            }
        )
    if isinstance(histogram, h5py.Dataset):
        details["has_histogram"] = True
        details["histogram_shape"] = [int(value) for value in histogram.shape]
    else:
        details["has_histogram"] = False
    return details


def extract_standard_ims_metadata(h5: h5py.File) -> dict[str, Any]:
    image_group = _image_info_group(h5)
    image_info = read_hdf5_attrs(image_group) if image_group is not None else {}
    metadata = {
        "top_level_keys": sorted(str(key) for key in h5.keys()),
        "dataset_info_keys": sorted(str(key) for key in h5["DataSetInfo"].keys()) if "DataSetInfo" in h5 else [],
        "image": image_info,
        "voxel_sizes_um": compute_voxel_sizes_um(image_info),
    }
    metadata["n_channels"] = _maybe_int(image_info.get("Noc")) or len(_channel_info_groups(h5))
    return metadata


def enumerate_channel_metadata(h5: h5py.File) -> list[dict[str, Any]]:
    channels: list[dict[str, Any]] = []
    for channel_index, channel_group in _channel_info_groups(h5):
        attrs = read_hdf5_attrs(channel_group)
        dataset_group = _channel_dataset_group(h5, channel_index)
        details = _dataset_details(dataset_group)
        combined_text = " ".join(
            filter(
                None,
                [
                    attrs.get("Name", ""),
                    attrs.get("Description", ""),
                    attrs.get("Color", ""),
                    attrs.get("LSMExcitationWavelength", ""),
                    attrs.get("LSMEmissionWavelength", ""),
                ],
            )
        ).lower()
        channels.append(
            {
                "channel_index": channel_index,
                "group_name": f"Channel {channel_index}",
                "name": attrs.get("Name", ""),
                "description": attrs.get("Description", ""),
                "color": attrs.get("Color", ""),
                "lsm_excitation_wavelength": attrs.get("LSMExcitationWavelength", ""),
                "lsm_emission_wavelength": attrs.get("LSMEmissionWavelength", ""),
                "color_range": attrs.get("ColorRange", ""),
                "min": attrs.get("Min", ""),
                "max": attrs.get("Max", ""),
                "attrs": attrs,
                "metadata_text": combined_text,
                **details,
            }
        )
    return channels


def _score_far_red_channel(channel: dict[str, Any]) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []
    metadata_text = str(channel.get("metadata_text", "")).lower()
    if any(hint in metadata_text for hint in EXPLICIT_RBPMS_HINTS):
        score += 100
        reasons.append("explicit_rbpms_metadata")
    far_red_matches = [hint for hint in FAR_RED_HINTS if hint in metadata_text]
    if far_red_matches:
        score += 35
        reasons.append("far_red_text:" + ",".join(sorted(set(far_red_matches))))

    excitation = _maybe_float(channel.get("lsm_excitation_wavelength"))
    emission = _maybe_float(channel.get("lsm_emission_wavelength"))
    if excitation is not None and 620.0 <= excitation <= 680.0:
        score += 15
        reasons.append("excitation_620_680nm")
    if emission is not None and 650.0 <= emission <= 720.0:
        score += 20
        reasons.append("emission_650_720nm")
    return score, reasons


def find_candidate_rbpms_channels(channels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for channel in channels:
        score, reasons = _score_far_red_channel(channel)
        ranked.append(
            {
                "channel_index": int(channel["channel_index"]),
                "name": str(channel.get("name", "")),
                "description": str(channel.get("description", "")),
                "score": int(score),
                "reason": ";".join(reasons) if reasons else "no_rbpms_signal_in_metadata",
                "explicit_rbpms": any(hint in str(channel.get("metadata_text", "")).lower() for hint in EXPLICIT_RBPMS_HINTS),
                "far_red_candidate": any("far_red" in reason or "nm" in reason for reason in reasons),
            }
        )
    ranked.sort(key=lambda row: (-int(row["score"]), int(row["channel_index"])))
    top_score = int(ranked[0]["score"]) if ranked else 0
    tied_top = sum(1 for row in ranked if int(row["score"]) == top_score and top_score > 0)
    for row in ranked:
        row["is_unambiguous_top_candidate"] = bool(top_score > 0 and int(row["score"]) == top_score and tied_top == 1)
    return ranked


def _annotation_like_payload(path: str, attrs: dict[str, str]) -> bool:
    haystack = f"{path} {' '.join(attrs.values())}".lower()
    return any(hint in haystack for hint in ANNOTATION_HINTS)


def find_scene_objects(h5: h5py.File) -> dict[str, Any]:
    scene_summary: dict[str, Any] = {
        "scene_groups": [],
        "objects": [],
        "annotation_candidates": [],
        "has_annotation_like_objects": False,
    }

    def visitor(name: str, node: h5py.Group | h5py.Dataset) -> None:
        attrs = read_hdf5_attrs(node)
        record: dict[str, Any] = {
            "path": name,
            "kind": "dataset" if isinstance(node, h5py.Dataset) else "group",
            "attrs": attrs,
        }
        if isinstance(node, h5py.Dataset):
            record["shape"] = [int(value) for value in node.shape]
            record["dtype"] = str(node.dtype)
        scene_summary["objects"].append(record)
        if _annotation_like_payload(name, attrs):
            scene_summary["annotation_candidates"].append(record)

    for key in ("Scene", "Scene8"):
        node = h5.get(key)
        if not isinstance(node, h5py.Group):
            continue
        scene_summary["scene_groups"].append(key)
        node.visititems(visitor)

    scene_summary["has_annotation_like_objects"] = bool(scene_summary["annotation_candidates"])
    return scene_summary


def _decode_thumbnail_from_bytes(payload: bytes) -> np.ndarray | None:
    try:
        from PIL import Image
    except Exception:
        return None
    try:
        with Image.open(io.BytesIO(payload)) as image:
            return np.asarray(image)
    except Exception:
        return None


def _reshape_thumbnail_array(array: np.ndarray, attrs: dict[str, str]) -> np.ndarray | None:
    data = np.asarray(array)
    if data.ndim == 2:
        return data
    if data.ndim == 3:
        return data
    flat = data.reshape(-1)
    if flat.dtype != np.uint8:
        return None
    width = _maybe_int(attrs.get("Width"))
    height = _maybe_int(attrs.get("Height"))
    if width and height:
        for channels in (4, 3, 1):
            expected = width * height * channels
            if flat.size == expected:
                shaped = flat.reshape(height, width, channels)
                return shaped[..., 0] if channels == 1 else shaped
    decoded = _decode_thumbnail_from_bytes(bytes(flat.tolist()))
    if decoded is not None:
        return decoded
    return None


def extract_ims_thumbnail(h5: h5py.File) -> np.ndarray | None:
    thumbnail_node = h5.get("Thumbnail")
    if thumbnail_node is None:
        return None
    candidates: list[tuple[h5py.Dataset, dict[str, str]]] = []
    if isinstance(thumbnail_node, h5py.Dataset):
        candidates.append((thumbnail_node, read_hdf5_attrs(thumbnail_node)))
    elif isinstance(thumbnail_node, h5py.Group):
        for key in ("Data", "Thumbnail", "Image"):
            value = thumbnail_node.get(key)
            if isinstance(value, h5py.Dataset):
                attrs = read_hdf5_attrs(thumbnail_node)
                attrs.update(read_hdf5_attrs(value))
                candidates.append((value, attrs))
        if not candidates:
            for key in thumbnail_node.keys():
                value = thumbnail_node.get(key)
                if isinstance(value, h5py.Dataset):
                    attrs = read_hdf5_attrs(thumbnail_node)
                    attrs.update(read_hdf5_attrs(value))
                    candidates.append((value, attrs))
    for dataset, attrs in candidates:
        array = _reshape_thumbnail_array(np.asarray(dataset), attrs)
        if array is not None:
            return array
    return None


def extract_scene_spot_points(h5: h5py.File) -> dict[str, Any] | None:
    dataset_candidates: list[tuple[str, h5py.Dataset]] = []
    for scene_group in ("Scene8", "Scene"):
        node = h5.get(scene_group)
        if not isinstance(node, h5py.Group):
            continue

        def visitor(name: str, obj: h5py.Group | h5py.Dataset) -> None:
            if not isinstance(obj, h5py.Dataset):
                return
            full_path = f"{scene_group}/{name}"
            dtype_names = tuple(obj.dtype.names or ())
            if {"PositionX", "PositionY", "PositionZ"}.issubset(dtype_names):
                dataset_candidates.append((full_path, obj))
            elif obj.shape and len(obj.shape) == 2 and int(obj.shape[1]) >= 4 and full_path.endswith("CoordsXYZR"):
                dataset_candidates.append((full_path, obj))

        node.visititems(visitor)

    if not dataset_candidates:
        return None

    dataset_path, dataset = dataset_candidates[0]
    parent = dataset.parent
    creation_parameters = None
    creation_node = parent.get("CreationParameters")
    if isinstance(creation_node, h5py.Dataset):
        creation_raw = np.asarray(creation_node)
        if creation_raw.size:
            creation_parameters = _stringify(creation_raw.reshape(-1)[0])

    if dataset.dtype.names and {"PositionX", "PositionY", "PositionZ"}.issubset(set(dataset.dtype.names)):
        data = np.asarray(dataset)
        xyzr = np.column_stack(
            [
                np.asarray(data["PositionX"], dtype=float),
                np.asarray(data["PositionY"], dtype=float),
                np.asarray(data["PositionZ"], dtype=float),
                np.asarray(data["Radius"], dtype=float) if "Radius" in data.dtype.names else np.full(len(data), np.nan, dtype=float),
            ]
        )
        point_ids = np.asarray(data["ID"], dtype=int) if "ID" in data.dtype.names else None
    else:
        raw = np.asarray(dataset, dtype=float)
        xyzr = raw[:, :4]
        point_ids = None

    return {
        "dataset_path": dataset_path,
        "parent_path": parent.name,
        "point_ids": point_ids,
        "xyzr_um": xyzr,
        "creation_parameters": creation_parameters,
    }


def stream_channel_crop(
    path: str | Path,
    *,
    channel_index: int,
    x0: int,
    y0: int,
    width: int,
    height: int,
    resolution_level: int = 0,
    timepoint: int = 0,
    z_start: int | None = None,
    z_end: int | None = None,
) -> np.ndarray:
    resolved = Path(path)
    with h5py.File(resolved, "r") as h5:
        dataset = _channel_dataset(
            h5,
            channel_index=int(channel_index),
            resolution_level=int(resolution_level),
            timepoint=int(timepoint),
        )
        if dataset is None:
            raise KeyError(
                f"Missing dataset for channel {int(channel_index)} at resolution level {int(resolution_level)} timepoint {int(timepoint)}"
            )
        if dataset.ndim == 2:
            return np.asarray(dataset[int(y0) : int(y0 + height), int(x0) : int(x0 + width)])
        if dataset.ndim != 3:
            raise ValueError(f"Expected a 2D or 3D channel dataset, got shape={tuple(int(v) for v in dataset.shape)}")
        z0 = 0 if z_start is None else max(int(z_start), 0)
        z1 = int(dataset.shape[0]) if z_end is None else min(int(z_end), int(dataset.shape[0]))
        if z0 >= z1:
            raise ValueError(f"Invalid z range [{z0}, {z1}) for dataset shape {tuple(int(v) for v in dataset.shape)}")
        if int(x0) < 0 or int(y0) < 0 or int(width) <= 0 or int(height) <= 0:
            raise ValueError("Crop bounds must be non-negative with positive width and height.")
        x1 = int(x0) + int(width)
        y1 = int(y0) + int(height)
        if x1 > int(dataset.shape[2]) or y1 > int(dataset.shape[1]):
            raise ValueError(f"Crop bounds {(x0, y0, width, height)} exceed dataset shape {tuple(int(v) for v in dataset.shape)}")
        return np.asarray(dataset[z0:z1, int(y0) : y1, int(x0) : x1])


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    resolved = Path(path)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def inspect_ims_file(path: Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    with h5py.File(resolved, "r") as h5:
        metadata = extract_standard_ims_metadata(h5)
        channels = enumerate_channel_metadata(h5)
        candidates = find_candidate_rbpms_channels(channels)
        scene = find_scene_objects(h5)
        thumbnail = extract_ims_thumbnail(h5)
    return {
        "path": str(resolved),
        "name": resolved.name,
        "stem": resolved.stem,
        "size_bytes": int(resolved.stat().st_size),
        "source_sha256": file_sha256(resolved),
        "metadata": metadata,
        "channels": channels,
        "candidate_rbpms_channels": candidates,
        "scene": scene,
        "thumbnail_available": thumbnail is not None,
        "thumbnail_shape": [int(value) for value in thumbnail.shape] if thumbnail is not None else None,
    }
