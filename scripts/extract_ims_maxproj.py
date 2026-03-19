from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ims_io import inspect_ims_file


def _log_progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _normalize_preview(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    low = float(np.percentile(arr, 1.0))
    high = float(np.percentile(arr, 99.5))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(arr.min())
        high = float(arr.max()) if arr.size else low + 1.0
    scaled = np.clip((arr - low) / max(high - low, 1e-6), 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def _resolve_channel_dataset(h5: h5py.File, *, channel_index: int, resolution_level: int, timepoint: int) -> h5py.Dataset:
    path = f"DataSet/ResolutionLevel {int(resolution_level)}/TimePoint {int(timepoint)}/Channel {int(channel_index)}/Data"
    dataset = h5.get(path)
    if not isinstance(dataset, h5py.Dataset):
        raise KeyError(f"Missing dataset path: {path}")
    return dataset


def stream_max_projection(
    path: str | Path,
    channel_index: int,
    *,
    resolution_level: int = 0,
    timepoint: int = 0,
    slab: tuple[int | None, int | None] | None = None,
) -> np.ndarray:
    with h5py.File(Path(path), "r") as h5:
        dataset = _resolve_channel_dataset(
            h5,
            channel_index=int(channel_index),
            resolution_level=int(resolution_level),
            timepoint=int(timepoint),
        )
        if dataset.ndim == 2:
            return np.asarray(dataset[...])
        if dataset.ndim != 3:
            raise ValueError(f"Expected a 2D or 3D per-channel dataset, got shape={dataset.shape}")
        start = 0 if slab is None or slab[0] is None else max(int(slab[0]), 0)
        stop = int(dataset.shape[0]) if slab is None or slab[1] is None else min(int(slab[1]), int(dataset.shape[0]))
        if start >= stop:
            raise ValueError(f"Invalid slab bounds for dataset shape {dataset.shape}: {slab}")
        projection = np.asarray(dataset[start], dtype=dataset.dtype)
        for index in range(start + 1, stop):
            projection = np.maximum(projection, np.asarray(dataset[index], dtype=dataset.dtype))
        return projection


def export_projected_tiff(image: np.ndarray, destination: str | Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(destination, np.asarray(image))
    return destination


def export_preview_png(image: np.ndarray, destination: str | Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(destination, _normalize_preview(image), cmap="gray")
    return destination


def _load_metadata_payload(row: pd.Series) -> dict[str, Any]:
    metadata_path = Path(str(row["metadata_json_path"]))
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _channel_indices(payload: dict[str, Any], channel_arg: str) -> list[int]:
    channels = [
        int(channel["channel_index"])
        for channel in payload.get("channels", [])
        if channel.get("data_dataset_path") or channel.get("data_shape")
    ]
    if channel_arg == "all":
        return channels
    if channel_arg == "auto":
        candidates = list(payload.get("candidate_rbpms_channels", []))
        if not candidates:
            raise ValueError(f"No candidate channel metadata for {payload['path']}")
        top = candidates[0]
        if not bool(top.get("is_unambiguous_top_candidate")):
            raise ValueError(f"Top candidate is ambiguous for {payload['path']}")
        return [int(top["channel_index"])]
    return [int(token.strip()) for token in channel_arg.split(",") if token.strip()]


def _projection_label(slab: tuple[int | None, int | None] | None) -> str:
    if slab is None or (slab[0] is None and slab[1] is None):
        return "max"
    start = "" if slab[0] is None else str(int(slab[0]))
    end = "" if slab[1] is None else str(int(slab[1]))
    return f"slab_max_{start}_{end}".strip("_")


def _contact_sheet(preview_paths: list[tuple[str, Path]], destination: Path) -> Path:
    if not preview_paths:
        return destination
    images = [plt.imread(path) for _, path in preview_paths]
    widths = [image.shape[1] for image in images]
    heights = [image.shape[0] for image in images]
    canvas = np.ones((sum(heights), max(widths), 3), dtype=np.float32)
    cursor = 0
    for image in images:
        rgb = image[..., :3] if image.ndim == 3 else np.repeat(image[..., None], 3, axis=2)
        canvas[cursor : cursor + rgb.shape[0], : rgb.shape[1], :] = rgb
        cursor += rgb.shape[0]
    destination.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(destination, canvas)
    return destination


def build_manifest_row(
    *,
    payload: dict[str, Any],
    channel_index: int,
    tiff_path: Path,
    preview_path: Path,
    selection_mode: str,
    projection: str,
) -> dict[str, Any]:
    metadata = dict(payload.get("metadata", {}))
    image = dict(metadata.get("image", {}))
    voxel = dict(metadata.get("voxel_sizes_um", {}))
    scene = dict(payload.get("scene", {}))
    channel_map = {int(channel["channel_index"]): channel for channel in payload.get("channels", [])}
    candidate_map = {int(candidate["channel_index"]): candidate for candidate in payload.get("candidate_rbpms_channels", [])}
    channel = channel_map[int(channel_index)]
    candidate = candidate_map.get(int(channel_index), {})
    data_shape = channel.get("data_shape") or [None, None, None]
    size_z = data_shape[0] if len(data_shape) > 2 else image.get("Z", "")
    size_y = data_shape[-2] if len(data_shape) >= 2 else image.get("Y", "")
    size_x = data_shape[-1] if len(data_shape) >= 1 else image.get("X", "")
    return {
        "image_id": f"{payload['stem']}__ch{int(channel_index)}",
        "source_ims_path": payload["path"],
        "projected_tiff_path": str(tiff_path),
        "preview_png_path": str(preview_path),
        "channel_index": int(channel_index),
        "channel_name": channel.get("name", ""),
        "channel_description": channel.get("description", ""),
        "size_x": size_x,
        "size_y": size_y,
        "size_z": size_z,
        "voxel_x_um": voxel.get("voxel_x_um"),
        "voxel_y_um": voxel.get("voxel_y_um"),
        "voxel_z_um": voxel.get("voxel_z_um"),
        "unit": image.get("Unit", ""),
        "recording_date": image.get("RecordingDate", ""),
        "lens_power": image.get("LensPower", ""),
        "projection": projection,
        "has_scene": bool(scene.get("scene_groups")),
        "candidate_rbpms_reason": candidate.get("reason", ""),
        "selection_mode": selection_mode,
        "source_sha256": payload["source_sha256"],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract streamed max projections from private `.ims` files.")
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--channel", default="all", help="`all`, `auto`, or a comma-separated list of channel indices.")
    parser.add_argument("--resolution-level", type=int, default=0)
    parser.add_argument("--timepoint", type=int, default=0)
    parser.add_argument("--slab-start", type=int, default=None)
    parser.add_argument("--slab-end", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    started_at = time.perf_counter()
    args = parse_args(argv)
    inventory = pd.read_csv(args.inventory)
    slab = (args.slab_start, args.slab_end) if args.slab_start is not None or args.slab_end is not None else None
    projection = _projection_label(slab)
    channel_mode = str(args.channel).strip().lower()
    if channel_mode == "all":
        base_dir = args.output_dir.resolve() / "all_channels"
    else:
        base_dir = args.output_dir.resolve() / "selected"
    base_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = base_dir / "previews"
    review_dir = base_dir / "review"

    manifest_rows: list[dict[str, Any]] = []
    total_images = len(inventory)
    total_channels_written = 0
    _log_progress(
        f"[extract_ims_maxproj] Starting extraction for {total_images} image(s) with channel mode '{channel_mode}' into {base_dir}"
    )
    for image_offset, (_, row) in enumerate(inventory.iterrows(), start=1):
        image_started_at = time.perf_counter()
        payload = _load_metadata_payload(row)
        if not payload.get("channels"):
            payload = inspect_ims_file(Path(str(row["source_ims_path"])))
        channel_indices = _channel_indices(payload, channel_mode)
        if not channel_indices:
            _log_progress(
                f"[extract_ims_maxproj] [{image_offset}/{total_images}] {payload['stem']}: no extractable channels found, skipping"
            )
            continue
        _log_progress(
            f"[extract_ims_maxproj] [{image_offset}/{total_images}] {payload['stem']}: extracting {len(channel_indices)} channel(s)"
        )
        extracted_preview_paths: list[tuple[str, Path]] = []
        for channel_offset, channel_index in enumerate(channel_indices, start=1):
            channel_started_at = time.perf_counter()
            _log_progress(
                f"[extract_ims_maxproj] [{image_offset}/{total_images}] {payload['stem']} channel {channel_index} ({channel_offset}/{len(channel_indices)})"
            )
            projected = stream_max_projection(
                payload["path"],
                channel_index=int(channel_index),
                resolution_level=int(args.resolution_level),
                timepoint=int(args.timepoint),
                slab=slab,
            )
            tiff_path = export_projected_tiff(projected, base_dir / f"{payload['stem']}__ch{int(channel_index)}_{projection}.tif")
            preview_path = export_preview_png(projected, preview_dir / f"{payload['stem']}__ch{int(channel_index)}_{projection}.png")
            extracted_preview_paths.append((f"ch{int(channel_index)}", preview_path))

            sidecar = build_manifest_row(
                payload=payload,
                channel_index=int(channel_index),
                tiff_path=tiff_path,
                preview_path=preview_path,
                selection_mode=channel_mode,
                projection=projection,
            )
            (tiff_path.with_suffix(".json")).write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
            manifest_rows.append(sidecar)
            total_channels_written += 1
            _log_progress(
                f"[extract_ims_maxproj] [{image_offset}/{total_images}] {payload['stem']} channel {channel_index}: wrote {tiff_path.name} in {time.perf_counter() - channel_started_at:.1f}s"
            )
        _contact_sheet(extracted_preview_paths, review_dir / f"{payload['stem']}__channels_contact_sheet.png")
        _log_progress(
            f"[extract_ims_maxproj] [{image_offset}/{total_images}] {payload['stem']}: completed in {time.perf_counter() - image_started_at:.1f}s"
        )

    manifest = pd.DataFrame(manifest_rows).sort_values(["source_ims_path", "channel_index"]).reset_index(drop=True)
    manifest.to_csv(base_dir / "projected_manifest.csv", index=False)
    _log_progress(
        f"[extract_ims_maxproj] Finished: wrote {total_channels_written} projection(s) across {total_images} image(s) in {time.perf_counter() - started_at:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
