from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ims_io import compute_voxel_sizes_um, extract_scene_spot_points, inspect_ims_file


SOURCE_CHANNEL_RE = re.compile(r'mSourceChannelIndex="(\d+)"')


def _parse_source_channel_index(creation_parameters: str | None) -> int | None:
    if not creation_parameters:
        return None
    match = SOURCE_CHANNEL_RE.search(creation_parameters)
    if match is None:
        return None
    return int(match.group(1))


def _image_geometry(payload: dict) -> tuple[dict[str, str], dict[str, float | None], int, int, int]:
    image = dict(payload.get("metadata", {}).get("image", {}))
    voxel = dict(payload.get("metadata", {}).get("voxel_sizes_um", {}))
    if not voxel:
        voxel = compute_voxel_sizes_um(image)
    size_x = int(image["X"])
    size_y = int(image["Y"])
    size_z = int(image["Z"])
    return image, voxel, size_x, size_y, size_z


def export_scene_points(*, ims_path: Path, metadata_json: Path | None = None) -> tuple[pd.DataFrame, dict]:
    payload = json.loads(metadata_json.read_text(encoding="utf-8")) if metadata_json is not None else inspect_ims_file(ims_path)
    image, voxel, size_x, size_y, size_z = _image_geometry(payload)
    ext_min_x = float(image.get("ExtMin0", 0.0))
    ext_min_y = float(image.get("ExtMin1", 0.0))
    ext_min_z = float(image.get("ExtMin2", 0.0))

    with h5py.File(ims_path, "r") as h5:
        scene_points = extract_scene_spot_points(h5)
    if scene_points is None:
        raise ValueError(f"No embedded scene spot points found in {ims_path}")

    xyzr = np.asarray(scene_points["xyzr_um"], dtype=float)
    point_ids = scene_points.get("point_ids")
    source_channel_index = _parse_source_channel_index(scene_points.get("creation_parameters"))

    frame = pd.DataFrame(
        {
            "point_id": np.arange(len(xyzr), dtype=int) if point_ids is None else np.asarray(point_ids, dtype=int),
            "x_um": xyzr[:, 0],
            "y_um": xyzr[:, 1],
            "z_um": xyzr[:, 2],
            "radius_um": xyzr[:, 3],
        }
    )
    frame["x_px"] = (frame["x_um"] - ext_min_x) / float(voxel["voxel_x_um"])
    frame["y_px"] = (frame["y_um"] - ext_min_y) / float(voxel["voxel_y_um"])
    frame["z_px"] = (frame["z_um"] - ext_min_z) / float(voxel["voxel_z_um"])
    frame["x_px_clipped"] = frame["x_px"].clip(lower=0.0, upper=max(size_x - 1, 0))
    frame["y_px_clipped"] = frame["y_px"].clip(lower=0.0, upper=max(size_y - 1, 0))
    frame["z_px_clipped"] = frame["z_px"].clip(lower=0.0, upper=max(size_z - 1, 0))
    frame["inside_image_bounds"] = (
        (frame["x_px"] >= 0.0)
        & (frame["x_px"] < float(size_x))
        & (frame["y_px"] >= 0.0)
        & (frame["y_px"] < float(size_y))
        & (frame["z_px"] >= 0.0)
        & (frame["z_px"] < float(size_z))
    )

    summary = {
        "ims_path": str(ims_path.resolve()),
        "scene_dataset_path": str(scene_points["dataset_path"]),
        "scene_parent_path": str(scene_points["parent_path"]),
        "source_channel_index": source_channel_index,
        "point_count": int(len(frame)),
        "inside_image_bounds_count": int(frame["inside_image_bounds"].sum()),
        "size_x": size_x,
        "size_y": size_y,
        "size_z": size_z,
        "voxel_x_um": voxel["voxel_x_um"],
        "voxel_y_um": voxel["voxel_y_um"],
        "voxel_z_um": voxel["voxel_z_um"],
    }
    return frame, summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export embedded Imaris scene spots to full-image pixel CSV.")
    parser.add_argument("--ims-path", required=True, type=Path)
    parser.add_argument("--metadata-json", default=None, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-json", default=None, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    frame, summary = export_scene_points(
        ims_path=args.ims_path.resolve(),
        metadata_json=args.metadata_json.resolve() if args.metadata_json is not None else None,
    )
    output_csv = args.output_csv.resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    output_json = args.output_json.resolve() if args.output_json is not None else output_csv.with_suffix(".json")
    output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"[export_ims_scene_points] wrote {len(frame)} points from {summary['scene_dataset_path']} to {output_csv}",
        file=sys.stderr,
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
