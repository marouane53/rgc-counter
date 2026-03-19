from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ims_io import inspect_ims_file


def _resolve_inputs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        expanded = glob.glob(str(Path(pattern).expanduser()))
        if expanded:
            paths.extend(Path(match).resolve() for match in expanded if Path(match).is_file())
        elif Path(pattern).expanduser().is_file():
            paths.append(Path(pattern).expanduser().resolve())
    unique = sorted(dict.fromkeys(paths))
    if not unique:
        raise ValueError("No .ims files matched the provided inputs.")
    return unique


def _save_thumbnail(array, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if array.ndim == 2:
        plt.imsave(destination, array, cmap="gray")
    else:
        plt.imsave(destination, array)


def build_inventory_row(payload: dict[str, object], metadata_json_path: Path, thumbnail_path: Path | None) -> dict[str, object]:
    metadata = dict(payload.get("metadata", {}))
    image = dict(metadata.get("image", {}))
    voxel = dict(metadata.get("voxel_sizes_um", {}))
    candidates = list(payload.get("candidate_rbpms_channels", []))
    top_candidate = candidates[0] if candidates else {}
    scene = dict(payload.get("scene", {}))
    return {
        "stem": payload["stem"],
        "source_ims_path": payload["path"],
        "metadata_json_path": str(metadata_json_path),
        "thumbnail_path": str(thumbnail_path) if thumbnail_path is not None else "",
        "source_sha256": payload["source_sha256"],
        "size_bytes": payload["size_bytes"],
        "recording_date": image.get("RecordingDate", ""),
        "lens_power": image.get("LensPower", ""),
        "unit": image.get("Unit", ""),
        "size_x": image.get("X", ""),
        "size_y": image.get("Y", ""),
        "size_z": image.get("Z", ""),
        "n_channels": metadata.get("n_channels", ""),
        "voxel_x_um": voxel.get("voxel_x_um"),
        "voxel_y_um": voxel.get("voxel_y_um"),
        "voxel_z_um": voxel.get("voxel_z_um"),
        "has_scene": bool(scene.get("scene_groups")),
        "scene_annotation_candidates": len(scene.get("annotation_candidates", [])),
        "top_candidate_channel_index": top_candidate.get("channel_index", ""),
        "top_candidate_score": top_candidate.get("score", ""),
        "top_candidate_reason": top_candidate.get("reason", ""),
        "top_candidate_unambiguous": bool(top_candidate.get("is_unambiguous_top_candidate", False)),
    }


def build_report(rows: list[dict[str, object]]) -> str:
    frame = pd.DataFrame(rows)
    def _markdown_table(local_frame: pd.DataFrame) -> str:
        if local_frame.empty:
            return "_No rows._"
        columns = list(local_frame.columns)
        lines = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join(["---"] * len(columns)) + " |",
        ]
        for row in local_frame.to_dict("records"):
            lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
        return "\n".join(lines)

    lines = [
        "# IMS Recon Report",
        "",
        f"- Files inspected: `{len(rows)}`",
        "",
    ]
    if frame.empty:
        lines.extend(["No rows produced.", ""])
        return "\n".join(lines)
    lines.extend(["## Inventory", "", _markdown_table(frame), ""])
    ambiguous = frame[~frame["top_candidate_unambiguous"].fillna(False)]
    if ambiguous.empty:
        lines.extend(["## Channel Selection Status", "", "Every file has one unambiguous top-ranked candidate channel in metadata.", ""])
    else:
        lines.extend(
            [
                "## Channel Selection Status",
                "",
                "Metadata alone does not resolve every file. Use the all-channel preview pass before selecting the benchmark lane.",
                "",
                _markdown_table(ambiguous[["stem", "top_candidate_channel_index", "top_candidate_score", "top_candidate_reason"]]),
                "",
            ]
        )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect private `.ims` files with h5py and emit metadata inventories.")
    parser.add_argument("--input", nargs="+", required=True, help="Input .ims paths or glob patterns.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Metadata output directory.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_paths = _resolve_inputs(args.input)
    metadata_dir = args.output_dir.resolve()
    thumbnail_dir = metadata_dir.parent / "thumbnails"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    thumbnail_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for path in input_paths:
        payload = inspect_ims_file(path)
        metadata_json_path = metadata_dir / f"{path.stem}.json"
        metadata_json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        thumbnail_path: Path | None = None
        if payload.get("thumbnail_available"):
            thumbnail_path = thumbnail_dir / f"{path.stem}.png"
            with metadata_json_path.open("r", encoding="utf-8"):
                pass
            from src.ims_io import extract_ims_thumbnail  # local import keeps script entrypoint simple
            import h5py

            with h5py.File(path, "r") as h5:
                thumbnail = extract_ims_thumbnail(h5)
            if thumbnail is not None:
                _save_thumbnail(thumbnail, thumbnail_path)
        rows.append(build_inventory_row(payload, metadata_json_path, thumbnail_path))

    inventory = pd.DataFrame(rows).sort_values("stem").reset_index(drop=True)
    inventory.to_csv(metadata_dir / "ims_inventory.csv", index=False)
    (metadata_dir / "ims_recon_report.md").write_text(build_report(rows) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
