from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_ome import load_any_image
from src.roi_data import crop_2d_or_yxc
from src.roi_selection import DEFAULT_SPLIT_TARGETS, parse_split_targets, remaining_split_targets, roi_row, select_fixed_rois_napari


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.to_dict("records"):
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def _projected_images(projected_dir: Path) -> list[Path]:
    paths = sorted(path for path in projected_dir.rglob("*_max.tif") if path.is_file())
    if not paths:
        raise ValueError(f"No projected TIFFs found under {projected_dir}")
    return paths


def _roi_id(image_path: Path, split: str, index: int) -> str:
    return f"{image_path.stem}__{split}_{int(index):03d}"


def _image_sidecar_payload(image_path: Path) -> dict:
    for candidate in (image_path.with_suffix(".json"), Path(str(image_path) + ".json")):
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return {}


def _write_preview(image, *, x0: int, y0: int, width: int, height: int, destination: Path) -> None:
    import matplotlib.pyplot as plt

    crop = crop_2d_or_yxc(image, x0=x0, y0=y0, width=width, height=height)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if crop.ndim == 2:
        plt.imsave(destination, crop, cmap="gray")
    else:
        plt.imsave(destination, crop)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cut a fixed-size ROI manifest from projected TIFF tiles with napari.")
    parser.add_argument("--projected-dir", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--roi-size", default=512, type=int)
    parser.add_argument("--target-count", default=28, type=int)
    parser.add_argument("--split-targets", default="dev=17,locked_eval=6,qc_or_exclude=5")
    parser.add_argument("--search-passes", default=2, type=int)
    parser.add_argument("--export-previews", action="store_true")
    parser.add_argument("--annotator", default=os.environ.get("USER", "unknown"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    projected_paths = _projected_images(args.projected_dir.resolve())
    output_manifest = args.output_manifest.resolve()
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    split_targets = parse_split_targets(args.split_targets)
    if sum(split_targets.values()) < int(args.target_count):
        raise ValueError("split target totals must cover the requested target count.")

    rows: list[dict[str, object]] = []
    split_counts: Counter[str] = Counter()
    _log(
        f"[cut_roi_manifest_from_tiles] Starting ROI selection across {len(projected_paths)} image(s); "
        f"target={int(args.target_count)} roi_size={int(args.roi_size)} split_targets={dict(split_targets)}"
    )

    for pass_index in range(int(args.search_passes)):
        if sum(split_counts.values()) >= int(args.target_count):
            break
        _log(
            f"[cut_roi_manifest_from_tiles] Pass {int(pass_index) + 1}/{int(args.search_passes)} "
            f"current_counts={dict(split_counts)}"
        )
        for image_path in projected_paths:
            remaining = remaining_split_targets(split_counts, split_targets)
            if sum(remaining.values()) <= 0:
                break
            _log(
                f"[cut_roi_manifest_from_tiles] Opening {image_path.name} with remaining_targets={dict(remaining)}"
            )
            image, _ = load_any_image(str(image_path))
            sidecar = _image_sidecar_payload(image_path)
            selections = select_fixed_rois_napari(
                image,
                roi_size=int(args.roi_size),
                split_targets=remaining,
                title=f"{image_path.name} | pass {int(pass_index) + 1}",
            )
            _log(
                f"[cut_roi_manifest_from_tiles] {image_path.name}: captured {len(selections)} selection(s)"
            )
            for selection in selections:
                split = str(selection["split"])
                if split_counts[split] >= int(split_targets.get(split, 0)):
                    continue
                split_counts[split] += 1
                roi_id = _roi_id(image_path, split, split_counts[split])
                row = roi_row(
                    roi_id=roi_id,
                    image_path=image_path,
                    split=split,
                    x0=int(selection["x0"]),
                    y0=int(selection["y0"]),
                    width=int(selection["width"]),
                    height=int(selection["height"]),
                    annotator=args.annotator,
                    image_marker="RBPMS",
                    image_source_channel=int(sidecar.get("channel_index")) if sidecar.get("channel_index") is not None else None,
                )
                rows.append(row)
                if args.export_previews:
                    _write_preview(
                        image,
                        x0=int(selection["x0"]),
                        y0=int(selection["y0"]),
                        width=int(selection["width"]),
                        height=int(selection["height"]),
                        destination=output_manifest.parent / "previews" / f"{roi_id}.png",
                    )
            if sum(split_counts.values()) >= int(args.target_count):
                break

    frame = pd.DataFrame(rows)
    frame.to_csv(output_manifest, index=False)
    (output_manifest.parent / "selection_summary.md").write_text(
        "# ROI Selection Summary\n\n"
        + _markdown_table(
            pd.DataFrame(
                [{"split": split, "selected": int(split_counts.get(split, 0)), "target": int(target)} for split, target in split_targets.items()]
            )
        )
        + "\n",
        encoding="utf-8",
    )
    _log(
        f"[cut_roi_manifest_from_tiles] Wrote {len(frame)} ROI row(s) to {output_manifest} "
        f"with final_counts={dict(split_counts)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
