from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ims_io import stream_channel_crop
from src.micro_roi_benchmark import canonical_recipe_json, default_preprocess_variants, default_projection_recipes
from src.rbpms_confocal import (
    build_void_fold_mask,
    compute_slice_focus_scores,
    enumerate_candidate_slabs,
    normalize_for_detection,
    project_focus_weighted,
    project_percentile,
    project_slab,
    project_topk_mean,
    subtract_background,
)
from src.roi_data import iter_roi_records, load_roi_manifest


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _image_sidecar_payload(image_path: Path) -> dict:
    for candidate in (image_path.with_suffix(".json"), Path(str(image_path) + ".json")):
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"Missing projected image sidecar for {image_path}")


def _save_png(image: np.ndarray, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(destination, image, cmap="gray")


def _best_slab_lookup(catalog_rows: list[dict]) -> dict[int, dict]:
    lookup: dict[int, dict] = {}
    for row in catalog_rows:
        if bool(row.get("is_best")):
            lookup[int(row["window_size"])] = dict(row)
    return lookup


def _apply_projection_recipe(volume: np.ndarray, focus_scores: pd.DataFrame, slab_lookup: dict[int, dict], recipe: dict) -> np.ndarray:
    kind = str(recipe["projection_kind"])
    if kind == "full_max":
        return project_slab(volume, 0, int(volume.shape[0]), mode="max")
    if kind == "best_slab_max":
        slab = slab_lookup[int(recipe["window_size"])]
        return project_slab(volume, int(slab["start"]), int(slab["end"]), mode="max")
    if kind == "best_slab_topkmean":
        slab = slab_lookup[int(recipe["window_size"])]
        return project_topk_mean(volume[int(slab["start"]) : int(slab["end"])], int(recipe["topk"]))
    if kind == "full_focus_weighted":
        return project_focus_weighted(volume, focus_scores)
    if kind == "full_percentile":
        return project_percentile(volume, float(recipe["percentile_q"]))
    raise ValueError(f"Unsupported projection recipe: {recipe}")


def _apply_preprocess(image: np.ndarray, preprocess: dict) -> np.ndarray:
    method = str(preprocess.get("background_subtraction", "none"))
    if method == "none":
        base = np.asarray(image)
    else:
        base = subtract_background(image, method=method, radius_px=int(preprocess["radius_px"]))
    return normalize_for_detection(base, mode=str(preprocess.get("normalization", "robust_float")))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the private channel-1 confocal projection ablation on micro-ROIs.")
    parser.add_argument("--roi-manifest", "--manifest", dest="roi_manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--resolution-level", default=0, type=int)
    parser.add_argument("--timepoint", default=0, type=int)
    parser.add_argument("--include-splits", nargs="*", default=None)
    parser.add_argument("--exclude-splits", nargs="*", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = load_roi_manifest(args.roi_manifest)
    records = iter_roi_records(
        manifest,
        manifest_path=args.roi_manifest,
        include_splits=args.include_splits,
        exclude_splits=args.exclude_splits,
    )
    output_dir = args.output_dir.resolve()
    raw_dir = output_dir / "raw_stacks"
    focus_dir = output_dir / "focus_scores"
    slab_dir = output_dir / "slab_catalogs"
    view_dir = output_dir / "views"
    for directory in (raw_dir, focus_dir, slab_dir, view_dir):
        directory.mkdir(parents=True, exist_ok=True)

    projection_recipes = default_projection_recipes()
    preprocess_variants = default_preprocess_variants()
    expected_views = len(records) * len(projection_recipes) * len(preprocess_variants)
    _log(
        f"[run_rbpms_projection_ablation] Starting projection lab for {len(records)} ROI(s); "
        f"{len(projection_recipes)} projection recipe(s) x {len(preprocess_variants)} preprocess variant(s) = "
        f"{expected_views} detector-input view(s)"
    )

    manifest_rows: list[dict[str, object]] = []
    for index, record in enumerate(records, start=1):
        image_sidecar = _image_sidecar_payload(record.image_path)
        source_ims_path = Path(str(image_sidecar.get("source_ims_path", ""))).resolve()
        if not source_ims_path.exists():
            raise FileNotFoundError(f"Missing source .ims path for ROI {record.roi_id}: {source_ims_path}")
        channel_index = record.image_source_channel if record.image_source_channel is not None else int(image_sidecar.get("channel_index", 1))
        _log(f"[run_rbpms_projection_ablation] [{index}/{len(records)}] {record.roi_id}: reading raw stack from {source_ims_path.name} channel {channel_index}")
        volume = stream_channel_crop(
            source_ims_path,
            channel_index=int(channel_index),
            x0=int(record.x0),
            y0=int(record.y0),
            width=int(record.width),
            height=int(record.height),
            resolution_level=int(args.resolution_level),
            timepoint=int(args.timepoint),
        )
        raw_stack_path = raw_dir / f"{record.roi_id}__raw_stack.tif"
        tifffile.imwrite(raw_stack_path, volume)
        focus_scores = compute_slice_focus_scores(volume)
        focus_path = focus_dir / f"{record.roi_id}__focus_scores.csv"
        focus_scores.to_csv(focus_path, index=False)
        slab_catalog_rows = enumerate_candidate_slabs(volume)
        slab_catalog = pd.DataFrame(slab_catalog_rows)
        slab_path = slab_dir / f"{record.roi_id}__slab_catalog.csv"
        slab_catalog.to_csv(slab_path, index=False)
        slab_lookup = _best_slab_lookup(slab_catalog_rows)
        _log(
            f"[run_rbpms_projection_ablation] [{index}/{len(records)}] {record.roi_id}: "
            f"raw stack shape={tuple(int(v) for v in volume.shape)} focus_rows={len(focus_scores)} slabs={len(slab_catalog_rows)}"
        )

        for recipe_index, recipe in enumerate(projection_recipes, start=1):
            projection_image = _apply_projection_recipe(volume, focus_scores, slab_lookup, recipe)
            projection_id = str(recipe["projection_id"])
            projection_recipe_json = canonical_recipe_json(recipe)
            projection_dir = view_dir / record.roi_id / projection_id
            projection_dir.mkdir(parents=True, exist_ok=True)
            fold_mask = build_void_fold_mask(projection_image)
            exclude_mask_path = projection_dir / f"{record.roi_id}__{projection_id}__exclude_mask.tif"
            tifffile.imwrite(exclude_mask_path, fold_mask.astype(np.uint8))
            _log(
                f"[run_rbpms_projection_ablation] [{index}/{len(records)}] {record.roi_id}: "
                f"projection {recipe_index}/{len(projection_recipes)} -> {projection_id} "
                f"(masked_px={int(fold_mask.sum())})"
            )

            for preprocess_index, preprocess in enumerate(preprocess_variants, start=1):
                preprocess_id = str(preprocess["preprocess_id"])
                preprocess_json = canonical_recipe_json(preprocess)
                current_dir = projection_dir / preprocess_id
                current_dir.mkdir(parents=True, exist_ok=True)
                detector_input = _apply_preprocess(projection_image, preprocess)
                detector_input_path = current_dir / f"{record.roi_id}__{projection_id}__{preprocess_id}__detector_input.tif"
                tifffile.imwrite(detector_input_path, detector_input.astype(np.float32))
                preview_png_path = current_dir / f"{record.roi_id}__{projection_id}__{preprocess_id}__preview.png"
                _save_png(normalize_for_detection(detector_input, mode="display_uint8"), preview_png_path)
                sidecar_path = current_dir / f"{record.roi_id}__{projection_id}__{preprocess_id}.json"
                sidecar_payload = {
                    "roi_id": record.roi_id,
                    "split": record.split,
                    "source_ims_path": str(source_ims_path),
                    "source_projected_image_path": str(record.image_path),
                    "image_source_channel": int(channel_index),
                    "raw_stack_path": str(raw_stack_path),
                    "focus_scores_path": str(focus_path),
                    "slab_catalog_path": str(slab_path),
                    "projection_recipe_json": projection_recipe_json,
                    "preprocess_json": preprocess_json,
                    "detector_input_path": str(detector_input_path),
                    "exclude_mask_path": str(exclude_mask_path),
                    "preview_png_path": str(preview_png_path),
                }
                sidecar_path.write_text(json.dumps(sidecar_payload, indent=2) + "\n", encoding="utf-8")
                manifest_rows.append(
                    {
                        "roi_id": record.roi_id,
                        "split": record.split,
                        "image_path": str(record.image_path),
                        "source_ims_path": str(source_ims_path),
                        "image_source_channel": int(channel_index),
                        "raw_stack_path": str(raw_stack_path),
                        "focus_scores_path": str(focus_path),
                        "slab_catalog_path": str(slab_path),
                        "projection_id": projection_id,
                        "preprocess_id": preprocess_id,
                        "projection_recipe_json": projection_recipe_json,
                        "preprocess_json": preprocess_json,
                        "detector_input_path": str(detector_input_path),
                        "preview_png_path": str(preview_png_path),
                        "exclude_mask_path": str(exclude_mask_path),
                        "sidecar_json_path": str(sidecar_path),
                    }
                )
                _log(
                    f"[run_rbpms_projection_ablation] [{index}/{len(records)}] {record.roi_id}: "
                    f"{projection_id} preprocess {preprocess_index}/{len(preprocess_variants)} -> {preprocess_id}"
                )

    view_manifest = pd.DataFrame(manifest_rows).sort_values(["roi_id", "projection_id", "preprocess_id"]).reset_index(drop=True)
    view_manifest_path = output_dir / "view_manifest.csv"
    view_manifest.to_csv(view_manifest_path, index=False)
    _log(
        f"[run_rbpms_projection_ablation] Finished: wrote {len(view_manifest)} view row(s) to {view_manifest_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
