from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_ome import load_any_image
from src.point_curation import merge_candidate_point_frames, summarize_curation_edits
from src.roi_data import crop_2d_or_yxc, iter_roi_records, load_roi_manifest


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def write_points_csv(path: Path, points_yx: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"x_px": points_yx[:, 1] if len(points_yx) else [], "y_px": points_yx[:, 0] if len(points_yx) else []}).to_csv(path, index=False)


def _load_candidates_for_roi(summary: pd.DataFrame, roi_id: str, candidate_configs: list[str] | None) -> list[pd.DataFrame]:
    roi_runs = summary.loc[summary["roi_id"].astype(str) == roi_id].copy()
    if roi_runs.empty:
        return []
    if candidate_configs:
        roi_runs = roi_runs.loc[roi_runs["config_id"].astype(str).isin(candidate_configs)].copy()
    elif "rank_within_roi" in roi_runs.columns:
        roi_runs = roi_runs.sort_values(["rank_within_roi", "config_id"]).head(2).copy()
    else:
        roi_runs = roi_runs.sort_values(["config_id"]).head(2).copy()
    frames: list[pd.DataFrame] = []
    for path in roi_runs["points_csv_path"].astype(str).tolist():
        candidate_path = Path(path)
        if not candidate_path.exists():
            continue
        frame = pd.read_csv(candidate_path)
        if not frame.empty:
            frames.append(frame)
    return frames


def curate_points_interactive(image: np.ndarray, initial_points_yx: np.ndarray, *, overlay_text: str, save_path: Path | None) -> np.ndarray:
    import napari

    holder: dict[str, np.ndarray] = {"points": np.asarray(initial_points_yx, dtype=float)}
    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(image, name="roi")
        viewer.text_overlay.visible = True
        viewer.text_overlay.text = overlay_text
        points_layer = viewer.add_points(
            initial_points_yx,
            name="curated_points",
            ndim=2,
            size=10,
            edge_color="yellow",
            face_color="transparent",
        )

        def _current_points() -> np.ndarray:
            return np.asarray(points_layer.data, dtype=float)

        def _save_points(*, close_viewer: bool) -> None:
            holder["points"] = _current_points()
            if save_path is not None:
                write_points_csv(save_path, holder["points"])
            if close_viewer:
                viewer.close()

        @viewer.bind_key("s")
        def _save_and_close(v):
            _save_points(close_viewer=True)

        @viewer.bind_key("p")
        def _save_partial(v):
            _save_points(close_viewer=False)

        @viewer.bind_key("u")
        def _undo_last_point(v):
            if len(points_layer.data):
                points_layer.data = points_layer.data[:-1]

        @viewer.bind_key("q")
        def _quit_without_save(v):
            v.close()

        print("Curate the preloaded candidate soma centers in the 'curated_points' layer.")
        print("Delete false positives, add missed centers, 'u' undo, 'p' partial save, 's' save and close, 'q' close.")

    return np.asarray(holder["points"], dtype=float)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assistively curate point-detector candidates into ROI-local truth.")
    parser.add_argument("--roi-manifest", "--manifest", dest="roi_manifest", required=True, type=Path)
    parser.add_argument("--probe-summary", required=True, type=Path)
    parser.add_argument("--candidate-configs", nargs="*", default=None)
    parser.add_argument("--annotator", default=os.environ.get("USER", "unknown"))
    parser.add_argument("--include-splits", nargs="*", default=None)
    parser.add_argument("--exclude-splits", nargs="*", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = pd.read_csv(args.probe_summary)
    manifest = load_roi_manifest(args.roi_manifest)
    records = iter_roi_records(
        manifest,
        manifest_path=args.roi_manifest,
        include_splits=args.include_splits,
        exclude_splits=args.exclude_splits,
    )
    _log(
        f"[curate_point_candidates] Starting assisted curation for {len(records)} ROI(s) "
        f"using {'explicit configs ' + ', '.join(args.candidate_configs) if args.candidate_configs else 'top-2 ranked configs per ROI'}"
    )

    curated_count = 0
    for index, record in enumerate(records, start=1):
        if record.manual_points_path is None:
            continue
        image, _ = load_any_image(str(record.image_path))
        crop = crop_2d_or_yxc(image, x0=record.x0, y0=record.y0, width=record.width, height=record.height)
        candidate_frames = _load_candidates_for_roi(summary, record.roi_id, args.candidate_configs)
        merged = merge_candidate_point_frames(candidate_frames, tolerance_px=3.0)
        initial_points = merged[["y_px", "x_px"]].to_numpy(dtype=float) if not merged.empty else np.empty((0, 2), dtype=float)
        _log(
            f"[curate_point_candidates] [{index}/{len(records)}] {record.roi_id}: "
            f"loaded {len(candidate_frames)} candidate set(s), seeded {len(initial_points)} point(s)"
        )
        started = time.perf_counter()
        final_points = curate_points_interactive(
            crop,
            initial_points,
            overlay_text=(
                f"ROI: {record.roi_id}\n"
                f"Split: {record.split}\n"
                f"Seed configs: {', '.join(args.candidate_configs) if args.candidate_configs else 'top-2 heuristic'}\n"
                "Delete false positives and add missed soma centers.\n"
                "Border rule: one point per soma center; avoid folds, debris, and border-only fragments."
            ),
            save_path=record.manual_points_path,
        )
        write_points_csv(record.manual_points_path, final_points)
        edit_summary = summarize_curation_edits(initial_points, final_points, tolerance_px=1.5)
        meta_payload = {
            "roi_id": record.roi_id,
            "annotator": args.annotator,
            "image_path": str(record.image_path),
            "marker": record.marker,
            "truth_marker": "RBPMS",
            "truth_source_channel": int(record.image_source_channel) if record.image_source_channel is not None else None,
            "truth_derivation": "assisted_curated_point_truth",
            "truth_source_path": str(record.image_path),
            "seed_config_ids": args.candidate_configs or summary.loc[summary["roi_id"].astype(str) == record.roi_id].sort_values(["rank_within_roi", "config_id"]).head(2)["config_id"].astype(str).tolist(),
            "elapsed_seconds": float(time.perf_counter() - started),
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "tool": "scripts/curate_point_candidates.py",
            **edit_summary,
        }
        meta_path = record.manual_points_path.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(meta_payload, indent=2) + "\n", encoding="utf-8")
        curated_count += 1
        _log(
            f"[curate_point_candidates] [{index}/{len(records)}] {record.roi_id}: "
            f"saved {len(final_points)} curated point(s) to {record.manual_points_path} "
            f"(added={edit_summary['added_count']} deleted={edit_summary['deleted_count']} "
            f"elapsed={meta_payload['elapsed_seconds']:.1f}s)"
        )
    _log(f"[curate_point_candidates] Finished: curated {curated_count} ROI(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
