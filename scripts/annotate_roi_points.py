from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_ome import load_any_image
from src.roi_data import crop_2d_or_yxc, filter_roi_manifest_by_split, iter_roi_records, load_roi_manifest


def read_existing_points(path: Path) -> np.ndarray:
    if not path.exists():
        return np.empty((0, 2), dtype=float)
    frame = pd.read_csv(path)
    if frame.empty:
        return np.empty((0, 2), dtype=float)
    return frame[["y_px", "x_px"]].to_numpy(dtype=float)


def write_points_csv(path: Path, points_yx: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "x_px": points_yx[:, 1] if len(points_yx) else [],
            "y_px": points_yx[:, 0] if len(points_yx) else [],
        }
    )
    frame.to_csv(path, index=False)


def build_points_metadata(
    *,
    roi_id: str,
    annotator: str,
    image_path: str | Path,
    marker: str,
    modality: str,
    roi_xywh: tuple[int, int, int, int],
    n_points: int,
) -> dict[str, object]:
    return {
        "roi_id": roi_id,
        "annotator": annotator,
        "image_path": str(image_path),
        "marker": marker,
        "modality": modality,
        "roi_xywh": [int(v) for v in roi_xywh],
        "n_points": int(n_points),
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "tool": "scripts/annotate_roi_points.py",
    }


def write_points_metadata(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def annotate_points_interactive(
    image: np.ndarray,
    existing_points_yx: np.ndarray,
    *,
    save_path: Path | None = None,
    overlay_text: str = "",
) -> np.ndarray:
    import napari

    holder: dict[str, np.ndarray] = {"points": np.asarray(existing_points_yx, dtype=float)}
    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(image, name="roi")
        viewer.text_overlay.visible = True
        viewer.text_overlay.text = overlay_text
        points_layer = viewer.add_points(
            existing_points_yx,
            name="manual_points",
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

        print("Add or edit soma-center points in the 'manual_points' layer.")
        print("Press 'p' to save partial progress, 'u' to undo the last point, 's' to save and close, or 'q' to close without saving.")

    return np.asarray(holder["points"], dtype=float)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Annotate ROI-local soma-center points with Napari.")
    parser.add_argument("--roi_manifest", "--manifest", dest="roi_manifest", required=True, type=Path)
    parser.add_argument("--roi_id", default=None)
    parser.add_argument("--annotator", default=os.environ.get("USER", "unknown"))
    parser.add_argument("--include-splits", nargs="*", default=None)
    parser.add_argument("--exclude-splits", nargs="*", default=None)
    args = parser.parse_args(argv)

    manifest = filter_roi_manifest_by_split(
        load_roi_manifest(args.roi_manifest),
        include_splits=args.include_splits,
        exclude_splits=args.exclude_splits,
    )
    records = iter_roi_records(manifest, manifest_path=args.roi_manifest)
    if args.roi_id:
        records = [record for record in records if record.roi_id == args.roi_id]
        if not records:
            raise SystemExit(f"ROI not found: {args.roi_id}")

    for record in records:
        if record.manual_points_path is None:
            raise ValueError(f"ROI {record.roi_id} is missing manual_points_path in the manifest.")
        image, _ = load_any_image(str(record.image_path))
        crop = crop_2d_or_yxc(image, x0=record.x0, y0=record.y0, width=record.width, height=record.height)
        existing_points = read_existing_points(record.manual_points_path)
        points = annotate_points_interactive(
            crop,
            existing_points,
            save_path=record.manual_points_path,
            overlay_text=(
                f"ROI: {record.roi_id}\n"
                f"Split: {record.split}\n"
                f"Notes: {record.notes or 'none'}\n"
                "Border rule: one point per soma center; do not count debris or border-only fragments."
            ),
        )

        write_points_csv(record.manual_points_path, points)
        write_points_metadata(
            record.manual_points_path.with_suffix(".meta.json"),
            build_points_metadata(
                roi_id=record.roi_id,
                annotator=args.annotator,
                image_path=record.image_path,
                marker=record.marker,
                modality=record.modality,
                roi_xywh=(record.x0, record.y0, record.width, record.height),
                n_points=len(points),
            ),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
