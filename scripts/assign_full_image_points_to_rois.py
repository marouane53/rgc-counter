from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.roi_data import iter_roi_records, load_roi_manifest


def _load_points_sidecar(path: Path) -> dict:
    candidates = [path.with_suffix(".json"), Path(str(path) + ".json")]
    for candidate in candidates:
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return {}


def assign_points_to_rois(*, roi_manifest: Path, full_image_points_csv: Path, image_path: Path | None = None) -> pd.DataFrame:
    manifest = load_roi_manifest(roi_manifest)
    records = iter_roi_records(manifest, manifest_path=roi_manifest)
    points = pd.read_csv(full_image_points_csv)
    sidecar = _load_points_sidecar(full_image_points_csv)
    required = {"x_px", "y_px"}
    missing = sorted(required - set(points.columns))
    if missing:
        raise ValueError(f"Full-image points CSV is missing required columns: {missing}")

    selected_image_path = image_path.resolve() if image_path is not None else None
    rows: list[dict[str, object]] = []
    for record in records:
        if selected_image_path is not None and record.image_path.resolve() != selected_image_path:
            continue
        if record.manual_points_path is None:
            continue
        x1 = record.x0 + record.width
        y1 = record.y0 + record.height
        inside = points.loc[
            (points["x_px"] >= float(record.x0))
            & (points["x_px"] < float(x1))
            & (points["y_px"] >= float(record.y0))
            & (points["y_px"] < float(y1))
        ].copy()
        if inside.empty:
            local = pd.DataFrame(columns=["x_px", "y_px"])
        else:
            local = pd.DataFrame(
                {
                    "x_px": inside["x_px"] - float(record.x0),
                    "y_px": inside["y_px"] - float(record.y0),
                }
            )
        record.manual_points_path.parent.mkdir(parents=True, exist_ok=True)
        local.to_csv(record.manual_points_path, index=False)
        meta_path = record.manual_points_path.with_suffix(".meta.json")
        meta_payload = {
            "roi_id": record.roi_id,
            "marker": record.marker,
            "truth_marker": sidecar.get("truth_marker"),
            "truth_source_channel": sidecar.get("truth_source_channel", sidecar.get("source_channel_index")),
            "truth_derivation": sidecar.get("truth_derivation", "full_image_points_subset"),
            "truth_source_path": sidecar.get("truth_source_path", str(full_image_points_csv.resolve())),
            "source_points_csv": str(full_image_points_csv.resolve()),
            "source_image_path": str(record.image_path),
            "roi_xywh": [int(record.x0), int(record.y0), int(record.width), int(record.height)],
            "n_points": int(len(local)),
            "tool": "scripts/assign_full_image_points_to_rois.py",
        }
        meta_path.write_text(json.dumps(meta_payload, indent=2) + "\n", encoding="utf-8")
        rows.append(
            {
                "roi_id": record.roi_id,
                "image_path": str(record.image_path),
                "manual_points_path": str(record.manual_points_path),
                "manual_points_meta_path": str(meta_path),
                "n_points_written": int(len(local)),
            }
        )
    return pd.DataFrame(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert full-image point truth into ROI-local point CSVs.")
    parser.add_argument("--roi-manifest", "--manifest", dest="roi_manifest", required=True, type=Path)
    parser.add_argument("--full-image-points-csv", required=True, type=Path)
    parser.add_argument("--image-path", default=None, type=Path)
    parser.add_argument("--summary-csv", default=None, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = assign_points_to_rois(
        roi_manifest=args.roi_manifest.resolve(),
        full_image_points_csv=args.full_image_points_csv.resolve(),
        image_path=args.image_path.resolve() if args.image_path is not None else None,
    )
    if args.summary_csv is not None:
        summary_path = args.summary_csv.resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path, index=False)
    print(
        f"[assign_full_image_points_to_rois] wrote ROI-local points for {len(summary)} ROI(s)",
        file=sys.stderr,
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
