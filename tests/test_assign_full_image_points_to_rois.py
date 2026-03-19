from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import tifffile

from scripts.assign_full_image_points_to_rois import assign_points_to_rois


def test_assign_points_to_rois_writes_roi_local_csvs(tmp_path: Path):
    image_path = tmp_path / "tile.tif"
    tifffile.imwrite(image_path, [[0] * 16 for _ in range(16)])

    roi_manifest = tmp_path / "roi_manifest.csv"
    pd.DataFrame(
        [
            {
                "roi_id": "roi_1",
                "image_path": str(image_path),
                "marker": "RBPMS",
                "modality": "flatmount",
                "x0": 4,
                "y0": 5,
                "width": 4,
                "height": 4,
                "annotator": "tester",
                "manual_points_path": "manual_points/roi_1.csv",
                "split": "dev",
                "notes": "",
            }
        ]
    ).to_csv(roi_manifest, index=False)

    points_csv = tmp_path / "points.csv"
    pd.DataFrame([{"x_px": 5.5, "y_px": 6.5}, {"x_px": 20.0, "y_px": 20.0}]).to_csv(points_csv, index=False)
    points_csv.with_suffix(".json").write_text(
        json.dumps(
            {
                "truth_source_channel": 0,
                "truth_derivation": "embedded_imaris_scene_spots",
                "truth_source_path": "/tmp/origin.ims",
            }
        ),
        encoding="utf-8",
    )

    summary = assign_points_to_rois(roi_manifest=roi_manifest, full_image_points_csv=points_csv)

    assert len(summary) == 1
    local = pd.read_csv(tmp_path / "manual_points" / "roi_1.csv")
    assert local.to_dict("records") == [{"x_px": 1.5, "y_px": 1.5}]
    meta = json.loads((tmp_path / "manual_points" / "roi_1.meta.json").read_text(encoding="utf-8"))
    assert meta["truth_source_channel"] == 0
    assert meta["truth_derivation"] == "embedded_imaris_scene_spots"
