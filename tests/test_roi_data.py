from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from src.roi_data import crop_2d_or_yxc, load_roi_manifest, qc_roi_manifest


def _write_manifest(root: Path, rows: list[dict[str, object]]) -> Path:
    manifest = root / "roi_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest


def test_qc_roi_manifest_flags_duplicate_file_hashes(tmp_path: Path):
    image = np.zeros((16, 16), dtype=np.uint8)
    shared = tmp_path / "shared.tif"
    tifffile.imwrite(shared, image)
    manual = tmp_path / "manual.csv"
    manual.write_text("x_px,y_px\n1,1\n", encoding="utf-8")
    manifest_path = _write_manifest(
        tmp_path,
        [
            {"roi_id": "R1", "image_path": shared.name, "marker": "RBPMS", "modality": "flatmount", "x0": 0, "y0": 0, "width": 8, "height": 8, "annotator": "A", "manual_points_path": manual.name},
            {"roi_id": "R2", "image_path": shared.name, "marker": "RBPMS", "modality": "flatmount", "x0": 8, "y0": 8, "width": 8, "height": 8, "annotator": "A", "manual_points_path": manual.name},
        ],
    )

    qc = qc_roi_manifest(load_roi_manifest(manifest_path), manifest_path=manifest_path)

    assert qc["duplicate_image"].all()


def test_qc_roi_manifest_flags_duplicate_crops(tmp_path: Path):
    image_a = np.zeros((16, 16), dtype=np.uint8)
    image_b = np.zeros((16, 16), dtype=np.uint8)
    tifffile.imwrite(tmp_path / "a.tif", image_a)
    tifffile.imwrite(tmp_path / "b.tif", image_b)
    manual = tmp_path / "manual.csv"
    manual.write_text("x_px,y_px\n1,1\n", encoding="utf-8")
    manifest_path = _write_manifest(
        tmp_path,
        [
            {"roi_id": "R1", "image_path": "a.tif", "marker": "RBPMS", "modality": "flatmount", "x0": 0, "y0": 0, "width": 8, "height": 8, "annotator": "A", "manual_points_path": manual.name},
            {"roi_id": "R2", "image_path": "b.tif", "marker": "RBPMS", "modality": "flatmount", "x0": 0, "y0": 0, "width": 8, "height": 8, "annotator": "A", "manual_points_path": manual.name},
        ],
    )

    qc = qc_roi_manifest(load_roi_manifest(manifest_path), manifest_path=manifest_path)

    assert qc["duplicate_crop"].all()


def test_qc_roi_manifest_flags_out_of_bounds_roi(tmp_path: Path):
    tifffile.imwrite(tmp_path / "a.tif", np.zeros((16, 16), dtype=np.uint8))
    manual = tmp_path / "manual.csv"
    manual.write_text("x_px,y_px\n1,1\n", encoding="utf-8")
    manifest_path = _write_manifest(
        tmp_path,
        [
            {"roi_id": "R1", "image_path": "a.tif", "marker": "RBPMS", "modality": "flatmount", "x0": 10, "y0": 10, "width": 16, "height": 16, "annotator": "A", "manual_points_path": manual.name},
        ],
    )

    qc = qc_roi_manifest(load_roi_manifest(manifest_path), manifest_path=manifest_path)

    assert bool(qc.loc[0, "bounds_ok"]) is False


def test_crop_2d_or_yxc_handles_grayscale_and_channels_last():
    gray = np.arange(100, dtype=np.uint8).reshape(10, 10)
    yxc = np.stack([gray, gray + 1], axis=-1)

    gray_crop = crop_2d_or_yxc(gray, x0=2, y0=3, width=4, height=5)
    yxc_crop = crop_2d_or_yxc(yxc, x0=2, y0=3, width=4, height=5)

    assert gray_crop.shape == (5, 4)
    assert yxc_crop.shape == (5, 4, 2)
