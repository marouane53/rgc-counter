from __future__ import annotations

from pathlib import Path

import json
import numpy as np
import pandas as pd
import tifffile

from scripts.qc_roi_sources import qc_has_blockers
from src.roi_data import (
    TRUTH_PROVENANCE_STATUS_INVALID,
    TRUTH_PROVENANCE_STATUS_UNKNOWN,
    crop_2d_or_yxc,
    iter_roi_records,
    load_roi_manifest,
    qc_roi_manifest,
)


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
    assert bool(qc["reused_source_image"].all()) is True


def test_qc_reused_source_images_are_not_blocking_when_rois_do_not_overlap(tmp_path: Path):
    image = np.arange(32 * 32, dtype=np.uint16).reshape(32, 32)
    shared = tmp_path / "shared.tif"
    tifffile.imwrite(shared, image)
    manual = tmp_path / "manual.csv"
    manual.write_text("x_px,y_px\n1,1\n", encoding="utf-8")
    manifest_path = _write_manifest(
        tmp_path,
        [
            {"roi_id": "R1", "image_path": shared.name, "marker": "RBPMS", "modality": "flatmount", "x0": 0, "y0": 0, "width": 8, "height": 8, "annotator": "A", "manual_points_path": manual.name},
            {"roi_id": "R2", "image_path": shared.name, "marker": "RBPMS", "modality": "flatmount", "x0": 16, "y0": 16, "width": 8, "height": 8, "annotator": "A", "manual_points_path": manual.name},
        ],
    )

    qc = qc_roi_manifest(load_roi_manifest(manifest_path), manifest_path=manifest_path)

    assert qc["reused_source_image"].all()
    assert not qc["overlaps_with_other_roi"].any()
    assert qc_has_blockers(qc) is False


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


def test_qc_roi_manifest_flags_overlapping_rois(tmp_path: Path):
    tifffile.imwrite(tmp_path / "a.tif", np.zeros((32, 32), dtype=np.uint8))
    manual = tmp_path / "manual.csv"
    manual.write_text("x_px,y_px\n1,1\n", encoding="utf-8")
    manifest_path = _write_manifest(
        tmp_path,
        [
            {"roi_id": "R1", "image_path": "a.tif", "marker": "RBPMS", "modality": "flatmount", "x0": 0, "y0": 0, "width": 16, "height": 16, "annotator": "A", "manual_points_path": manual.name},
            {"roi_id": "R2", "image_path": "a.tif", "marker": "RBPMS", "modality": "flatmount", "x0": 8, "y0": 8, "width": 16, "height": 16, "annotator": "A", "manual_points_path": manual.name},
        ],
    )

    qc = qc_roi_manifest(load_roi_manifest(manifest_path), manifest_path=manifest_path)

    assert qc["overlaps_with_other_roi"].all()
    assert qc_has_blockers(qc) is True


def test_crop_2d_or_yxc_handles_grayscale_and_channels_last():
    gray = np.arange(100, dtype=np.uint8).reshape(10, 10)
    yxc = np.stack([gray, gray + 1], axis=-1)

    gray_crop = crop_2d_or_yxc(gray, x0=2, y0=3, width=4, height=5)
    yxc_crop = crop_2d_or_yxc(yxc, x0=2, y0=3, width=4, height=5)

    assert gray_crop.shape == (5, 4)
    assert yxc_crop.shape == (5, 4, 2)


def test_iter_roi_records_marks_cross_channel_truth_invalid(tmp_path: Path):
    image_path = tmp_path / "tile.tif"
    tifffile.imwrite(image_path, np.zeros((32, 32), dtype=np.uint8))
    image_path.with_suffix(".json").write_text(json.dumps({"channel_index": 1}), encoding="utf-8")
    manual = tmp_path / "manual.csv"
    manual.write_text("x_px,y_px\n1,1\n", encoding="utf-8")
    manual.with_suffix(".meta.json").write_text(
        json.dumps({"truth_marker": "RBPMS", "truth_source_channel": 0, "truth_derivation": "embedded_imaris_scene_spots"}),
        encoding="utf-8",
    )
    manifest_path = _write_manifest(
        tmp_path,
        [
            {
                "roi_id": "R1",
                "image_path": image_path.name,
                "marker": "RBPMS",
                "modality": "flatmount",
                "x0": 0,
                "y0": 0,
                "width": 16,
                "height": 16,
                "annotator": "A",
                "manual_points_path": manual.name,
                "split": "dev",
                "notes": "",
            }
        ],
    )

    records = iter_roi_records(load_roi_manifest(manifest_path), manifest_path=manifest_path)

    assert len(records) == 1
    assert records[0].image_source_channel == 1
    assert records[0].truth_source_channel == 0
    assert records[0].truth_provenance_valid is False
    assert records[0].truth_provenance_status == TRUTH_PROVENANCE_STATUS_INVALID


def test_iter_roi_records_leaves_legacy_truth_unknown_but_valid(tmp_path: Path):
    image_path = tmp_path / "tile.tif"
    tifffile.imwrite(image_path, np.zeros((32, 32), dtype=np.uint8))
    manual = tmp_path / "manual.csv"
    manual.write_text("x_px,y_px\n1,1\n", encoding="utf-8")
    manifest_path = _write_manifest(
        tmp_path,
        [
            {
                "roi_id": "R1",
                "image_path": image_path.name,
                "marker": "RBPMS",
                "modality": "flatmount",
                "x0": 0,
                "y0": 0,
                "width": 16,
                "height": 16,
                "annotator": "A",
                "manual_points_path": manual.name,
                "split": "dev",
                "notes": "",
            }
        ],
    )

    records = iter_roi_records(load_roi_manifest(manifest_path), manifest_path=manifest_path)

    assert records[0].truth_provenance_valid is True
    assert records[0].truth_provenance_status == TRUTH_PROVENANCE_STATUS_UNKNOWN
