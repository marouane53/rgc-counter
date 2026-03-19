from __future__ import annotations

import json
from pathlib import Path

import h5py
import pandas as pd

from scripts.export_ims_scene_points import export_scene_points, main as export_main
from src.ims_io import inspect_ims_file


def _write_scene_points_fixture(path: Path) -> Path:
    with h5py.File(path, "w") as h5:
        dataset_info = h5.create_group("DataSetInfo")
        image = dataset_info.create_group("Image")
        image.attrs.update(
            {
                "X": "10",
                "Y": "10",
                "Z": "5",
                "Unit": "um",
                "ExtMin0": "0",
                "ExtMax0": "10",
                "ExtMin1": "0",
                "ExtMax1": "10",
                "ExtMin2": "0",
                "ExtMax2": "5",
            }
        )
        scene = h5.create_group("Scene8")
        content = scene.create_group("Content")
        points = content.create_group("Points0")
        points.create_dataset(
            "Spot",
            data=[
                (1, 1.0, 2.0, 3.0, 0.5),
                (2, 4.5, 5.5, 1.5, 0.5),
            ],
            dtype=[("ID", "<i8"), ("PositionX", "<f4"), ("PositionY", "<f4"), ("PositionZ", "<f4"), ("Radius", "<f4")],
        )
        points.create_dataset("CreationParameters", data=[b'<bpPointsCreationParameters mSourceChannelIndex="1"/>'])
    return path


def test_export_scene_points_converts_to_pixel_coordinates(tmp_path: Path):
    ims_path = _write_scene_points_fixture(tmp_path / "fixture.ims")
    payload = inspect_ims_file(ims_path)
    metadata_json = tmp_path / "metadata.json"
    metadata_json.write_text(json.dumps(payload), encoding="utf-8")

    frame, summary = export_scene_points(ims_path=ims_path, metadata_json=metadata_json)

    assert summary["source_channel_index"] == 1
    assert summary["truth_source_channel"] == 1
    assert summary["truth_derivation"] == "embedded_imaris_scene_spots"
    assert len(frame) == 2
    assert frame.loc[0, "x_px"] == 1.0
    assert frame.loc[0, "y_px"] == 2.0
    assert bool(frame.loc[0, "inside_image_bounds"]) is True


def test_export_scene_points_script_writes_outputs(tmp_path: Path):
    ims_path = _write_scene_points_fixture(tmp_path / "fixture.ims")
    payload = inspect_ims_file(ims_path)
    metadata_json = tmp_path / "metadata.json"
    metadata_json.write_text(json.dumps(payload), encoding="utf-8")
    output_csv = tmp_path / "points.csv"

    exit_code = export_main(["--ims-path", str(ims_path), "--metadata-json", str(metadata_json), "--output-csv", str(output_csv)])

    assert exit_code == 0
    frame = pd.read_csv(output_csv)
    assert list(frame.columns[:5]) == ["point_id", "x_um", "y_um", "z_um", "radius_um"]
    assert output_csv.with_suffix(".json").exists()
