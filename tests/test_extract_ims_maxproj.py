from __future__ import annotations

import json
from pathlib import Path

import h5py
import pandas as pd
import tifffile

from scripts.extract_ims_maxproj import export_preview_png, main as extract_main, stream_max_projection
from src.ims_io import inspect_ims_file


def _write_projection_fixture(path: Path) -> Path:
    with h5py.File(path, "w") as h5:
        dataset_info = h5.create_group("DataSetInfo")
        image = dataset_info.create_group("Image")
        image.attrs.update({"X": "4", "Y": "3", "Z": "2", "Noc": "2", "Unit": "um", "ExtMin0": "0", "ExtMax0": "4", "ExtMin1": "0", "ExtMax1": "3", "ExtMin2": "0", "ExtMax2": "2"})
        channel0 = dataset_info.create_group("Channel 0")
        channel0.attrs.update({"Name": "DAPI", "Description": "nuclei"})
        channel1 = dataset_info.create_group("Channel 1")
        channel1.attrs.update({"Name": "RBPMS Cy5", "Description": "RBPMS", "LSMExcitationWavelength": "647", "LSMEmissionWavelength": "668"})

        dataset = h5.create_group("DataSet")
        resolution = dataset.create_group("ResolutionLevel 0")
        timepoint = resolution.create_group("TimePoint 0")
        channel0_group = timepoint.create_group("Channel 0")
        channel0_group.create_dataset("Data", data=[[[1, 2, 3, 4], [4, 3, 2, 1], [0, 1, 0, 1]], [[5, 6, 7, 8], [8, 7, 6, 5], [1, 2, 1, 2]]])
        channel1_group = timepoint.create_group("Channel 1")
        channel1_group.create_dataset("Data", data=[[[10, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], [[2, 20, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]]])
    return path


def test_stream_max_projection_matches_expected_output(tmp_path: Path):
    path = _write_projection_fixture(tmp_path / "fixture.ims")

    projected = stream_max_projection(path, channel_index=0)

    assert projected.tolist() == [[5, 6, 7, 8], [8, 7, 6, 5], [1, 2, 1, 2]]


def test_extract_script_writes_selected_manifest_and_sidecars(tmp_path: Path):
    ims_path = _write_projection_fixture(tmp_path / "fixture.ims")
    payload = inspect_ims_file(ims_path)
    metadata_json = tmp_path / "metadata.json"
    metadata_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    inventory = tmp_path / "ims_inventory.csv"
    pd.DataFrame(
        [
            {
                "source_ims_path": str(ims_path),
                "metadata_json_path": str(metadata_json),
            }
        ]
    ).to_csv(inventory, index=False)

    output_dir = tmp_path / "projected"
    exit_code = extract_main(["--inventory", str(inventory), "--channel", "auto", "--output-dir", str(output_dir)])

    assert exit_code == 0
    manifest = pd.read_csv(output_dir / "selected" / "projected_manifest.csv")
    assert len(manifest) == 1
    projected_tiff = Path(str(manifest.loc[0, "projected_tiff_path"]))
    preview_png = Path(str(manifest.loc[0, "preview_png_path"]))
    assert projected_tiff.exists()
    assert preview_png.exists()
    assert projected_tiff.with_suffix(".json").exists()
    projected = tifffile.imread(projected_tiff)
    assert projected.tolist() == [[10, 20, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]]


def test_export_preview_png_writes_file(tmp_path: Path):
    destination = tmp_path / "preview.png"
    export_preview_png([[0, 1], [2, 3]], destination)
    assert destination.exists()


def test_extract_script_skips_metadata_only_channels(tmp_path: Path):
    ims_path = tmp_path / "fixture.ims"
    with h5py.File(ims_path, "w") as h5:
        dataset_info = h5.create_group("DataSetInfo")
        image = dataset_info.create_group("Image")
        image.attrs.update({"X": "4", "Y": "3", "Z": "2", "Noc": "3", "Unit": "um", "ExtMin0": "0", "ExtMax0": "4", "ExtMin1": "0", "ExtMax1": "3", "ExtMin2": "0", "ExtMax2": "2"})
        for index, name in enumerate(("DAPI", "RBPMS Cy5", "Metadata Only")):
            channel = dataset_info.create_group(f"Channel {index}")
            channel.attrs.update({"Name": name})

        dataset = h5.create_group("DataSet")
        resolution = dataset.create_group("ResolutionLevel 0")
        timepoint = resolution.create_group("TimePoint 0")
        for index in (0, 1):
            channel_group = timepoint.create_group(f"Channel {index}")
            channel_group.create_dataset("Data", data=[[[1, 2, 3, 4], [4, 3, 2, 1], [0, 1, 0, 1]], [[5, 6, 7, 8], [8, 7, 6, 5], [1, 2, 1, 2]]])

    payload = inspect_ims_file(ims_path)
    metadata_json = tmp_path / "metadata.json"
    metadata_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    inventory = tmp_path / "ims_inventory.csv"
    pd.DataFrame([{"source_ims_path": str(ims_path), "metadata_json_path": str(metadata_json)}]).to_csv(inventory, index=False)

    output_dir = tmp_path / "projected"
    exit_code = extract_main(["--inventory", str(inventory), "--channel", "all", "--output-dir", str(output_dir)])

    assert exit_code == 0
    manifest = pd.read_csv(output_dir / "all_channels" / "projected_manifest.csv")
    assert manifest["channel_index"].tolist() == [0, 1]


def test_extract_script_emits_progress_updates(tmp_path: Path, capsys):
    ims_path = _write_projection_fixture(tmp_path / "fixture.ims")
    payload = inspect_ims_file(ims_path)
    metadata_json = tmp_path / "metadata.json"
    metadata_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    inventory = tmp_path / "ims_inventory.csv"
    pd.DataFrame([{"source_ims_path": str(ims_path), "metadata_json_path": str(metadata_json)}]).to_csv(inventory, index=False)

    output_dir = tmp_path / "projected"
    exit_code = extract_main(["--inventory", str(inventory), "--channel", "all", "--output-dir", str(output_dir)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Starting extraction for 1 image(s)" in captured.err
    assert "extracting 2 channel(s)" in captured.err
    assert "Finished: wrote 2 projection(s)" in captured.err
