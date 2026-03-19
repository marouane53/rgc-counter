from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from src.ims_io import (
    compute_voxel_sizes_um,
    extract_scene_spot_points,
    extract_standard_ims_metadata,
    find_candidate_rbpms_channels,
    inspect_ims_file,
    stream_channel_crop,
)


def _write_ims_fixture(path: Path) -> Path:
    with h5py.File(path, "w") as h5:
        dataset_info = h5.create_group("DataSetInfo")
        image = dataset_info.create_group("Image")
        image.attrs.update(
            {
                "X": "5",
                "Y": "4",
                "Z": "3",
                "Noc": "2",
                "Unit": "um",
                "ExtMin0": "0",
                "ExtMax0": "10",
                "ExtMin1": "0",
                "ExtMax1": "8",
                "ExtMin2": "0",
                "ExtMax2": "6",
                "LensPower": "20",
                "RecordingDate": "2025-02-17",
            }
        )
        channel0 = dataset_info.create_group("Channel 0")
        channel0.attrs.update({"Name": "DAPI", "Description": "nuclei", "Color": "blue"})
        channel1 = dataset_info.create_group("Channel 1")
        channel1.attrs.update(
            {
                "Name": "RBPMS Alexa647",
                "Description": "RBPMS ganglion cells",
                "Color": "far red",
                "LSMExcitationWavelength": "647",
                "LSMEmissionWavelength": "668",
            }
        )

        dataset = h5.create_group("DataSet")
        resolution = dataset.create_group("ResolutionLevel 0")
        timepoint = resolution.create_group("TimePoint 0")
        channel0_group = timepoint.create_group("Channel 0")
        channel0_group.create_dataset("Data", data=np.arange(3 * 4 * 5, dtype=np.uint16).reshape(3, 4, 5))
        channel0_group.create_dataset("Histogram", data=np.arange(8, dtype=np.uint16))
        channel1_group = timepoint.create_group("Channel 1")
        channel1_group.create_dataset("Data", data=np.arange(3 * 4 * 5, dtype=np.uint16).reshape(3, 4, 5) + 100)

        thumbnail = h5.create_group("Thumbnail")
        thumbnail.create_dataset("Data", data=np.zeros((8, 8, 3), dtype=np.uint8))

        scene = h5.create_group("Scene8")
        spots = scene.create_group("Spots")
        spots.attrs["Name"] = "Manual Spots"
        spots.create_dataset("Position", data=np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
    return path


def test_compute_voxel_sizes_um_from_extents():
    voxel = compute_voxel_sizes_um(
        {
            "Unit": "um",
            "ExtMin0": "0",
            "ExtMax0": "10",
            "ExtMin1": "0",
            "ExtMax1": "8",
            "ExtMin2": "0",
            "ExtMax2": "6",
            "X": "5",
            "Y": "4",
            "Z": "3",
        }
    )

    assert voxel["voxel_x_um"] == 2.0
    assert voxel["voxel_y_um"] == 2.0
    assert voxel["voxel_z_um"] == 2.0


def test_inspect_ims_file_extracts_metadata_and_scene_candidates(tmp_path: Path):
    path = _write_ims_fixture(tmp_path / "sample.ims")

    payload = inspect_ims_file(path)

    assert payload["metadata"]["n_channels"] == 2
    assert payload["thumbnail_available"] is True
    assert payload["scene"]["has_annotation_like_objects"] is True
    assert payload["scene"]["annotation_candidates"]
    assert payload["channels"][1]["name"] == "RBPMS Alexa647"


def test_candidate_ranking_prefers_explicit_rbpms():
    ranked = find_candidate_rbpms_channels(
        [
            {"channel_index": 0, "metadata_text": "dapi nuclei", "name": "DAPI", "description": "", "lsm_excitation_wavelength": "", "lsm_emission_wavelength": ""},
            {"channel_index": 1, "metadata_text": "rbpms alexa 647 far red", "name": "RBPMS", "description": "", "lsm_excitation_wavelength": "647", "lsm_emission_wavelength": "668"},
        ]
    )

    assert ranked[0]["channel_index"] == 1
    assert ranked[0]["explicit_rbpms"] is True
    assert ranked[0]["is_unambiguous_top_candidate"] is True


def test_extract_standard_ims_metadata_includes_voxel_sizes(tmp_path: Path):
    path = _write_ims_fixture(tmp_path / "sample.ims")
    with h5py.File(path, "r") as h5:
        metadata = extract_standard_ims_metadata(h5)

    assert metadata["image"]["RecordingDate"] == "2025-02-17"
    assert metadata["voxel_sizes_um"]["voxel_x_um"] == 2.0


def test_extract_standard_ims_metadata_decodes_char_array_attrs(tmp_path: Path):
    path = tmp_path / "char-array.ims"
    with h5py.File(path, "w") as h5:
        dataset_info = h5.create_group("DataSetInfo")
        image = dataset_info.create_group("Image")
        image.attrs.update(
            {
                "X": np.array(list("1024"), dtype="S1"),
                "Y": np.array(list("1024"), dtype="S1"),
                "Z": np.array(list("535"), dtype="S1"),
                "Unit": np.array(list("um"), dtype="S1"),
                "ExtMin0": np.array(list("0"), dtype="S1"),
                "ExtMax0": np.array(list("775.758"), dtype="S1"),
                "ExtMin1": np.array(list("0"), dtype="S1"),
                "ExtMax1": np.array(list("775.758"), dtype="S1"),
                "ExtMin2": np.array(list("25.4147"), dtype="S1"),
                "ExtMax2": np.array(list("92.7633"), dtype="S1"),
                "RecordingDate": np.array(list("2025-01-29 19:51:45.365"), dtype="S1"),
                "LensPower": np.array(list("20"), dtype="S1"),
            }
        )

        channel0 = dataset_info.create_group("Channel 0")
        channel0.attrs["Name"] = np.array(list("Magenta"), dtype="S1")

    with h5py.File(path, "r") as h5:
        metadata = extract_standard_ims_metadata(h5)

    assert metadata["image"]["RecordingDate"] == "2025-01-29 19:51:45.365"
    assert metadata["image"]["LensPower"] == "20"
    assert metadata["image"]["Unit"] == "um"
    assert metadata["voxel_sizes_um"]["voxel_x_um"] == 775.758 / 1024
    assert metadata["voxel_sizes_um"]["voxel_z_um"] == (92.7633 - 25.4147) / 535


def test_extract_scene_spot_points_prefers_structured_spot_dataset(tmp_path: Path):
    path = tmp_path / "spots.ims"
    with h5py.File(path, "w") as h5:
        dataset_info = h5.create_group("DataSetInfo")
        image = dataset_info.create_group("Image")
        image.attrs.update({"X": "10", "Y": "10", "Z": "5", "Unit": "um", "ExtMin0": "0", "ExtMax0": "10", "ExtMin1": "0", "ExtMax1": "10", "ExtMin2": "0", "ExtMax2": "5"})
        scene = h5.create_group("Scene8")
        content = scene.create_group("Content")
        points = content.create_group("Points0")
        points.create_dataset(
            "Spot",
            data=np.array(
                [
                    (1, 1.0, 2.0, 3.0, 0.5),
                    (2, 4.0, 5.0, 1.0, 0.75),
                ],
                dtype=[("ID", "<i8"), ("PositionX", "<f4"), ("PositionY", "<f4"), ("PositionZ", "<f4"), ("Radius", "<f4")],
            ),
        )
        points.create_dataset("CreationParameters", data=np.array([b'<bpPointsCreationParameters mSourceChannelIndex="1"/>']))

    with h5py.File(path, "r") as h5:
        payload = extract_scene_spot_points(h5)

    assert payload is not None
    assert payload["dataset_path"] == "Scene8/Content/Points0/Spot"
    assert payload["point_ids"].tolist() == [1, 2]
    assert payload["xyzr_um"].shape == (2, 4)
    assert 'mSourceChannelIndex="1"' in payload["creation_parameters"]


def test_stream_channel_crop_reads_raw_zyx_subvolume(tmp_path: Path):
    path = _write_ims_fixture(tmp_path / "sample.ims")

    crop = stream_channel_crop(path, channel_index=1, x0=1, y0=1, width=3, height=2, z_start=1, z_end=3)

    assert crop.shape == (2, 2, 3)
    assert crop[0, 0, 0] == 126
    assert crop[-1, -1, -1] == 153
