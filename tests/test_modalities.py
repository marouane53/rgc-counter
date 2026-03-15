import numpy as np

from src.modalities import adapt_image_for_modality


def test_oct_adapter_projects_volume_to_2d():
    image = np.zeros((4, 6, 6), dtype=np.uint16)
    image[2, 1:3, 1:3] = 10
    adapted, meta = adapt_image_for_modality(image, {}, modality="oct", projection="max")

    assert adapted.shape == (6, 6)
    assert adapted.max() == 10
    assert meta["modality_adapter"] == "oct"


def test_vis_octf_adapter_projects_depth_and_keeps_channels():
    image = np.zeros((3, 5, 5, 2), dtype=np.uint16)
    image[1, 2:4, 2:4, 0] = 7
    image[2, 1:3, 1:3, 1] = 9

    adapted, meta = adapt_image_for_modality(image, {}, modality="vis_octf", projection="max")

    assert adapted.shape == (5, 5, 2)
    assert adapted[..., 0].max() == 7
    assert adapted[..., 1].max() == 9
    assert meta["modality_adapter"] == "vis_octf"


def test_lightsheet_adapter_projects_volume():
    image = np.zeros((5, 8, 8), dtype=np.uint16)
    image[3, 4:6, 4:6] = 12

    adapted, meta = adapt_image_for_modality(image, {}, modality="lightsheet", projection="max")

    assert adapted.shape == (8, 8)
    assert adapted.max() == 12
    assert meta["modality_adapter"] == "lightsheet"
