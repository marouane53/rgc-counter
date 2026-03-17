import numpy as np
from PIL import Image
import tifffile

from src.io_ome import load_any_image


def test_load_any_image_supports_jpeg_fallback(tmp_path):
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[4:12, 4:12, 0] = 255

    path = tmp_path / "sample.jpg"
    Image.fromarray(image).save(path)

    loaded, meta = load_any_image(str(path))

    assert loaded.shape == image.shape
    assert loaded.dtype == np.uint8
    assert meta["reader"] == "pillow"
    assert meta["reader_raw_shape"] == [16, 16, 3]
    assert meta["canonical_loaded_shape"] == [16, 16, 3]


def test_load_any_image_squeezes_singleton_tiff_dimensions(tmp_path):
    image = np.zeros((1, 16, 16, 1), dtype=np.uint8)
    image[0, 4:12, 4:12, 0] = 255

    path = tmp_path / "sample.ome.tif"
    tifffile.imwrite(path, image)

    loaded, meta = load_any_image(str(path))

    assert loaded.shape == (16, 16)
    assert meta["canonical_loaded_shape"] == [16, 16]
    if meta["reader"] == "aicsimageio":
        assert meta["aics_requested_dims"] == "CZYX"
        assert meta["aics_raw_shape"] == [1, 1, 16, 16]
