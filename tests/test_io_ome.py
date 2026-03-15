import numpy as np
from PIL import Image

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
