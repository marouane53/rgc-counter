import numpy as np

from src.uncertainty_io import read_float_map, save_float_map


def test_save_and_read_float_map_round_trip(tmp_path):
    array = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4)
    path = save_float_map(array, tmp_path / "fgprob.tif", metadata={"kind": "foreground_probability"})

    loaded, metadata = read_float_map(path)

    assert loaded.dtype == np.float32
    assert loaded.shape == (4, 4)
    assert np.allclose(loaded, array)
    assert "ImageDescription" in metadata
