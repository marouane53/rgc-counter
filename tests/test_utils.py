import numpy as np

from src.utils import ensure_grayscale


def test_ensure_grayscale_supports_yx():
    image = np.arange(36, dtype=np.uint8).reshape(6, 6)

    out = ensure_grayscale(image)

    assert out.shape == (6, 6)
    assert np.array_equal(out, image)


def test_ensure_grayscale_supports_yxc():
    image = np.zeros((6, 6, 2), dtype=np.uint8)
    image[..., 1] = 7

    out = ensure_grayscale(image, channel_index=1)

    assert out.shape == (6, 6)
    assert np.all(out == 7)


def test_ensure_grayscale_supports_cyx():
    image = np.zeros((2, 6, 6), dtype=np.uint8)
    image[1, ...] = 5

    out = ensure_grayscale(image, channel_index=1)

    assert out.shape == (6, 6)
    assert np.all(out == 5)


def test_ensure_grayscale_supports_zyx():
    image = np.zeros((5, 6, 6), dtype=np.uint8)
    image[4, ...] = 9

    out = ensure_grayscale(image)

    assert out.shape == (6, 6)
    assert np.all(out == 9)


def test_ensure_grayscale_supports_zyxc():
    image = np.zeros((5, 6, 6, 2), dtype=np.uint8)
    image[4, ..., 1] = 11

    out = ensure_grayscale(image, channel_index=1)

    assert out.shape == (6, 6)
    assert np.all(out == 11)


def test_ensure_grayscale_supports_czyx():
    image = np.zeros((2, 5, 6, 6), dtype=np.uint8)
    image[1, 4, ...] = 13

    out = ensure_grayscale(image, channel_index=1)

    assert out.shape == (6, 6)
    assert np.all(out == 13)


def test_ensure_grayscale_squeezes_singleton_four_dimensional_input():
    image = np.zeros((1, 6, 6, 1), dtype=np.uint8)
    image[0, 1:3, 1:3, 0] = 17

    out = ensure_grayscale(image)

    assert out.shape == (6, 6)
    assert np.max(out) == 17
