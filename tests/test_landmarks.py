import numpy as np

from src.landmarks import build_tissue_mask, detect_onh_hole


def test_detect_onh_hole_finds_central_void():
    yy, xx = np.mgrid[:128, :128]
    center = np.array([64, 64])
    dist = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    image = np.zeros((128, 128), dtype=np.uint8)
    image[(dist <= 45) & (dist >= 10)] = 220

    onh_xy, info = detect_onh_hole(image)

    assert onh_xy is not None
    assert abs(onh_xy[0] - 64.0) < 3.0
    assert abs(onh_xy[1] - 64.0) < 3.0
    assert info["confidence"] > 0.0


def test_build_tissue_mask_returns_nonempty_mask_for_bright_disc():
    yy, xx = np.mgrid[:64, :64]
    image = np.zeros((64, 64), dtype=np.uint8)
    image[((xx - 32) ** 2 + (yy - 32) ** 2) <= 20 ** 2] = 200

    mask = build_tissue_mask(image)

    assert mask.dtype == bool
    assert mask.sum() > 0
