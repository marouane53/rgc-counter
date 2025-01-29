# src/focus_detection.py

import numpy as np
import cv2

def compute_in_focus_mask_auto(image,
                               tile_size=64,
                               focus_threshold=50,
                               brightness_min=20,
                               brightness_max=230,
                               morph_kernel=5):
    """
    Computes a binary mask of in-focus regions using:
      1) Brightness filtering
      2) Variance of Laplacian
      3) Morphological cleanup
      
    :param image: 2D numpy array (grayscale)
    :param tile_size: tile size in pixels
    :param focus_threshold: min Laplacian variance to consider tile in focus
    :param brightness_min: min average intensity for tile (exclude if below)
    :param brightness_max: max average intensity for tile (exclude if above)
    :param morph_kernel: size for morphological closing
    :return: boolean mask (True = in-focus, False = out-of-focus)
    """
    if image.dtype != np.float32:
        img_float = image.astype(np.float32)
    else:
        img_float = image
    
    height, width = img_float.shape
    focus_mask = np.zeros((height, width), dtype=np.uint8)

    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            y_end = min(y + tile_size, height)
            x_end = min(x + tile_size, width)
            
            tile = img_float[y:y_end, x:x_end]

            # 1) Brightness check
            tile_mean = tile.mean()
            if tile_mean < brightness_min or tile_mean > brightness_max:
                continue  # skip marking as in-focus

            # 2) Sharpness check via Laplacian variance
            lap = cv2.Laplacian(tile, cv2.CV_32F)
            variance = lap.var()
            
            if variance >= focus_threshold:
                focus_mask[y:y_end, x:x_end] = 1

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
    focus_mask = cv2.morphologyEx(focus_mask, cv2.MORPH_CLOSE, kernel)

    return focus_mask.astype(bool)
