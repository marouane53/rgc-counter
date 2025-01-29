# src/focus_detection.py

import numpy as np
import cv2

def compute_in_focus_mask(image, tile_size=64, focus_threshold=50, morph_kernel=5):
    """
    Computes a binary mask of in-focus regions using a tile-based 
    Variance of Laplacian approach.
    
    :param image: 2D numpy array (grayscale)
    :param tile_size: size of the patch/tile in pixels (e.g. 64)
    :param focus_threshold: threshold for deciding if a tile is in focus 
                            (larger = stricter, smaller = more lenient)
    :param morph_kernel: size for morphological closing (to smooth boundaries)
    :return: boolean mask (True = in-focus, False = out-of-focus)
    """
    # Ensure image is float for Laplacian calculations
    if image.dtype != np.float32:
        img_float = image.astype(np.float32)
    else:
        img_float = image
    
    height, width = img_float.shape
    # We'll create an empty mask to store tile-based decisions
    focus_mask = np.zeros((height, width), dtype=np.uint8)

    # Slide over the image in tile_size steps
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            y_end = min(y + tile_size, height)
            x_end = min(x + tile_size, width)
            
            tile = img_float[y:y_end, x:x_end]
            # Compute variance of Laplacian
            lap = cv2.Laplacian(tile, cv2.CV_32F)
            variance = lap.var()
            
            if variance >= focus_threshold:
                focus_mask[y:y_end, x:x_end] = 1

    # Morphological cleaning to ensure smooth edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
    # Close small holes
    focus_mask = cv2.morphologyEx(focus_mask, cv2.MORPH_CLOSE, kernel)
    # Optionally, you could do open as well if you want to remove small bright spots:
    # focus_mask = cv2.morphologyEx(focus_mask, cv2.MORPH_OPEN, kernel)

    # Convert to boolean
    in_focus_bool = focus_mask.astype(bool)
    return in_focus_bool
