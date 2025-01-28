# src/postprocessing.py

import numpy as np
import cv2
from src.config import MIN_CELL_SIZE, MAX_CELL_SIZE

def filter_masks_by_size(masks, min_size=MIN_CELL_SIZE, max_size=MAX_CELL_SIZE):
    """
    Remove segmented objects that are too small or too large.
    masks is a 2D integer array where each object has an integer ID > 0.
    """
    filtered = np.zeros_like(masks)
    object_ids = np.unique(masks)
    object_id_counter = 1
    
    for oid in object_ids:
        if oid == 0:
            continue
        mask_area = np.sum(masks == oid)
        if min_size <= mask_area <= max_size:
            filtered[masks == oid] = object_id_counter
            object_id_counter += 1
    return filtered

def postprocess_masks(masks, min_size=MIN_CELL_SIZE, max_size=MAX_CELL_SIZE):
    """
    A convenience function that filters masks by size
    or can do morphological ops if needed.
    """
    filtered_masks = filter_masks_by_size(masks, min_size, max_size)
    return filtered_masks

def apply_gaussian_blur(image, ksize=3):
    """
    Applies a mild Gaussian blur to the input image.
    :param image: 2D numpy array (grayscale)
    :param ksize: kernel size for the blur, must be odd (e.g. 3, 5)
    :return: blurred image
    """
    # Ensure image is uint8 or float for OpenCV
    if image.dtype != np.uint8:
        img_norm = image.astype(np.float32)
        img_norm = img_norm / img_norm.max() * 255.0
        img_uint8 = img_norm.astype(np.uint8)
    else:
        img_uint8 = image

    blurred = cv2.GaussianBlur(img_uint8, (ksize, ksize), 0)
    return blurred

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8,8)):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance contrast.

    :param image: 2D numpy array (grayscale)
    :param clip_limit: Threshold for contrast limiting
    :param tile_grid_size: Size of grid for histogram equalization
    :return: Contrast-enhanced image (uint8)
    """
    # Ensure the image is uint8
    if image.dtype != np.uint8:
        img_norm = image.astype(np.float32)
        img_norm = img_norm / img_norm.max() * 255.0
        image_uint8 = img_norm.astype(np.uint8)
    else:
        image_uint8 = image
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(image_uint8)
    return enhanced
