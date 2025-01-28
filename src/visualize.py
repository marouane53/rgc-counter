# src/visualize.py

import numpy as np
import cv2
import matplotlib.pyplot as plt
from src.config import OVERLAY_ALPHA

def create_debug_overlay(image, masks, alpha=OVERLAY_ALPHA):
    """
    Draws colored labels on top of the grayscale image for verification.
    Returns an RGB image.
    """
    # Make sure image is 0-255 range for visualization
    # If it's floating or >255, rescale
    if image.dtype != np.uint8:
        disp_img = (image / image.max() * 255).astype(np.uint8)
    else:
        disp_img = image.copy()
    
    # Convert grayscale to RGB
    disp_img = cv2.cvtColor(disp_img, cv2.COLOR_GRAY2RGB)
    
    # Generate random colors for each mask ID
    unique_ids = np.unique(masks)
    unique_ids = unique_ids[unique_ids != 0]
    
    color_map = {}
    np.random.seed(42)  # for reproducible colors
    for oid in unique_ids:
        color_map[oid] = (
            np.random.randint(0, 255),
            np.random.randint(0, 255),
            np.random.randint(0, 255),
        )
    
    overlay = disp_img.copy()
    
    for oid in unique_ids:
        overlay[masks == oid] = color_map[oid]
    
    # Blend
    debug_image = cv2.addWeighted(overlay, alpha, disp_img, 1 - alpha, 0)
    return debug_image

def save_debug_image(debug_image, output_path):
    """
    Save the debug overlay as a PNG or JPEG.
    """
    cv2.imwrite(output_path, debug_image[..., ::-1])  # cv2 uses BGR by default, so flip to save in correct color
