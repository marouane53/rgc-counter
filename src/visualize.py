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

def apply_out_of_focus_overlay(image_rgb, in_focus_mask, alpha=0.3, color=(128,128,128)):
    """
    Overlays a semi-transparent color on the OUT-of-focus region (i.e. where in_focus_mask is False).
    image_rgb: an RGB image (uint8)
    in_focus_mask: boolean mask, True for in-focus, False for out-of-focus
    alpha: blending factor for the overlay
    color: BGR or RGB color for out-of-focus region (as a tuple)
    """
    # Make a copy to avoid modifying original
    output = image_rgb.copy()

    # We want to highlight the out-of-focus region, i.e. ~in_focus_mask
    h, w, _ = output.shape
    out_of_focus_mask = ~in_focus_mask
    
    # Create a solid color image
    overlay_color = np.zeros((h, w, 3), dtype=np.uint8)
    overlay_color[:] = color  # fill with chosen color (assume it's RGB)
    
    # Blend only where out_of_focus_mask is True
    # We'll do: output = alpha*overlay_color + (1-alpha)*output, but only in out_of_focus_mask
    overlay_indices = np.where(out_of_focus_mask)
    output[overlay_indices] = (
        alpha * overlay_color[overlay_indices] + 
        (1 - alpha) * output[overlay_indices]
    ).astype(np.uint8)
    
    return output

def save_debug_image(debug_image, output_path):
    """
    Save the debug overlay as a PNG or JPEG.
    """
    # OpenCV expects BGR for imwrite. 
    # If our debug_image is in RGB, convert it:
    debug_bgr = cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, debug_bgr)
