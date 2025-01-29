# src/analysis.py

import numpy as np
from src.config import MICRONS_PER_PIXEL

def compute_cell_count_and_density(masks, in_focus_mask):
    """
    Given a 2D integer mask from Cellpose and a boolean in_focus_mask,
    compute:
      1) total cell count (segmented objects)
      2) effective area in mm^2 (only in-focus region)
      3) density = cell_count / area_mm^2
    """
    # Unique IDs (excluding 0 which is background)
    object_ids = np.unique(masks)
    object_ids = object_ids[object_ids != 0]
    cell_count = len(object_ids)
    
    # Only count the in-focus pixels
    in_focus_area_pixels = np.sum(in_focus_mask)

    # Convert pixel area to mm^2 
    # (MICRONS_PER_PIXEL * MICRONS_PER_PIXEL) => area in micron^2
    # 1 mm^2 = 1e6 micron^2
    area_in_microns2 = in_focus_area_pixels * (MICRONS_PER_PIXEL ** 2)
    area_in_mm2 = area_in_microns2 / 1e6
    
    if area_in_mm2 == 0:
        density_cells_per_mm2 = 0
    else:
        density_cells_per_mm2 = cell_count / area_in_mm2
    
    return cell_count, area_in_mm2, density_cells_per_mm2
