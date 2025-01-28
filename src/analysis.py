# src/analysis.py

import numpy as np
from src.config import MICRONS_PER_PIXEL

def compute_cell_count_and_density(masks):
    """
    Given a 2D integer mask from Cellpose where each cell has a unique ID,
    compute total cell count and density (cells per mm^2, for instance).
    """
    # Unique IDs (excluding 0 which is background)
    object_ids = np.unique(masks)
    object_ids = object_ids[object_ids != 0]
    cell_count = len(object_ids)
    
    # total area in pixel^2
    height, width = masks.shape
    total_area_pixels = height * width
    
    # convert to mm^2 
    # (MICRONS_PER_PIXEL * MICRONS_PER_PIXEL) => area in micron^2
    # 1 mm^2 = 1e6 micron^2
    area_in_microns2 = total_area_pixels * (MICRONS_PER_PIXEL ** 2)
    area_in_mm2 = area_in_microns2 / 1e6
    
    if area_in_mm2 == 0:
        density_cells_per_mm2 = 0
    else:
        density_cells_per_mm2 = cell_count / area_in_mm2
    
    return cell_count, area_in_mm2, density_cells_per_mm2
