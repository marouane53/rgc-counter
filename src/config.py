# src/config.py

import os
import yaml

# Build the path to config.yaml in the project root
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')

if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Could not find config.yaml at {CONFIG_PATH}")

with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    data = yaml.safe_load(f)

# Basic settings
MICRONS_PER_PIXEL = data.get('microns_per_pixel', 0.227)
"""
This value indicates how many microns each pixel covers, e.g. 0.227 um.
Used to compute cell density in cells / mm^2 or cells / um^2, etc.
"""

# If you wanted the pixel area, you could do:
# PIXEL_AREA_MICRONS2 = MICRONS_PER_PIXEL * MICRONS_PER_PIXEL
# or you can compute it on-the-fly in analysis.py

# Cell detection settings
cell_detection = data.get('cell_detection', {})
CELL_DIAMETER = cell_detection.get('diameter', None)
MODEL_TYPE = cell_detection.get('model_type', 'cyto')
USE_GPU = cell_detection.get('use_gpu', True)

# Analysis settings
analysis = data.get('analysis', {})
MIN_CELL_SIZE = analysis.get('min_cell_size', 5)
MAX_CELL_SIZE = analysis.get('max_cell_size', 1000000)

# Visualization settings
visualization = data.get('visualization', {})
OVERLAY_ALPHA = visualization.get('overlay_alpha', 0.5)
