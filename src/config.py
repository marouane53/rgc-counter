# src/config.py

import os
import yaml

# Build the path to config.yaml in the project root
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')

if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Could not find config.yaml at {CONFIG_PATH}")

with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    data = yaml.safe_load(f)

MICRONS_PER_PIXEL = data.get('microns_per_pixel', 0.227)
"""
This value indicates how many microns each pixel covers, e.g. 0.227 um.
Used to compute cell density in cells / mm^2 or cells / um^2, etc.
"""

# If you wanted the pixel area, you could do:
# PIXEL_AREA_MICRONS2 = MICRONS_PER_PIXEL * MICRONS_PER_PIXEL
# or you can compute it on-the-fly in analysis.py
