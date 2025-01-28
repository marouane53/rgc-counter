# src/utils.py

import os
import glob
import tifffile
import numpy as np

def load_tiff_images(folder_path):
    """
    Loads all .tif or .tiff images in the given folder.
    Returns a list of (image_path, image_array).
    """
    images_list = []
    for ext in ("*.tif", "*.tiff"):
        for filepath in glob.glob(os.path.join(folder_path, ext)):
            # read with tifffile
            img = tifffile.imread(filepath)
            images_list.append((filepath, img))
    return images_list


def save_results_to_csv(results, output_csv_path):
    """
    Saves the cell counts/density results to a CSV file.
    results is expected to be a list of dict, e.g.:
    [
      {
         'filename': 'Snap-001.tif',
         'cell_count': 120,
         'area_mm2': 0.12,
         'density_cells_per_mm2': 1000
      },
      ...
    ]
    """
    import csv
    
    fieldnames = ['filename', 'cell_count', 'area_mm2', 'density_cells_per_mm2']
    with open(output_csv_path, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def ensure_grayscale(image, channel_index=0):
    """
    If the image has multiple channels, extract the one we need.
    If the image is already single-channel, just return it.
    """
    if len(image.shape) == 2:
        # Already grayscale
        return image
    elif len(image.shape) == 3:
        # e.g., (Height, Width, Channels) or (Channels, Height, Width)
        # We need to guess the dimension. 
        # Common TIF shape could be (z, y, x) or (y, x, c).
        # We'll try to pick channel_index from the last dim if it's < 10 (assuming it's color channels)
        if image.shape[-1] < 10:  
            # shape = (H, W, Channels)
            return image[..., channel_index]
        else:
            # shape = (Channels, H, W)
            return image[channel_index, ...]
    else:
        raise ValueError("Unexpected image dimensions: {}".format(image.shape))
