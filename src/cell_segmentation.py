# src/cell_segmentation.py

import numpy as np
from cellpose import models

def segment_cells_cellpose(
    image,
    diameter=None,
    model_type='cyto',
    channels=[0, 0],
    use_gpu=True
):
    """
    Use Cellpose to segment cells from an image.
    
    :param image: 2D numpy array (grayscale).
    :param diameter: approximate diameter of your RGCs in pixels.
                     If None, Cellpose will estimate automatically.
    :param model_type: 'cyto' or 'nuclei' or a custom model path.
    :param channels: Tells Cellpose which channel is the cell channel 
                     and which is the nuclear channel. [0,0] for grayscale.
    :param use_gpu: boolean, whether to enable GPU for Cellpose if available.
    :return: (masks, flows, styles, diams)
    """
    model = models.Cellpose(model_type=model_type, gpu=use_gpu)
    
    masks, flows, styles, diams = model.eval(
        image, 
        diameter=diameter,
        channels=channels,
        progress=True
    )
    return masks, flows, styles, diams
