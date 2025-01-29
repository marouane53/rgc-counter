# manual_roi.py

import napari
import numpy as np
import warnings

def select_bounding_box_napari(image):
    """
    Launches Napari, displays the image, and lets the user draw
    a single rectangular ROI (bounding box).
    
    Returns (y1, y2, x1, x2) as integers indicating the selected region.
    """
    # We need a shapes layer for the user to draw the rectangle
    # Napari usage note:
    #  - The user draws a shape in the Shapes layer
    #  - Then we read the layer data after they close the viewer

    with napari.gui_qt():
        viewer = napari.Viewer()
        # Add the image
        layer_image = viewer.add_image(image, name="Grayscale Image")

        # Add a shapes layer
        shapes_layer = viewer.add_shapes(name="ROI", shape_type='rectangle', edge_color='yellow', face_color='yellow', opacity=0.4)

        print("Draw a rectangle in the Shapes layer. Then close Napari to confirm.")

        @viewer.bind_key('q')
        def close_viewer(viewer):
            viewer.close()

    # After Napari closes, we should have shapes_layer.data
    if len(shapes_layer.data) == 0:
        warnings.warn("No bounding box was drawn. Using the entire image as default ROI.")
        return (0, image.shape[0], 0, image.shape[1])
    
    # Grab the first shape (we assume only 1 rectangle)
    # Napari shapes store rectangle corners in a Nx2 array
    rect_coords = shapes_layer.data[0]  # shape: (4, 2)
    # rect_coords has corners in (y, x) format
    ys = rect_coords[:, 0]
    xs = rect_coords[:, 1]

    y1, y2 = int(np.min(ys)), int(np.max(ys))
    x1, x2 = int(np.min(xs)), int(np.max(xs))

    return (y1, y2, x1, x2)
