import numpy as np
import pandas as pd

from src.marker_metrics import add_marker_metrics


def test_add_marker_metrics_populates_geometry_channel_and_relation_features():
    image = np.zeros((32, 32, 2), dtype=np.uint8)
    labels = np.zeros((32, 32), dtype=np.uint16)
    labels[4:10, 4:10] = 1
    labels[18:24, 18:24] = 2
    image[4:10, 4:10, 0] = 220
    image[18:24, 18:24, 0] = 220
    image[18:24, 18:24, 1] = 230

    object_table = pd.DataFrame(
        {
            "object_id": [1, 2],
            "area_px": [36, 36],
            "centroid_x_px": [6.5, 20.5],
            "centroid_y_px": [6.5, 20.5],
        }
    )
    config = {
        "channels": {"RBPMS": 0, "MICROGLIA": 1},
        "masks": {"microglia": {"channel": "MICROGLIA", "min_intensity": 180}},
    }

    out = add_marker_metrics(object_table, image, labels, config=config)

    assert "geometry.circularity" in out.columns
    assert "channel.mean.RBPMS" in out.columns
    assert "channel.mean_z.RBPMS" in out.columns
    assert "relation.overlap_fraction.microglia" in out.columns
    assert out.loc[0, "relation.overlap_fraction.microglia"] == 0.0
    assert out.loc[1, "relation.overlap_fraction.microglia"] > 0.9
