import pandas as pd
import pytest

from src.retina_coords import register_cells, retina_frame_from_points


def test_register_cells_projects_centroids_into_retina_frame():
    frame = retina_frame_from_points(
        onh_xy_px=(100.0, 100.0),
        dorsal_xy_px=(100.0, 50.0),
        um_per_px=2.0,
        source="cli",
    )
    table = pd.DataFrame(
        {
            "centroid_x_px": [110.0, 100.0],
            "centroid_y_px": [100.0, 90.0],
        }
    )

    out = register_cells(table, frame)

    assert out.loc[0, "ret_x_um"] == pytest.approx(20.0)
    assert out.loc[0, "ret_y_um"] == pytest.approx(0.0)
    assert out.loc[0, "theta_deg"] == pytest.approx(0.0)
    assert out.loc[1, "ret_x_um"] == pytest.approx(0.0)
    assert out.loc[1, "ret_y_um"] == pytest.approx(20.0)
    assert out.loc[1, "theta_deg"] == pytest.approx(90.0)
