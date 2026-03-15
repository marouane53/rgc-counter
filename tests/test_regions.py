import numpy as np
import pandas as pd

from src.regions import assign_regions, summarize_regions
from src.retina_coords import retina_frame_from_points


def test_assign_regions_adds_expected_labels():
    table = pd.DataFrame(
        {
            "ret_x_um": [10.0, -10.0],
            "ret_y_um": [0.0, -10.0],
            "ecc_um": [10.0, 14.1421356],
            "theta_deg": [0.0, 225.0],
        }
    )

    out = assign_regions(table, schema_name="mouse_flatmount_v1", max_ecc_um=100.0)

    assert list(out["ring"]) == ["central", "central"]
    assert list(out["quadrant"]) == ["dorsal_temporal", "ventral_nasal"]
    assert list(out["sector"]) == ["temporal", "ventral_nasal"]
    assert list(out["peripapillary_bin"]) == ["peripapillary", "peripapillary"]


def test_summarize_regions_returns_area_and_count_tables():
    object_table = pd.DataFrame(
        {
            "ring": ["central", "peripheral"],
            "quadrant": ["dorsal_temporal", "ventral_nasal"],
            "sector": ["temporal", "ventral_nasal"],
            "peripapillary_bin": ["peripapillary", "non_peripapillary"],
        }
    )
    focus_pixels = pd.DataFrame(
        {
            "ret_x_um": [1.0, 2.0, -50.0, -60.0],
            "ret_y_um": [1.0, 2.0, -50.0, -60.0],
            "ecc_um": [1.4, 2.8, 70.7, 84.8],
            "theta_deg": [45.0, 45.0, 225.0, 225.0],
        }
    )
    tissue_pixels = focus_pixels.copy()
    tissue_mask = np.ones((8, 8), dtype=bool)
    frame = retina_frame_from_points(
        onh_xy_px=(4.0, 4.0),
        dorsal_xy_px=(4.0, 0.0),
        um_per_px=1.0,
        source="cli",
    )

    summary = summarize_regions(
        object_table=object_table,
        focus_pixels=focus_pixels,
        tissue_pixels=tissue_pixels,
        tissue_mask=tissue_mask,
        frame=frame,
        schema_name="mouse_flatmount_v1",
        source_path="sample.tif",
    )

    assert set(summary["region_axis"]) == {"ring", "quadrant", "sector", "peripapillary_bin"}
    ring_rows = summary[summary["region_axis"] == "ring"]
    assert int(ring_rows["object_count"].sum()) == 2
    assert ring_rows["area_px"].sum() > 0
