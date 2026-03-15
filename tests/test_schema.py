import pandas as pd
import pytest

from src.schema import validate_object_table, validate_region_table, validate_study_table


def test_validate_object_table_adds_contract_columns_in_permissive_mode():
    frame = pd.DataFrame(
        [
            {
                "filename": "sample.tif",
                "source_path": "/tmp/sample.tif",
                "object_id": 1,
                "centroid_x_px": 10.0,
                "centroid_y_px": 20.0,
                "area_px": 42,
            }
        ]
    )

    out = validate_object_table(frame, strict=False)

    assert out.loc[0, "schema_version"] == "1.0.0"
    assert out.loc[0, "table_kind"] == "object"
    assert out.loc[0, "image_id"] == "sample"
    assert out.loc[0, "phenotype"] == "unclassified"
    assert bool(out.loc[0, "kept"]) is True


def test_validate_region_table_maps_legacy_schema_alias():
    frame = pd.DataFrame(
        [
            {
                "source_path": "/tmp/sample.tif",
                "retina_region_schema": "mouse_flatmount_v1",
                "region_axis": "ring",
                "region_label": "central",
                "area_mm2": 0.1,
                "object_count": 5,
                "density_cells_per_mm2": 50.0,
            }
        ]
    )

    out = validate_region_table(frame, strict=False)

    assert out.loc[0, "region_schema"] == "mouse_flatmount_v1"
    assert out.loc[0, "table_kind"] == "region"


def test_validate_study_table_strict_raises_on_missing_required_columns():
    frame = pd.DataFrame([{"sample_id": "S1"}])

    with pytest.raises(ValueError):
        validate_study_table(frame, strict=True)
