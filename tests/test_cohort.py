from pathlib import Path

import pandas as pd

from src.cohort import build_sample_table, build_study_region_table
from src.context import RunContext


def test_build_sample_table_merges_manifest_and_context():
    manifest = pd.DataFrame(
        [
            {
                "sample_id": "S1",
                "animal_id": "A1",
                "eye": "OD",
                "condition": "treated",
                "genotype": "WT",
                "timepoint_dpi": 7,
                "modality": "flatmount",
                "stain_panel": "RBPMS",
                "path": "/tmp/sample.tif",
            }
        ]
    )
    ctx = RunContext(path=Path("/tmp/sample.tif"), image=None, meta={})  # type: ignore[arg-type]
    ctx.summary_row = {"filename": "sample.tif", "cell_count": 10}
    ctx.artifacts["object_table"] = Path("/tmp/objects.parquet")

    out = build_sample_table(manifest, [ctx])

    assert out.loc[0, "sample_id"] == "S1"
    assert out.loc[0, "cell_count"] == 10
    assert out.loc[0, "artifact_object_table"] == "/tmp/objects.parquet"


def test_build_study_region_table_adds_manifest_metadata():
    manifest = pd.DataFrame(
        [
            {
                "sample_id": "S1",
                "animal_id": "A1",
                "eye": "OD",
                "condition": "treated",
                "genotype": "WT",
                "timepoint_dpi": 7,
                "modality": "flatmount",
                "stain_panel": "RBPMS",
                "path": "/tmp/sample.tif",
            }
        ]
    )
    ctx = RunContext(path=Path("/tmp/sample.tif"), image=None, meta={})  # type: ignore[arg-type]
    ctx.region_table = pd.DataFrame([{"region_axis": "ring", "region_label": "central", "object_count": 3}])

    out = build_study_region_table(manifest, [ctx])

    assert out.loc[0, "sample_id"] == "S1"
    assert out.loc[0, "region_label"] == "central"
