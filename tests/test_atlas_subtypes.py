from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.atlas_subtypes import load_atlas_subtype_priors, score_atlas_subtypes


def _demo_priors() -> dict:
    return {
        "schema_version": 1,
        "atlas_name": "demo",
        "retina_region_schema": "mouse_flatmount_v1",
        "location_weight": 0.7,
        "marker_weight": 0.3,
        "channels": {"RBPMS": 0, "MELANOPSIN": 1},
        "subtypes": {
            "alpha_rgc": {
                "location_priors": {
                    "quadrant": {"weight": 1.0, "priors": {"dorsal_temporal": 0.9, "ventral_nasal": 0.2}}
                },
                "markers": [
                    {
                        "feature": "channel.mean_bgsub.RBPMS",
                        "direction": "high",
                        "center": 2.0,
                        "scale": 0.5,
                        "weight": 1.0,
                    }
                ],
            },
            "iprgc": {
                "location_priors": {
                    "quadrant": {"weight": 1.0, "priors": {"dorsal_temporal": 0.2, "ventral_nasal": 0.9}}
                },
                "markers": [
                    {
                        "feature": "channel.mean_bgsub.MELANOPSIN",
                        "direction": "high",
                        "center": 2.0,
                        "scale": 0.5,
                        "weight": 1.0,
                    }
                ],
            },
        },
    }


def test_load_atlas_subtype_priors_validates_axes(tmp_path: Path):
    path = tmp_path / "priors.yaml"
    path.write_text(
        """
schema_version: 1
atlas_name: demo
subtypes:
  bad:
    location_priors:
      bogus:
        weight: 1.0
        priors: {a: 0.5}
        """.strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported atlas subtype axis"):
        load_atlas_subtype_priors(path)


def test_score_atlas_subtypes_emits_probabilities_and_top1():
    frame = pd.DataFrame(
        {
            "image_id": ["img1", "img1"],
            "retina_region_schema": ["mouse_flatmount_v1", "mouse_flatmount_v1"],
            "quadrant": ["dorsal_temporal", "ventral_nasal"],
            "channel.mean_bgsub.RBPMS": [3.0, 0.1],
            "channel.mean_bgsub.MELANOPSIN": [0.1, 3.0],
        }
    )

    result = score_atlas_subtypes(frame, _demo_priors())
    out = result["object_table"]

    prob_columns = [column for column in out.columns if column.startswith("atlas_subtype_prob__")]
    assert np.allclose(out[prob_columns].sum(axis=1).to_numpy(dtype=float), 1.0)
    assert list(out["atlas_subtype_top1"]) == ["alpha_rgc", "iprgc"]
    assert (out["atlas_subtype_margin"] > 0).all()
    assert not result["summary"].empty
    assert not result["region_summary"].empty


def test_score_atlas_subtypes_uses_neutral_marker_score_for_missing_values():
    frame = pd.DataFrame(
        {
            "image_id": ["img1"],
            "retina_region_schema": ["mouse_flatmount_v1"],
            "quadrant": ["dorsal_temporal"],
            "channel.mean_bgsub.RBPMS": [np.nan],
            "channel.mean_bgsub.MELANOPSIN": [np.nan],
        }
    )

    result = score_atlas_subtypes(frame, _demo_priors())
    out = result["object_table"]

    assert out.loc[0, "atlas_subtype_marker_score__alpha_rgc"] == 0.5
    assert out.loc[0, "atlas_subtype_marker_score__iprgc"] == 0.5
    assert np.isclose(out.filter(like="atlas_subtype_prob__").iloc[0].sum(), 1.0)


def test_score_atlas_subtypes_rejects_schema_mismatch():
    frame = pd.DataFrame(
        {
            "image_id": ["img1"],
            "retina_region_schema": ["rat_flatmount_v1"],
            "quadrant": ["dorsal_temporal"],
            "channel.mean_bgsub.RBPMS": [1.0],
            "channel.mean_bgsub.MELANOPSIN": [1.0],
        }
    )

    with pytest.raises(ValueError, match="expect retina_region_schema"):
        score_atlas_subtypes(frame, _demo_priors())
