import pandas as pd

from src.phenotype_engine import assign_phenotypes, normalize_engine_config


def test_assign_phenotypes_supports_schema_v2_rules():
    table = pd.DataFrame(
        {
            "object_id": [1, 2],
            "geometry.area_px": [120.0, 120.0],
            "geometry.circularity": [0.7, 0.7],
            "relation.overlap_fraction.rgc_positive": [1.0, 1.0],
            "relation.overlap_fraction.microglia": [0.0, 1.0],
        }
    )
    config = {
        "schema_version": 2,
        "channels": {"RBPMS": 0, "IBA1": 1},
        "masks": {},
        "classes": {
            "rgc": {
                "priority": 100,
                "include": [
                    {"feature": "relation.overlap_fraction", "target": "rgc_positive", "op": "gt", "value": 0.0},
                    {"feature": "geometry.area_px", "op": "ge", "value": 100},
                ],
                "exclude": [
                    {"feature": "relation.overlap_fraction", "target": "microglia", "op": "gt", "value": 0.0}
                ],
            },
            "microglia": {
                "priority": 90,
                "include": [
                    {"feature": "relation.overlap_fraction", "target": "microglia", "op": "gt", "value": 0.0}
                ],
                "exclude": [],
            },
        },
    }

    out = assign_phenotypes(table, normalize_engine_config(config))

    assert list(out["phenotype"]) == ["rgc", "microglia"]
    assert list(out["phenotype_engine"]) == ["v2", "v2"]


def test_normalize_engine_config_converts_legacy_rules():
    legacy = {
        "channels": {"rgc_channel": 0, "microglia_channel": 1},
        "thresholds": {"rgc_min_intensity": 180, "microglia_min_intensity": 180},
        "logic": {"require_rgc_positive": True, "exclude_microglia_overlap": True},
        "morphology_priors": {"min_area_px": 100, "max_area_px": 1000, "min_circularity": 0.2},
    }

    out = normalize_engine_config(legacy)

    assert out["schema_version"] == 2
    assert "rgc" in out["classes"]
    assert "microglia" in out["classes"]
