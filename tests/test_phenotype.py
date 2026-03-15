import numpy as np

from src.phenotype import apply_marker_rules


def test_apply_marker_rules_excludes_microglia_overlap_objects():
    image = np.zeros((32, 32, 2), dtype=np.uint8)
    masks = np.zeros((32, 32), dtype=np.uint16)

    masks[4:10, 4:10] = 1
    masks[18:24, 18:24] = 2

    image[4:10, 4:10, 0] = 220
    image[18:24, 18:24, 0] = 220
    image[18:24, 18:24, 1] = 230

    rules = {
        "channels": {"rgc_channel": 0, "microglia_channel": 1},
        "thresholds": {"rgc_min_intensity": 180, "microglia_min_intensity": 180},
        "logic": {"require_rgc_positive": True, "exclude_microglia_overlap": True},
        "morphology_priors": {"min_area_px": 10, "max_area_px": 1000, "min_circularity": 0.1},
    }

    filtered, annotations = apply_marker_rules(image, masks, rules)

    assert set(np.unique(filtered)) == {0, 1}
    assert annotations[1]["kept"] is True
    assert annotations[2]["kept"] is False
    assert annotations[2]["reason"] == "microglia_overlap"
