import numpy as np
import pandas as pd

from src.edits import apply_edit_log


def test_apply_edit_log_relabels_and_tracks_mapping():
    labels = np.zeros((6, 6), dtype=np.uint16)
    labels[1:3, 1:3] = 1
    labels[1:3, 4:6] = 2
    labels[4:6, 4:6] = 3
    table = pd.DataFrame(
        [
            {"object_id": 1, "phenotype": "rgc"},
            {"object_id": 2, "phenotype": "rgc"},
            {"object_id": 3, "phenotype": "rgc"},
        ]
    )
    edit_doc = {
        "edits": [
            {"op": "delete_object", "object_id": 1},
            {"op": "merge_objects", "keep_object_id": 2, "object_ids": [2, 3]},
            {"op": "relabel_phenotype", "object_id": 2, "phenotype": "microglia"},
            {"op": "set_landmarks", "onh_xy": [10, 10], "dorsal_xy": [10, 1]},
        ]
    }

    relabeled, out_table, meta = apply_edit_log(labels, table, edit_doc)

    assert int(relabeled.max()) == 1
    assert list(out_table["object_id"]) == [1]
    assert meta["phenotype_overrides"] == {1: "microglia"}
    assert meta["onh_xy"] == (10, 10)
    assert meta["dorsal_xy"] == (10, 1)
