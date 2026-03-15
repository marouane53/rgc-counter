import pandas as pd

from src.methods import build_methods_appendix


def test_build_methods_appendix_includes_core_sections():
    resolved = {
        "backend": "cellpose",
        "focus_mode": "none",
        "use_gpu": False,
        "diameter": None,
        "min_size": 5,
        "max_size": 1000,
        "tta": False,
        "spatial_stats": False,
        "phenotype_engine": "v2",
        "marker_metrics": True,
        "interaction_metrics": True,
        "register_retina": True,
        "region_schema": "mouse_flatmount_v1",
        "onh_mode": "cli",
    }
    sample_table = pd.DataFrame([{"animal_id": "A1", "condition": "treated", "timepoint_dpi": 7}])
    text = build_methods_appendix(resolved_config=resolved, sample_table=sample_table)

    assert "# Methods Appendix" in text
    assert "Pipeline Configuration" in text
    assert "Study Cohort" in text
    assert "`v2`" in text
