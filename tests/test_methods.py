import pandas as pd

from src.methods import build_methods_appendix


def test_build_methods_appendix_includes_core_sections():
    resolved = {
        "backend": "cellpose",
        "model_spec": {
            "backend": "cellpose",
            "source": "builtin",
            "model_label": "cellpose_builtin:cyto",
            "display_label": "cellpose_builtin:cyto",
            "builtin_name": "cyto",
            "asset_path": None,
            "model_type": "cyto",
            "alias": None,
            "trust_mode": "builtin",
        },
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
    assert "Model Selection" in text
    assert "Study Cohort" in text
    assert "`v2`" in text


def test_build_methods_appendix_includes_statistical_analysis_when_provided():
    text = build_methods_appendix(
        resolved_config={"backend": "cellpose", "modality": "flatmount", "modality_projection": "max", "focus_mode": "none", "use_gpu": False, "diameter": None, "min_size": 5, "max_size": 1000, "tta": False, "spatial_stats": False, "phenotype_engine": "legacy", "marker_metrics": False, "interaction_metrics": False, "register_retina": False, "atlas_reference": None, "track_longitudinal": False},
        study_statistics={
            "requested_mode": "auto",
            "sample_outcome": {
                "analysis_level": "sample",
                "outcome": "cell_count",
                "selected_mode": "mixed",
                "selection_reason": "Detected repeated structure.",
                "fallback_reason": None,
                "warnings": [],
                "mixed_formula": "cell_count ~ C(condition)",
                "grouping_factor": "animal_id",
                "variance_components": ["animal_eye"],
            }
        },
    )

    assert "Statistical Analysis" in text
    assert "Requested statistics mode" in text
    assert "cell_count" in text


def test_build_methods_appendix_describes_rigorous_spatial_analysis():
    text = build_methods_appendix(
        resolved_config={
            "backend": "cellpose",
            "modality": "flatmount",
            "modality_projection": "max",
            "focus_mode": "none",
            "use_gpu": False,
            "diameter": None,
            "min_size": 5,
            "max_size": 1000,
            "tta": False,
            "spatial_stats": True,
            "spatial_mode": "rigorous",
            "spatial_envelope_sims": 999,
            "spatial_random_seed": 1337,
            "phenotype_engine": "legacy",
            "marker_metrics": False,
            "interaction_metrics": False,
            "register_retina": False,
            "atlas_reference": None,
            "track_longitudinal": False,
        }
    )

    assert "Spatial Analysis" in text
    assert "Rigorous spatial inference" in text
    assert "CSR envelopes" in text


def test_build_methods_appendix_describes_atlas_subtype_priors():
    text = build_methods_appendix(
        resolved_config={
            "backend": "cellpose",
            "modality": "flatmount",
            "modality_projection": "max",
            "focus_mode": "none",
            "use_gpu": False,
            "diameter": None,
            "min_size": 5,
            "max_size": 1000,
            "tta": False,
            "spatial_stats": False,
            "phenotype_engine": "legacy",
            "marker_metrics": True,
            "interaction_metrics": False,
            "register_retina": True,
            "atlas_reference": None,
            "atlas_subtype_priors": "examples/atlas/atlas_subtype_priors.example.yaml",
            "track_longitudinal": False,
        },
        atlas_subtypes={
            "enabled": True,
            "config_path": "examples/atlas/atlas_subtype_priors.example.yaml",
            "atlas_name": "demo_subtypes",
            "retina_region_schema": "mouse_flatmount_v1",
            "location_weight": 0.7,
            "marker_weight": 0.3,
            "subtypes": ["alpha_rgc", "iprgc"],
            "used_location_evidence": True,
        },
    )

    assert "Atlas Subtype Priors" in text
    assert "probabilistic atlas-prior subtype scoring" in text.lower()
    assert "validated subtype truth" in text


def test_build_methods_appendix_describes_registration_aware_tracking():
    text = build_methods_appendix(
        resolved_config={
            "backend": "cellpose",
            "modality": "flatmount",
            "modality_projection": "max",
            "focus_mode": "none",
            "use_gpu": False,
            "diameter": None,
            "min_size": 5,
            "max_size": 1000,
            "tta": False,
            "spatial_stats": False,
            "phenotype_engine": "legacy",
            "marker_metrics": False,
            "interaction_metrics": False,
            "register_retina": False,
            "atlas_reference": None,
            "atlas_subtype_priors": None,
            "track_longitudinal": True,
            "tracking_mode": "registered",
        },
        tracking={
            "enabled": True,
            "tracking_mode": "registered",
            "max_disp_px": 20.0,
            "alignment_method": "phase_cross_correlation",
            "fallback_policy": "centroid_fallback_with_qc_flag",
            "n_pairs_registered": 3,
            "n_pairs_fallback": 1,
        },
    )

    assert "Longitudinal Tracking" in text
    assert "phase cross-correlation" in text.lower()
    assert "best-effort correspondences" in text
