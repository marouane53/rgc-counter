from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def build_methods_appendix(
    *,
    resolved_config: dict[str, Any],
    sample_table: pd.DataFrame | None = None,
    region_table: pd.DataFrame | None = None,
    validation_summary: pd.DataFrame | None = None,
    study_statistics: dict[str, Any] | None = None,
) -> str:
    model_spec = resolved_config.get("model_spec", {}) if isinstance(resolved_config.get("model_spec"), dict) else {}
    lines: list[str] = []
    lines.append("# Methods Appendix")
    lines.append("")
    lines.append("## Pipeline Configuration")
    lines.append("")
    lines.append(f"- Backend: `{resolved_config.get('backend')}`")
    lines.append(f"- Modality: `{resolved_config.get('modality')}`")
    lines.append(f"- Modality projection: `{resolved_config.get('modality_projection')}`")
    lines.append(f"- Focus mode: `{resolved_config.get('focus_mode')}`")
    lines.append(f"- GPU enabled: `{resolved_config.get('use_gpu')}`")
    lines.append(f"- Diameter override: `{resolved_config.get('diameter')}`")
    lines.append(f"- Size filter: min `{resolved_config.get('min_size')}`, max `{resolved_config.get('max_size')}`")
    lines.append(f"- TTA enabled: `{resolved_config.get('tta')}`")
    lines.append(f"- Spatial stats enabled: `{resolved_config.get('spatial_stats')}`")
    lines.append(f"- Phenotype engine: `{resolved_config.get('phenotype_engine')}`")
    lines.append(f"- Marker metrics: `{resolved_config.get('marker_metrics')}`")
    lines.append(f"- Interaction metrics: `{resolved_config.get('interaction_metrics')}`")
    lines.append(f"- Retina registration: `{resolved_config.get('register_retina')}`")
    if resolved_config.get("register_retina"):
        lines.append(f"- Region schema: `{resolved_config.get('region_schema')}`")
        lines.append(f"- ONH mode: `{resolved_config.get('onh_mode')}`")
    lines.append(f"- Atlas reference: `{resolved_config.get('atlas_reference')}`")
    lines.append(f"- Longitudinal tracking: `{resolved_config.get('track_longitudinal')}`")
    lines.append("")

    if model_spec:
        lines.append("## Model Selection")
        lines.append("")
        lines.append(f"- Model label: `{model_spec.get('model_label')}`")
        lines.append(f"- Model source: `{model_spec.get('source')}`")
        lines.append(f"- Display label: `{model_spec.get('display_label')}`")
        lines.append(f"- Backend: `{model_spec.get('backend')}`")
        lines.append(f"- Built-in model name: `{model_spec.get('builtin_name')}`")
        lines.append(f"- Asset path: `{model_spec.get('asset_path')}`")
        lines.append(f"- Trust mode: `{model_spec.get('trust_mode')}`")
        if model_spec.get("alias"):
            lines.append(f"- Alias: `{model_spec.get('alias')}`")
        if model_spec.get("source") == "legacy_custom":
            lines.append("- Compatibility note: this run used the legacy custom Cellpose path through `--model_type`.")
        lines.append("")

    if sample_table is not None and not sample_table.empty:
        lines.append("## Study Cohort")
        lines.append("")
        lines.append(f"- Samples processed: `{len(sample_table)}`")
        if "animal_id" in sample_table.columns:
            lines.append(f"- Animals represented: `{sample_table['animal_id'].nunique()}`")
        if "condition" in sample_table.columns:
            counts = sample_table["condition"].astype(str).value_counts().to_dict()
            lines.append(f"- Conditions: `{counts}`")
        if "timepoint_dpi" in sample_table.columns:
            timepoints = sorted(sample_table["timepoint_dpi"].dropna().unique().tolist())
            lines.append(f"- Timepoints (dpi): `{timepoints}`")
        lines.append("")

    if region_table is not None and not region_table.empty:
        lines.append("## Region Analysis")
        lines.append("")
        axes = sorted(region_table["region_axis"].astype(str).unique().tolist())
        lines.append(f"- Region axes summarized: `{axes}`")
        lines.append("")

    if validation_summary is not None and not validation_summary.empty:
        row = validation_summary.iloc[0]
        lines.append("## Validation Summary")
        lines.append("")
        lines.append(f"- Samples with manual reference: `{int(row['n_samples'])}`")
        lines.append(f"- Mean bias: `{row['mean_bias']:.3f}`")
        lines.append(f"- MAE: `{row['mae']:.3f}`")
        lines.append(f"- RMSE: `{row['rmse']:.3f}`")
        lines.append(f"- Pearson r: `{row['pearson_r']:.3f}`")
        lines.append(f"- ICC(2,1): `{row['icc_2_1']:.3f}`")
        lines.append("")

    if study_statistics:
        lines.append("## Statistical Analysis")
        lines.append("")
        lines.append(f"- Requested statistics mode: `{study_statistics.get('requested_mode')}`")
        for key in ("sample_outcome", "region_outcome"):
            payload = study_statistics.get(key, {})
            if not payload:
                continue
            lines.append(
                f"- {payload.get('analysis_level', 'analysis').title()} outcome "
                f"`{payload.get('outcome')}` used `{payload.get('selected_mode')}` mode."
            )
            if payload.get("selection_reason"):
                lines.append(f"  Selection reason: {payload['selection_reason']}")
            if payload.get("fallback_reason"):
                lines.append(f"  Fallback reason: {payload['fallback_reason']}")
            if payload.get("mixed_formula"):
                lines.append(f"  Mixed-effects formula: `{payload['mixed_formula']}`")
            if payload.get("grouping_factor"):
                lines.append(f"  Grouping factor: `{payload['grouping_factor']}`")
            if payload.get("variance_components"):
                lines.append(f"  Variance components: `{payload['variance_components']}`")
            for warning in payload.get("warnings", []):
                lines.append(f"  Warning: {warning}")
        lines.append("")

    lines.append("## Output Policy")
    lines.append("")
    lines.append("- Summary tables were generated from saved per-image outputs rather than notebook-only analysis.")
    lines.append("- Study figures were generated from cohort tables after image processing completed.")
    lines.append("- Any warnings or missing artifacts should be inspected in the run provenance JSON.")
    lines.append("")
    return "\n".join(lines)


def write_methods_appendix(path: str | Path, content: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path
