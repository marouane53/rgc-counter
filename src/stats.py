from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.stats_mixed import empty_mixed_frame, fit_region_mixed_effects, fit_sample_mixed_effects

SIMPLE_STATS_COLUMNS = [
    "outcome",
    "design",
    "test",
    "condition_a",
    "condition_b",
    "n",
    "mean_a",
    "mean_b",
    "mean_difference_b_minus_a",
    "effect_size",
    "p_value",
    "ci_low",
    "ci_high",
    "p_value_fdr",
]

REGION_STATS_COLUMNS = SIMPLE_STATS_COLUMNS + ["region_axis", "region_label", "p_value_fdr_regions"]

DESIGN_AUDIT_COLUMNS = ["category", "key", "value"]


@dataclass
class StudyStatisticsResult:
    study_stats: pd.DataFrame
    region_stats: pd.DataFrame
    sample_mixed: pd.DataFrame
    region_mixed: pd.DataFrame
    design_audit: pd.DataFrame
    design_audit_markdown: str
    decision: dict[str, Any]


def _fdr_bh(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        value = min(prev, ranked[i] * n / rank)
        adjusted[i] = value
        prev = value
    out = np.empty(n, dtype=float)
    out[order] = adjusted
    return out.tolist()


def _paired_design(sample_table: pd.DataFrame, outcome: str) -> tuple[bool, pd.DataFrame]:
    if "animal_id" not in sample_table.columns or "condition" not in sample_table.columns:
        return False, pd.DataFrame()
    pivot = sample_table.pivot_table(index="animal_id", columns="condition", values=outcome, aggfunc="mean")
    pivot = pivot.dropna()
    return pivot.shape[1] == 2 and len(pivot) >= 2, pivot


def _effect_size_independent(a: np.ndarray, b: np.ndarray) -> float:
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / max(len(a) + len(b) - 2, 1))
    if pooled <= 1e-12:
        return 0.0
    return float((b.mean() - a.mean()) / pooled)


def _effect_size_paired(diff: np.ndarray) -> float:
    std = diff.std(ddof=1) if len(diff) > 1 else 0.0
    if std <= 1e-12:
        return 0.0
    return float(diff.mean() / std)


def _ci_mean_difference(diff: np.ndarray) -> tuple[float, float]:
    if len(diff) < 2:
        value = float(diff.mean()) if len(diff) else 0.0
        return value, value
    sem = stats.sem(diff)
    margin = stats.t.ppf(0.975, df=len(diff) - 1) * sem
    mean = float(diff.mean())
    return mean - float(margin), mean + float(margin)


def empty_simple_stats_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=SIMPLE_STATS_COLUMNS)


def empty_region_stats_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=REGION_STATS_COLUMNS)


def compute_outcome_stats(sample_table: pd.DataFrame, outcome: str = "cell_count") -> pd.DataFrame:
    if outcome not in sample_table.columns or "condition" not in sample_table.columns:
        return empty_simple_stats_frame()

    working = sample_table.dropna(subset=[outcome, "condition"]).copy()
    conditions = sorted(working["condition"].astype(str).unique())
    if len(conditions) != 2:
        return empty_simple_stats_frame()

    cond_a, cond_b = conditions
    paired, pivot = _paired_design(working, outcome)
    rows: list[dict[str, Any]] = []

    if paired:
        a = pivot[cond_a].to_numpy(dtype=float)
        b = pivot[cond_b].to_numpy(dtype=float)
        diff = b - a
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            t_result = stats.ttest_rel(a, b)
        ci_low, ci_high = _ci_mean_difference(diff)
        rows.append(
            {
                "outcome": outcome,
                "design": "paired",
                "test": "paired_t",
                "condition_a": cond_a,
                "condition_b": cond_b,
                "n": int(len(diff)),
                "mean_a": float(a.mean()),
                "mean_b": float(b.mean()),
                "mean_difference_b_minus_a": float(diff.mean()),
                "effect_size": _effect_size_paired(diff),
                "p_value": float(t_result.pvalue),
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
        if len(diff) >= 2 and not np.allclose(diff, 0.0):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                w_result = stats.wilcoxon(a, b)
            rows.append(
                {
                    "outcome": outcome,
                    "design": "paired",
                    "test": "wilcoxon",
                    "condition_a": cond_a,
                    "condition_b": cond_b,
                    "n": int(len(diff)),
                    "mean_a": float(a.mean()),
                    "mean_b": float(b.mean()),
                    "mean_difference_b_minus_a": float(diff.mean()),
                    "effect_size": _effect_size_paired(diff),
                    "p_value": float(w_result.pvalue),
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                }
            )
    else:
        a = working.loc[working["condition"] == cond_a, outcome].to_numpy(dtype=float)
        b = working.loc[working["condition"] == cond_b, outcome].to_numpy(dtype=float)
        if len(a) < 2 or len(b) < 2:
            return empty_simple_stats_frame()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            t_result = stats.ttest_ind(a, b, equal_var=False)
        rows.append(
            {
                "outcome": outcome,
                "design": "independent",
                "test": "welch_t",
                "condition_a": cond_a,
                "condition_b": cond_b,
                "n": int(len(a) + len(b)),
                "mean_a": float(a.mean()),
                "mean_b": float(b.mean()),
                "mean_difference_b_minus_a": float(b.mean() - a.mean()),
                "effect_size": _effect_size_independent(a, b),
                "p_value": float(t_result.pvalue),
                "ci_low": np.nan,
                "ci_high": np.nan,
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            u_result = stats.mannwhitneyu(a, b, alternative="two-sided")
        rows.append(
            {
                "outcome": outcome,
                "design": "independent",
                "test": "mannwhitney",
                "condition_a": cond_a,
                "condition_b": cond_b,
                "n": int(len(a) + len(b)),
                "mean_a": float(a.mean()),
                "mean_b": float(b.mean()),
                "mean_difference_b_minus_a": float(b.mean() - a.mean()),
                "effect_size": _effect_size_independent(a, b),
                "p_value": float(u_result.pvalue),
                "ci_low": np.nan,
                "ci_high": np.nan,
            }
        )

    frame = pd.DataFrame(rows, columns=SIMPLE_STATS_COLUMNS[:-1])
    if not frame.empty:
        frame["p_value_fdr"] = _fdr_bh(frame["p_value"].tolist())
    else:
        frame = empty_simple_stats_frame()
    return frame[SIMPLE_STATS_COLUMNS]


def compute_region_stats(region_table: pd.DataFrame, outcome: str = "density_cells_per_mm2") -> pd.DataFrame:
    if region_table.empty or outcome not in region_table.columns:
        return empty_region_stats_frame()

    frames: list[pd.DataFrame] = []
    for (axis, label), subset in region_table.groupby(["region_axis", "region_label"], dropna=False):
        stats_frame = compute_outcome_stats(subset, outcome=outcome)
        if stats_frame.empty:
            continue
        stats_frame["region_axis"] = axis
        stats_frame["region_label"] = label
        frames.append(stats_frame)
    if not frames:
        return empty_region_stats_frame()
    combined = pd.concat(frames, ignore_index=True)
    combined["p_value_fdr_regions"] = _fdr_bh(combined["p_value"].tolist())
    return combined.reindex(columns=REGION_STATS_COLUMNS)


def build_design_audit(sample_table: pd.DataFrame, region_table: pd.DataFrame | None = None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    if not sample_table.empty:
        if {"condition", "animal_id"}.issubset(sample_table.columns):
            counts = sample_table.groupby("condition", dropna=False)["animal_id"].nunique(dropna=True)
            for condition, count in counts.items():
                rows.append({"category": "animals_per_condition", "key": str(condition), "value": int(count)})

        if {"animal_id", "eye"}.issubset(sample_table.columns):
            eye_counts = sample_table.groupby("animal_id", dropna=False)["eye"].nunique(dropna=True)
            for animal_id, count in eye_counts.items():
                rows.append({"category": "eyes_per_animal", "key": str(animal_id), "value": int(count)})

            sample_table = sample_table.copy()
            sample_table["animal_eye"] = sample_table["animal_id"].astype(str) + ":" + sample_table["eye"].astype(str)
            sample_counts = sample_table.groupby("animal_eye", dropna=False).size()
            for animal_eye, count in sample_counts.items():
                rows.append({"category": "samples_per_eye", "key": str(animal_eye), "value": int(count)})

            if "timepoint_dpi" in sample_table.columns:
                tp_counts = sample_table.groupby("animal_eye", dropna=False)["timepoint_dpi"].nunique(dropna=True)
                for animal_eye, count in tp_counts.items():
                    rows.append({"category": "timepoints_per_animal_eye", "key": str(animal_eye), "value": int(count)})

        sample_missing_columns = ["sample_id", "animal_id", "eye", "condition", "timepoint_dpi", "cell_count", "density_cells_per_mm2"]
        for column in sample_missing_columns:
            if column in sample_table.columns:
                rows.append(
                    {
                        "category": "missingness_sample",
                        "key": column,
                        "value": int(sample_table[column].isna().sum()),
                    }
                )

    if region_table is not None and not region_table.empty:
        if {"sample_id", "region_axis", "region_label"}.issubset(region_table.columns):
            frame = region_table.copy()
            frame["region_key"] = frame["region_axis"].astype(str) + ":" + frame["region_label"].astype(str)
            region_counts = frame.groupby("sample_id", dropna=False)["region_key"].nunique(dropna=True)
            for sample_id, count in region_counts.items():
                rows.append({"category": "regions_per_sample", "key": str(sample_id), "value": int(count)})

        region_missing_columns = ["sample_id", "animal_id", "condition", "region_axis", "region_label", "density_cells_per_mm2"]
        for column in region_missing_columns:
            if column in region_table.columns:
                rows.append(
                    {
                        "category": "missingness_region",
                        "key": column,
                        "value": int(region_table[column].isna().sum()),
                    }
                )

    return pd.DataFrame(rows, columns=DESIGN_AUDIT_COLUMNS)


def render_design_audit_markdown(audit_frame: pd.DataFrame) -> str:
    lines = ["# Study Design Audit", ""]
    if audit_frame.empty:
        lines.append("No design-audit rows were generated.")
        lines.append("")
        return "\n".join(lines)

    for category, frame in audit_frame.groupby("category", dropna=False):
        lines.append(f"## {str(category).replace('_', ' ').title()}")
        lines.append("")
        for _, row in frame.iterrows():
            lines.append(f"- {row['key']}: {row['value']}")
        lines.append("")
    return "\n".join(lines)


def build_statistics_decision_frame(decision: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key in ("sample_outcome", "region_outcome"):
        payload = decision.get(key, {})
        if not payload:
            continue
        rows.append(
            {
                "analysis_level": payload.get("analysis_level"),
                "outcome": payload.get("outcome"),
                "requested_mode": decision.get("requested_mode"),
                "selected_mode": payload.get("selected_mode"),
                "selection_reason": payload.get("selection_reason"),
                "fallback_reason": payload.get("fallback_reason"),
                "simple_written": payload.get("simple_written"),
                "mixed_written": payload.get("mixed_written"),
                "warnings": "; ".join(payload.get("warnings", [])),
            }
        )
    return pd.DataFrame(rows)


def _has_more_than_two_conditions(frame: pd.DataFrame) -> bool:
    return "condition" in frame.columns and frame["condition"].astype(str).dropna().nunique() > 2


def _sample_mixed_eligibility(sample_table: pd.DataFrame, outcome: str) -> tuple[bool, str]:
    required = [outcome, "animal_id", "condition"]
    missing = [column for column in required if column not in sample_table.columns]
    if missing:
        return False, f"Missing required columns for mixed-effects: {missing}"

    frame = sample_table.dropna(subset=required).copy()
    if frame.empty:
        return False, "No non-null rows were available for the sample outcome."
    if frame["animal_id"].astype(str).nunique() < 2:
        return False, "Mixed-effects requires at least two animals."
    if frame["condition"].astype(str).nunique() < 2:
        return False, "Mixed-effects requires at least two conditions."

    repeated_reasons: list[str] = []
    condition_count = frame["condition"].astype(str).nunique()
    if frame.groupby("animal_id", dropna=False).size().max() > condition_count:
        repeated_reasons.append("multiple samples per animal")
    if "eye" in frame.columns and frame.groupby("animal_id", dropna=False)["eye"].nunique(dropna=True).max() > 1:
        repeated_reasons.append("multiple eyes per animal")
    if "timepoint_dpi" in frame.columns and frame.groupby("animal_id", dropna=False)["timepoint_dpi"].nunique(dropna=True).max() > 1:
        repeated_reasons.append("multiple timepoints per animal")
    if not repeated_reasons:
        return False, "No repeated structure was detected in the sample table."
    return True, "Detected repeated structure: " + ", ".join(repeated_reasons) + "."


def _region_mixed_eligibility(region_table: pd.DataFrame, outcome: str) -> tuple[bool, str]:
    required = [outcome, "animal_id", "condition", "region_axis", "region_label"]
    missing = [column for column in required if column not in region_table.columns]
    if missing:
        return False, f"Missing required columns for mixed-effects: {missing}"

    frame = region_table.dropna(subset=required).copy()
    if frame.empty:
        return False, "No non-null rows were available for the region outcome."
    if frame["animal_id"].astype(str).nunique() < 2:
        return False, "Mixed-effects requires at least two animals."
    if frame["condition"].astype(str).nunique() < 2:
        return False, "Mixed-effects requires at least two conditions."
    if "sample_id" not in frame.columns:
        return False, "Region mixed-effects requires sample_id to model repeated regions within samples."

    region_counts = frame.groupby("sample_id", dropna=False).size()
    if region_counts.max() <= 1:
        return False, "No repeated region rows were detected within samples."
    return True, "Detected repeated region structure within samples."


def _simple_unavailable_reason(frame: pd.DataFrame, outcome: str) -> str:
    if outcome not in frame.columns or "condition" not in frame.columns:
        return "Simple statistics were not run because required columns were missing."

    working = frame.dropna(subset=[outcome, "condition"]).copy()
    condition_count = working["condition"].astype(str).nunique()
    if condition_count != 2:
        return "Simple statistics were not run because the current simple path supports exactly two conditions."

    paired, pivot = _paired_design(working, outcome)
    if paired and len(pivot) < 2:
        return "Simple paired statistics were not run because fewer than two animals had both conditions."
    if not paired:
        counts = working["condition"].astype(str).value_counts()
        if counts.min() < 2:
            return "Simple independent statistics were not run because fewer than two observations were available per condition."
    return "Simple statistics were selected."


def _region_simple_unavailable_reason(region_table: pd.DataFrame, outcome: str) -> str:
    if outcome not in region_table.columns or "condition" not in region_table.columns:
        return "Simple region statistics were not run because required columns were missing."
    if _has_more_than_two_conditions(region_table):
        return "Simple region statistics were not run because the current simple path supports exactly two conditions."
    if "region_axis" not in region_table.columns or "region_label" not in region_table.columns:
        return "Simple region statistics were not run because region labels were missing."

    for _, subset in region_table.groupby(["region_axis", "region_label"], dropna=False):
        if not compute_outcome_stats(subset, outcome=outcome).empty:
            return "Simple region statistics were selected."
    return "Simple region statistics were not run because no region had enough observations for the current two-condition tests."


def _serialize_outcome_decision(
    *,
    analysis_level: str,
    outcome: str,
    selected_mode: str,
    selection_reason: str,
    fallback_reason: str | None = None,
    warnings_list: list[str] | None = None,
    simple_frame: pd.DataFrame | None = None,
    mixed_frame: pd.DataFrame | None = None,
    mixed_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "analysis_level": analysis_level,
        "outcome": outcome,
        "selected_mode": selected_mode,
        "selection_reason": selection_reason,
        "fallback_reason": fallback_reason,
        "warnings": list(warnings_list or []),
        "simple_written": bool(simple_frame is not None and not simple_frame.empty),
        "mixed_written": bool(mixed_frame is not None and not mixed_frame.empty),
        "mixed_formula": (mixed_meta or {}).get("formula"),
        "grouping_factor": (mixed_meta or {}).get("grouping_factor"),
        "variance_components": (mixed_meta or {}).get("variance_components", []),
        "n_obs": (mixed_meta or {}).get("n_obs"),
        "n_groups": (mixed_meta or {}).get("n_groups"),
        "converged": (mixed_meta or {}).get("converged"),
    }


def _run_sample_statistics(sample_table: pd.DataFrame, requested_mode: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    outcome = "cell_count"
    warnings_list: list[str] = []
    simple_frame = empty_simple_stats_frame()
    mixed_frame = empty_mixed_frame()
    eligible, eligibility_reason = _sample_mixed_eligibility(sample_table, outcome)

    if requested_mode == "simple":
        simple_frame = compute_outcome_stats(sample_table, outcome=outcome)
        return simple_frame, mixed_frame, _serialize_outcome_decision(
            analysis_level="sample",
            outcome=outcome,
            selected_mode="simple",
            selection_reason="Explicit simple mode requested.",
            warnings_list=[] if not simple_frame.empty else [_simple_unavailable_reason(sample_table, outcome)],
            simple_frame=simple_frame,
        )

    if requested_mode == "mixed":
        if not eligible:
            raise RuntimeError(f"Sample mixed-effects was requested but is not eligible: {eligibility_reason}")
        result = fit_sample_mixed_effects(sample_table, outcome=outcome)
        return simple_frame, result.frame, _serialize_outcome_decision(
            analysis_level="sample",
            outcome=outcome,
            selected_mode="mixed",
            selection_reason=eligibility_reason,
            mixed_frame=result.frame,
            mixed_meta=result.meta,
        )

    if eligible:
        try:
            result = fit_sample_mixed_effects(sample_table, outcome=outcome)
            return simple_frame, result.frame, _serialize_outcome_decision(
                analysis_level="sample",
                outcome=outcome,
                selected_mode="mixed",
                selection_reason=eligibility_reason,
                mixed_frame=result.frame,
                mixed_meta=result.meta,
            )
        except Exception as exc:
            fallback_reason = f"Mixed-effects failed and fell back to simple statistics: {exc}"
            warnings_list.append(fallback_reason)
            simple_frame = compute_outcome_stats(sample_table, outcome=outcome)
            selected = "simple" if not simple_frame.empty else "none"
            return simple_frame, mixed_frame, _serialize_outcome_decision(
                analysis_level="sample",
                outcome=outcome,
                selected_mode=selected,
                selection_reason=eligibility_reason,
                fallback_reason=fallback_reason,
                warnings_list=warnings_list,
                simple_frame=simple_frame,
            )

    simple_frame = compute_outcome_stats(sample_table, outcome=outcome)
    selected = "simple" if not simple_frame.empty else "none"
    warnings_list = [] if not simple_frame.empty else [_simple_unavailable_reason(sample_table, outcome)]
    return simple_frame, mixed_frame, _serialize_outcome_decision(
        analysis_level="sample",
        outcome=outcome,
        selected_mode=selected,
        selection_reason=eligibility_reason,
        warnings_list=warnings_list,
        simple_frame=simple_frame,
    )


def _run_region_statistics(region_table: pd.DataFrame, requested_mode: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    outcome = "density_cells_per_mm2"
    warnings_list: list[str] = []
    simple_frame = empty_region_stats_frame()
    mixed_frame = empty_mixed_frame()

    if region_table.empty:
        return simple_frame, mixed_frame, _serialize_outcome_decision(
            analysis_level="region",
            outcome=outcome,
            selected_mode="none",
            selection_reason="No region table was available for region-level statistics.",
        )

    eligible, eligibility_reason = _region_mixed_eligibility(region_table, outcome)

    if requested_mode == "simple":
        simple_frame = compute_region_stats(region_table, outcome=outcome)
        warnings_list = [] if not simple_frame.empty else [_region_simple_unavailable_reason(region_table, outcome)]
        return simple_frame, mixed_frame, _serialize_outcome_decision(
            analysis_level="region",
            outcome=outcome,
            selected_mode="simple" if not simple_frame.empty else "none",
            selection_reason="Explicit simple mode requested.",
            warnings_list=warnings_list,
            simple_frame=simple_frame,
        )

    if requested_mode == "mixed":
        if not eligible:
            raise RuntimeError(f"Region mixed-effects was requested but is not eligible: {eligibility_reason}")
        result = fit_region_mixed_effects(region_table, outcome=outcome)
        return simple_frame, result.frame, _serialize_outcome_decision(
            analysis_level="region",
            outcome=outcome,
            selected_mode="mixed",
            selection_reason=eligibility_reason,
            mixed_frame=result.frame,
            mixed_meta=result.meta,
        )

    if eligible:
        try:
            result = fit_region_mixed_effects(region_table, outcome=outcome)
            return simple_frame, result.frame, _serialize_outcome_decision(
                analysis_level="region",
                outcome=outcome,
                selected_mode="mixed",
                selection_reason=eligibility_reason,
                mixed_frame=result.frame,
                mixed_meta=result.meta,
            )
        except Exception as exc:
            fallback_reason = f"Region mixed-effects failed and fell back to simple statistics: {exc}"
            warnings_list.append(fallback_reason)
            simple_frame = compute_region_stats(region_table, outcome=outcome)
            selected = "simple" if not simple_frame.empty else "none"
            return simple_frame, mixed_frame, _serialize_outcome_decision(
                analysis_level="region",
                outcome=outcome,
                selected_mode=selected,
                selection_reason=eligibility_reason,
                fallback_reason=fallback_reason,
                warnings_list=warnings_list,
                simple_frame=simple_frame,
            )

    simple_frame = compute_region_stats(region_table, outcome=outcome)
    selected = "simple" if not simple_frame.empty else "none"
    warnings_list = [] if not simple_frame.empty else [_region_simple_unavailable_reason(region_table, outcome)]
    return simple_frame, mixed_frame, _serialize_outcome_decision(
        analysis_level="region",
        outcome=outcome,
        selected_mode=selected,
        selection_reason=eligibility_reason,
        warnings_list=warnings_list,
        simple_frame=simple_frame,
    )


def run_study_statistics(
    sample_table: pd.DataFrame,
    region_table: pd.DataFrame | None = None,
    *,
    requested_mode: str = "auto",
) -> StudyStatisticsResult:
    region_table = region_table if region_table is not None else pd.DataFrame()
    design_audit = build_design_audit(sample_table, region_table)
    design_audit_markdown = render_design_audit_markdown(design_audit)

    study_stats, sample_mixed, sample_decision = _run_sample_statistics(sample_table, requested_mode)
    region_stats, region_mixed, region_decision = _run_region_statistics(region_table, requested_mode)

    decision = {
        "requested_mode": requested_mode,
        "sample_outcome": sample_decision,
        "region_outcome": region_decision,
        "warnings": sample_decision.get("warnings", []) + region_decision.get("warnings", []),
    }
    return StudyStatisticsResult(
        study_stats=study_stats,
        region_stats=region_stats,
        sample_mixed=sample_mixed,
        region_mixed=region_mixed,
        design_audit=design_audit,
        design_audit_markdown=design_audit_markdown,
        decision=decision,
    )


def write_study_statistics_artifacts(
    result: StudyStatisticsResult,
    *,
    stats_dir: str | Path,
    stats_mixed_dir: str | Path,
) -> dict[str, Path]:
    stats_dir = Path(stats_dir)
    stats_dir.mkdir(parents=True, exist_ok=True)
    stats_mixed_dir = Path(stats_mixed_dir)
    stats_mixed_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    study_stats_path = stats_dir / "study_stats.csv"
    result.study_stats.to_csv(study_stats_path, index=False)
    written["study_stats"] = study_stats_path

    region_stats_path = stats_dir / "region_stats.csv"
    result.region_stats.to_csv(region_stats_path, index=False)
    written["region_stats"] = region_stats_path

    design_audit_csv = stats_dir / "design_audit.csv"
    result.design_audit.to_csv(design_audit_csv, index=False)
    written["design_audit_csv"] = design_audit_csv

    design_audit_md = stats_dir / "design_audit.md"
    design_audit_md.write_text(result.design_audit_markdown, encoding="utf-8")
    written["design_audit_md"] = design_audit_md

    decision_json = stats_dir / "statistics_decision.json"
    decision_json.write_text(json.dumps(result.decision, indent=2), encoding="utf-8")
    written["statistics_decision"] = decision_json

    if not result.sample_mixed.empty:
        sample_mixed_path = stats_mixed_dir / "sample_mixed_effects.csv"
        result.sample_mixed.to_csv(sample_mixed_path, index=False)
        written["sample_mixed"] = sample_mixed_path
    if not result.region_mixed.empty:
        region_mixed_path = stats_mixed_dir / "region_mixed_effects.csv"
        result.region_mixed.to_csv(region_mixed_path, index=False)
        written["region_mixed"] = region_mixed_path

    return written
