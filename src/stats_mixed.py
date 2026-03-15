from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np
import pandas as pd


@dataclass
class MixedModelResult:
    frame: pd.DataFrame
    meta: dict[str, Any]


MIXED_COLUMNS = [
    "outcome",
    "analysis_level",
    "formula",
    "grouping_factor",
    "variance_components",
    "n_obs",
    "n_groups",
    "converged",
    "term",
    "estimate",
    "stderr",
    "statistic",
    "p_value",
    "ci_low",
    "ci_high",
    "p_value_fdr",
]


def empty_mixed_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=MIXED_COLUMNS)


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


def _finalize_mixed_result(
    fit,
    *,
    frame: pd.DataFrame,
    formula: str,
    grouping_factor: str,
    variance_components: list[str],
    analysis_level: str,
    outcome: str,
) -> MixedModelResult:
    converged = bool(getattr(fit, "converged", False))
    terms = list(fit.fe_params.index)
    if not terms:
        raise RuntimeError("Mixed-effects fit produced no fixed-effect coefficients.")

    conf = fit.conf_int().loc[terms]
    out = pd.DataFrame(
        {
            "outcome": outcome,
            "analysis_level": analysis_level,
            "formula": formula,
            "grouping_factor": grouping_factor,
            "variance_components": ";".join(variance_components),
            "n_obs": int(len(frame)),
            "n_groups": int(frame[grouping_factor].astype(str).nunique()),
            "converged": converged,
            "term": terms,
            "estimate": fit.fe_params.loc[terms].astype(float).values,
            "stderr": fit.bse_fe.loc[terms].astype(float).values,
            "statistic": fit.tvalues.loc[terms].astype(float).values,
            "p_value": fit.pvalues.loc[terms].astype(float).values,
            "ci_low": conf[0].astype(float).values,
            "ci_high": conf[1].astype(float).values,
        }
    )
    out["p_value_fdr"] = _fdr_bh(out["p_value"].tolist())

    numeric = out[["estimate", "stderr", "statistic", "p_value", "ci_low", "ci_high"]].to_numpy(dtype=float)
    if out.empty or not np.isfinite(numeric).all():
        raise RuntimeError("Mixed-effects fit produced a non-finite coefficient table.")
    if not converged:
        raise RuntimeError("Mixed-effects fit did not converge.")

    meta = {
        "formula": formula,
        "grouping_factor": grouping_factor,
        "variance_components": variance_components,
        "n_obs": int(len(frame)),
        "n_groups": int(frame[grouping_factor].astype(str).nunique()),
        "converged": converged,
    }
    return MixedModelResult(frame=out[MIXED_COLUMNS], meta=meta)


def _fit_mixedlm_with_retries(model):
    last_fit = None
    errors: list[str] = []
    for method in ("lbfgs", "powell", "cg", "nm"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = model.fit(reml=False, method=method, disp=False)
            if bool(getattr(fit, "converged", False)):
                return fit
            last_fit = fit
            errors.append(f"{method}: not converged")
        except Exception as exc:
            errors.append(f"{method}: {exc}")
    if last_fit is not None:
        return last_fit
    raise RuntimeError("Mixed-effects fit failed across all optimizers: " + "; ".join(errors))


def fit_sample_mixed_effects(sample_table: pd.DataFrame, outcome: str = "cell_count") -> MixedModelResult:
    try:
        import statsmodels.formula.api as smf
    except ImportError as exc:
        raise RuntimeError("statsmodels is required for mixed-effects statistics.") from exc

    required = [outcome, "animal_id", "condition"]
    if any(column not in sample_table.columns for column in required):
        raise RuntimeError("Sample table is missing required columns for mixed-effects modeling.")

    frame = sample_table.dropna(subset=required).copy()
    if frame.empty or frame["animal_id"].astype(str).nunique() < 2 or frame["condition"].astype(str).nunique() < 2:
        raise RuntimeError("Sample table does not have enough groups or conditions for mixed-effects modeling.")
    if frame[outcome].astype(float).std(ddof=1) <= 1e-12:
        raise RuntimeError("Sample outcome variance is effectively zero; mixed-effects fit is not usable.")

    grouping_factor = "animal_id"
    fixed_terms = ["C(condition)"]
    if "timepoint_dpi" in frame.columns and frame["timepoint_dpi"].dropna().nunique() > 1:
        fixed_terms.append("timepoint_dpi")

    variance_components: dict[str, str] = {}
    vc_names: list[str] = []
    if "eye" in frame.columns and frame["eye"].notna().any():
        frame["animal_eye"] = frame["animal_id"].astype(str) + ":" + frame["eye"].astype(str)
        if frame["animal_eye"].nunique() > frame["animal_id"].astype(str).nunique():
            variance_components["animal_eye"] = "0 + C(animal_eye)"
            vc_names.append("animal_eye")

    formula = f"{outcome} ~ " + " + ".join(fixed_terms)
    candidate_vc = [(variance_components, vc_names)]
    if variance_components:
        candidate_vc.append(({}, []))

    errors: list[str] = []
    for vc_formula, vc_used in candidate_vc:
        try:
            model = smf.mixedlm(
                formula=formula,
                data=frame,
                groups=frame[grouping_factor],
                re_formula="1",
                vc_formula=vc_formula or None,
            )
            fit = _fit_mixedlm_with_retries(model)
            return _finalize_mixed_result(
                fit,
                frame=frame,
                formula=formula,
                grouping_factor=grouping_factor,
                variance_components=vc_used,
                analysis_level="sample",
                outcome=outcome,
            )
        except Exception as exc:
            errors.append(str(exc))
    raise RuntimeError("Sample mixed-effects fit failed: " + "; ".join(errors))


def fit_region_mixed_effects(region_table: pd.DataFrame, outcome: str = "density_cells_per_mm2") -> MixedModelResult:
    try:
        import statsmodels.formula.api as smf
    except ImportError as exc:
        raise RuntimeError("statsmodels is required for mixed-effects statistics.") from exc

    required = [outcome, "animal_id", "condition", "region_axis", "region_label"]
    if any(column not in region_table.columns for column in required):
        raise RuntimeError("Region table is missing required columns for mixed-effects modeling.")

    frame = region_table.dropna(subset=required).copy()
    if frame.empty or frame["animal_id"].astype(str).nunique() < 2 or frame["condition"].astype(str).nunique() < 2:
        raise RuntimeError("Region table does not have enough groups or conditions for mixed-effects modeling.")
    if frame[outcome].astype(float).std(ddof=1) <= 1e-12:
        raise RuntimeError("Region outcome variance is effectively zero; mixed-effects fit is not usable.")

    frame["region_key"] = frame["region_axis"].astype(str) + ":" + frame["region_label"].astype(str)
    grouping_factor = "animal_id"
    fixed_terms = ["C(condition)", "C(region_key)", "C(condition):C(region_key)"]
    if "timepoint_dpi" in frame.columns and frame["timepoint_dpi"].dropna().nunique() > 1:
        fixed_terms.append("timepoint_dpi")

    variance_components: dict[str, str] = {}
    vc_names: list[str] = []
    if "eye" in frame.columns and frame["eye"].notna().any():
        frame["animal_eye"] = frame["animal_id"].astype(str) + ":" + frame["eye"].astype(str)
        if frame["animal_eye"].nunique() > frame["animal_id"].astype(str).nunique():
            variance_components["animal_eye"] = "0 + C(animal_eye)"
            vc_names.append("animal_eye")
    if "sample_id" in frame.columns and frame["sample_id"].astype(str).nunique() < len(frame):
        variance_components["sample_id"] = "0 + C(sample_id)"
        vc_names.append("sample_id")

    formula = f"{outcome} ~ " + " + ".join(fixed_terms)
    candidate_vc: list[tuple[dict[str, str], list[str]]] = [(variance_components, vc_names)]
    if "sample_id" in variance_components:
        reduced = {key: value for key, value in variance_components.items() if key != "sample_id"}
        candidate_vc.append((reduced, [name for name in vc_names if name != "sample_id"]))
    if "animal_eye" in variance_components:
        reduced = {key: value for key, value in variance_components.items() if key != "animal_eye"}
        candidate_vc.append((reduced, [name for name in vc_names if name != "animal_eye"]))
    if variance_components:
        candidate_vc.append(({}, []))

    seen: set[tuple[str, ...]] = set()
    errors: list[str] = []
    for vc_formula, vc_used in candidate_vc:
        signature = tuple(sorted(vc_formula))
        if signature in seen:
            continue
        seen.add(signature)
        try:
            model = smf.mixedlm(
                formula=formula,
                data=frame,
                groups=frame[grouping_factor],
                re_formula="1",
                vc_formula=vc_formula or None,
            )
            fit = _fit_mixedlm_with_retries(model)
            return _finalize_mixed_result(
                fit,
                frame=frame,
                formula=formula,
                grouping_factor=grouping_factor,
                variance_components=vc_used,
                analysis_level="region",
                outcome=outcome,
            )
        except Exception as exc:
            errors.append(str(exc))
    raise RuntimeError("Region mixed-effects fit failed: " + "; ".join(errors))
