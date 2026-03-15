from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


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


def compute_outcome_stats(sample_table: pd.DataFrame, outcome: str = "cell_count") -> pd.DataFrame:
    if outcome not in sample_table.columns or "condition" not in sample_table.columns:
        return pd.DataFrame()

    working = sample_table.dropna(subset=[outcome, "condition"]).copy()
    conditions = sorted(working["condition"].astype(str).unique())
    if len(conditions) != 2:
        return pd.DataFrame()

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
            return pd.DataFrame()
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

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["p_value_fdr"] = _fdr_bh(frame["p_value"].tolist())
    return frame


def compute_region_stats(region_table: pd.DataFrame, outcome: str = "density_cells_per_mm2") -> pd.DataFrame:
    if region_table.empty or outcome not in region_table.columns:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for (axis, label), subset in region_table.groupby(["region_axis", "region_label"], dropna=False):
        stats_frame = compute_outcome_stats(subset, outcome=outcome)
        if stats_frame.empty:
            continue
        stats_frame["region_axis"] = axis
        stats_frame["region_label"] = label
        frames.append(stats_frame)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["p_value_fdr_regions"] = _fdr_bh(combined["p_value"].tolist())
    return combined
