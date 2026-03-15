from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_condition_summary_plot(sample_table: pd.DataFrame, destination: str | Path, outcome: str = "cell_count") -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    grouped = sample_table.groupby("condition", dropna=False)[outcome]
    means = grouped.mean()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(means.index.astype(str), means.values, color="#4c72b0", alpha=0.7)
    for idx, condition in enumerate(means.index.astype(str)):
        y = sample_table.loc[sample_table["condition"].astype(str) == condition, outcome]
        ax.scatter([idx] * len(y), y, color="black", zorder=3)
    ax.set_ylabel(outcome)
    ax.set_title(f"Condition Summary: {outcome}")
    fig.tight_layout()
    fig.savefig(destination, dpi=200)
    plt.close(fig)
    return destination


def save_paired_plot(sample_table: pd.DataFrame, destination: str | Path, outcome: str = "cell_count") -> Path | None:
    if "animal_id" not in sample_table.columns or "condition" not in sample_table.columns:
        return None
    pivot = sample_table.pivot_table(index="animal_id", columns="condition", values=outcome, aggfunc="mean").dropna()
    if pivot.shape[1] != 2 or pivot.empty:
        return None

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    conditions = list(pivot.columns.astype(str))
    fig, ax = plt.subplots(figsize=(6, 4))
    for _, row in pivot.iterrows():
        ax.plot(conditions, row.to_numpy(dtype=float), marker="o", color="#4c72b0", alpha=0.8)
    ax.set_ylabel(outcome)
    ax.set_title(f"Paired {outcome} by animal")
    fig.tight_layout()
    fig.savefig(destination, dpi=200)
    plt.close(fig)
    return destination


def save_region_density_plot(region_table: pd.DataFrame, destination: str | Path) -> Path | None:
    if region_table.empty:
        return None
    subset = region_table[region_table["region_axis"] == "ring"].copy()
    if subset.empty or "condition" not in subset.columns:
        return None

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pivot = subset.pivot_table(
        index="region_label",
        columns="condition",
        values="density_cells_per_mm2",
        aggfunc="mean",
    ).fillna(0)
    pivot = pivot.loc[(pivot.sum(axis=1) > 0)]
    if pivot.empty:
        return None

    ax = pivot.plot(kind="bar", figsize=(7, 4), color=["#4c72b0", "#dd8452", "#55a868", "#c44e52"])
    ax.set_ylabel("Density (cells/mm^2)")
    ax.set_title("Regional Density by Condition")
    ax.tick_params(axis="x", rotation=30)
    ax.figure.tight_layout()
    ax.figure.savefig(destination, dpi=200)
    plt.close(ax.figure)
    return destination


def save_phenotype_composition_plot(sample_table: pd.DataFrame, destination: str | Path) -> Path | None:
    phenotype_columns = [column for column in sample_table.columns if column.startswith("phenotype_count_")]
    if not phenotype_columns:
        return None
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    melted = sample_table[["condition", *phenotype_columns]].copy()
    melted = melted.melt(id_vars=["condition"], var_name="phenotype", value_name="count")
    melted["phenotype"] = melted["phenotype"].str.replace("phenotype_count_", "", regex=False)
    grouped = melted.groupby(["condition", "phenotype"], dropna=False)["count"].mean().reset_index()
    pivot = grouped.pivot(index="condition", columns="phenotype", values="count").fillna(0)
    if pivot.empty:
        return None

    ax = pivot.plot(kind="bar", stacked=True, figsize=(7, 4), colormap="tab20")
    ax.set_ylabel("Mean phenotype count")
    ax.set_title("Phenotype Composition by Condition")
    ax.figure.tight_layout()
    ax.figure.savefig(destination, dpi=200)
    plt.close(ax.figure)
    return destination


def save_atlas_deviation_plot(atlas_comparison: pd.DataFrame, destination: str | Path) -> Path | None:
    if atlas_comparison.empty:
        return None
    subset = atlas_comparison[atlas_comparison["region_axis"] == "ring"].copy()
    if subset.empty:
        return None

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    value_column = "delta_density_cells_per_mm2"
    if "condition" in subset.columns:
        pivot = subset.pivot_table(
            index="region_label",
            columns="condition",
            values=value_column,
            aggfunc="mean",
        ).fillna(0)
        ax = pivot.plot(kind="bar", figsize=(7, 4), color=["#4c72b0", "#dd8452", "#55a868", "#c44e52"])
        ax.set_ylabel("Observed - atlas density")
        ax.set_title("Atlas Deviation by Region")
        ax.tick_params(axis="x", rotation=30)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.figure.tight_layout()
        ax.figure.savefig(destination, dpi=200)
        plt.close(ax.figure)
        return destination

    grouped = subset.groupby("region_label", dropna=False)[value_column].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(grouped.index.astype(str), grouped.values, color="#c44e52", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Observed - atlas density")
    ax.set_title("Atlas Deviation by Region")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(destination, dpi=200)
    plt.close(fig)
    return destination
