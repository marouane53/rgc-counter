from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
import yaml


SUPPORTED_LOCATION_AXES = ("ring", "quadrant", "sector", "peripapillary_bin")
DEFAULT_LOCATION_WEIGHT = 0.7
DEFAULT_MARKER_WEIGHT = 0.3


def subtype_slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")
    return slug or "subtype"


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def load_atlas_subtype_priors(path: str | Path) -> dict[str, Any]:
    priors_path = Path(path)
    with priors_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    missing = [field for field in ("schema_version", "atlas_name", "subtypes") if field not in raw]
    if missing:
        raise ValueError(f"Atlas subtype priors missing required field(s): {missing}")
    if int(raw["schema_version"]) != 1:
        raise ValueError(f"Unsupported atlas subtype schema_version: {raw['schema_version']}")
    if not isinstance(raw["subtypes"], dict) or not raw["subtypes"]:
        raise ValueError("Atlas subtype priors must define at least one subtype.")

    normalized_subtypes: dict[str, dict[str, Any]] = {}
    seen_slugs: set[str] = set()
    for subtype_name, subtype_cfg in raw["subtypes"].items():
        if not isinstance(subtype_cfg, dict):
            raise ValueError(f"Subtype '{subtype_name}' must map to an object.")
        slug = subtype_slug(str(subtype_name))
        if slug in seen_slugs:
            raise ValueError(f"Subtype slug collision for '{subtype_name}' -> '{slug}'.")
        seen_slugs.add(slug)

        location_priors: dict[str, dict[str, Any]] = {}
        for axis, axis_cfg in (subtype_cfg.get("location_priors") or {}).items():
            if axis not in SUPPORTED_LOCATION_AXES:
                raise ValueError(f"Unsupported atlas subtype axis '{axis}'.")
            if not isinstance(axis_cfg, dict):
                raise ValueError(f"Location prior for '{subtype_name}' axis '{axis}' must be an object.")
            priors = axis_cfg.get("priors")
            if not isinstance(priors, dict) or not priors:
                raise ValueError(f"Location prior for '{subtype_name}' axis '{axis}' must define non-empty priors.")
            weight = float(axis_cfg.get("weight", 1.0))
            if weight <= 0:
                raise ValueError(f"Location prior for '{subtype_name}' axis '{axis}' must have positive weight.")
            location_priors[axis] = {
                "weight": weight,
                "priors": {str(label): float(score) for label, score in priors.items()},
            }

        markers: list[dict[str, Any]] = []
        for index, marker_cfg in enumerate(subtype_cfg.get("markers") or []):
            if not isinstance(marker_cfg, dict):
                raise ValueError(f"Marker prior #{index + 1} for '{subtype_name}' must be an object.")
            feature = marker_cfg.get("feature")
            direction = marker_cfg.get("direction")
            center = marker_cfg.get("center")
            scale = marker_cfg.get("scale")
            if not feature or not isinstance(feature, str):
                raise ValueError(f"Marker prior #{index + 1} for '{subtype_name}' is missing 'feature'.")
            if direction not in {"high", "low"}:
                raise ValueError(f"Marker prior '{feature}' for '{subtype_name}' must use direction 'high' or 'low'.")
            if center is None:
                raise ValueError(f"Marker prior '{feature}' for '{subtype_name}' is missing 'center'.")
            if scale is None or float(scale) <= 0:
                raise ValueError(f"Marker prior '{feature}' for '{subtype_name}' must have positive 'scale'.")
            weight = float(marker_cfg.get("weight", 1.0))
            if weight <= 0:
                raise ValueError(f"Marker prior '{feature}' for '{subtype_name}' must have positive weight.")
            markers.append(
                {
                    "feature": feature,
                    "direction": direction,
                    "center": float(center),
                    "scale": float(scale),
                    "weight": weight,
                }
            )

        normalized_subtypes[str(subtype_name)] = {
            "slug": slug,
            "location_priors": location_priors,
            "markers": markers,
        }

    location_weight = float(raw.get("location_weight", DEFAULT_LOCATION_WEIGHT))
    marker_weight = float(raw.get("marker_weight", DEFAULT_MARKER_WEIGHT))
    if location_weight < 0 or marker_weight < 0 or (location_weight + marker_weight) <= 0:
        raise ValueError("Atlas subtype prior weights must be non-negative and not both zero.")

    return {
        "schema_version": 1,
        "config_path": str(priors_path),
        "atlas_name": str(raw["atlas_name"]),
        "retina_region_schema": raw.get("retina_region_schema"),
        "location_weight": location_weight,
        "marker_weight": marker_weight,
        "channels": dict(raw.get("channels", {})),
        "compose": dict(raw.get("compose", {})),
        "subtypes": normalized_subtypes,
    }


def _validate_schema_compatibility(object_table: pd.DataFrame, config: dict[str, Any]) -> None:
    expected_schema = config.get("retina_region_schema")
    if not expected_schema:
        return
    for column in ("retina_region_schema", "region_schema"):
        if column not in object_table.columns:
            continue
        values = object_table[column].dropna().astype(str).unique().tolist()
        if values and any(value != str(expected_schema) for value in values):
            raise ValueError(
                f"Atlas subtype priors expect retina_region_schema '{expected_schema}', "
                f"but run produced {values}."
            )


def _validate_marker_features(object_table: pd.DataFrame, config: dict[str, Any]) -> None:
    features = {
        marker["feature"]
        for subtype_cfg in config["subtypes"].values()
        for marker in subtype_cfg.get("markers", [])
    }
    missing = sorted(feature for feature in features if feature not in object_table.columns)
    if missing:
        raise ValueError(f"Atlas subtype priors reference missing object-table feature(s): {missing}")


def _location_score_for_subtype(frame: pd.DataFrame, subtype_cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    axis_scores: list[np.ndarray] = []
    axis_weights: list[np.ndarray] = []

    for axis, axis_cfg in subtype_cfg.get("location_priors", {}).items():
        if axis not in frame.columns:
            continue
        mapped = frame[axis].map(axis_cfg["priors"])
        values = pd.to_numeric(mapped, errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(values)
        if not valid.any():
            continue
        axis_scores.append(np.where(valid, values, 0.0))
        axis_weights.append(np.where(valid, float(axis_cfg["weight"]), 0.0))

    if not axis_scores:
        size = len(frame)
        return np.full(size, 0.5, dtype=float), np.zeros(size, dtype=bool)

    score_stack = np.vstack(axis_scores)
    weight_stack = np.vstack(axis_weights)
    denom = weight_stack.sum(axis=0)
    score = np.divide(
        (score_stack * weight_stack).sum(axis=0),
        denom,
        out=np.full(len(frame), 0.5, dtype=float),
        where=denom > 0,
    )
    return score, denom > 0


def _marker_score_for_subtype(frame: pd.DataFrame, subtype_cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    marker_scores: list[np.ndarray] = []
    marker_weights: list[np.ndarray] = []

    for marker in subtype_cfg.get("markers", []):
        values = pd.to_numeric(frame[marker["feature"]], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(values)
        centered = (values - marker["center"]) / marker["scale"]
        if marker["direction"] == "low":
            centered = -centered
        score = _sigmoid(centered)
        marker_scores.append(np.where(valid, score, 0.0))
        marker_weights.append(np.where(valid, float(marker["weight"]), 0.0))

    if not marker_scores:
        size = len(frame)
        return np.full(size, 0.5, dtype=float), np.zeros(size, dtype=bool)

    score_stack = np.vstack(marker_scores)
    weight_stack = np.vstack(marker_weights)
    denom = weight_stack.sum(axis=0)
    score = np.divide(
        (score_stack * weight_stack).sum(axis=0),
        denom,
        out=np.full(len(frame), 0.5, dtype=float),
        where=denom > 0,
    )
    return score, denom > 0


def _combined_score(
    *,
    location_score: np.ndarray,
    location_available: np.ndarray,
    marker_score: np.ndarray,
    marker_available: np.ndarray,
    location_weight: float,
    marker_weight: float,
) -> np.ndarray:
    loc_weight = np.where(location_available, location_weight, 0.0)
    mark_weight = np.where(marker_available, marker_weight, 0.0)
    denom = loc_weight + mark_weight
    combined = np.divide(
        location_score * loc_weight + marker_score * mark_weight,
        denom,
        out=np.full(location_score.shape[0], 0.5, dtype=float),
        where=denom > 0,
    )
    combined[~np.isfinite(combined)] = 0.5
    return combined


def score_atlas_subtypes(object_table: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    out = object_table.copy()
    if out.empty:
        return {
            "object_table": out,
            "summary": pd.DataFrame(),
            "region_summary": pd.DataFrame(),
            "atlas_name": config["atlas_name"],
            "used_location_evidence": False,
            "subtypes": [],
            "top1_counts": {},
        }

    _validate_schema_compatibility(out, config)
    _validate_marker_features(out, config)

    subtype_names = list(config["subtypes"].keys())
    subtype_slugs = [config["subtypes"][name].get("slug", subtype_slug(name)) for name in subtype_names]
    subtype_map = {slug: name for name, slug in zip(subtype_names, subtype_slugs)}

    combined_columns: list[np.ndarray] = []
    used_location_evidence = False
    for subtype_name in subtype_names:
        subtype_cfg = config["subtypes"][subtype_name]
        slug = subtype_cfg.get("slug", subtype_slug(subtype_name))
        location_score, location_available = _location_score_for_subtype(out, subtype_cfg)
        marker_score, marker_available = _marker_score_for_subtype(out, subtype_cfg)
        combined = _combined_score(
            location_score=location_score,
            location_available=location_available,
            marker_score=marker_score,
            marker_available=marker_available,
            location_weight=float(config["location_weight"]),
            marker_weight=float(config["marker_weight"]),
        )
        used_location_evidence = used_location_evidence or bool(location_available.any())
        out[f"atlas_subtype_location_score__{slug}"] = location_score
        out[f"atlas_subtype_marker_score__{slug}"] = marker_score
        combined_columns.append(combined)

    combined_matrix = np.vstack(combined_columns).T if combined_columns else np.zeros((len(out), 0), dtype=float)
    row_sums = combined_matrix.sum(axis=1)
    probabilities = np.divide(
        combined_matrix,
        row_sums[:, None],
        out=np.full_like(combined_matrix, np.nan, dtype=float),
        where=row_sums[:, None] > 0,
    )
    invalid_rows = ~np.isfinite(probabilities).all(axis=1)
    if invalid_rows.any():
        probabilities[invalid_rows, :] = 1.0 / max(len(subtype_names), 1)

    for index, subtype_name in enumerate(subtype_names):
        slug = subtype_slugs[index]
        out[f"atlas_subtype_prob__{slug}"] = probabilities[:, index]

    if len(subtype_names) == 1:
        out["atlas_subtype_top1"] = subtype_names[0]
        out["atlas_subtype_top1_probability"] = probabilities[:, 0]
        out["atlas_subtype_margin"] = probabilities[:, 0]
    else:
        top_indices = np.argmax(probabilities, axis=1)
        ordered = np.sort(probabilities, axis=1)
        out["atlas_subtype_top1"] = [subtype_names[int(index)] for index in top_indices]
        out["atlas_subtype_top1_probability"] = probabilities[np.arange(len(out)), top_indices]
        out["atlas_subtype_margin"] = ordered[:, -1] - ordered[:, -2]

    summary = summarize_atlas_subtypes(out, config["atlas_name"], subtype_map=subtype_map)
    region_summary = summarize_atlas_subtypes_by_region(out, config["atlas_name"], subtype_map=subtype_map)
    counts = out["atlas_subtype_top1"].value_counts().to_dict()
    return {
        "object_table": out,
        "summary": summary,
        "region_summary": region_summary,
        "atlas_name": config["atlas_name"],
        "used_location_evidence": used_location_evidence,
        "subtypes": subtype_names,
        "top1_counts": {
            config["subtypes"][name].get("slug", subtype_slug(name)): int(counts.get(name, 0))
            for name in subtype_names
        },
    }


def summarize_atlas_subtypes(
    object_table: pd.DataFrame,
    atlas_name: str,
    *,
    subtype_map: dict[str, str],
    id_column: str = "image_id",
    id_value: str | None = None,
) -> pd.DataFrame:
    if object_table.empty or "atlas_subtype_top1" not in object_table.columns:
        return pd.DataFrame()

    resolved_id = id_value
    if resolved_id is None:
        if id_column in object_table.columns and not object_table[id_column].empty:
            resolved_id = str(object_table[id_column].iloc[0])
        else:
            resolved_id = ""

    subtype_pairs = [
        (column.replace("atlas_subtype_prob__", "", 1), column)
        for column in object_table.columns
        if column.startswith("atlas_subtype_prob__")
    ]
    rows: list[dict[str, Any]] = []
    total = max(len(object_table), 1)
    for slug, column in subtype_pairs:
        subtype_name = subtype_map[slug]
        top1 = int((object_table["atlas_subtype_top1"] == subtype_name).sum())
        rows.append(
            {
                id_column: resolved_id,
                "subtype": subtype_name,
                "top1_count": top1,
                "top1_fraction": float(top1 / total),
                "mean_probability": float(pd.to_numeric(object_table[column], errors="coerce").mean()),
                "atlas_name": atlas_name,
            }
        )
    return pd.DataFrame(rows)


def summarize_atlas_subtypes_by_region(
    object_table: pd.DataFrame,
    atlas_name: str,
    *,
    subtype_map: dict[str, str],
    id_column: str = "image_id",
    id_value: str | None = None,
) -> pd.DataFrame:
    if object_table.empty or "atlas_subtype_top1" not in object_table.columns:
        return pd.DataFrame()

    resolved_id = id_value
    if resolved_id is None:
        if id_column in object_table.columns and not object_table[id_column].empty:
            resolved_id = str(object_table[id_column].iloc[0])
        else:
            resolved_id = ""

    subtype_pairs = [
        (column.replace("atlas_subtype_prob__", "", 1), column)
        for column in object_table.columns
        if column.startswith("atlas_subtype_prob__")
    ]

    rows: list[dict[str, Any]] = []
    for axis in SUPPORTED_LOCATION_AXES:
        if axis not in object_table.columns:
            continue
        subset = object_table[object_table[axis].notna()].copy()
        if subset.empty:
            continue
        for label, frame in subset.groupby(axis, dropna=False):
            total = max(len(frame), 1)
            for slug, column in subtype_pairs:
                subtype_name = subtype_map[slug]
                top1 = int((frame["atlas_subtype_top1"] == subtype_name).sum())
                rows.append(
                    {
                        id_column: resolved_id,
                        "region_axis": axis,
                        "region_label": str(label),
                        "subtype": subtype_name,
                        "top1_count": top1,
                        "top1_fraction": float(top1 / total),
                        "mean_probability": float(pd.to_numeric(frame[column], errors="coerce").mean()),
                        "atlas_name": atlas_name,
                    }
                )
    return pd.DataFrame(rows)


def atlas_subtype_summary_output_path(output_dir: str | Path, filepath: str | Path) -> Path:
    return Path(output_dir) / "atlas_subtypes" / f"{Path(filepath).name.rsplit('.', 1)[0]}_atlas_subtype_summary.csv"


def atlas_subtype_region_summary_output_path(output_dir: str | Path, filepath: str | Path) -> Path:
    return Path(output_dir) / "atlas_subtypes" / f"{Path(filepath).name.rsplit('.', 1)[0]}_atlas_subtype_region_summary.csv"


def write_atlas_subtype_table(frame: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path
