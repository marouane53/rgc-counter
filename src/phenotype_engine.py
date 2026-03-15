from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd
import yaml


def load_engine_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return normalize_engine_config(raw)


def normalize_engine_config(raw: dict[str, Any]) -> dict[str, Any]:
    schema_version = raw.get("schema_version")
    if schema_version == 2:
        cfg = dict(raw)
        cfg.setdefault("channels", {})
        cfg.setdefault("compose", {})
        cfg.setdefault("masks", {})
        cfg.setdefault("classes", {})
        return cfg
    return convert_schema_v1_to_v2(raw)


def convert_schema_v1_to_v2(raw: dict[str, Any]) -> dict[str, Any]:
    channels_raw = raw.get("channels", {})
    thresholds = raw.get("thresholds", {})
    logic = raw.get("logic", {})
    morph = raw.get("morphology_priors", {})

    channels: dict[str, int] = {}
    if channels_raw.get("rgc_channel") is not None:
        channels["RGC"] = int(channels_raw["rgc_channel"])
    if channels_raw.get("microglia_channel") is not None:
        channels["MICROGLIA"] = int(channels_raw["microglia_channel"])

    masks: dict[str, Any] = {}
    if "RGC" in channels:
        masks["rgc_positive"] = {
            "channel": "RGC",
            "min_intensity": float(thresholds.get("rgc_min_intensity", 120)),
        }
    if "MICROGLIA" in channels:
        masks["microglia"] = {
            "channel": "MICROGLIA",
            "min_intensity": float(thresholds.get("microglia_min_intensity", 120)),
        }

    include: list[dict[str, Any]] = []
    exclude: list[dict[str, Any]] = []
    if logic.get("require_rgc_positive", True) and "rgc_positive" in masks:
        include.append(
            {"feature": "relation.overlap_fraction", "target": "rgc_positive", "op": "gt", "value": 0.0}
        )
    if logic.get("exclude_microglia_overlap", True) and "microglia" in masks:
        exclude.append(
            {"feature": "relation.overlap_fraction", "target": "microglia", "op": "gt", "value": 0.0}
        )
    if morph.get("min_area_px") is not None:
        include.append({"feature": "geometry.area_px", "op": "ge", "value": float(morph["min_area_px"])})
    if morph.get("max_area_px") is not None:
        include.append({"feature": "geometry.area_px", "op": "le", "value": float(morph["max_area_px"])})
    if morph.get("min_circularity") is not None:
        include.append(
            {"feature": "geometry.circularity", "op": "ge", "value": float(morph["min_circularity"])}
        )

    classes: dict[str, Any] = {}
    if include or exclude:
        classes["rgc"] = {"priority": 100, "include": include, "exclude": exclude}
    if "microglia" in masks:
        classes["microglia"] = {
            "priority": 90,
            "include": [
                {"feature": "relation.overlap_fraction", "target": "microglia", "op": "gt", "value": 0.0}
            ],
            "exclude": [],
        }

    return {
        "schema_version": 2,
        "channels": channels,
        "compose": {},
        "masks": masks,
        "classes": classes,
        "source_schema_version": 1,
    }


def _resolve_feature_value(row: pd.Series, rule: dict[str, Any], matches_class: Callable[[str], bool]) -> Any:
    feature = str(rule["feature"])
    if feature == "class.is":
        return matches_class(str(rule["value"]))
    if feature in {"geometry.area_px", "geometry.perimeter_px", "geometry.circularity", "geometry.eccentricity"}:
        return row.get(feature)
    if feature.startswith("channel."):
        channel_name = str(rule["channel"])
        return row.get(f"{feature}.{channel_name}")
    if feature.startswith("relation.overlap_fraction") or feature.startswith("relation.distance_to_mask_px"):
        target = str(rule["target"])
        return row.get(f"{feature}.{target}")
    raise ValueError(f"Unsupported phenotype feature: {feature}")


def _evaluate_comparison(value: Any, op: str, target: Any) -> bool:
    if value is None or pd.isna(value):
        return False
    if op == "ge":
        return value >= target
    if op == "gt":
        return value > target
    if op == "le":
        return value <= target
    if op == "lt":
        return value < target
    if op == "eq":
        return value == target
    if op == "ne":
        return value != target
    raise ValueError(f"Unsupported op: {op}")


def _evaluate_rule(row: pd.Series, rule: dict[str, Any], matches_class: Callable[[str], bool]) -> bool:
    feature = str(rule["feature"])
    if feature == "class.is":
        target_class = str(rule["value"])
        return bool(matches_class(target_class))
    value = _resolve_feature_value(row, rule, matches_class)
    return _evaluate_comparison(value, str(rule["op"]), rule["value"])


def assign_phenotypes(object_table: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    classes = config.get("classes", {})
    if object_table.empty or not classes:
        out = object_table.copy()
        out["phenotype"] = "unclassified"
        out["phenotype_priority"] = -1
        out["phenotype_engine"] = "v2"
        return out

    ordered = sorted(classes.items(), key=lambda item: item[1].get("priority", 0), reverse=True)
    phenotypes: list[str] = []
    priorities: list[int] = []

    for _, row in object_table.iterrows():
        cache: dict[str, bool] = {}

        def matches_class(class_name: str, stack: tuple[str, ...] = ()) -> bool:
            if class_name in cache:
                return cache[class_name]
            if class_name in stack:
                raise ValueError(f"Circular phenotype class dependency detected: {' -> '.join(stack + (class_name,))}")
            rule = classes[class_name]
            include_ok = all(
                _evaluate_rule(row, item, lambda nested: matches_class(nested, stack + (class_name,)))
                for item in rule.get("include", [])
            )
            exclude_hit = any(
                _evaluate_rule(row, item, lambda nested: matches_class(nested, stack + (class_name,)))
                for item in rule.get("exclude", [])
            )
            cache[class_name] = include_ok and not exclude_hit
            return cache[class_name]

        assigned = "unclassified"
        assigned_priority = -1
        for name, rule in ordered:
            if matches_class(name):
                assigned = name
                assigned_priority = int(rule.get("priority", 0))
                break
        phenotypes.append(assigned)
        priorities.append(assigned_priority)

    out = object_table.copy()
    out["phenotype"] = phenotypes
    out["phenotype_priority"] = priorities
    out["phenotype_engine"] = "v2"
    return out
