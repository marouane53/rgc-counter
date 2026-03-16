from __future__ import annotations

import json
import platform
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from src.context import RunContext
from src.schema import PROVENANCE_VERSION


def safe_package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def collect_environment_info() -> dict[str, Any]:
    packages = [
        "numpy",
        "pandas",
        "torch",
        "cellpose",
        "aicsimageio",
        "ome-zarr",
        "scikit-image",
        "scikit-learn",
    ]
    package_versions = {name: safe_package_version(name) for name in packages}
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def summarize_context(ctx: RunContext) -> dict[str, Any]:
    return {
        "path": str(ctx.path),
        "meta": ctx.meta,
        "metrics": ctx.metrics,
        "warnings": ctx.warnings,
        "summary_row": ctx.summary_row,
        "artifacts": {key: str(value) for key, value in ctx.artifacts.items()},
    }


def build_run_provenance(
    *,
    args: dict[str, Any],
    resolved_config: dict[str, Any],
    contexts: list[RunContext],
    run_started_at: datetime,
    run_finished_at: datetime,
    results_csv_path: str | Path,
    study_statistics: dict[str, Any] | None = None,
    model_spec: dict[str, Any] | None = None,
    spatial_analysis: dict[str, Any] | None = None,
    atlas_subtypes: dict[str, Any] | None = None,
    tracking: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "provenance_version": PROVENANCE_VERSION,
        "run_started_at": run_started_at,
        "run_finished_at": run_finished_at,
        "args": args,
        "resolved_config": resolved_config,
        "environment": collect_environment_info(),
        "results_csv_path": str(results_csv_path),
        "images": [summarize_context(ctx) for ctx in contexts],
    }
    if study_statistics is not None:
        payload["study_statistics"] = study_statistics
    if model_spec is not None:
        payload["model_spec"] = model_spec
    if spatial_analysis is not None:
        payload["spatial_analysis"] = spatial_analysis
    if atlas_subtypes is not None:
        payload["atlas_subtypes"] = atlas_subtypes
    if tracking is not None:
        payload["tracking"] = tracking
    return payload


def write_provenance(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)
    return path
