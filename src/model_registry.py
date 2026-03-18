from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


TRUSTED_LOCAL_ONLY = "trusted_local_only"
DEFAULT_STARDIST_MODEL = "2D_versatile_fluo"
DEFAULT_SAM_MODEL_TYPE = "vit_h"


@dataclass(frozen=True)
class ModelSpec:
    backend: str
    source: str
    model_label: str
    display_label: str
    builtin_name: str | None
    asset_path: str | None
    model_type: str | None
    alias: str | None
    trust_mode: str


def _normalize_backend(value: str | None) -> str | None:
    if value is None:
        return None
    return str(value).strip().lower() or None


def _existing_path(value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.exists():
        return path.resolve()
    return None


def _custom_asset_flags(
    *,
    cellpose_model: str | None,
    stardist_weights: str | None,
    sam_checkpoint: str | None,
) -> dict[str, str]:
    flags: dict[str, str] = {}
    if cellpose_model:
        flags["cellpose"] = cellpose_model
    if stardist_weights:
        flags["stardist"] = stardist_weights
    if sam_checkpoint:
        flags["sam"] = sam_checkpoint
    return flags


def _display_label(model_label: str, alias: str | None) -> str:
    return str(alias).strip() if alias and str(alias).strip() else model_label


def _builtin_spec(*, backend: str, builtin_name: str, alias: str | None, model_type: str | None = None) -> ModelSpec:
    label = f"{backend}_builtin:{builtin_name}"
    return ModelSpec(
        backend=backend,
        source="builtin",
        model_label=label,
        display_label=_display_label(label, alias),
        builtin_name=builtin_name,
        asset_path=None,
        model_type=model_type or builtin_name,
        alias=alias,
        trust_mode="builtin",
    )


def _custom_spec(
    *,
    backend: str,
    asset_path: str,
    alias: str | None,
    source: str = "custom",
    model_type: str | None = None,
    builtin_name: str | None = None,
) -> ModelSpec:
    normalized = str(Path(asset_path).expanduser().resolve())
    label = f"{backend}_{source}:{normalized}"
    return ModelSpec(
        backend=backend,
        source=source,
        model_label=label,
        display_label=_display_label(label, alias),
        builtin_name=builtin_name,
        asset_path=normalized,
        model_type=model_type,
        alias=alias,
        trust_mode=TRUSTED_LOCAL_ONLY,
    )


def _validate_path(value: str, *, label: str) -> str:
    path = _existing_path(value)
    if path is None:
        raise FileNotFoundError(f"{label} not found: {value}")
    return str(path)


def resolve_model_spec(
    *,
    backend: str | None,
    model_type: str | None,
    cellpose_model: str | None,
    stardist_weights: str | None,
    sam_checkpoint: str | None,
    model_alias: str | None,
) -> ModelSpec:
    alias = str(model_alias).strip() if model_alias is not None and str(model_alias).strip() else None
    requested_backend = _normalize_backend(backend)
    custom_flags = _custom_asset_flags(
        cellpose_model=cellpose_model,
        stardist_weights=stardist_weights,
        sam_checkpoint=sam_checkpoint,
    )
    if len(custom_flags) > 1:
        raise ValueError("Only one custom model asset may be provided per run.")

    inferred_backend = requested_backend
    if inferred_backend is None:
        if cellpose_model:
            inferred_backend = "cellpose"
        elif stardist_weights:
            inferred_backend = "stardist"
        elif sam_checkpoint:
            inferred_backend = "sam"
        else:
            inferred_backend = "cellpose"

    if inferred_backend not in {"cellpose", "stardist", "sam", "blob_watershed"}:
        raise ValueError(f"Unsupported backend: {inferred_backend}")

    for flag_backend, value in custom_flags.items():
        if requested_backend is not None and flag_backend != inferred_backend:
            raise ValueError(
                f"Custom asset for backend '{flag_backend}' cannot be used with explicit backend '{requested_backend}'."
            )
        if value is None:
            continue

    if inferred_backend == "cellpose":
        if cellpose_model:
            asset_path = _validate_path(cellpose_model, label="Cellpose checkpoint")
            return _custom_spec(
                backend="cellpose",
                asset_path=asset_path,
                alias=alias,
                source="custom",
                model_type=asset_path,
            )

        legacy_path = _existing_path(model_type)
        if legacy_path is not None:
            return _custom_spec(
                backend="cellpose",
                asset_path=str(legacy_path),
                alias=alias,
                source="legacy_custom",
                model_type=str(legacy_path),
            )

        builtin_name = str(model_type).strip() if model_type is not None and str(model_type).strip() else "cyto"
        return _builtin_spec(backend="cellpose", builtin_name=builtin_name, alias=alias, model_type=builtin_name)

    if inferred_backend == "stardist":
        if stardist_weights:
            asset_path = _validate_path(stardist_weights, label="StarDist weights")
            return _custom_spec(
                backend="stardist",
                asset_path=asset_path,
                alias=alias,
                source="custom",
                model_type=None,
            )
        return _builtin_spec(
            backend="stardist",
            builtin_name=DEFAULT_STARDIST_MODEL,
            alias=alias,
            model_type=None,
        )

    if inferred_backend == "blob_watershed":
        builtin_name = str(model_type).strip() if model_type is not None and str(model_type).strip() else "blob_watershed"
        return _builtin_spec(
            backend="blob_watershed",
            builtin_name=builtin_name,
            alias=alias,
            model_type=builtin_name,
        )

    if not sam_checkpoint:
        raise ValueError("SAM backend requires --sam_checkpoint.")
    asset_path = _validate_path(sam_checkpoint, label="SAM checkpoint")
    return _custom_spec(
        backend="sam",
        asset_path=asset_path,
        alias=alias,
        source="custom",
        model_type=DEFAULT_SAM_MODEL_TYPE,
        builtin_name=None,
    )


def model_spec_to_dict(model_spec: ModelSpec) -> dict[str, Any]:
    return asdict(model_spec)


def model_summary_fields(model_spec: ModelSpec) -> dict[str, Any]:
    return {
        "model_label": model_spec.model_label,
        "model_source": model_spec.source,
        "model_alias": model_spec.alias,
        "model_asset_path": model_spec.asset_path,
        "model_builtin_name": model_spec.builtin_name,
        "model_trust_mode": model_spec.trust_mode,
    }


def model_warning(model_spec: ModelSpec) -> str | None:
    if model_spec.source in {"custom", "legacy_custom"}:
        return (
            f"Using {model_spec.source.replace('_', ' ')} model asset '{model_spec.display_label}'. "
            "Custom checkpoints and weights are treated as trusted local assets only."
        )
    return None
