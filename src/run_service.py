from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from src import utils
from src.config import (
    CELL_DIAMETER,
    MAX_CELL_SIZE,
    MICRONS_PER_PIXEL,
    MIN_CELL_SIZE,
    MODEL_TYPE,
    OVERLAY_ALPHA,
    USE_GPU,
    data as CONFIG_DATA,
)
from src.context import RunContext
from src.io_ome import save_labels_to_ome_zarr
from src.measurements import object_table_path_for, write_object_table
from src.model_registry import ModelSpec, model_spec_to_dict, model_summary_fields, model_warning, resolve_model_spec
from src.modalities import adapt_image_for_modality
from src.models import build_segmenter
from src.phenotype import load_rules
from src.phenotype_engine import load_engine_config
from src.pipeline import build_default_pipeline
from src.provenance import build_run_provenance, write_provenance
from src.regions import region_table_path_for, write_region_table
from src.report import write_html_report
from src.retina_coords import (
    registered_density_plot_path,
    retina_frame_output_path,
    save_registered_density_plot,
    write_retina_frame_json,
)
from src.review import resolve_edit_log_path
from src.uncertainty_io import save_float_map
from src.visualize import apply_out_of_focus_overlay, create_debug_overlay, save_debug_image


BBoxSelector = Callable[[np.ndarray], tuple[int, int, int, int]]


@dataclass
class RuntimeOptions:
    backend: str = "cellpose"
    diameter: float | None = None
    model_type: str | None = None
    cellpose_model: str | None = None
    stardist_weights: str | None = None
    model_alias: str | None = None
    min_size: int | None = None
    max_size: int | None = None
    use_gpu: bool | None = None
    modality: str = "flatmount"
    modality_projection: str = "max"
    modality_channel_index: int | None = 0
    modality_slab_start: int | None = None
    modality_slab_end: int | None = None
    apply_clahe: bool = False
    focus_mode: str = "none"
    sam_checkpoint: str | None = None
    phenotype_config: str | None = None
    phenotype_engine: str = "legacy"
    marker_metrics: bool = False
    interaction_metrics: bool = False
    tta: bool = False
    tta_transforms: list[str] | None = None
    spatial_stats: bool = False
    register_retina: bool = False
    region_schema: str = "mouse_flatmount_v1"
    onh_mode: str = "cli"
    onh_xy: tuple[float, float] | None = None
    dorsal_xy: tuple[float, float] | None = None
    retina_frame_path: str | None = None
    save_debug: bool = True
    save_ome_zarr: bool = False
    write_html_report: bool = True
    write_object_table: bool = True
    write_provenance: bool = True
    write_uncertainty_maps: bool = False
    write_qc_maps: bool = False
    strict_schemas: bool = False
    apply_edits: str | None = None
    tiling: bool = False
    tile_size: int = 1024
    tile_overlap: int = 128


@dataclass
class AppRuntime:
    options: RuntimeOptions
    pipeline: Any
    pipeline_cfg: dict[str, Any]
    resolved_config: dict[str, Any]
    model_spec: ModelSpec
    backend: str
    diameter: float | None
    model_type: str
    min_size: int
    max_size: int
    use_gpu: bool
    created_at: datetime


def _load_phenotype_configs(options: RuntimeOptions) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not options.phenotype_config:
        return None, None
    if options.phenotype_engine == "legacy":
        return load_rules(options.phenotype_config), None
    return None, load_engine_config(options.phenotype_config)


def _resolved_config(runtime: AppRuntime) -> dict[str, Any]:
    options = runtime.options
    return {
        "backend": runtime.backend,
        "diameter": runtime.diameter,
        "model_type": runtime.model_type,
        "cellpose_model": options.cellpose_model,
        "stardist_weights": options.stardist_weights,
        "model_alias": options.model_alias,
        **model_summary_fields(runtime.model_spec),
        "model_spec": model_spec_to_dict(runtime.model_spec),
        "min_size": runtime.min_size,
        "max_size": runtime.max_size,
        "use_gpu": runtime.use_gpu,
        "modality": options.modality,
        "modality_projection": options.modality_projection,
        "modality_channel_index": options.modality_channel_index,
        "modality_slab_start": options.modality_slab_start,
        "modality_slab_end": options.modality_slab_end,
        "apply_clahe": options.apply_clahe,
        "focus_mode": options.focus_mode,
        "tta": options.tta,
        "tta_transforms": options.tta_transforms,
        "spatial_stats": options.spatial_stats,
        "phenotype_engine": options.phenotype_engine,
        "marker_metrics": options.marker_metrics,
        "interaction_metrics": options.interaction_metrics,
        "register_retina": options.register_retina,
        "region_schema": options.region_schema,
        "onh_mode": options.onh_mode,
        "onh_xy": options.onh_xy,
        "dorsal_xy": options.dorsal_xy,
        "retina_frame_path": options.retina_frame_path,
        "save_debug": options.save_debug,
        "save_ome_zarr": options.save_ome_zarr,
        "write_html_report": options.write_html_report,
        "write_object_table": options.write_object_table,
        "write_provenance": options.write_provenance,
        "write_uncertainty_maps": options.write_uncertainty_maps,
        "write_qc_maps": options.write_qc_maps,
        "strict_schemas": options.strict_schemas,
        "apply_edits": options.apply_edits,
        "tiling": options.tiling,
        "tile_size": options.tile_size,
        "tile_overlap": options.tile_overlap,
        "source": "napari",
    }


def build_runtime(
    options: RuntimeOptions,
    *,
    bbox_selector: BBoxSelector | None = None,
    segmenter_override: Any | None = None,
) -> AppRuntime:
    diameter = options.diameter if options.diameter is not None else CELL_DIAMETER
    model_type = options.model_type or MODEL_TYPE
    min_size = int(options.min_size if options.min_size is not None else MIN_CELL_SIZE)
    max_size = int(options.max_size if options.max_size is not None else MAX_CELL_SIZE)
    use_gpu = bool(options.use_gpu) if options.use_gpu is not None else bool(USE_GPU and torch.cuda.is_available())
    if segmenter_override is not None and (options.backend or "cellpose").lower() not in {"cellpose", "stardist", "sam"}:
        backend_name = (options.backend or "override").lower()
        model_spec = ModelSpec(
            backend=backend_name,
            source="builtin",
            model_label=f"{backend_name}_override:injected",
            display_label=options.model_alias or f"{backend_name}_override:injected",
            builtin_name=model_type,
            asset_path=None,
            model_type=model_type,
            alias=options.model_alias,
            trust_mode="builtin",
        )
    else:
        model_spec = resolve_model_spec(
            backend=options.backend,
            model_type=model_type,
            cellpose_model=options.cellpose_model,
            stardist_weights=options.stardist_weights,
            sam_checkpoint=options.sam_checkpoint,
            model_alias=options.model_alias,
        )
    backend = model_spec.backend
    warning = model_warning(model_spec)
    if warning:
        print(f"[WARNING] {warning}")

    phenotype_rules, phenotype_engine_config = _load_phenotype_configs(options)
    segmenter = segmenter_override or build_segmenter(
        model_spec=model_spec,
        diameter=diameter,
        use_gpu=use_gpu,
    )
    pipeline_cfg = {
        "apply_clahe": options.apply_clahe,
        "focus_mode": options.focus_mode,
        "tta": options.tta,
        "tta_transforms": options.tta_transforms,
        "tiling": options.tiling,
        "tile_size": options.tile_size,
        "tile_overlap": options.tile_overlap,
        "min_size": min_size,
        "max_size": max_size,
        "qc_config": copy.deepcopy(CONFIG_DATA.get("qc", {})),
        "spatial_stats": options.spatial_stats,
        "backend": backend,
        "use_gpu": use_gpu,
        "model_spec": model_summary_fields(model_spec),
        "phenotype_engine": options.phenotype_engine,
        "marker_metrics": options.marker_metrics,
        "interaction_metrics": options.interaction_metrics,
        "phenotype_rules": phenotype_rules,
        "phenotype_engine_config": phenotype_engine_config,
        "register_retina": options.register_retina,
        "region_schema": options.region_schema,
        "onh_mode": options.onh_mode,
        "onh_xy": options.onh_xy,
        "dorsal_xy": options.dorsal_xy,
        "retina_frame_path": options.retina_frame_path,
        "apply_edits": options.apply_edits,
    }
    pipeline = build_default_pipeline(
        segmenter,
        bbox_selector=bbox_selector,
        phenotype_rules=phenotype_rules,
        phenotype_engine_config=phenotype_engine_config,
    )
    runtime = AppRuntime(
        options=options,
        pipeline=pipeline,
        pipeline_cfg=pipeline_cfg,
        resolved_config={},
        model_spec=model_spec,
        backend=backend,
        diameter=diameter,
        model_type=model_spec.model_type or model_type,
        min_size=min_size,
        max_size=max_size,
        use_gpu=use_gpu,
        created_at=datetime.now(),
    )
    runtime.resolved_config = _resolved_config(runtime)
    return runtime


def run_array(
    runtime: AppRuntime,
    *,
    image: np.ndarray,
    source_path: str | Path,
    meta: dict[str, Any] | None = None,
) -> RunContext:
    adapted_image, adapted_meta = adapt_image_for_modality(
        image,
        meta,
        modality=runtime.options.modality,
        projection=runtime.options.modality_projection,
        channel_index=runtime.options.modality_channel_index,
        slab_start=runtime.options.modality_slab_start,
        slab_end=runtime.options.modality_slab_end,
    )
    ctx = RunContext(path=Path(source_path), image=adapted_image, meta=adapted_meta or {})
    ctx = runtime.pipeline.run(ctx, dict(runtime.pipeline_cfg))
    ctx.metrics["modality"] = runtime.options.modality
    ctx.summary_row["modality"] = runtime.options.modality
    return ctx


def summarize_context(ctx: RunContext) -> str:
    density = ctx.metrics.get("density_cells_per_mm2")
    density_text = f"{density:.2f}" if isinstance(density, (int, float)) else "n/a"
    lines = [
        f"File: {ctx.path.name}",
        f"Cells: {ctx.metrics.get('cell_count', 'n/a')}",
        f"Density: {density_text} cells/mm^2",
        f"Backend: {ctx.metrics.get('backend', ctx.seg_info.get('backend', 'unknown'))}",
        f"Model: {ctx.metrics.get('model_label', 'unknown')}",
    ]
    if ctx.warnings:
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in ctx.warnings)
    return "\n".join(lines)


def build_debug_preview(ctx: RunContext, *, focus_mode: str) -> np.ndarray | None:
    if ctx.gray is None or ctx.labels is None:
        return None
    debug_image = create_debug_overlay(ctx.gray, ctx.labels, alpha=OVERLAY_ALPHA)
    if focus_mode in ("auto", "qc") and ctx.qc_mask is not None:
        debug_image = apply_out_of_focus_overlay(debug_image, ctx.qc_mask, alpha=0.3)
    return debug_image


def _write_debug_artifact(
    *,
    ctx: RunContext,
    runtime: AppRuntime,
    output_dir: Path,
    saved_images_for_report: list[tuple[str, str]],
) -> None:
    if not runtime.options.save_debug:
        return
    debug_image = build_debug_preview(ctx, focus_mode=runtime.options.focus_mode)
    if debug_image is None:
        return
    debug_path = output_dir / f"{ctx.path.name.rsplit('.', 1)[0]}_debug.png"
    save_debug_image(debug_image, str(debug_path))
    ctx.artifacts["debug_overlay"] = debug_path
    saved_images_for_report.append((f"Debug overlay {ctx.path.name}", debug_path.name))

    if runtime.options.spatial_stats and "isodensity_map" in ctx.state:
        import cv2

        iso8 = utils.safe_uint8(ctx.state["isodensity_map"])
        iso_color = cv2.applyColorMap(iso8, cv2.COLORMAP_JET)
        iso_path = output_dir / f"{ctx.path.name.rsplit('.', 1)[0]}_isodensity.png"
        cv2.imwrite(str(iso_path), iso_color)
        ctx.artifacts["isodensity_map"] = iso_path
        saved_images_for_report.append((f"Isodensity {ctx.path.name}", iso_path.name))


def _save_map_preview(array: np.ndarray, destination: Path) -> Path:
    import cv2

    destination.parent.mkdir(parents=True, exist_ok=True)
    preview = cv2.applyColorMap(utils.safe_uint8(array), cv2.COLORMAP_VIRIDIS)
    cv2.imwrite(str(destination), preview)
    return destination


def export_context(
    runtime: AppRuntime,
    ctx: RunContext,
    output_dir: str | Path,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_images_for_report: list[tuple[str, str]] = []
    report_assets: list[tuple[str, str]] = []

    csv_path = output_dir / "results.csv"
    utils.save_results_to_csv([dict(ctx.summary_row)], str(csv_path))
    ctx.artifacts["results_csv"] = csv_path

    _write_debug_artifact(
        ctx=ctx,
        runtime=runtime,
        output_dir=output_dir,
        saved_images_for_report=saved_images_for_report,
    )

    if runtime.options.save_ome_zarr and ctx.gray is not None and ctx.labels is not None:
        zarr_dir = output_dir / f"{ctx.path.name.rsplit('.', 1)[0]}.zarr"
        save_labels_to_ome_zarr(
            ctx.gray,
            ctx.labels,
            str(zarr_dir),
            {
                "backend": ctx.seg_info.get("backend", runtime.backend),
                "use_gpu": runtime.use_gpu,
                "microns_per_pixel": MICRONS_PER_PIXEL,
            },
            chunk=256,
        )
        ctx.artifacts["ome_zarr"] = zarr_dir

    if runtime.options.write_object_table and ctx.object_table is not None:
        object_path = write_object_table(
            ctx.object_table,
            object_table_path_for(output_dir, ctx.path),
            strict=runtime.options.strict_schemas,
        )
        ctx.artifacts["object_table"] = object_path
        if object_path.suffix != ".parquet":
            ctx.warnings.append("Object table written as CSV because no parquet engine was available.")

    if runtime.options.write_uncertainty_maps and ctx.state.get("foreground_probability") is not None:
        uncertainty_dir = output_dir / "uncertainty"
        tif_path = save_float_map(
            ctx.state["foreground_probability"],
            uncertainty_dir / f"{ctx.path.name.rsplit('.', 1)[0]}_fgprob.tif",
            metadata={"kind": "foreground_probability", "image_id": ctx.path.name.rsplit('.', 1)[0]},
        )
        ctx.artifacts["foreground_probability"] = tif_path
        report_assets.append(("Foreground probability map", str(tif_path.relative_to(output_dir))))
        preview = _save_map_preview(ctx.state["foreground_probability"], uncertainty_dir / f"{ctx.path.name.rsplit('.', 1)[0]}_fgprob_preview.png")
        ctx.artifacts["foreground_probability_preview"] = preview
        saved_images_for_report.append(("Foreground probability preview", str(preview.relative_to(output_dir))))

    if runtime.options.write_qc_maps and ctx.state.get("focus_score_map") is not None:
        qc_dir = output_dir / "qc_maps"
        tif_path = save_float_map(
            ctx.state["focus_score_map"],
            qc_dir / f"{ctx.path.name.rsplit('.', 1)[0]}_focus_score.tif",
            metadata={"kind": "focus_score_map", "image_id": ctx.path.name.rsplit('.', 1)[0]},
        )
        ctx.artifacts["focus_score_map"] = tif_path
        report_assets.append(("Focus score map", str(tif_path.relative_to(output_dir))))
        preview = _save_map_preview(ctx.state["focus_score_map"], qc_dir / f"{ctx.path.name.rsplit('.', 1)[0]}_focus_score_preview.png")
        ctx.artifacts["focus_score_map_preview"] = preview
        saved_images_for_report.append(("Focus score preview", str(preview.relative_to(output_dir))))

    if runtime.options.register_retina and "retina_frame" in ctx.state:
        frame_path = write_retina_frame_json(ctx.state["retina_frame"], retina_frame_output_path(output_dir, ctx.path))
        ctx.artifacts["retina_frame"] = frame_path

        if ctx.region_table is not None:
            region_path = write_region_table(
                ctx.region_table,
                region_table_path_for(output_dir, ctx.path),
                strict=runtime.options.strict_schemas,
            )
            ctx.artifacts["region_table"] = region_path

        if ctx.object_table is not None and not ctx.object_table.empty:
            map_png = save_registered_density_plot(
                ctx.object_table,
                registered_density_plot_path(output_dir, ctx.path, suffix=".png"),
            )
            ctx.artifacts["registered_density_map_png"] = map_png
            saved_images_for_report.append(("Registered density map", str(map_png.relative_to(output_dir))))

            map_svg = save_registered_density_plot(
                ctx.object_table,
                registered_density_plot_path(output_dir, ctx.path, suffix=".svg"),
            )
            ctx.artifacts["registered_density_map_svg"] = map_svg

    edit_log = resolve_edit_log_path(ctx.path, runtime.options.apply_edits)
    if edit_log is not None and edit_log.exists():
        ctx.artifacts.setdefault("edit_log", edit_log)
        report_assets.append(("Review edits", str(edit_log.relative_to(output_dir)) if edit_log.is_relative_to(output_dir) else edit_log.name))

    if runtime.options.write_html_report:
        report_path = write_html_report(
            str(output_dir),
            {
                "source": "napari",
                "backend": runtime.backend,
                "model_label": runtime.model_spec.model_label,
                "model_source": runtime.model_spec.source,
                "modality": runtime.options.modality,
                "diameter": runtime.diameter,
                "min_size": runtime.min_size,
                "max_size": runtime.max_size,
                "gpu": runtime.use_gpu,
                "focus_mode": runtime.options.focus_mode,
                "tta": runtime.options.tta,
            },
            [dict(ctx.summary_row)],
            images=saved_images_for_report,
            notes="Exported from the napari dock widget.",
            assets=report_assets,
        )
        ctx.artifacts["html_report"] = Path(report_path)

    if runtime.options.write_provenance:
        provenance_path = write_provenance(
            output_dir / "provenance.json",
            build_run_provenance(
                args={"source": "napari", "output_dir": str(output_dir)},
                resolved_config=runtime.resolved_config,
                contexts=[ctx],
                run_started_at=runtime.created_at,
                run_finished_at=datetime.now(),
                results_csv_path=csv_path,
                model_spec=model_spec_to_dict(runtime.model_spec),
            ),
        )
        ctx.artifacts["provenance"] = provenance_path

    return {key: value for key, value in ctx.artifacts.items()}
