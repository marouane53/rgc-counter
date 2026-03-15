# main.py

from __future__ import annotations

import argparse
import copy
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

from src import utils
from src.atlas import compare_region_table_to_atlas, load_atlas_reference, summarize_atlas_comparison
from src.calibration import (
    apply_dotted_overrides,
    evaluate_count_agreement,
    load_calibration_grid,
    rank_grid_results,
    write_best_params,
    write_calibration_report,
)
from src.cohort import build_sample_table, build_study_region_table, write_table_bundle
from src.figures import (
    save_atlas_deviation_plot,
    save_condition_summary_plot,
    save_paired_plot,
    save_phenotype_composition_plot,
    save_region_density_plot,
)
from src.methods import build_methods_appendix, write_methods_appendix
from src.visualize import create_debug_overlay, save_debug_image, apply_out_of_focus_overlay
from src.config import (
    CELL_DIAMETER, MODEL_TYPE, USE_GPU,
    MIN_CELL_SIZE, MAX_CELL_SIZE, OVERLAY_ALPHA,
    MICRONS_PER_PIXEL, data as CONFIG_DATA
)
from src.context import RunContext
from src.io_ome import load_any_image
from src.manifest import load_manifest
from src.measurements import object_table_path_for, write_object_table
from src.modalities import adapt_image_for_modality
from src.models import build_segmenter
from src.io_ome import save_labels_to_ome_zarr
from src.pipeline import build_default_pipeline
from src.phenotype_engine import load_engine_config
from src.phenotype import load_rules
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
from src.stats import compute_outcome_stats, compute_region_stats
from src.track import build_longitudinal_track_table, summarize_tracks
from src.uncertainty_io import save_float_map
from src.validation import (
    build_validation_table,
    save_agreement_scatter_plot,
    save_bland_altman_plot,
    summarize_validation,
)


def _output_stem(filepath: str | Path) -> str:
    return Path(filepath).name.rsplit(".", 1)[0]


def _resolve_focus_mode(args: argparse.Namespace) -> str:
    if args.focus_bbox:
        return "bbox"
    if args.focus_auto:
        return "auto"
    if args.focus_qc:
        return "qc"
    return "none"


def _resolve_modality(args: argparse.Namespace, manifest_row: dict[str, object] | None = None) -> str:
    if manifest_row is not None and manifest_row.get("modality"):
        return str(manifest_row["modality"])
    return args.modality


def _adapt_loaded_image(
    image,
    meta: dict,
    *,
    modality: str,
    args: argparse.Namespace,
):
    adapted_image, adapted_meta = adapt_image_for_modality(
        image,
        meta,
        modality=modality,
        projection=args.modality_projection,
        channel_index=args.modality_channel_index,
        slab_start=args.modality_slab_start,
        slab_end=args.modality_slab_end,
    )
    adapted_meta["modality"] = modality
    return adapted_image, adapted_meta


def _build_pipeline_cfg(
    args: argparse.Namespace,
    *,
    backend: str,
    use_gpu: bool,
    min_size: int,
    max_size: int,
    phenotype_rules: dict | None = None,
    phenotype_engine_config: dict | None = None,
) -> dict[str, object]:
    return {
        "apply_clahe": args.apply_clahe,
        "focus_mode": _resolve_focus_mode(args),
        "tta": args.tta,
        "tta_transforms": args.tta_transforms,
        "tiling": args.tiling,
        "tile_size": args.tile_size,
        "tile_overlap": args.tile_overlap,
        "min_size": min_size,
        "max_size": max_size,
        "qc_config": copy.deepcopy(CONFIG_DATA.get("qc", {})),
        "spatial_stats": args.spatial_stats,
        "backend": backend,
        "use_gpu": use_gpu,
        "phenotype_engine": args.phenotype_engine,
        "marker_metrics": args.marker_metrics,
        "interaction_metrics": args.interaction_metrics,
        "phenotype_rules": phenotype_rules,
        "phenotype_engine_config": phenotype_engine_config,
        "register_retina": args.register_retina,
        "region_schema": args.region_schema,
        "onh_mode": args.onh_mode,
        "onh_xy": tuple(args.onh_xy) if args.onh_xy is not None else None,
        "dorsal_xy": tuple(args.dorsal_xy) if args.dorsal_xy is not None else None,
        "retina_frame_path": args.retina_frame_path,
        "apply_edits": args.apply_edits,
    }


def _load_phenotype_configs(args: argparse.Namespace) -> tuple[dict | None, dict | None]:
    ph_rules = None
    phenotype_engine_config = None
    if args.phenotype_config:
        if args.phenotype_engine == "legacy":
            ph_rules = load_rules(args.phenotype_config)
        else:
            phenotype_engine_config = load_engine_config(args.phenotype_config)
    return ph_rules, phenotype_engine_config


def _save_map_preview(array, destination: str | Path) -> Path:
    import cv2

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    array8 = utils.safe_uint8(array)
    preview = cv2.applyColorMap(array8, cv2.COLORMAP_VIRIDIS)
    cv2.imwrite(str(destination), preview)
    return destination


def _build_sample_cfg(
    base_cfg: dict[str, object],
    row: dict[str, object] | None,
    *,
    register_retina: bool,
    retina_frame_path: str | None,
) -> dict[str, object]:
    sample_cfg = copy.deepcopy(base_cfg)
    if row is None or not register_retina or retina_frame_path:
        return sample_cfg
    if all(column in row and pd.notna(row[column]) for column in ("onh_x_px", "onh_y_px", "dorsal_x_px", "dorsal_y_px")):
        sample_cfg["onh_mode"] = "cli"
        sample_cfg["onh_xy"] = (float(row["onh_x_px"]), float(row["onh_y_px"]))
        sample_cfg["dorsal_xy"] = (float(row["dorsal_x_px"]), float(row["dorsal_y_px"]))
    return sample_cfg


def _run_pipeline_context(
    *,
    filepath: str | Path,
    image,
    meta: dict,
    pipeline,
    pipeline_cfg: dict[str, object],
) -> RunContext:
    ctx = RunContext(path=Path(filepath), image=image, meta=meta)
    return pipeline.run(ctx, copy.deepcopy(pipeline_cfg))


def _write_context_artifacts(
    *,
    ctx: RunContext,
    filepath: str | Path,
    output_dir: str | Path,
    report_root: str | Path,
    args: argparse.Namespace,
    use_gpu: bool,
    focus_mode: str,
    saved_images_for_report: list[tuple[str, str]],
    report_assets: list[tuple[str, str]],
) -> None:
    output_dir = str(output_dir)
    report_root = str(report_root)
    filepath = str(filepath)

    if args.save_debug and ctx.gray is not None and ctx.labels is not None:
        debug_image = create_debug_overlay(ctx.gray, ctx.labels, alpha=OVERLAY_ALPHA)
        if focus_mode in ("auto", "qc"):
            debug_image = apply_out_of_focus_overlay(debug_image, ctx.qc_mask, alpha=0.3)
        debug_filename = _output_stem(filepath) + "_debug.png"
        out_path = os.path.join(output_dir, debug_filename)
        save_debug_image(debug_image, out_path)
        saved_images_for_report.append(("Debug overlay " + os.path.basename(filepath), os.path.relpath(out_path, report_root)))
        ctx.artifacts["debug_overlay"] = Path(out_path)

        if args.spatial_stats and "isodensity_map" in ctx.state:
            iso = ctx.state["isodensity_map"]
            import cv2
            iso8 = utils.safe_uint8(iso)
            iso_color = cv2.applyColorMap(iso8, cv2.COLORMAP_JET)
            iso_path = os.path.join(output_dir, _output_stem(filepath) + "_isodensity.png")
            cv2.imwrite(iso_path, iso_color)
            saved_images_for_report.append(("Isodensity " + os.path.basename(filepath), os.path.relpath(iso_path, report_root)))
            ctx.artifacts["isodensity_map"] = Path(iso_path)

    if args.write_uncertainty_maps and ctx.state.get("foreground_probability") is not None:
        uncertainty_dir = Path(output_dir) / "uncertainty"
        tif_path = uncertainty_dir / f"{_output_stem(filepath)}_fgprob.tif"
        written_tif = save_float_map(
            ctx.state["foreground_probability"],
            tif_path,
            metadata={"kind": "foreground_probability", "image_id": _output_stem(filepath)},
        )
        ctx.artifacts["foreground_probability"] = written_tif
        report_assets.append((f"Foreground probability map {os.path.basename(filepath)}", os.path.relpath(written_tif, report_root)))
        preview_path = uncertainty_dir / f"{_output_stem(filepath)}_fgprob_preview.png"
        preview = _save_map_preview(ctx.state["foreground_probability"], preview_path)
        ctx.artifacts["foreground_probability_preview"] = preview
        saved_images_for_report.append((f"Foreground probability preview {os.path.basename(filepath)}", os.path.relpath(preview, report_root)))

    if args.write_qc_maps and ctx.state.get("focus_score_map") is not None:
        qc_dir = Path(output_dir) / "qc_maps"
        tif_path = qc_dir / f"{_output_stem(filepath)}_focus_score.tif"
        written_tif = save_float_map(
            ctx.state["focus_score_map"],
            tif_path,
            metadata={"kind": "focus_score_map", "image_id": _output_stem(filepath)},
        )
        ctx.artifacts["focus_score_map"] = written_tif
        report_assets.append((f"Focus score map {os.path.basename(filepath)}", os.path.relpath(written_tif, report_root)))
        preview_path = qc_dir / f"{_output_stem(filepath)}_focus_score_preview.png"
        preview = _save_map_preview(ctx.state["focus_score_map"], preview_path)
        ctx.artifacts["focus_score_map_preview"] = preview
        saved_images_for_report.append((f"Focus score preview {os.path.basename(filepath)}", os.path.relpath(preview, report_root)))

    if args.save_ome_zarr and ctx.gray is not None and ctx.labels is not None:
        zarr_dir = os.path.join(output_dir, _output_stem(filepath) + ".zarr")
        try:
            meta_to_write = {
                "backend": ctx.seg_info.get("backend", "unknown"),
                "use_gpu": use_gpu,
                "microns_per_pixel": MICRONS_PER_PIXEL,
            }
            save_labels_to_ome_zarr(ctx.gray, ctx.labels, zarr_dir, meta_to_write, chunk=256)
            ctx.artifacts["ome_zarr"] = Path(zarr_dir)
        except Exception as exc:
            print(f"[WARN] Failed to save OME-Zarr: {exc}")
            ctx.warnings.append(f"Failed to save OME-Zarr: {exc}")

    if args.write_object_table and ctx.object_table is not None:
        object_table_path = object_table_path_for(output_dir, filepath)
        written_path = write_object_table(ctx.object_table, object_table_path, strict=args.strict_schemas)
        ctx.artifacts["object_table"] = written_path
        if written_path.suffix != ".parquet":
            ctx.warnings.append("Object table written as CSV because no parquet engine was available.")

    if args.register_retina and "retina_frame" in ctx.state:
        frame_path = retina_frame_output_path(output_dir, filepath)
        written_frame = write_retina_frame_json(ctx.state["retina_frame"], frame_path)
        ctx.artifacts["retina_frame"] = written_frame

        if ctx.region_table is not None:
            region_path = region_table_path_for(output_dir, filepath)
            written_region = write_region_table(ctx.region_table, region_path, strict=args.strict_schemas)
            ctx.artifacts["region_table"] = written_region

        if ctx.object_table is not None:
            map_png_path = registered_density_plot_path(output_dir, filepath, suffix=".png")
            written_png = save_registered_density_plot(ctx.object_table, map_png_path)
            ctx.artifacts["registered_density_map_png"] = written_png
            saved_images_for_report.append(
                ("Registered density map " + os.path.basename(filepath), os.path.relpath(written_png, report_root))
            )
            map_svg_path = registered_density_plot_path(output_dir, filepath, suffix=".svg")
            written_svg = save_registered_density_plot(ctx.object_table, map_svg_path)
            ctx.artifacts["registered_density_map_svg"] = written_svg

    edit_log = resolve_edit_log_path(filepath, args.apply_edits)
    if edit_log is not None and edit_log.exists():
        ctx.artifacts.setdefault("edit_log", edit_log)
        report_assets.append((f"Review edits {os.path.basename(filepath)}", os.path.relpath(edit_log, report_root)))


def _process_single_image(
    *,
    filepath: str | Path,
    image,
    meta: dict,
    pipeline,
    pipeline_cfg: dict[str, object],
    args: argparse.Namespace,
    output_dir: str | Path,
    report_root: str | Path,
    use_gpu: bool,
    focus_mode: str,
    saved_images_for_report: list[tuple[str, str]],
    report_assets: list[tuple[str, str]],
) -> RunContext:
    ctx = _run_pipeline_context(
        filepath=filepath,
        image=image,
        meta=meta,
        pipeline=pipeline,
        pipeline_cfg=pipeline_cfg,
    )
    _write_context_artifacts(
        ctx=ctx,
        filepath=filepath,
        output_dir=output_dir,
        report_root=report_root,
        args=args,
        use_gpu=use_gpu,
        focus_mode=focus_mode,
        saved_images_for_report=saved_images_for_report,
        report_assets=report_assets,
    )
    return ctx


def _resolved_config_dict(
    args: argparse.Namespace,
    *,
    diameter: float | None,
    model_type: str,
    min_size: int,
    max_size: int,
    use_gpu: bool,
    focus_mode: str,
    backend: str,
) -> dict[str, object]:
    return {
        "diameter": diameter,
        "model_type": model_type,
        "min_size": min_size,
        "max_size": max_size,
        "use_gpu": use_gpu,
        "focus_mode": focus_mode,
        "backend": backend,
        "save_debug": args.save_debug,
        "save_ome_zarr": args.save_ome_zarr,
        "write_html_report": args.write_html_report,
        "write_object_table": args.write_object_table,
        "write_provenance": args.write_provenance,
        "write_uncertainty_maps": args.write_uncertainty_maps,
        "write_qc_maps": args.write_qc_maps,
        "strict_schemas": args.strict_schemas,
        "spatial_stats": args.spatial_stats,
        "tta": args.tta,
        "tiling": args.tiling,
        "tile_size": args.tile_size,
        "tile_overlap": args.tile_overlap,
        "modality": args.modality,
        "modality_projection": args.modality_projection,
        "modality_channel_index": args.modality_channel_index,
        "modality_slab_start": args.modality_slab_start,
        "modality_slab_end": args.modality_slab_end,
        "phenotype_engine": args.phenotype_engine,
        "marker_metrics": args.marker_metrics,
        "interaction_metrics": args.interaction_metrics,
        "register_retina": args.register_retina,
        "region_schema": args.region_schema,
        "onh_mode": args.onh_mode,
        "onh_xy": args.onh_xy,
        "dorsal_xy": args.dorsal_xy,
        "retina_frame_path": args.retina_frame_path,
        "apply_edits": args.apply_edits,
        "atlas_reference": args.atlas_reference,
        "track_longitudinal": args.track_longitudinal,
        "track_max_disp_px": args.track_max_disp_px,
        "manifest": args.manifest,
        "study_output_dir": args.study_output_dir,
        "calibration_grid": args.calibration_grid,
    }


def _merge_manual_annotations(manifest_df: pd.DataFrame, path: str | None) -> pd.DataFrame:
    if not path:
        return manifest_df
    manual_df = pd.read_csv(path)
    if "sample_id" not in manual_df.columns:
        raise ValueError("Manual annotations CSV must contain a sample_id column.")
    merged = manifest_df.merge(manual_df, on="sample_id", how="left", suffixes=("", "_manual"))
    if "label_path_manual" in merged.columns:
        merged["label_path"] = merged["label_path"].fillna(merged["label_path_manual"])
        merged = merged.drop(columns=["label_path_manual"])
    if "manual_count" in merged.columns:
        if "expected_total_objects" in merged.columns:
            merged["expected_total_objects"] = merged["manual_count"].fillna(merged["expected_total_objects"])
        else:
            merged["expected_total_objects"] = merged["manual_count"]
    return merged


def _write_atlas_outputs(
    *,
    atlas_reference_path: str,
    region_table: pd.DataFrame,
    output_dir: Path,
    saved_images_for_report: list[tuple[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if region_table.empty:
        return pd.DataFrame(), pd.DataFrame()

    atlas_reference = load_atlas_reference(atlas_reference_path)
    atlas_comparison = compare_region_table_to_atlas(region_table, atlas_reference)
    if atlas_comparison.empty:
        return pd.DataFrame(), pd.DataFrame()

    atlas_dir = output_dir / "atlas"
    atlas_dir.mkdir(parents=True, exist_ok=True)
    atlas_comparison.to_csv(atlas_dir / "atlas_comparison.csv", index=False)
    atlas_summary = summarize_atlas_comparison(atlas_comparison)
    if not atlas_summary.empty:
        atlas_summary.to_csv(atlas_dir / "atlas_summary.csv", index=False)

    atlas_plot = save_atlas_deviation_plot(atlas_comparison, output_dir / "figures" / "atlas_deviation_by_region.png")
    if atlas_plot is not None:
        saved_images_for_report.append(("Atlas deviation by region", os.path.relpath(atlas_plot, output_dir)))
    return atlas_comparison, atlas_summary


def _write_tracking_outputs(
    *,
    manifest_df: pd.DataFrame,
    contexts: list[RunContext],
    output_dir: Path,
    max_disp_px: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    track_table = build_longitudinal_track_table(manifest_df, contexts, max_disp_px=max_disp_px)
    if track_table.empty:
        return pd.DataFrame(), pd.DataFrame()

    track_dir = output_dir / "tracking"
    track_dir.mkdir(parents=True, exist_ok=True)
    track_table.to_csv(track_dir / "track_observations.csv", index=False)
    track_summary = summarize_tracks(track_table)
    if not track_summary.empty:
        track_summary.to_csv(track_dir / "track_summary.csv", index=False)
    return track_table, track_summary


def _run_manifest_contexts(
    *,
    args: argparse.Namespace,
    manifest_df: pd.DataFrame,
    pipeline,
    pipeline_cfg: dict[str, object],
    use_gpu: bool,
    focus_mode: str,
    output_root: Path | None,
    write_artifacts: bool,
) -> tuple[list[RunContext], list[tuple[str, str]], list[tuple[str, str]]]:
    processed_contexts: list[RunContext] = []
    saved_images_for_report: list[tuple[str, str]] = []
    report_assets: list[tuple[str, str]] = []

    for idx, row in enumerate(manifest_df.to_dict("records"), start=1):
        sample_id = str(row["sample_id"])
        print(f"[INFO] [STUDY] ({idx}/{len(manifest_df)}) {sample_id}")
        image_path = str(row["path"])
        image, meta = load_any_image(image_path)
        modality = _resolve_modality(args, row)
        image, meta = _adapt_loaded_image(image, meta, modality=modality, args=args)
        sample_cfg = _build_sample_cfg(
            pipeline_cfg,
            row,
            register_retina=args.register_retina,
            retina_frame_path=args.retina_frame_path,
        )

        if write_artifacts:
            assert output_root is not None
            sample_output_dir = output_root / "samples" / sample_id
            sample_output_dir.mkdir(parents=True, exist_ok=True)
            ctx = _process_single_image(
                filepath=image_path,
                image=image,
                meta=meta,
                pipeline=pipeline,
                pipeline_cfg=sample_cfg,
                args=args,
                output_dir=sample_output_dir,
                report_root=output_root,
                use_gpu=use_gpu,
                focus_mode=focus_mode,
                saved_images_for_report=saved_images_for_report,
                report_assets=report_assets,
            )
        else:
            ctx = _run_pipeline_context(
                filepath=image_path,
                image=image,
                meta=meta,
                pipeline=pipeline,
                pipeline_cfg=sample_cfg,
            )

        ctx.metrics["modality"] = modality
        ctx.summary_row["modality"] = modality
        processed_contexts.append(ctx)
        print(
            f"Processed {sample_id} | "
            f"Cells: {ctx.metrics['cell_count']} | "
            f"Density: {ctx.metrics['density_cells_per_mm2']:.2f} cells/mm^2"
        )

    return processed_contexts, saved_images_for_report, report_assets


def _normalized_calibration_params(
    params: dict[str, object],
    *,
    phenotype_engine: str,
) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for key, value in params.items():
        if key.startswith("qc."):
            normalized[f"qc_config.{key[len('qc.') :]}"] = value
        elif key.startswith("phenotype."):
            prefix = "phenotype_engine_config" if phenotype_engine == "v2" else "phenotype_rules"
            normalized[f"{prefix}.{key[len('phenotype.') :]}"] = value
        else:
            normalized[key] = value
    return normalized


def _run_calibration_mode(
    *,
    args: argparse.Namespace,
    manifest_df: pd.DataFrame,
    pipeline,
    base_pipeline_cfg: dict[str, object],
    focus_mode: str,
    diameter: float | None,
    model_type: str,
    min_size: int,
    max_size: int,
    use_gpu: bool,
    backend: str,
) -> None:
    if not args.manifest:
        raise ValueError("--calibration_grid requires --manifest.")

    calibration_spec = load_calibration_grid(args.calibration_grid)
    calibration_root = Path(args.study_output_dir or args.output_dir) / "calibration"
    calibration_root.mkdir(parents=True, exist_ok=True)

    best_validation = pd.DataFrame()
    best_sample_table = pd.DataFrame()
    best_row: dict[str, object] | None = None
    result_rows: list[dict[str, object]] = []

    for index, params in enumerate(calibration_spec["grid"], start=1):
        print(f"[INFO] [CALIBRATION] ({index}/{len(calibration_spec['grid'])}) {params}")
        normalized = _normalized_calibration_params(params, phenotype_engine=args.phenotype_engine)
        candidate_cfg = apply_dotted_overrides(base_pipeline_cfg, normalized)
        contexts, _, _ = _run_manifest_contexts(
            args=args,
            manifest_df=manifest_df,
            pipeline=pipeline,
            pipeline_cfg=candidate_cfg,
            use_gpu=use_gpu,
            focus_mode=focus_mode,
            output_root=None,
            write_artifacts=False,
        )
        sample_table = build_sample_table(manifest_df, contexts)
        validation_table = build_validation_table(sample_table)
        metrics = evaluate_count_agreement(validation_table)
        result_rows.append(
            {
                **{str(key): value for key, value in params.items()},
                "params_json": json.dumps(params, sort_keys=True),
                "params": params,
                **metrics,
            }
        )

        if best_row is None:
            best_row = result_rows[-1]
            best_validation = validation_table
            best_sample_table = sample_table

    grid_frame = pd.DataFrame(result_rows)
    ranked = rank_grid_results(grid_frame, calibration_spec["selection_metric"])
    ranked.to_csv(calibration_root / "grid_search.csv", index=False)
    if ranked.empty:
        raise RuntimeError("Calibration produced no grid results.")

    best_row = ranked.iloc[0].to_dict()
    best_params = json.loads(str(best_row["params_json"]))
    write_best_params(calibration_root / "best_params.json", best_params)

    normalized_best = _normalized_calibration_params(best_params, phenotype_engine=args.phenotype_engine)
    best_cfg = apply_dotted_overrides(base_pipeline_cfg, normalized_best)
    contexts, _, _ = _run_manifest_contexts(
        args=args,
        manifest_df=manifest_df,
        pipeline=pipeline,
        pipeline_cfg=best_cfg,
        use_gpu=use_gpu,
        focus_mode=focus_mode,
        output_root=None,
        write_artifacts=False,
    )
    best_sample_table = build_sample_table(manifest_df, contexts)
    best_validation = build_validation_table(best_sample_table)

    agreement_dir = calibration_root / "agreement"
    agreement_dir.mkdir(parents=True, exist_ok=True)
    if not best_validation.empty:
        save_bland_altman_plot(best_validation, agreement_dir / "bland_altman.png")
        save_agreement_scatter_plot(best_validation, agreement_dir / "agreement_scatter.png")

    write_calibration_report(
        calibration_root / "calibration_report.md",
        selection_metric=calibration_spec["selection_metric"],
        best_row={**best_row, "params": best_params},
        grid_size=len(calibration_spec["grid"]),
    )

    validation_summary = summarize_validation(best_validation) if not best_validation.empty else pd.DataFrame()
    if not validation_summary.empty:
        validation_summary.to_csv(calibration_root / "validation_summary.csv", index=False)
    if not best_sample_table.empty:
        best_sample_table.to_csv(calibration_root / "best_sample_table.csv", index=False)

    resolved_config = _resolved_config_dict(
        args,
        diameter=diameter,
        model_type=model_type,
        min_size=min_size,
        max_size=max_size,
        use_gpu=use_gpu,
        focus_mode=focus_mode,
        backend=backend,
    )
    write_html_report(
        str(calibration_root),
        {**resolved_config, "mode": "calibration"},
        ranked.drop(columns=["params"], errors="ignore").to_dict("records"),
        images=[
            ("Calibration Bland-Altman", os.path.relpath(agreement_dir / "bland_altman.png", calibration_root))
        ] if (agreement_dir / "bland_altman.png").exists() else [],
        tables=[
            {"title": "Calibration Ranking", "html": ranked.drop(columns=["params"], errors="ignore").to_html(index=False)}
        ],
        notes=f"Best selection metric: {calibration_spec['selection_metric']}",
        assets=[("Best parameters", "best_params.json"), ("Calibration report", "calibration_report.md")],
    )


def _run_study_mode(
    *,
    args: argparse.Namespace,
    run_started_at: datetime,
    segmenter,
    pipeline,
    pipeline_cfg: dict[str, object],
    use_gpu: bool,
    focus_mode: str,
    diameter: float | None,
    model_type: str,
    min_size: int,
    max_size: int,
    backend: str,
) -> None:
    manifest_df = _merge_manual_annotations(load_manifest(args.manifest), args.manual_annotations)
    study_output_dir = Path(args.study_output_dir or args.output_dir)
    study_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[INFO] Processing study manifest with {len(manifest_df)} sample(s) from {args.manifest}")

    processed_contexts, saved_images_for_report, report_assets = _run_manifest_contexts(
        args=args,
        manifest_df=manifest_df,
        pipeline=pipeline,
        pipeline_cfg=pipeline_cfg,
        use_gpu=use_gpu,
        focus_mode=focus_mode,
        output_root=study_output_dir,
        write_artifacts=True,
    )

    sample_table = build_sample_table(manifest_df, processed_contexts)
    study_summary_csv = study_output_dir / "study_summary.csv"
    study_summary_parquet = study_output_dir / "study_summary.parquet"
    write_table_bundle(sample_table, study_summary_csv, study_summary_parquet, table_kind="study", strict=args.strict_schemas)
    print(f"\nStudy summary saved to {study_summary_csv}")

    region_table = build_study_region_table(manifest_df, processed_contexts)
    if not region_table.empty:
        write_table_bundle(
            region_table,
            study_output_dir / "study_regions.csv",
            study_output_dir / "study_regions.parquet",
            table_kind="region",
            strict=args.strict_schemas,
        )

    atlas_comparison = pd.DataFrame()
    atlas_summary = pd.DataFrame()
    if args.atlas_reference and not region_table.empty:
        atlas_comparison, atlas_summary = _write_atlas_outputs(
            atlas_reference_path=args.atlas_reference,
            region_table=region_table,
            output_dir=study_output_dir,
            saved_images_for_report=saved_images_for_report,
        )

    track_table = pd.DataFrame()
    track_summary = pd.DataFrame()
    if args.track_longitudinal:
        track_table, track_summary = _write_tracking_outputs(
            manifest_df=manifest_df,
            contexts=processed_contexts,
            output_dir=study_output_dir,
            max_disp_px=args.track_max_disp_px,
        )

    stats_dir = study_output_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    stats_frame = compute_outcome_stats(sample_table, outcome="cell_count")
    if not stats_frame.empty:
        stats_frame.to_csv(stats_dir / "study_stats.csv", index=False)

    region_stats = compute_region_stats(region_table, outcome="density_cells_per_mm2") if not region_table.empty else pd.DataFrame()
    if not region_stats.empty:
        region_stats.to_csv(stats_dir / "region_stats.csv", index=False)

    validation_dir = study_output_dir / "validation"
    validation_table = build_validation_table(sample_table)
    validation_summary = summarize_validation(validation_table) if not validation_table.empty else pd.DataFrame()
    if not validation_table.empty:
        validation_dir.mkdir(parents=True, exist_ok=True)
        validation_table.to_csv(validation_dir / "validation_details.csv", index=False)
        validation_summary.to_csv(validation_dir / "validation_summary.csv", index=False)
        bland = save_bland_altman_plot(validation_table, validation_dir / "bland_altman.png")
        scatter = save_agreement_scatter_plot(validation_table, validation_dir / "agreement_scatter.png")
        saved_images_for_report.append(("Bland-Altman", os.path.relpath(bland, study_output_dir)))
        saved_images_for_report.append(("Manual vs automated", os.path.relpath(scatter, study_output_dir)))

    figures_dir = study_output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    condition_plot = save_condition_summary_plot(sample_table, figures_dir / "cell_count_by_condition.png", outcome="cell_count")
    saved_images_for_report.append(("Cell count by condition", os.path.relpath(condition_plot, study_output_dir)))
    paired_plot = save_paired_plot(sample_table, figures_dir / "cell_count_paired.png", outcome="cell_count")
    if paired_plot is not None:
        saved_images_for_report.append(("Paired cell counts", os.path.relpath(paired_plot, study_output_dir)))
    region_plot = save_region_density_plot(region_table, figures_dir / "region_density_by_condition.png") if not region_table.empty else None
    if region_plot is not None:
        saved_images_for_report.append(("Regional density by condition", os.path.relpath(region_plot, study_output_dir)))
    phenotype_plot = save_phenotype_composition_plot(sample_table, figures_dir / "phenotype_composition.png")
    if phenotype_plot is not None:
        saved_images_for_report.append(("Phenotype composition", os.path.relpath(phenotype_plot, study_output_dir)))

    resolved_config = _resolved_config_dict(
        args,
        diameter=diameter,
        model_type=model_type,
        min_size=min_size,
        max_size=max_size,
        use_gpu=use_gpu,
        focus_mode=focus_mode,
        backend=backend,
    )
    methods_appendix = build_methods_appendix(
        resolved_config=resolved_config,
        sample_table=sample_table,
        region_table=region_table,
        validation_summary=validation_summary,
    )
    methods_path = write_methods_appendix(study_output_dir / "methods_appendix.md", methods_appendix)

    if args.write_html_report:
        run_info = {
            "manifest": args.manifest,
            "study_output_dir": str(study_output_dir),
            "backend": backend,
            "modality": args.modality,
            "diameter": diameter,
            "min_size": min_size,
            "max_size": max_size,
            "gpu": use_gpu,
            "focus_mode": focus_mode,
            "tta": args.tta,
        }
        notes = f"Study mode with {len(sample_table)} samples."
        extra_tables = []
        if not stats_frame.empty:
            extra_tables.append({"title": "Study Statistics", "html": stats_frame.to_html(index=False)})
        if not region_stats.empty:
            extra_tables.append({"title": "Region Statistics", "html": region_stats.head(40).to_html(index=False)})
        if not validation_summary.empty:
            extra_tables.append({"title": "Validation Summary", "html": validation_summary.to_html(index=False)})
        if not atlas_summary.empty:
            extra_tables.append({"title": "Atlas Summary", "html": atlas_summary.to_html(index=False)})
        if not track_summary.empty:
            extra_tables.append({"title": "Tracking Summary", "html": track_summary.to_html(index=False)})
        report_path = write_html_report(
            str(study_output_dir),
            run_info,
            sample_table.to_dict("records"),
            saved_images_for_report,
            notes=notes,
            tables=extra_tables,
            methods_appendix=methods_appendix,
            assets=report_assets + [("Methods appendix", os.path.relpath(methods_path, study_output_dir))],
        )
        print(f"[INFO] HTML report written to {report_path}")
        for ctx in processed_contexts:
            ctx.artifacts["html_report"] = Path(report_path)
            ctx.artifacts["methods_appendix"] = methods_path

    if args.write_provenance:
        provenance_payload = build_run_provenance(
            args=vars(args),
            resolved_config=resolved_config,
            contexts=processed_contexts,
            run_started_at=run_started_at,
            run_finished_at=datetime.now(),
            results_csv_path=study_summary_csv,
        )
        provenance_path = write_provenance(study_output_dir / "provenance.json", provenance_payload)
        print(f"[INFO] Provenance written to {provenance_path}")


def main():
    run_started_at = datetime.now()
    parser = argparse.ArgumentParser(description="Automated RGC Counting Suite")

    parser.add_argument("--input_dir", type=str, default="input", help="Folder containing images")
    parser.add_argument("--output_dir", type=str, default="Outputs", help="Folder for outputs")
    parser.add_argument("--manifest", type=str, default=None, help="CSV manifest for study-mode processing")
    parser.add_argument("--study_output_dir", type=str, default=None, help="Folder for study-mode outputs")
    parser.add_argument("--manual_annotations", type=str, default=None, help="Optional CSV with sample_id plus manual_count or label_path")
    parser.add_argument("--diameter", type=float, default=None, help="Override config.yaml cell diameter in pixels")
    parser.add_argument("--model_type", type=str, default=None, help="Cellpose model type: 'cyto', 'nuclei' or custom path")
    parser.add_argument("--min_size", type=int, default=None, help="Override minimum mask area in pixels")
    parser.add_argument("--max_size", type=int, default=None, help="Override maximum mask area in pixels")

    parser.add_argument("--save_debug", action="store_true", help="Save debug overlays")
    parser.add_argument("--use_gpu", action="store_true", help="Force GPU on")
    parser.add_argument("--no_gpu", action="store_true", help="Force GPU off")

    parser.add_argument("--apply_clahe", action="store_true", help="Apply CLAHE contrast enhancement")
    parser.add_argument("--modality", type=str, default="flatmount", help="Input modality: flatmount | oct | vis_octf | lightsheet")
    parser.add_argument("--modality_projection", type=str, default="max", help="Volume projection mode for OCT/light-sheet inputs")
    parser.add_argument("--modality_channel_index", type=int, default=0, help="Channel index for modality adapters when channel reduction is needed")
    parser.add_argument("--modality_slab_start", type=int, default=None, help="Optional starting depth index for volume projection")
    parser.add_argument("--modality_slab_end", type=int, default=None, help="Optional ending depth index for volume projection")

    # Focus modes
    focus_group = parser.add_mutually_exclusive_group()
    focus_group.add_argument("--focus_none", action="store_true", help="Analyze the entire image")
    focus_group.add_argument("--focus_bbox", action="store_true", help="Manual bounding box in Napari")
    focus_group.add_argument("--focus_auto", action="store_true", help="Legacy auto focus (Laplacian + brightness)")
    focus_group.add_argument("--focus_qc", action="store_true", help="Multi-metric focus QC (recommended)")

    # Backend models
    parser.add_argument("--backend", type=str, default=None, help="Segmentation backend: cellpose | stardist | sam")
    parser.add_argument("--sam_checkpoint", type=str, default=None, help="Path to SAM model checkpoint if backend=sam")

    # Phenotype logic
    parser.add_argument("--phenotype_config", type=str, default=None, help="YAML file with marker-aware rules")
    parser.add_argument("--phenotype_engine", type=str, choices=["legacy", "v2"], default="legacy", help="Phenotype processing mode")
    parser.add_argument("--marker_metrics", action="store_true", help="Add per-object marker and morphology metrics")
    parser.add_argument("--interaction_metrics", action="store_true", help="Add phenotype interaction metrics to object tables")

    # TTA
    parser.add_argument("--tta", action="store_true", help="Enable Test-Time Augmentation")
    parser.add_argument("--tta_transforms", type=str, nargs="*", default=None, help="TTA transforms, e.g., flip_h flip_v rot90")

    # Spatial statistics
    parser.add_argument("--spatial_stats", action="store_true", help="Compute spatial mosaic metrics")

    # Retina registration
    parser.add_argument("--register_retina", action="store_true", help="Register cells into an ONH-centered retina coordinate frame")
    parser.add_argument("--region_schema", type=str, default="mouse_flatmount_v1", help="Named region schema for retina registration")
    parser.add_argument("--onh_mode", type=str, choices=["cli", "sidecar", "auto_hole", "auto_combined"], default="cli", help="How to resolve ONH/orientation inputs")
    parser.add_argument("--onh_xy", type=float, nargs=2, default=None, metavar=("X", "Y"), help="ONH center in image pixel coordinates")
    parser.add_argument("--dorsal_xy", type=float, nargs=2, default=None, metavar=("X", "Y"), help="Point indicating the dorsal direction in pixel coordinates")
    parser.add_argument("--retina_frame_path", type=str, default=None, help="Optional retina frame JSON sidecar")
    parser.add_argument("--apply_edits", type=str, default=None, help="Optional edit-log JSON to replay during the pipeline")
    parser.add_argument("--atlas_reference", type=str, default=None, help="Optional atlas reference CSV for region-wise observed vs expected comparison")
    parser.add_argument("--track_longitudinal", action="store_true", help="Build longitudinal cell tracks across timepoints in study mode")
    parser.add_argument("--track_max_disp_px", type=float, default=20.0, help="Maximum centroid displacement for longitudinal matching")

    # I/O options
    parser.add_argument("--save_ome_zarr", action="store_true", help="Write image + masks as OME-Zarr")
    parser.add_argument("--write_html_report", action="store_true", help="Generate HTML report")
    parser.add_argument("--write_object_table", action="store_true", help="Write per-image object tables")
    parser.add_argument("--write_provenance", action="store_true", help="Write run provenance JSON")
    parser.add_argument("--write_uncertainty_maps", action="store_true", help="Write foreground-probability maps when available")
    parser.add_argument("--write_qc_maps", action="store_true", help="Write focus-score maps when available")
    parser.add_argument("--strict_schemas", action="store_true", help="Fail if written tables violate required schema contracts")

    # Tiling and calibration
    parser.add_argument("--tiling", action="store_true", help="Run segmentation in overlapping tiles")
    parser.add_argument("--tile_size", type=int, default=1024, help="Tile size in pixels for tiled inference")
    parser.add_argument("--tile_overlap", type=int, default=128, help="Tile overlap in pixels for tiled inference")
    parser.add_argument("--calibration_grid", type=str, default=None, help="YAML file describing a calibration sweep over a manifest")

    # Optic nerve axon module
    parser.add_argument("--axon_dir", type=str, default=None, help="Optional: folder of optic nerve images for AxonDeepSeg")

    args = parser.parse_args()

    # Merge CLI with config.yaml
    diameter = args.diameter if args.diameter is not None else CELL_DIAMETER
    model_type = args.model_type if args.model_type is not None else MODEL_TYPE
    min_size = args.min_size if args.min_size is not None else MIN_CELL_SIZE
    max_size = args.max_size if args.max_size is not None else MAX_CELL_SIZE

    # GPU handling
    if args.no_gpu:
        use_gpu = False
    elif args.use_gpu:
        use_gpu = True
    else:
        use_gpu = USE_GPU

    if use_gpu:
        if torch.cuda.is_available():
            print("[INFO] GPU is enabled and CUDA is available.")
        else:
            print("[WARNING] GPU requested but CUDA is not available. Falling back to CPU.")
            use_gpu = False
    else:
        print("[INFO] Using CPU.")

    # Determine focus mode
    focus_mode = _resolve_focus_mode(args)

    # Build segmenter backend
    backend = args.backend
    # If not provided on CLI, try to read from config via src.config import; but we passed earlier
    if backend is None:
        backend = model_type if model_type in ("cellpose", "stardist", "sam") else "cellpose"
    segmenter = build_segmenter(backend=backend,
                                diameter=diameter,
                                model_type=model_type if backend == "cellpose" else "ignored",
                                use_gpu=use_gpu,
                                sam_checkpoint=args.sam_checkpoint)

    # Prepare outputs
    os.makedirs(args.output_dir, exist_ok=True)

    # Load phenotype rules if requested
    ph_rules, phenotype_engine_config = _load_phenotype_configs(args)

    bbox_selector = None
    if focus_mode == "bbox":
        from manual_roi import select_bounding_box_napari
        bbox_selector = select_bounding_box_napari

    pipeline_cfg = _build_pipeline_cfg(
        args,
        backend=backend,
        use_gpu=use_gpu,
        min_size=min_size,
        max_size=max_size,
        phenotype_rules=ph_rules,
        phenotype_engine_config=phenotype_engine_config,
    )
    pipeline = build_default_pipeline(
        segmenter,
        phenotype_rules=ph_rules,
        phenotype_engine_config=phenotype_engine_config,
        bbox_selector=bbox_selector,
    )

    if args.manifest and args.calibration_grid:
        manifest_df = _merge_manual_annotations(load_manifest(args.manifest), args.manual_annotations)
        _run_calibration_mode(
            args=args,
            manifest_df=manifest_df,
            pipeline=pipeline,
            base_pipeline_cfg=pipeline_cfg,
            focus_mode=focus_mode,
            diameter=diameter,
            model_type=model_type,
            min_size=min_size,
            max_size=max_size,
            use_gpu=use_gpu,
            backend=backend,
        )
        print("\nCalibration completed successfully.")
        return

    if args.manifest:
        _run_study_mode(
            args=args,
            run_started_at=run_started_at,
            segmenter=segmenter,
            pipeline=pipeline,
            pipeline_cfg=pipeline_cfg,
            use_gpu=use_gpu,
            focus_mode=focus_mode,
            diameter=diameter,
            model_type=model_type,
            min_size=min_size,
            max_size=max_size,
            backend=backend,
        )
        print("\nAll files processed successfully.")
        return

    # Load images
    image_list = utils.load_images_any(args.input_dir)
    if not image_list:
        print(f"[ERROR] No images found in {args.input_dir}.")
        return

    print(f"\n[INFO] Processing {len(image_list)} image(s) from {args.input_dir}")

    # Track rows for report and provenance
    rows = []
    saved_images_for_report = []
    report_assets: list[tuple[str, str]] = []
    processed_contexts = []

    for idx, (filepath, img, meta) in enumerate(image_list, start=1):
        print(f"[INFO] ({idx}/{len(image_list)}) {os.path.basename(filepath)}")
        modality = _resolve_modality(args)
        img, meta = _adapt_loaded_image(img, meta, modality=modality, args=args)
        ctx = _process_single_image(
            filepath=filepath,
            image=img,
            meta=meta,
            pipeline=pipeline,
            pipeline_cfg=pipeline_cfg,
            args=args,
            output_dir=args.output_dir,
            report_root=args.output_dir,
            use_gpu=use_gpu,
            focus_mode=focus_mode,
            saved_images_for_report=saved_images_for_report,
            report_assets=report_assets,
        )
        ctx.metrics["modality"] = modality
        ctx.summary_row["modality"] = modality
        processed_contexts.append(ctx)

        # Collect row
        row = dict(ctx.summary_row)
        rows.append(row)

        print(
            f"Processed {os.path.basename(filepath)} | "
            f"Cells: {ctx.metrics['cell_count']} | "
            f"Density: {ctx.metrics['density_cells_per_mm2']:.2f} cells/mm^2"
        )

    # Save CSV per directory root
    csv_path = os.path.join(args.output_dir, "results.csv")
    utils.save_results_to_csv(rows, csv_path)
    print(f"\nResults saved to {csv_path}")

    region_frames = [
        ctx.region_table
        for ctx in processed_contexts
        if ctx.region_table is not None and not ctx.region_table.empty
    ]
    combined_region_table = pd.concat(region_frames, ignore_index=True) if region_frames else pd.DataFrame()
    atlas_summary = pd.DataFrame()
    if args.atlas_reference and not combined_region_table.empty:
        _, atlas_summary = _write_atlas_outputs(
            atlas_reference_path=args.atlas_reference,
            region_table=combined_region_table,
            output_dir=Path(args.output_dir),
            saved_images_for_report=saved_images_for_report,
        )

    # Write HTML report if requested
    if args.write_html_report:
        run_info = {
            "input_dir": args.input_dir,
            "output_dir": args.output_dir,
            "backend": backend,
            "modality": args.modality,
            "diameter": diameter,
            "min_size": min_size,
            "max_size": max_size,
            "gpu": use_gpu,
            "focus_mode": focus_mode,
            "tta": args.tta,
        }
        extra_tables = []
        if not atlas_summary.empty:
            extra_tables.append({"title": "Atlas Summary", "html": atlas_summary.to_html(index=False)})
        report_path = write_html_report(
            args.output_dir,
            run_info,
            rows,
            saved_images_for_report,
            notes="",
            tables=extra_tables,
            assets=report_assets,
        )
        print(f"[INFO] HTML report written to {report_path}")
        for ctx in processed_contexts:
            ctx.artifacts["html_report"] = Path(report_path)

    if args.write_provenance:
        provenance_payload = build_run_provenance(
            args=vars(args),
            resolved_config=_resolved_config_dict(
                args,
                diameter=diameter,
                model_type=model_type,
                min_size=min_size,
                max_size=max_size,
                use_gpu=use_gpu,
                focus_mode=focus_mode,
                backend=backend,
            ),
            contexts=processed_contexts,
            run_started_at=run_started_at,
            run_finished_at=datetime.now(),
            results_csv_path=csv_path,
        )
        provenance_path = write_provenance(Path(args.output_dir) / "provenance.json", provenance_payload)
        print(f"[INFO] Provenance written to {provenance_path}")

    # Optic nerve axon analysis (optional)
    if args.axon_dir:
        try:
            from src.axon import analyze_optic_nerve
            axon_out = os.path.join(args.output_dir, "axon")
            os.makedirs(axon_out, exist_ok=True)
            results = analyze_optic_nerve(args.axon_dir, axon_out, model="default")
            # Log into a JSON file
            axon_json = os.path.join(axon_out, "axon_results.json")
            with open(axon_json, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            print(f"[INFO] Axon analysis results saved to {axon_json}")
        except Exception as e:
            print(f"[WARN] Optic nerve analysis skipped: {e}")

    print("\nAll files processed successfully.")

if __name__ == "__main__":
    main()
