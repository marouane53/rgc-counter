from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np

from src.analysis import compute_cell_count_and_density
from src.atlas_subtypes import score_atlas_subtypes
from src.edits import apply_edit_log, load_edit_log
from src.interactions import add_interaction_metrics
from src.landmarks import build_tissue_mask
from src.marker_metrics import add_marker_metrics
from src.config import data as CONFIG_DATA
from src.context import RunContext
from src.focus_detection import compute_in_focus_mask_auto
from src.measurements import add_uncertainty_summary_columns, build_object_table
from src.phenotype import apply_marker_rules
from src.phenotype_engine import assign_phenotypes
from src.postprocessing import apply_clahe, postprocess_masks
from src.qc import focus_mask_multimetric
from src.regions import assign_regions, summarize_regions
from src.retina_coords import register_cells, register_focus_mask_pixels, resolve_retina_frame
from src.review import resolve_edit_log_path
from src.spatial import (
    DEFAULT_RIGOROUS_RADII_PX,
    compute_rigorous_spatial_bundle,
    centroids_from_masks,
    isodensity_map,
    kept_object_table,
    nn_regularity_index,
    ripley_k,
    rigorous_points_from_object_table,
    voronoi_regulariry_index,
)
from src.tiling import segment_tiled
from src.uncertainty import segment_with_tta
from src.utils import ensure_grayscale, image_is_multichannel


BBoxSelector = Callable[[np.ndarray], tuple[int, int, int, int]]


class Stage(Protocol):
    name: str

    def run(self, ctx: RunContext, cfg: dict[str, Any]) -> RunContext:
        ...


class PipelineRunner:
    def __init__(self, stages: list[Stage]):
        self.stages = stages

    def run(self, ctx: RunContext, cfg: dict[str, Any]) -> RunContext:
        for stage in self.stages:
            ctx = stage.run(ctx, cfg)
        return ctx


def _resolved_qc_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(cfg.get("qc_config", CONFIG_DATA.get("qc", {})))


def _resolved_model_fields(ctx: RunContext, cfg: dict[str, Any]) -> dict[str, Any]:
    model_fields = {
        "model_label": ctx.seg_info.get("model_label"),
        "model_source": ctx.seg_info.get("model_source"),
        "model_alias": ctx.seg_info.get("model_alias"),
        "model_asset_path": ctx.seg_info.get("model_asset_path"),
        "model_builtin_name": ctx.seg_info.get("model_builtin_name"),
        "model_trust_mode": ctx.seg_info.get("model_trust_mode"),
    }
    fallback = cfg.get("model_spec") or {}
    for key in list(model_fields):
        if model_fields[key] is None:
            model_fields[key] = fallback.get(key)
    return model_fields


def _append_warning(ctx: RunContext, message: str) -> None:
    if message not in ctx.warnings:
        ctx.warnings.append(message)


def _count_labeled_objects(labels: np.ndarray | None) -> int:
    if labels is None:
        return 0
    object_ids = np.unique(labels)
    object_ids = object_ids[object_ids != 0]
    return int(len(object_ids))


def _set_object_flow_metrics(ctx: RunContext, **updates: int) -> None:
    flow = dict(ctx.metrics.get("object_flow") or {})
    for key, value in updates.items():
        flow[key] = int(value)
        ctx.summary_row[f"object_flow_{key}"] = int(value)
    ctx.metrics["object_flow"] = flow


def _update_measurements_and_summary(ctx: RunContext, cfg: dict[str, Any]) -> None:
    if ctx.labels is None or ctx.qc_mask is None:
        raise ValueError("Measurement refresh requires labels and qc_mask.")

    cell_count, area_mm2, density_cells_per_mm2 = compute_cell_count_and_density(ctx.labels, ctx.qc_mask)
    ctx.metrics["cell_count"] = int(cell_count)
    ctx.metrics["area_mm2"] = float(area_mm2)
    ctx.metrics["density_cells_per_mm2"] = float(density_cells_per_mm2)

    ctx.object_table = build_object_table(
        path=ctx.path,
        labels=ctx.labels,
        focus_mask=ctx.qc_mask,
        gray_image=ctx.gray,
        meta=ctx.meta,
    )
    ctx.object_table = add_uncertainty_summary_columns(
        ctx.object_table,
        ctx.labels,
        ctx.state.get("foreground_probability"),
    )
    kept = kept_object_table(ctx.object_table)
    focus_overlap_positive = (
        int((kept["focus_overlap_px"].fillna(0).astype(float) > 0).sum())
        if not kept.empty and "focus_overlap_px" in kept.columns
        else 0
    )
    _set_object_flow_metrics(
        ctx,
        n_labels_postprocess=_count_labeled_objects(ctx.labels),
        n_objects_object_table=len(ctx.object_table),
        n_objects_kept=len(kept),
        n_objects_focus_overlap_gt0=focus_overlap_positive,
    )

    ctx.summary_row["filename"] = ctx.path.name
    ctx.summary_row["cell_count"] = cell_count
    ctx.summary_row["area_mm2"] = area_mm2
    ctx.summary_row["density_cells_per_mm2"] = density_cells_per_mm2
    ctx.summary_row["backend"] = ctx.seg_info.get("backend", cfg.get("backend", "unknown"))
    ctx.summary_row["use_gpu"] = cfg.get("use_gpu", False)
    for key, value in _resolved_model_fields(ctx, cfg).items():
        ctx.metrics[key] = value
        ctx.summary_row[key] = value


@dataclass
class PrepareImageStage:
    name: str = "prepare_image"

    def run(self, ctx: RunContext, cfg: dict[str, Any]) -> RunContext:
        gray = ensure_grayscale(ctx.image)
        if cfg.get("apply_clahe"):
            gray = apply_clahe(gray, clip_limit=2.0, tile_grid_size=(8, 8))
        ctx.gray = gray
        ctx.metrics["image_shape"] = list(gray.shape)
        ctx.metrics["image_dtype"] = str(gray.dtype)
        if any(int(dim) <= 1 for dim in gray.shape):
            _append_warning(ctx, f"Degenerate grayscale shape detected: {tuple(int(dim) for dim in gray.shape)}")
        return ctx


@dataclass
class FocusMaskStage:
    bbox_selector: BBoxSelector | None = None
    name: str = "focus_mask"

    def run(self, ctx: RunContext, cfg: dict[str, Any]) -> RunContext:
        if ctx.gray is None:
            raise ValueError("FocusMaskStage requires ctx.gray to be populated.")

        focus_mode = cfg.get("focus_mode", "none")
        gray = ctx.gray

        if focus_mode == "none":
            in_focus_mask = np.ones_like(gray, dtype=bool)
            segmentation_input = gray
        elif focus_mode == "bbox":
            if self.bbox_selector is None:
                raise ValueError("Focus bbox mode requires a bbox selector callback.")
            y1, y2, x1, x2 = self.bbox_selector(gray)
            in_focus_mask = np.zeros_like(gray, dtype=bool)
            in_focus_mask[y1:y2, x1:x2] = True
            segmentation_input = gray[y1:y2, x1:x2]
            ctx.state["bbox"] = (y1, y2, x1, x2)
        elif focus_mode == "qc":
            qc_cfg = _resolved_qc_config(cfg)
            weights = {
                "lap": qc_cfg.get("laplacian_z", 1.0),
                "ten": qc_cfg.get("tenengrad_z", 1.0),
                "hf": qc_cfg.get("highfreq_z", 1.0),
            }
            in_focus_mask, score_map = focus_mask_multimetric(
                gray,
                tile_size=qc_cfg.get("tile_size", 64),
                brightness_min=qc_cfg.get("brightness_min", 20),
                brightness_max=qc_cfg.get("brightness_max", 230),
                weights=weights,
                threshold_z=qc_cfg.get("threshold_z", 0.0),
                morph_kernel=qc_cfg.get("morph_kernel", 5),
            )
            segmentation_input = gray.copy()
            segmentation_input[~in_focus_mask] = 0
            ctx.state["focus_score_map"] = score_map
        else:
            in_focus_mask = compute_in_focus_mask_auto(
                gray,
                tile_size=32,
                focus_threshold=50,
                brightness_min=20,
                brightness_max=230,
                morph_kernel=5,
            )
            segmentation_input = gray.copy()
            segmentation_input[~in_focus_mask] = 0

        ctx.qc_mask = in_focus_mask
        ctx.state["segmentation_input"] = segmentation_input
        ctx.metrics["focus_mode"] = focus_mode
        ctx.metrics["focus_area_px"] = int(in_focus_mask.sum())
        return ctx


@dataclass
class SegmentationStage:
    segmenter: Any
    name: str = "segment"

    def run(self, ctx: RunContext, cfg: dict[str, Any]) -> RunContext:
        segmentation_input = ctx.state.get("segmentation_input")
        if segmentation_input is None:
            raise ValueError("SegmentationStage requires segmentation_input in ctx.state.")

        if cfg.get("tiling"):
            masks, seg_info = segment_tiled(
                self.segmenter,
                segmentation_input,
                tile_size=int(cfg.get("tile_size", 1024)),
                overlap=int(cfg.get("tile_overlap", 128)),
                use_tta=bool(cfg.get("tta")),
                transforms=cfg.get("tta_transforms"),
            )
        elif cfg.get("tta"):
            masks, seg_info = segment_with_tta(
                self.segmenter,
                segmentation_input,
                transforms=cfg.get("tta_transforms"),
            )
        else:
            masks, seg_info = self.segmenter.segment(segmentation_input)

        bbox = ctx.state.get("bbox")
        if bbox is not None and ctx.gray is not None:
            y1, y2, x1, x2 = bbox
            full_masks = np.zeros_like(ctx.gray, dtype=np.uint32)
            next_id = 1
            for object_id in np.unique(masks):
                if int(object_id) == 0:
                    continue
                full_masks[y1:y2, x1:x2][masks == object_id] = next_id
                next_id += 1
            masks = full_masks
            if seg_info.get("foreground_probability") is not None:
                full_probability = np.zeros_like(ctx.gray, dtype=np.float32)
                full_probability[y1:y2, x1:x2] = np.asarray(seg_info["foreground_probability"], dtype=np.float32)
                seg_info["foreground_probability"] = full_probability

        if seg_info.get("foreground_probability") is not None:
            ctx.state["foreground_probability"] = np.asarray(seg_info["foreground_probability"], dtype=np.float32)

        label_dtype = np.uint32 if int(np.max(masks)) > np.iinfo(np.uint16).max else np.uint16
        ctx.labels = masks.astype(label_dtype, copy=False)
        _set_object_flow_metrics(ctx, n_labels_raw=_count_labeled_objects(ctx.labels))
        ctx.seg_info = seg_info
        ctx.metrics["backend"] = seg_info.get("backend", cfg.get("backend", "unknown"))
        for key, value in _resolved_model_fields(ctx, cfg).items():
            if value is not None:
                ctx.metrics[key] = value
        if cfg.get("tiling"):
            ctx.metrics["tiling"] = {
                "tile_size": int(cfg.get("tile_size", 1024)),
                "tile_overlap": int(cfg.get("tile_overlap", 128)),
                "tile_count": int(seg_info.get("tile_count", 0)),
                "stitching": seg_info.get("stitching", "unknown"),
                "matched_overlap_pairs": int(seg_info.get("matched_overlap_pairs", 0)),
            }
        return ctx


@dataclass
class PostprocessStage:
    name: str = "postprocess"

    def run(self, ctx: RunContext, cfg: dict[str, Any]) -> RunContext:
        if ctx.labels is None:
            raise ValueError("PostprocessStage requires ctx.labels.")
        ctx.labels = postprocess_masks(ctx.labels, cfg["min_size"], cfg["max_size"])
        _set_object_flow_metrics(ctx, n_labels_postprocess=_count_labeled_objects(ctx.labels))
        return ctx


@dataclass
class PhenotypeStage:
    phenotype_rules: dict[str, Any] | None = None
    name: str = "phenotype"

    def run(self, ctx: RunContext, cfg: dict[str, Any]) -> RunContext:
        if cfg.get("phenotype_engine", "legacy") != "legacy":
            return ctx
        rules = cfg.get("phenotype_rules", self.phenotype_rules)
        if rules is None or ctx.labels is None:
            return ctx
        if not image_is_multichannel(ctx.image):
            return ctx
        try:
            filtered, annotations = apply_marker_rules(ctx.image, ctx.labels, rules)
            ctx.labels = filtered
            ctx.state["phenotype_annotations"] = annotations
        except Exception as exc:
            ctx.warnings.append(f"Phenotype rules failed: {exc}")
        return ctx


@dataclass
class MeasurementStage:
    name: str = "measure"

    def run(self, ctx: RunContext, cfg: dict[str, Any]) -> RunContext:
        _update_measurements_and_summary(ctx, cfg)
        return ctx


@dataclass
class MarkerMetricsStage:
    phenotype_engine_config: dict[str, Any] | None = None
    name: str = "marker_metrics"

    def run(self, ctx: RunContext, cfg: dict[str, Any]) -> RunContext:
        if ctx.object_table is None or ctx.labels is None:
            raise ValueError("MarkerMetricsStage requires object_table and labels.")
        if (
            not cfg.get("marker_metrics")
            and cfg.get("phenotype_engine", "legacy") != "v2"
            and cfg.get("atlas_subtype_priors_config") is None
        ):
            return ctx

        engine_config = None
        if cfg.get("phenotype_engine", "legacy") == "v2":
            engine_config = cfg.get("phenotype_engine_config", self.phenotype_engine_config)
        elif cfg.get("atlas_subtype_priors_config") is not None:
            engine_config = cfg.get("atlas_subtype_priors_config")
        ctx.object_table = add_marker_metrics(ctx.object_table, ctx.image, ctx.labels, config=engine_config)
        return ctx


@dataclass
class PhenotypeEngineStage:
    phenotype_engine_config: dict[str, Any] | None = None
    name: str = "phenotype_engine"

    def run(self, ctx: RunContext, cfg: dict[str, Any]) -> RunContext:
        if cfg.get("phenotype_engine", "legacy") != "v2":
            return ctx
        engine_config = cfg.get("phenotype_engine_config", self.phenotype_engine_config)
        if engine_config is None or ctx.object_table is None:
            return ctx

        ctx.object_table = assign_phenotypes(ctx.object_table, engine_config)
        counts = ctx.object_table["phenotype"].value_counts().to_dict()
        ctx.metrics["phenotype_counts"] = counts
        ctx.summary_row["phenotype_engine"] = "v2"
        return ctx


@dataclass
class ReviewStage:
    phenotype_engine_config: dict[str, Any] | None = None
    name: str = "review"

    def run(self, ctx: RunContext, cfg: dict[str, Any]) -> RunContext:
        edit_path = resolve_edit_log_path(ctx.path, cfg.get("apply_edits"))
        if edit_path is None:
            return ctx
        if ctx.labels is None or ctx.object_table is None:
            raise ValueError("ReviewStage requires labels and object_table.")

        document = load_edit_log(edit_path)
        structural_ops = {"delete_object", "merge_objects"}
        has_structural_edits = any(edit.get("op") in structural_ops for edit in document.get("edits", []))
        labels, table_after_edits, review_meta = apply_edit_log(ctx.labels, ctx.object_table, document)
        ctx.labels = labels

        if has_structural_edits:
            _update_measurements_and_summary(ctx, cfg)
            if cfg.get("marker_metrics") or cfg.get("phenotype_engine", "legacy") == "v2":
                engine_config = cfg.get("phenotype_engine_config", self.phenotype_engine_config)
                ctx.object_table = add_marker_metrics(ctx.object_table, ctx.image, ctx.labels, config=engine_config)
            if cfg.get("phenotype_engine", "legacy") == "v2":
                engine_config = cfg.get("phenotype_engine_config", self.phenotype_engine_config)
                if engine_config is not None and ctx.object_table is not None:
                    ctx.object_table = assign_phenotypes(ctx.object_table, engine_config)
        else:
            ctx.object_table = table_after_edits

        phenotype_overrides = review_meta.get("phenotype_overrides", {})
        if phenotype_overrides and ctx.object_table is not None and "phenotype" in ctx.object_table.columns:
            for object_id, phenotype in phenotype_overrides.items():
                ctx.object_table.loc[ctx.object_table["object_id"].astype(int) == int(object_id), "phenotype"] = str(phenotype)

        if ctx.object_table is not None and "phenotype" in ctx.object_table.columns:
            ctx.metrics["phenotype_counts"] = ctx.object_table["phenotype"].value_counts().to_dict()

        if review_meta.get("onh_xy") is not None and review_meta.get("dorsal_xy") is not None:
            cfg["onh_mode"] = "cli"
            cfg["onh_xy"] = tuple(review_meta["onh_xy"])
            cfg["dorsal_xy"] = tuple(review_meta["dorsal_xy"])

        ctx.artifacts["edit_log"] = Path(edit_path)
        ctx.metrics["review_edits_applied"] = len(document.get("edits", []))
        ctx.summary_row["review_edits_applied"] = len(document.get("edits", []))
        return ctx


@dataclass
class RetinaRegistrationStage:
    name: str = "retina_registration"

    def run(self, ctx: RunContext, cfg: dict[str, Any]) -> RunContext:
        if not cfg.get("register_retina"):
            return ctx
        if ctx.object_table is None or ctx.qc_mask is None or ctx.gray is None:
            raise ValueError("RetinaRegistrationStage requires object_table, qc_mask, and gray image.")

        frame = resolve_retina_frame(
            image_path=ctx.path,
            gray_image=ctx.gray,
            meta=ctx.meta,
            onh_mode=cfg.get("onh_mode", "cli"),
            onh_xy=cfg.get("onh_xy"),
            dorsal_xy=cfg.get("dorsal_xy"),
            retina_frame_path=cfg.get("retina_frame_path"),
        )
        focus_pixels = register_focus_mask_pixels(ctx.qc_mask, frame)
        tissue_mask = build_tissue_mask(ctx.gray)
        tissue_pixels = register_focus_mask_pixels(tissue_mask, frame)
        max_ecc_um = float(tissue_pixels["ecc_um"].max()) if not tissue_pixels.empty else float(focus_pixels["ecc_um"].max()) if not focus_pixels.empty else 0.0
        registered = register_cells(ctx.object_table, frame)
        registered = assign_regions(
            registered,
            schema_name=cfg.get("region_schema", "mouse_flatmount_v1"),
            max_ecc_um=max_ecc_um,
        )
        region_table = summarize_regions(
            object_table=registered,
            focus_pixels=focus_pixels,
            tissue_pixels=tissue_pixels,
            tissue_mask=tissue_mask,
            frame=frame,
            schema_name=cfg.get("region_schema", "mouse_flatmount_v1"),
            source_path=ctx.path,
        )

        ctx.object_table = registered
        ctx.region_table = region_table
        ctx.state["retina_frame"] = frame
        ctx.state["registered_focus_pixels"] = focus_pixels
        ctx.state["registered_tissue_pixels"] = tissue_pixels
        ctx.metrics["retina_registration"] = {
            "source": frame.source,
            "onh_source": frame.onh_source,
            "onh_confidence": frame.onh_confidence,
            "tissue_coverage_fraction": frame.tissue_coverage_fraction,
            "um_per_px": frame.um_per_px,
            "region_schema": cfg.get("region_schema", "mouse_flatmount_v1"),
            "max_ecc_um": max_ecc_um,
        }
        ctx.summary_row["retina_registered"] = True
        ctx.summary_row["region_schema"] = cfg.get("region_schema", "mouse_flatmount_v1")
        ctx.summary_row["max_ecc_um"] = max_ecc_um
        if float(frame.tissue_coverage_fraction) < 0.01:
            _append_warning(
                ctx,
                f"Near-zero tissue coverage after retina registration: {float(frame.tissue_coverage_fraction):.6f}",
            )
        return ctx


@dataclass
class AtlasSubtypeStage:
    name: str = "atlas_subtypes"

    def run(self, ctx: RunContext, cfg: dict[str, Any]) -> RunContext:
        config = cfg.get("atlas_subtype_priors_config")
        if config is None:
            return ctx
        if ctx.object_table is None:
            raise ValueError("AtlasSubtypeStage requires object_table.")

        result = score_atlas_subtypes(ctx.object_table, config)
        ctx.object_table = result["object_table"]
        ctx.state["atlas_subtypes"] = {
            "summary": result["summary"],
            "region_summary": result["region_summary"],
            "atlas_name": result["atlas_name"],
            "used_location_evidence": result["used_location_evidence"],
            "subtypes": result["subtypes"],
            "top1_counts": result["top1_counts"],
            "retina_region_schema": config.get("retina_region_schema"),
            "location_weight": float(config.get("location_weight", 0.7)),
            "marker_weight": float(config.get("marker_weight", 0.3)),
            "config_path": config.get("config_path"),
        }
        ctx.metrics["atlas_subtypes"] = {
            "enabled": True,
            "atlas_name": result["atlas_name"],
            "config_path": config.get("config_path"),
            "retina_region_schema": config.get("retina_region_schema"),
            "location_weight": float(config.get("location_weight", 0.7)),
            "marker_weight": float(config.get("marker_weight", 0.3)),
            "subtypes": result["subtypes"],
            "used_location_evidence": result["used_location_evidence"],
            "top1_counts": result["top1_counts"],
        }
        ctx.metrics["atlas_subtype_top1_counts"] = result["top1_counts"]
        if not result["used_location_evidence"]:
            ctx.warnings.append(
                "Atlas subtype priors ran without registration-backed location evidence; marker evidence only."
            )
        return ctx


@dataclass
class InteractionMetricsStage:
    name: str = "interaction_metrics"

    def run(self, ctx: RunContext, cfg: dict[str, Any]) -> RunContext:
        if not cfg.get("interaction_metrics"):
            return ctx
        if ctx.object_table is None or "phenotype" not in ctx.object_table.columns:
            return ctx
        ctx.object_table = add_interaction_metrics(ctx.object_table)
        return ctx


@dataclass
class SpatialStatsStage:
    name: str = "spatial_stats"

    def run(self, ctx: RunContext, cfg: dict[str, Any]) -> RunContext:
        if not cfg.get("spatial_stats"):
            return ctx
        if ctx.labels is None or ctx.qc_mask is None or ctx.gray is None or ctx.object_table is None:
            raise ValueError("SpatialStatsStage requires labels, qc_mask, gray image, and object_table.")

        cents = centroids_from_masks(ctx.labels)
        rr = nn_regularity_index(cents)
        vr = voronoi_regulariry_index(cents, ctx.gray.shape)
        radii = [25, 50, 75, 100, 150, 200]
        kfun = ripley_k(cents, radii, area_px=float(ctx.qc_mask.sum()))

        spatial = {
            "nn_mean": rr["mean"],
            "nn_std": rr["std"],
            "nnri": rr["nnri"],
            "vdri": vr["vdri"],
            **{f"k_{radius}": kfun[radius] for radius in radii},
        }
        ctx.metrics["spatial"] = spatial
        ctx.state["centroids"] = cents
        ctx.state["isodensity_map"] = isodensity_map(cents, ctx.gray.shape, sigma_px=50.0)
        ctx.summary_row["spatial_mode"] = cfg.get("spatial_mode", "legacy")
        ctx.summary_row.update(spatial)

        if cfg.get("spatial_mode", "legacy") == "rigorous":
            if len(cents) < 3:
                _append_warning(
                    ctx,
                    f"Rigorous spatial analysis requested with too few points: {int(len(cents))}",
                )
            tissue_mask = build_tissue_mask(ctx.gray)
            um_per_px = None
            max_ecc_um = None
            registered_tissue_pixels = None
            if "retina_frame" in ctx.state:
                um_per_px = float(ctx.state["retina_frame"].um_per_px)
            if "registered_tissue_pixels" in ctx.state:
                registered_tissue_pixels = ctx.state["registered_tissue_pixels"]
            retina_metrics = ctx.metrics.get("retina_registration", {})
            if isinstance(retina_metrics, dict):
                max_ecc_um = retina_metrics.get("max_ecc_um")

            rigorous = compute_rigorous_spatial_bundle(
                image_id=ctx.path.name.rsplit(".", 1)[0],
                object_table=ctx.object_table,
                image_shape=ctx.gray.shape,
                tissue_mask=tissue_mask,
                um_per_px=um_per_px,
                registered_tissue_pixels=registered_tissue_pixels,
                schema_name=cfg.get("region_schema") if registered_tissue_pixels is not None else None,
                max_ecc_um=float(max_ecc_um) if max_ecc_um is not None else None,
                radii_px=cfg.get("spatial_radii_px", DEFAULT_RIGOROUS_RADII_PX),
                simulation_count=int(cfg.get("spatial_envelope_sims", 999)),
                base_seed=int(cfg.get("spatial_random_seed", 1337)),
            )
            ctx.state["rigorous_spatial"] = rigorous
            ctx.metrics["spatial_analysis"] = rigorous["spatial_analysis"]
            ctx.summary_row.update(rigorous["global_summary"])
            spatial_input_points = rigorous_points_from_object_table(ctx.object_table, level="global")
            global_point_count = int(rigorous["global_summary"].get("rigorous_global_point_count", 0))
            _set_object_flow_metrics(
                ctx,
                n_points_spatial_input=len(spatial_input_points),
                n_points_global_domain=global_point_count,
            )
            if int(ctx.metrics.get("cell_count", 0)) > 0 and global_point_count == 0:
                message = "Counting/spatial mismatch detected: nonzero cell_count but zero rigorous global points."
                _append_warning(ctx, message)
                raise RuntimeError(message)
        return ctx


def build_default_pipeline(
    segmenter: Any,
    *,
    phenotype_rules: dict[str, Any] | None = None,
    phenotype_engine_config: dict[str, Any] | None = None,
    bbox_selector: BBoxSelector | None = None,
) -> PipelineRunner:
    stages: list[Stage] = [
        PrepareImageStage(),
        FocusMaskStage(bbox_selector=bbox_selector),
        SegmentationStage(segmenter=segmenter),
        PostprocessStage(),
        PhenotypeStage(phenotype_rules=phenotype_rules),
        MeasurementStage(),
        MarkerMetricsStage(phenotype_engine_config=phenotype_engine_config),
        PhenotypeEngineStage(phenotype_engine_config=phenotype_engine_config),
        ReviewStage(phenotype_engine_config=phenotype_engine_config),
        RetinaRegistrationStage(),
        AtlasSubtypeStage(),
        InteractionMetricsStage(),
        SpatialStatsStage(),
    ]
    return PipelineRunner(stages)
