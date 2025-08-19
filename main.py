# main.py

import os
import argparse
import json
import torch
import numpy as np

from src import utils
from src.cell_segmentation import segment_cells_cellpose  # kept for legacy, not used directly below
from src.postprocessing import postprocess_masks, apply_clahe
from src.analysis import compute_cell_count_and_density
from src.visualize import create_debug_overlay, save_debug_image, apply_out_of_focus_overlay
from src.config import (
    CELL_DIAMETER, MODEL_TYPE, USE_GPU,
    MIN_CELL_SIZE, MAX_CELL_SIZE, OVERLAY_ALPHA,
    MICRONS_PER_PIXEL
)
from src.focus_detection import compute_in_focus_mask_auto  # legacy fast path
from src.qc import focus_mask_multimetric
from src.models import build_segmenter
from src.io_ome import save_labels_to_ome_zarr
from src.uncertainty import segment_with_tta
from src.spatial import centroids_from_masks, nn_regularity_index, voronoi_regulariry_index, ripley_k, isodensity_map
from src.phenotype import load_rules, apply_marker_rules
from src.report import write_html_report


def main():
    parser = argparse.ArgumentParser(description="Automated RGC Counting Suite")

    parser.add_argument("--input_dir", type=str, default="input", help="Folder containing images")
    parser.add_argument("--output_dir", type=str, default="Outputs", help="Folder for outputs")
    parser.add_argument("--diameter", type=float, default=None, help="Override config.yaml cell diameter in pixels")
    parser.add_argument("--model_type", type=str, default=None, help="Cellpose model type: 'cyto', 'nuclei' or custom path")
    parser.add_argument("--min_size", type=int, default=None, help="Override minimum mask area in pixels")
    parser.add_argument("--max_size", type=int, default=None, help="Override maximum mask area in pixels")

    parser.add_argument("--save_debug", action="store_true", help="Save debug overlays")
    parser.add_argument("--use_gpu", action="store_true", help="Force GPU on")
    parser.add_argument("--no_gpu", action="store_true", help="Force GPU off")

    parser.add_argument("--apply_clahe", action="store_true", help="Apply CLAHE contrast enhancement")

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

    # TTA
    parser.add_argument("--tta", action="store_true", help="Enable Test-Time Augmentation")
    parser.add_argument("--tta_transforms", type=str, nargs="*", default=None, help="TTA transforms, e.g., flip_h flip_v rot90")

    # Spatial statistics
    parser.add_argument("--spatial_stats", action="store_true", help="Compute spatial mosaic metrics")

    # I/O options
    parser.add_argument("--save_ome_zarr", action="store_true", help="Write image + masks as OME-Zarr")
    parser.add_argument("--write_html_report", action="store_true", help="Generate HTML report")

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
    if args.focus_bbox:
        focus_mode = "bbox"
    elif args.focus_auto:
        focus_mode = "auto"
    elif args.focus_qc:
        focus_mode = "qc"
    else:
        focus_mode = "none"

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
    ph_rules = None
    if args.phenotype_config:
        ph_rules = load_rules(args.phenotype_config)

    # Load images
    image_list = utils.load_images_any(args.input_dir)
    if not image_list:
        print(f"[ERROR] No images found in {args.input_dir}.")
        return

    print(f"\n[INFO] Processing {len(image_list)} image(s) from {args.input_dir}")

    # Track rows for report
    rows = []
    saved_images_for_report = []

    # If bbox mode, import napari ROI tool
    if focus_mode == "bbox":
        from manual_roi import select_bounding_box_napari

    for idx, (filepath, img, meta) in enumerate(image_list, start=1):
        print(f"[INFO] ({idx}/{len(image_list)}) {os.path.basename(filepath)}")
        # Keep a copy for phenotype processing if multichannel
        img_multi = img
        gray_img = utils.ensure_grayscale(img)

        # Optional CLAHE
        if args.apply_clahe:
            gray_img = apply_clahe(gray_img, clip_limit=2.0, tile_grid_size=(8, 8))

        # Focus mask decision
        if focus_mode == "none":
            in_focus_mask = np.ones_like(gray_img, dtype=bool)
            segmentation_input = gray_img
        elif focus_mode == "bbox":
            y1, y2, x1, x2 = select_bounding_box_napari(gray_img)
            cropped = gray_img[y1:y2, x1:x2]
            in_focus_mask = np.zeros_like(gray_img, dtype=bool)
            in_focus_mask[y1:y2, x1:x2] = True
            segmentation_input = cropped
        elif focus_mode == "qc":
            # Multi-metric focus map
            from src.config import data as CFG
            qc_cfg = CFG.get("qc", {})
            weights = {"lap": qc_cfg.get("laplacian_z", 1.0),
                       "ten": qc_cfg.get("tenengrad_z", 1.0),
                       "hf": qc_cfg.get("highfreq_z", 1.0)}
            in_focus_mask, score_map = focus_mask_multimetric(
                gray_img,
                tile_size=qc_cfg.get("tile_size", 64),
                brightness_min=qc_cfg.get("brightness_min", 20),
                brightness_max=qc_cfg.get("brightness_max", 230),
                weights=weights,
                threshold_z=qc_cfg.get("threshold_z", 0.0),
                morph_kernel=qc_cfg.get("morph_kernel", 5)
            )
            masked = gray_img.copy()
            masked[~in_focus_mask] = 0
            segmentation_input = masked
        else:
            # legacy auto
            in_focus_mask = compute_in_focus_mask_auto(
                gray_img,
                tile_size=32,
                focus_threshold=50,
                brightness_min=20,
                brightness_max=230,
                morph_kernel=5
            )
            masked = gray_img.copy()
            masked[~in_focus_mask] = 0
            segmentation_input = masked

        # Segmentation with or without TTA
        if args.tta:
            masks, seg_info = segment_with_tta(segmenter, segmentation_input, transforms=args.tta_transforms)
        else:
            masks, seg_info = segmenter.segment(segmentation_input)

        # If bbox mode, expand to full image
        if focus_mode == "bbox":
            full_masks = np.zeros_like(gray_img, dtype=np.uint16)
            object_ids = np.unique(masks)
            next_id = 1
            for oid in object_ids:
                if oid == 0:
                    continue
                full_masks[y1:y2, x1:x2][masks == oid] = next_id
                next_id += 1
            masks = full_masks

        # Postprocess by size
        masks_filtered = postprocess_masks(masks, min_size, max_size)

        # Phenotype-aware filtering if rules provided and image is multichannel
        annotations = {}
        if ph_rules is not None and utils.image_is_multichannel(img_multi):
            try:
                masks_filtered, annotations = apply_marker_rules(img_multi, masks_filtered, ph_rules)
            except Exception as e:
                print(f"[WARN] Phenotype rules failed on {os.path.basename(filepath)}: {e}")

        # Analysis
        cell_count, area_mm2, density_cells_per_mm2 = compute_cell_count_and_density(masks_filtered, in_focus_mask)

        # Spatial stats (optional)
        spatial = {}
        if args.spatial_stats:
            cents = centroids_from_masks(masks_filtered)
            rr = nn_regularity_index(cents)
            vr = voronoi_regulariry_index(cents, gray_img.shape)
            radii = [25, 50, 75, 100, 150, 200]
            kfun = ripley_k(cents, radii, area_px=float(in_focus_mask.sum()))
            spatial = {
                "nn_mean": rr["mean"],
                "nn_std": rr["std"],
                "nnri": rr["nnri"],
                "vdri": vr["vdri"],
                **{f"k_{r}": kfun[r] for r in radii}
            }

        # Debug overlay
        if args.save_debug:
            debug_image = create_debug_overlay(gray_img, masks_filtered, alpha=OVERLAY_ALPHA)
            if focus_mode in ("auto", "qc"):
                debug_image = apply_out_of_focus_overlay(debug_image, in_focus_mask, alpha=0.3)
            debug_filename = os.path.basename(filepath).rsplit('.', 1)[0] + "_debug.png"
            out_path = os.path.join(args.output_dir, debug_filename)
            save_debug_image(debug_image, out_path)
            saved_images_for_report.append(("Debug overlay " + os.path.basename(filepath), os.path.relpath(out_path, args.output_dir)))

            # Optional isodensity map for visuals
            if args.spatial_stats:
                from src.spatial import isodensity_map
                cents = centroids_from_masks(masks_filtered)
                iso = isodensity_map(cents, gray_img.shape, sigma_px=50.0)
                # Save as image
                import cv2
                iso8 = utils.safe_uint8(iso)
                iso_color = cv2.applyColorMap(iso8, cv2.COLORMAP_JET)
                iso_path = os.path.join(args.output_dir, os.path.basename(filepath).rsplit('.', 1)[0] + "_isodensity.png")
                cv2.imwrite(iso_path, iso_color)
                saved_images_for_report.append(("Isodensity " + os.path.basename(filepath), os.path.relpath(iso_path, args.output_dir)))

        # Save OME-Zarr if requested
        if args.save_ome_zarr:
            zarr_dir = os.path.join(args.output_dir, os.path.basename(filepath).rsplit('.', 1)[0] + ".zarr")
            try:
                meta_to_write = {
                    "backend": seg_info.get("backend", "unknown"),
                    "use_gpu": use_gpu,
                    "microns_per_pixel": MICRONS_PER_PIXEL,
                }
                save_labels_to_ome_zarr(gray_img, masks_filtered, zarr_dir, meta_to_write, chunk=256)
            except Exception as e:
                print(f"[WARN] Failed to save OME-Zarr: {e}")

        # Collect row
        row = {
            "filename": os.path.basename(filepath),
            "cell_count": cell_count,
            "area_mm2": area_mm2,
            "density_cells_per_mm2": density_cells_per_mm2,
            "backend": seg_info.get("backend", backend),
            "use_gpu": use_gpu
        }
        row.update(spatial)
        rows.append(row)

        print(f"Processed {os.path.basename(filepath)} | Cells: {cell_count} | Density: {density_cells_per_mm2:.2f} cells/mm^2")

    # Save CSV per directory root
    csv_path = os.path.join(args.output_dir, "results.csv")
    utils.save_results_to_csv(rows, csv_path)
    print(f"\nResults saved to {csv_path}")

    # Write HTML report if requested
    if args.write_html_report:
        run_info = {
            "input_dir": args.input_dir,
            "output_dir": args.output_dir,
            "backend": backend,
            "diameter": diameter,
            "min_size": min_size,
            "max_size": max_size,
            "gpu": use_gpu,
            "focus_mode": focus_mode,
            "tta": args.tta,
        }
        report_path = write_html_report(args.output_dir, run_info, rows, saved_images_for_report, notes="")
        print(f"[INFO] HTML report written to {report_path}")

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
