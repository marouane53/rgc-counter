# main.py

import os
import argparse
import torch  # to check if GPU is available
import numpy as np

from src import utils
from src.cell_segmentation import segment_cells_cellpose
from src.postprocessing import (
    postprocess_masks,
    apply_clahe
)
from src.analysis import compute_cell_count_and_density
from src.visualize import (
    create_debug_overlay,
    save_debug_image,
    apply_out_of_focus_overlay
)
from src.utils import save_results_to_csv
from src.config import (
    CELL_DIAMETER,
    MODEL_TYPE,
    USE_GPU,
    MIN_CELL_SIZE,
    MAX_CELL_SIZE,
    OVERLAY_ALPHA
)

from src.focus_detection import compute_in_focus_mask_auto
from manual_roi import select_bounding_box_napari

def main():
    parser = argparse.ArgumentParser(description="Automated RGC Counting with various Focus Modes")
    parser.add_argument("--input_dir", type=str, default="input",
                        help="Folder containing .tif images")
    parser.add_argument("--output_dir", type=str, default="Outputs",
                        help="Folder for output CSV and debug images")
    parser.add_argument("--diameter", type=float, default=None,
                        help="Override config.yaml cell diameter (in pixels).")
    parser.add_argument("--model_type", type=str, default=None,
                        help="Override config.yaml model type: 'cyto', 'nuclei', or a custom model path.")
    parser.add_argument("--min_size", type=int, default=None,
                        help="Override config.yaml minimum mask area in pixels.")
    parser.add_argument("--max_size", type=int, default=None,
                        help="Override config.yaml maximum mask area in pixels.")
    parser.add_argument("--save_debug", action="store_true",
                        help="If set, generate and save debug overlays.")
    parser.add_argument("--use_gpu", action="store_true",
                        help="Override config.yaml GPU setting to True.")
    parser.add_argument("--no_gpu", action="store_true",
                        help="Override config.yaml GPU setting to False.")
    parser.add_argument("--apply_clahe", action="store_true",
                        help="Apply CLAHE to enhance contrast before segmentation.")

    focus_group = parser.add_mutually_exclusive_group()
    focus_group.add_argument("--focus_none", action="store_true",
                             help="No bounding or masking. Use entire image.")
    focus_group.add_argument("--focus_bbox", action="store_true",
                             help="Manually define bounding-box for each image in Napari.")
    focus_group.add_argument("--focus_auto", action="store_true",
                             help="Use improved auto tile-based approach (brightness + Laplacian).")

    args = parser.parse_args()
    
    # Create output directory if not existing
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Merge command line args with config.yaml settings
    diameter = args.diameter if args.diameter is not None else CELL_DIAMETER
    model_type = args.model_type if args.model_type is not None else MODEL_TYPE
    min_size = args.min_size if args.min_size is not None else MIN_CELL_SIZE
    max_size = args.max_size if args.max_size is not None else MAX_CELL_SIZE
    
    # Special handling for GPU flag
    if args.no_gpu:
        use_gpu = False
    elif args.use_gpu:
        use_gpu = True
    else:
        use_gpu = USE_GPU
    
    # GPU feedback
    if use_gpu:
        if torch.cuda.is_available():
            print("[INFO] GPU is enabled and CUDA is available. Cellpose will run on GPU.")
        else:
            print("[WARNING] GPU is enabled, but CUDA is NOT available.")
            print("[WARNING] Falling back to CPU for Cellpose segmentation.")
    else:
        print("[INFO] Using CPU for Cellpose (GPU not enabled).")

    # Determine focus mode
    if args.focus_none:
        focus_mode = "none"
    elif args.focus_bbox:
        focus_mode = "bbox"
    elif args.focus_auto:
        focus_mode = "auto"
    else:
        # default if user didn't pick
        focus_mode = "none"

    # Walk through all subdirectories
    for root, dirs, files in os.walk(args.input_dir):
        tiff_files = [f for f in files if f.lower().endswith(('.tif', '.tiff'))]
        if not tiff_files:
            continue

        # Create corresponding output subdirectory
        rel_path = os.path.relpath(root, args.input_dir)
        if rel_path == '.':
            current_output_dir = args.output_dir
        else:
            current_output_dir = os.path.join(args.output_dir, rel_path)
        if not os.path.exists(current_output_dir):
            os.makedirs(current_output_dir)

        image_list = utils.load_tiff_images(root)
        results = []
        
        print(f"\n[INFO] Processing directory: {root}")
        print(f"[INFO] Found {len(image_list)} images to process")
        
        for i, (filepath, img) in enumerate(image_list, start=1):
            print(f"[INFO] Processing image {i} of {len(image_list)}: {os.path.basename(filepath)}")

            # Convert to grayscale
            gray_img = utils.ensure_grayscale(img)
            
            # (Optional) apply CLAHE
            if args.apply_clahe:
                gray_img = apply_clahe(gray_img, clip_limit=2.0, tile_grid_size=(8,8))

            # Decide how to get our final "analysis mask"
            if focus_mode == "none":
                # Entire image is "in focus"
                in_focus_mask = np.ones_like(gray_img, dtype=bool)
                segmentation_input = gray_img

            elif focus_mode == "bbox":
                # Use Napari to get bounding box
                y1, y2, x1, x2 = select_bounding_box_napari(gray_img)
                cropped_img = gray_img[y1:y2, x1:x2]
                in_focus_submask = np.ones_like(cropped_img, dtype=bool)
                # We'll store the bounding box mask in the context of the full image
                in_focus_mask = np.zeros_like(gray_img, dtype=bool)
                in_focus_mask[y1:y2, x1:x2] = in_focus_submask
                segmentation_input = cropped_img

            else:  # focus_mode == "auto"
                in_focus_mask = compute_in_focus_mask_auto(
                    gray_img,
                    tile_size=64,
                    focus_threshold=50,
                    brightness_min=20,
                    brightness_max=230,
                    morph_kernel=5
                )
                # Zero out out-of-focus pixels
                masked_img = gray_img.copy()
                masked_img[~in_focus_mask] = 0
                segmentation_input = masked_img
            
            # Segment
            masks, flows, styles, diams = segment_cells_cellpose(
                segmentation_input,
                diameter=diameter,
                model_type=model_type,
                channels=[0, 0],
                use_gpu=use_gpu
            )
            
            # If we used bounding-box mode, the masks we get are for the cropped image
            # so we might need to place them back into the full resolution if we want debug overlay.
            if focus_mode == "bbox":
                # Expand the cropped masks back into the full-size array
                full_masks = np.zeros_like(gray_img, dtype=np.uint16)
                object_ids = np.unique(masks)
                next_id = 1
                for oid in object_ids:
                    if oid == 0:
                        continue
                    full_masks[y1:y2, x1:x2][masks == oid] = next_id
                    next_id += 1
                masks_filtered = postprocess_masks(full_masks, min_size, max_size)
            else:
                # normal case
                masks_filtered = postprocess_masks(masks, min_size, max_size)

            # Analysis
            cell_count, area_mm2, density_cells_per_mm2 = compute_cell_count_and_density(masks_filtered, in_focus_mask)
            
            # Debug overlay if requested
            if args.save_debug:
                debug_image = create_debug_overlay(gray_img, masks_filtered, alpha=OVERLAY_ALPHA)
                if focus_mode == "auto":
                    # Gray out or color out-of-focus
                    out_of_focus_overlay = apply_out_of_focus_overlay(debug_image, in_focus_mask, alpha=0.3)
                    debug_final = out_of_focus_overlay
                elif focus_mode == "bbox":
                    # Mark out-of-ROI region in a distinct color
                    debug_final = apply_out_of_focus_overlay(debug_image, in_focus_mask, alpha=0.3, color=(255,0,0))
                else:
                    debug_final = debug_image

                debug_filename = os.path.basename(filepath).replace('.tif', '_debug.png')
                save_debug_image(debug_final, os.path.join(current_output_dir, debug_filename))

            # Save results
            results.append({
                'filename': os.path.basename(filepath),
                'cell_count': cell_count,
                'area_mm2': area_mm2,
                'density_cells_per_mm2': density_cells_per_mm2
            })

            print(f"Processed {os.path.basename(filepath)} | "
                  f"Cells: {cell_count}, Density: {density_cells_per_mm2:.2f} cells/mm^2")
        
        # Save CSV
        csv_path = os.path.join(current_output_dir, "results.csv")
        save_results_to_csv(results, csv_path)
        print(f"\nResults for {root} saved to {csv_path}")

    print("\nAll directories processed successfully!")

if __name__ == "__main__":
    main()
