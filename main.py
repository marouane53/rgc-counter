# main.py

import os
import argparse
import torch  # to check if GPU is available

from src import utils
from src.cell_segmentation import segment_cells_cellpose
from src.postprocessing import (
    postprocess_masks,
    apply_gaussian_blur
)
from src.analysis import compute_cell_count_and_density
from src.visualize import create_debug_overlay, save_debug_image
from src.utils import save_results_to_csv
from src.config import (
    CELL_DIAMETER,
    MODEL_TYPE,
    USE_GPU,
    MIN_CELL_SIZE,
    MAX_CELL_SIZE,
    OVERLAY_ALPHA
)

def main():
    parser = argparse.ArgumentParser(description="Automated RGC Counting")
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
                        help="Override config.yaml GPU setting.")
    parser.add_argument("--no_gpu", action="store_true",
                        help="Override config.yaml GPU setting to False.")
    parser.add_argument("--apply_blur", action="store_true",
                        help="Apply Gaussian blur to the image before segmentation.")

    args = parser.parse_args()
    
    # Create output directory if not existing
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Merge command line args with config.yaml settings
    # Command line args take precedence when specified
    diameter = args.diameter if args.diameter is not None else CELL_DIAMETER
    model_type = args.model_type if args.model_type is not None else MODEL_TYPE
    min_size = args.min_size if args.min_size is not None else MIN_CELL_SIZE
    max_size = args.max_size if args.max_size is not None else MAX_CELL_SIZE
    
    # Special handling for GPU flag to allow explicit disable
    if args.no_gpu:
        use_gpu = False
    elif args.use_gpu:
        use_gpu = True
    else:
        use_gpu = USE_GPU
    
    # ---- GPU Feedback Logic ----
    if use_gpu:
        if torch.cuda.is_available():
            print("[INFO] GPU is enabled and CUDA is available. Cellpose will run on GPU.")
        else:
            print("[WARNING] GPU is enabled, but CUDA is NOT available.")
            print("[WARNING] Falling back to CPU for Cellpose segmentation.")
    else:
        print("[INFO] Using CPU for Cellpose (GPU not enabled).")
    # ----------------------------

    # Walk through all subdirectories
    for root, dirs, files in os.walk(args.input_dir):
        # Skip if no TIFF files in this directory
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

        # Load images from current directory
        image_list = utils.load_tiff_images(root)
        results = []
        
        print(f"\n[INFO] Processing directory: {root}")
        print(f"[INFO] Found {len(image_list)} images to process")
        
        # Process each image
        for i, (filepath, img) in enumerate(image_list, start=1):
            print(f"[INFO] Processing image {i} of {len(image_list)}: {os.path.basename(filepath)}")
            # Ensure grayscale
            gray_img = utils.ensure_grayscale(img)
            
            # (Optional) Apply Gaussian blur if requested
            if args.apply_blur:
                gray_img = apply_gaussian_blur(gray_img, ksize=3)
            
            # Segment cells with Cellpose
            masks, flows, styles, diams = segment_cells_cellpose(
                gray_img,
                diameter=diameter,
                model_type=model_type,
                channels=[0, 0],
                use_gpu=use_gpu
            )
            
            # Postprocessing
            masks_filtered = postprocess_masks(
                masks, 
                min_size=min_size, 
                max_size=max_size
            )
            
            # Analysis
            cell_count, area_mm2, density_cells_per_mm2 = compute_cell_count_and_density(masks_filtered)
            
            # (Optional) Debug overlay
            if args.save_debug:
                debug_image = create_debug_overlay(gray_img, masks_filtered, alpha=OVERLAY_ALPHA)
                debug_filename = os.path.basename(filepath).replace('.tif','_debug.png')
                save_debug_image(debug_image, os.path.join(current_output_dir, debug_filename))
            
            # Store result for CSV
            results.append({
                'filename': os.path.basename(filepath),
                'cell_count': cell_count,
                'area_mm2': area_mm2,
                'density_cells_per_mm2': density_cells_per_mm2
            })
            
            print(f"Processed {os.path.basename(filepath)} | "
                  f"Cells: {cell_count}, Density: {density_cells_per_mm2:.2f} cells/mm^2")
        
        # Save CSV for current directory
        csv_path = os.path.join(current_output_dir, "results.csv")
        save_results_to_csv(results, csv_path)
        print(f"\nResults for {root} saved to {csv_path}")

    print("\nAll directories processed successfully!")

if __name__ == "__main__":
    main()
