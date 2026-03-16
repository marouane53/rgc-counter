#!/usr/bin/env bash
# Deprecated legacy wrapper retained for backward reference only.
# Use `python main.py ...` for CLI runs or `python scripts/run_napari_ui.py` for the local UI.
# --------------------------------------------------
#  run_pipeline.sh
#  Cross-platform shell script for Cell Counting Project
# --------------------------------------------------

set -e

if [ ! -d ".venv" ]; then
    echo "[ERROR] .venv folder not found in the current directory." >&2
    echo "Please create a virtual environment named .venv first." >&2
    exit 1
fi

# Activate the virtual environment
source .venv/bin/activate

echo "Installing or updating Python requirements (please wait)..."
if ! pip install --upgrade pip --quiet > pip_install_log.txt 2>&1; then
    echo "[ERROR] Failed to upgrade pip." >&2
    echo "Check pip_install_log.txt for details." >&2
    exit 1
fi

if ! pip install -r requirements.txt --quiet >> pip_install_log.txt 2>&1; then
    echo "[ERROR] Failed to install dependencies." >&2
    echo "Check pip_install_log.txt for details." >&2
    exit 1
fi

echo "Requirements installed successfully."
echo

read -p "Enter the folder path that contains the .tif images (optional, default 'input'): " input_dir
input_dir=${input_dir:-input}
echo "Using input folder: $input_dir"

read -p "Enter the folder path where you want the results saved (optional, default 'Outputs'): " output_dir
output_dir=${output_dir:-Outputs}
echo "Using output folder: $output_dir"

read -p "Approximate RGC diameter (in pixels), or leave blank: " diameter_input
if [ -z "$diameter_input" ]; then
    diameter_arg=""
else
    diameter_arg="--diameter $diameter_input"
fi

read -p "Do you want to save debug overlays? (y/n): " debug_choice
if [[ "$debug_choice" =~ ^[Yy]$ ]]; then
    debug_arg="--save_debug"
else
    debug_arg=""
fi

read -p "Do you want to use GPU for Cellpose? (y/n): " gpu_choice
if [[ "$gpu_choice" =~ ^[Yy]$ ]]; then
    gpu_arg="--use_gpu"
else
    gpu_arg=""
fi

read -p "Apply CLAHE for contrast enhancement? (y/n): " clahe_choice
if [[ "$clahe_choice" =~ ^[Yy]$ ]]; then
    clahe_arg="--apply_clahe"
else
    clahe_arg=""
fi

read -p "Segmentation backend (cellpose/stardist/sam) [default cellpose]: " backend_choice
if [ -n "$backend_choice" ]; then backend_arg="--backend $backend_choice"; else backend_arg=""; fi

read -p "Enable TTA? (y/n): " tta_choice
if [[ "$tta_choice" =~ ^[Yy]$ ]]; then tta_arg="--tta"; else tta_arg=""; fi

read -p "Compute spatial stats (NNRI/VDRI/Ripley)? (y/n): " spatial_choice
if [[ "$spatial_choice" =~ ^[Yy]$ ]]; then spatial_arg="--spatial_stats"; else spatial_arg=""; fi

read -p "Save OME-Zarr (image + labels)? (y/n): " zarr_choice
if [[ "$zarr_choice" =~ ^[Yy]$ ]]; then zarr_arg="--save_ome_zarr"; else zarr_arg=""; fi

read -p "Write HTML report? (y/n): " report_choice
if [[ "$report_choice" =~ ^[Yy]$ ]]; then report_arg="--write_html_report"; else report_arg=""; fi

echo "(Focus) 1 None  2 BBox  3 Legacy Auto  4 QC Multi-metric"
read -p "Enter 1-4: " focus_mode_choice
case "$focus_mode_choice" in
  2) focus_mode_arg="--focus_bbox" ;;
  3) focus_mode_arg="--focus_auto" ;;
  4) focus_mode_arg="--focus_qc" ;;
  *) focus_mode_arg="--focus_none" ;;
esac

echo "--------------------------------------------------"
echo "Now running the main pipeline..."
echo "--------------------------------------------------"

python main.py \
  --input_dir "$input_dir" \
  --output_dir "$output_dir" \
  $diameter_arg \
  $debug_arg \
  $gpu_arg \
  $clahe_arg \
  $focus_mode_arg \
  $backend_arg \
  $tta_arg \
  $spatial_arg \
  $zarr_arg \
  $report_arg

echo "Return code: $?"
