#!/usr/bin/env bash
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

echo "--------------------------------------------------"
echo "Choose focus mode for each image:"
echo "(1) No focus bounding (analyze entire image)"
echo "(2) Manual bounding-box in Napari"
echo "(3) Automatic tile-based focus detection (improved)"
echo "--------------------------------------------------"
read -p "Enter 1, 2, or 3: " focus_mode_choice

case "$focus_mode_choice" in
    1) focus_mode_arg="--focus_none" ;;
    2) focus_mode_arg="--focus_bbox" ;;
    3) focus_mode_arg="--focus_auto" ;;
    *)
        echo "Invalid choice. Defaulting to (1) No focus bounding."
        focus_mode_arg="--focus_none"
        ;;
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
  $focus_mode_arg

echo "Return code: $?"

