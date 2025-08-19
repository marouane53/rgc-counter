# RGC Counter

Modular, production‑ready RGC analysis suite for automated Retinal Ganglion Cell (RGC) counting with pluggable segmenters, multi‑metric focus QC, optional marker‑aware filtering, spatial statistics, TTA uncertainty, OME‑TIFF/OME‑Zarr I/O, and an optional optic‑nerve axon module.

## Description

This project provides an automated pipeline for:
- Cell segmentation via pluggable backends (Cellpose default, StarDist optional, experimental SAM)
- Focus region detection (legacy auto and robust multi‑metric QC)
- Optional marker‑aware inclusion/exclusion using phenotype rules
- Post-processing, counts, densities, and spatial mosaic statistics (NNRI, VDRI, Ripley’s K)
- Test‑time augmentation (TTA) with pixel‑vote uncertainty
- OME‑TIFF/OME‑Zarr I/O and HTML reporting
- Optional optic nerve axon analysis via AxonDeepSeg

If you used the previous “solid script” pipeline, all new features are opt‑in — your existing workflow still works.

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for faster processing)
- Windows 10 or higher
- macOS 11 or higher

## Installation

1. Clone the repository:
```bash
git clone https://github.com/marouane53/rgc-counter.git
cd rgc-counter
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

Notes:
- StarDist backend requires `stardist` and `csbdeep` (already listed in requirements).
- SAM backend requires installing `segment-anything` and providing a model checkpoint manually (not pinned here).
- Optic nerve module calls the `AxonDeepSeg` CLI if it is installed system‑wide.

## Project Structure

```
rgc-counter/
├── input/
├── Outputs/
├── src/
│   ├── __init__.py
│   ├── analysis.py
│   ├── axon.py                 # Optic nerve (AxonDeepSeg) integration
│   ├── cell_segmentation.py
│   ├── config.py
│   ├── focus_detection.py
│   ├── io_ome.py               # OME‑TIFF/OME‑Zarr I/O
│   ├── models.py               # Segmenter factory (Cellpose/StarDist/SAM)
│   ├── phenotype.py            # Marker‑aware filtering rules
│   ├── postprocessing.py
│   ├── qc.py                   # Multi‑metric focus QC
│   ├── report.py               # HTML report generator
│   ├── spatial.py              # Spatial statistics
│   ├── uncertainty.py          # TTA uncertainty
│   └── utils.py                # Universal loader + helpers
├── config.yaml                 # Suite configuration (see below)
├── phenotype_rules.example.yaml# Template for marker‑aware rules
├── main.py                     # CLI wiring for all modules
├── requirements.txt
├── run_pipeline.bat            # Interactive batch (Windows)
└── run_pipeline.sh             # Interactive batch (macOS/Linux)
```

## Key Features
- Pluggable segmenters: `cellpose` (default), `stardist`, or `sam`.
- Focus QC: combine Laplacian variance, Tenengrad, and high‑frequency metrics per tile.
- Marker‑aware logic: require positive markers and exclude confounds via phenotype rules.
- Spatial stats: NNRI, Voronoi regularity (VDRI), Ripley’s K; optional isodensity maps.
- TTA uncertainty: flips/rotations with pixel‑level voting.
- I/O: robust OME‑TIFF loading, optional OME‑Zarr writing for napari/QuPath.
- Reporting: HTML report with run info, summary table, and saved visuals.

## Usage

### Interactive Batch Processing

The easiest way to use the pipeline is through the interactive batch script:

1. Place your .tif images in the `input/` folder
2. On **Windows**, double-click `run_pipeline.bat` or run it from the command prompt.
   On **macOS/Linux**, run `./run_pipeline.sh` from the terminal.
3. Follow the interactive prompts to configure:
   - Input/output directories
   - Cell diameter (or let Cellpose auto-estimate)
   - Debug overlay options
   - GPU usage
   - CLAHE contrast enhancement
   - Focus detection mode (None / BBox / Legacy Auto / QC)
   - Segmentation backend, TTA, spatial stats, OME‑Zarr, and HTML report

### Focus Detection Modes

The pipeline offers four focus modes:
1. **None**: Analyze the entire image
2. **BBox**: Napari-assisted bounding box per image
3. **Legacy Auto**: Brightness + Laplacian thresholding (fast)
4. **QC**: Multi‑metric focus QC (recommended for batch data)

### Command Line Usage

For direct command line usage:

```bash
python main.py --input_dir "input" --output_dir "Outputs" [OPTIONS]
```

Common options (see `python main.py -h` for full list):
- `--backend {cellpose,stardist,sam}`: Choose segmenter backend
- `--focus_none|--focus_bbox|--focus_auto|--focus_qc`: Focus modes
- `--tta [--tta_transforms flip_h flip_v rot90 ...]`: Enable TTA
- `--phenotype_config phenotype_rules.yaml`: Marker‑aware filtering
- `--spatial_stats`: Compute NNRI/VDRI/Ripley’s K
- `--save_ome_zarr`: Save image + labels as OME‑Zarr
- `--write_html_report`: Generate HTML report
- `--save_debug`, `--apply_clahe`, `--use_gpu/--no_gpu`, `--diameter`, `--min_size`, `--max_size`

Examples:
```bash
# Backward‑compatible run
python main.py --input_dir input --output_dir Outputs --save_debug

# Robust focus QC, spatial stats, and report
python main.py \
  --input_dir input --output_dir Outputs \
  --focus_qc --backend cellpose --spatial_stats \
  --write_html_report --save_ome_zarr --save_debug

# Marker‑aware counting
python main.py --input_dir input --output_dir Outputs \
  --focus_qc --phenotype_config phenotype_rules.example.yaml --save_debug

# Test‑time augmentation
python main.py --input_dir input --output_dir Outputs \
  --tta --tta_transforms flip_h flip_v rot90

# Optic nerve axons (requires AxonDeepSeg CLI)
python main.py --input_dir input --output_dir Outputs --axon_dir optic_nerve_images
```

## Outputs

- `Outputs/results.csv`: Summary table with counts, area, density, backend, and optional spatial metrics.
- Debug overlays: `<image>_debug.png` when `--save_debug` is given.
- Isodensity maps: `<image>_isodensity.png` when `--spatial_stats` and `--save_debug` are used.
- OME‑Zarr stores: `<image>.zarr` when `--save_ome_zarr` is used.
- HTML report: `Outputs/report.html` when `--write_html_report` is used.

## Configuration

Edit `config.yaml` to customize:
- `cell_detection`: diameter, model_type, backend, use_gpu
- `analysis`: min/max cell size, spatial settings
- `visualization`: overlay_alpha, save_debug
- `io`: save_ome_zarr, write_html_report, chunk size
- `qc`: tile_size, brightness range, metric weights, threshold
- `tta`: enabled, transforms, combiner
- `phenotype_rules`: path to a rules YAML (see `phenotype_rules.example.yaml`)

Phenotype rules let you require marker positivity and exclude confounds like microglia overlap. See the example YAML for channel mapping, thresholds, and morphology priors.

## Performance Tips

1. **GPU Acceleration**: Enable GPU support for significantly faster processing
2. **Focus Detection**: 
   - Use automatic mode for batch processing
   - Manual mode provides highest accuracy but requires user interaction
3. **CLAHE Enhancement**: Enable for images with poor contrast
4. **Cell Diameter**: 
   - Specify if known for better accuracy
   - Leave blank for automatic estimation

## Git Integration

The project is set up to:
- Track all source code and configuration files
- Ignore input images and output results
- Maintain the `input/` and `Outputs/` directory structure
- Prevent accidental commits of large data files

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Cellpose](https://github.com/mouseland/cellpose)
- [StarDist](https://github.com/stardist/stardist) and [CSBDeep](https://github.com/CSBDeep/CSBDeep)
- [AICSImageIO](https://github.com/AllenCellModeling/aicsimageio) / [OME‑Zarr](https://github.com/ome/ome-zarr-py)
- [Napari](https://napari.org/)
- [AxonDeepSeg](https://github.com/neuropoly/axondeepseg)

## Support

For support, please open an issue in the GitHub repository or contact the maintainers.
