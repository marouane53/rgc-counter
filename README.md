# RGC Counter

Local-first retinal spatial phenotyping suite for automated Retinal Ganglion Cell (RGC) analysis. The project now goes beyond counting: it produces schema-validated per-object tables, registered retina maps, phenotype-aware measurements, study-level statistics, atlas comparisons, modality-adapted runs for non-flat-mount data, longitudinal tracking, calibration outputs, HTML reports, replayable review sidecars, and a local napari workflow.

## Description

This project provides an automated pipeline for:
- Cell segmentation via pluggable backends (Cellpose default, StarDist optional, experimental SAM)
- Focus region detection (legacy auto and robust multi‑metric QC)
- Marker-aware filtering and a richer phenotype engine with per-object measurements
- Versioned per-object/study tables, provenance, uncertainty/QC map persistence, and reproducible output bundles
- ONH-centered retina registration, automatic ONH detection, biology-native region summaries, and registered density maps
- Study-mode cohort analysis with paired-eye, calibration, and validation outputs
- Publishable reports, figures, and a generated methods appendix
- Atlas comparison against expected regional density priors
- Modality adaptation for flat-mount, OCT, vis-OCT/fluorescence, and light-sheet style inputs
- Longitudinal point matching across repeated timepoints
- Tiled inference for large wholemounts, OME‑TIFF/OME‑Zarr I/O, a local napari dock, and optional optic-nerve axon analysis

If you used the previous “solid script” pipeline, all new features are opt‑in — your existing workflow still works.

## Prerequisites

- Python 3.9 or higher
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

3. Install the package:
```bash
pip install -e .[dev]
```

Fallback, if you prefer the existing requirements files:
```bash
pip install -r requirements.txt
```

Notes:
- StarDist backend requires `stardist` and `csbdeep` (already listed in requirements).
- The napari UI requires the `ui` extras (`pip install -e .[ui]`) or the existing requirements install.
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
│   ├── atlas.py                # Atlas reference loading and deviation scoring
│   ├── axon.py                 # Optic nerve (AxonDeepSeg) integration
│   ├── cell_segmentation.py
│   ├── config.py
│   ├── focus_detection.py
│   ├── io_ome.py               # OME‑TIFF/OME‑Zarr I/O
│   ├── calibration.py          # Manifest-driven threshold sweeps and ranking
│   ├── edits.py                # Replayable human-review sidecars
│   ├── landmarks.py            # Automatic ONH detection helpers
│   ├── modalities/
│   │   ├── __init__.py
│   │   ├── oct.py
│   │   ├── vis_octf.py
│   │   └── lightsheet.py
│   ├── models.py               # Segmenter factory (Cellpose/StarDist/SAM)
│   ├── review.py               # Edit-sidecar application stage
│   ├── run_service.py          # Reusable runtime for napari and future UIs
│   ├── schema.py               # Table contracts and schema validation
│   ├── tiling.py               # Overlapping tiled inference and stitching
│   ├── uncertainty_io.py       # Foreground/QC map writers
│   ├── phenotype.py            # Marker‑aware filtering rules
│   ├── postprocessing.py
│   ├── qc.py                   # Multi‑metric focus QC
│   ├── report.py               # HTML report generator
│   ├── spatial.py              # Spatial statistics
│   ├── track.py                # Longitudinal centroid matching
│   ├── uncertainty.py          # TTA uncertainty
│   ├── ui_napari/
│   │   ├── __init__.py
│   │   ├── dock_widget.py
│   │   └── helpers.py
│   └── utils.py                # Universal loader + helpers
├── config.yaml                 # Suite configuration (see below)
├── phenotype_rules.example.yaml# Template for marker‑aware rules
├── main.py                     # CLI wiring for all modules
├── pyproject.toml
├── requirements.txt
├── examples/
├── run_pipeline.bat            # Interactive batch (Windows)
└── run_pipeline.sh             # Interactive batch (macOS/Linux)
```

## Key Features
- Pluggable segmenters: `cellpose` (default), `stardist`, or `sam`.
- Focus QC: combine Laplacian variance, Tenengrad, and high‑frequency metrics per tile.
- Marker-aware logic: legacy rule filtering plus a richer phenotype engine with per-object marker metrics.
- Hard contracts: saved object/region/study tables carry `schema_version` and `table_kind`, with opt-in strict validation.
- Uncertainty persistence: save TTA foreground-probability maps, focus-score maps, and per-object uncertainty summaries.
- Spatial outputs: NNRI, Voronoi regularity (VDRI), Ripley’s K, optional isodensity maps, and registered retina heatmaps.
- Structured outputs: per-object tables, region tables, provenance JSON, and study-level bundles.
- Review loop: replayable sidecars for delete/merge/relabel/landmark corrections from napari or CLI reruns.
- Retina registration: ONH-centered coordinates, `auto_hole` / `auto_combined` ONH modes, tissue-aware region areas, and region summaries.
- Cohort analysis: manifest-driven study tables, paired-eye stats, validation summaries, calibration sweeps, and figure generation.
- Atlas comparison: observed-vs-expected regional density deltas from a reference CSV.
- Modality adapters: flat-mount, OCT, vis-OCT/fluorescence, and light-sheet style data can feed the same downstream contracts.
- Longitudinal tracking: opt-in centroid matching across repeated timepoints in study mode.
- Scale-up path: tiled inference for large images without changing downstream measurements.
- Local UI: a napari dock for interactive review, sidecar export, and bundle export.

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

### Napari UI

For local interactive review, launch the dock directly:

```bash
python scripts/run_napari_ui.py
```

The dock reuses the same core pipeline as the CLI. A practical flow is:

1. Open an image in napari
2. Optionally add a points layer with two landmarks: ONH first, dorsal second
3. Choose backend and analysis options
4. Run segmentation and inspect labels, focus mask, and centroids
5. Optionally queue delete/merge/relabel edits and save a replayable sidecar
6. Export the current run as a local artifact bundle

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
- `--modality flatmount|oct|vis_octf|lightsheet`: Adapt non-flat-mount inputs into the shared analysis path
- `--modality_projection max|mean|sum`: Projection mode for volumetric modality adapters
- `--modality_channel_index N`: Channel index to use when a modality adapter needs one
- `--focus_none|--focus_bbox|--focus_auto|--focus_qc`: Focus modes
- `--tta [--tta_transforms flip_h flip_v rot90 ...]`: Enable TTA
- `--tiling --tile_size 1024 --tile_overlap 128`: Enable tiled inference for large inputs
- `--phenotype_config phenotype_rules.yaml`: Marker‑aware filtering
- `--phenotype_engine legacy|v2`: Use the legacy filter or the v2 phenotype engine
- `--marker_metrics`: Add per-object marker and morphology metrics
- `--interaction_metrics`: Add phenotype interaction metrics
- `--spatial_stats`: Compute NNRI/VDRI/Ripley’s K
- `--register_retina`: Register cells into an ONH-centered retina coordinate frame
- `--region_schema mouse_flatmount_v1|rat_flatmount_v1`: Choose a biology-native region schema
- `--onh_mode cli|sidecar|auto_hole|auto_combined`: Supply manual landmarks or detect the ONH automatically
- `--onh_xy X Y`, `--dorsal_xy X Y`: Explicit retina landmarks for registration
- `--retina_frame_path path/to/frame.json`: Optional sidecar frame JSON
- `--apply_edits path/to/image.edits.json`: Replay review edits before registration and reporting
- `--atlas_reference path/to/atlas.csv`: Compare registered regional densities against a reference atlas
- `--manifest path/to/manifest.csv`: Run study mode from a manifest instead of scanning an input folder
- `--study_output_dir path/to/output`: Separate output root for study mode
- `--manual_annotations path/to/manual.csv`: Optional per-sample manual count or label-path overrides
- `--calibration_grid path/to/grid.yaml`: Sweep QC/phenotype parameters over a manifest with manual references
- `--track_longitudinal --track_max_disp_px 20`: Build opt-in longitudinal tracks in study mode
- `--save_ome_zarr`: Save image + labels as OME‑Zarr
- `--write_html_report`: Generate HTML report
- `--write_object_table`: Save per-image object tables
- `--write_provenance`: Save run provenance metadata
- `--write_uncertainty_maps`, `--write_qc_maps`: Persist TTA and QC maps as TIFFs
- `--strict_schemas`: Fail fast if saved tables violate required contracts
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

# Phenotype engine v2
python main.py --input_dir input --output_dir Outputs \
  --focus_none \
  --phenotype_engine v2 \
  --phenotype_config phenotype_rules.v2.example.yaml \
  --marker_metrics --interaction_metrics \
  --write_object_table --write_provenance

# Retina registration with explicit ONH + dorsal coordinates
python main.py --input_dir input --output_dir Outputs \
  --focus_none --register_retina \
  --onh_mode cli --onh_xy 512 498 --dorsal_xy 530 120 \
  --region_schema mouse_flatmount_v1 \
  --write_object_table --write_provenance

# OCT volume projected into the shared 2D analysis path
python main.py --input_dir oct_input --output_dir Outputs_oct \
  --modality oct --modality_projection max --focus_none \
  --write_object_table --write_provenance

# Study mode from a manifest
python main.py \
  --manifest test_subjects/synthetic/cohort/cohort_manifest.csv \
  --study_output_dir Outputs_study \
  --focus_none --register_retina \
  --write_object_table --write_provenance --write_html_report

# Study mode with atlas comparison and longitudinal tracking
python main.py \
  --manifest test_subjects/synthetic/cohort/cohort_manifest.csv \
  --study_output_dir Outputs_study \
  --focus_none --register_retina \
  --atlas_reference path/to/atlas.csv \
  --track_longitudinal --track_max_disp_px 20 \
  --write_object_table --write_provenance --write_html_report

# Tracked example smoke run with QC, TTA, schema validation, and auto-ONH
python main.py \
  --input_dir examples/smoke_data \
  --output_dir Outputs_example_maps \
  --focus_qc \
  --tta \
  --register_retina \
  --onh_mode auto_combined \
  --dorsal_xy 48 14 \
  --strict_schemas \
  --write_object_table --write_provenance --write_html_report \
  --write_uncertainty_maps --write_qc_maps \
  --no_gpu

# Tracked example study bundle
python main.py \
  --manifest examples/manifests/example_study_manifest.csv \
  --manual_annotations examples/manual_annotations/example_manual_annotations.csv \
  --study_output_dir Outputs_example \
  --focus_none \
  --register_retina \
  --strict_schemas \
  --write_object_table --write_provenance --write_html_report \
  --no_gpu

# Calibration sweep on a manifest
python main.py \
  --manifest examples/manifests/example_study_manifest.csv \
  --manual_annotations examples/manual_annotations/example_manual_annotations.csv \
  --study_output_dir Outputs_example \
  --focus_none \
  --register_retina \
  --calibration_grid examples/calibration/calibration_grid.example.yaml \
  --no_gpu

# Test‑time augmentation
python main.py --input_dir input --output_dir Outputs \
  --tta --tta_transforms flip_h flip_v rot90

# Optic nerve axons (requires AxonDeepSeg CLI)
python main.py --input_dir input --output_dir Outputs --axon_dir optic_nerve_images
```

## Outputs

- `Outputs/results.csv`: Summary table with counts, area, density, backend, and optional spatial metrics.
- `Outputs/objects/*_objects.parquet`: Per-object measurement tables when `--write_object_table` is used.
- `Outputs/uncertainty/*_fgprob.tif`: Foreground-probability maps when `--write_uncertainty_maps` is used.
- `Outputs/qc_maps/*_focus_score.tif`: Focus-score maps when `--write_qc_maps` is used.
- `Outputs/provenance.json`: Run metadata, resolved config, environment details, and per-image artifacts when `--write_provenance` is used.
- `Outputs/regions/*_region_summary.csv`: Region-wise area, count, and density summaries when `--register_retina` is used.
- `Outputs/retina_frames/*_retina_frame.json`: Resolved ONH/orientation frame for each registered image.
- `Outputs/registered_maps/*_registered_density_map.png|svg`: ONH-centered registered retina maps.
- `<image>.<ext>.edits.json`: Optional replayable review sidecars for delete/merge/relabel/landmark corrections.
- `Outputs_study/study_summary.csv|parquet`: Manifest-level sample table in study mode.
- `Outputs_study/study_regions.csv|parquet`: Combined region table across the cohort.
- `Outputs_study/stats/*.csv`: Study-level and region-level statistical summaries.
- `Outputs_study/validation/*`: Agreement plots and validation summaries when manual references are available.
- `Outputs_study/figures/*`: Cohort-level condition and paired-eye figures.
- `Outputs_study/methods_appendix.md`: Table-driven methods appendix generated from the actual run configuration.
- `Outputs_study/calibration/*`: Grid-search ranking, best parameters, agreement plots, and a calibration report when `--calibration_grid` is used.
- `Outputs/atlas/*` or `Outputs_study/atlas/*`: Optional atlas comparison tables when `--atlas_reference` is used.
- `Outputs_study/tracking/*`: Optional longitudinal track observations and summaries when `--track_longitudinal` is used.
- `Outputs_napari/*`: Optional single-image export bundle from the napari dock, including `results.csv`, overlays, object tables, report, and provenance.
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

For the richer v2 engine, see `phenotype_rules.v2.example.yaml`. It adds class priorities, named masks, and reusable per-object feature rules.

## Atlas Reference Format

Atlas comparison uses a CSV with at least these columns:

- `region_axis`
- `region_label`
- `expected_density_cells_per_mm2`

Optional columns:

- `atlas_name`
- `retina_region_schema`
- `expected_sd`

The region keys should match the labels produced by `--register_retina`, for example `ring` plus `central`, `pericentral`, and `peripheral`.

A tracked example atlas is included in `examples/atlas/atlas_reference.example.csv`.

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
