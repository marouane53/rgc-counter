# retinal-phenotyper

retinal-phenotyper is a local-first retinal image analysis platform, not just a cell counter. It processes flat-mount and related retinal imaging inputs into reproducible object tables, retina-registered region summaries, spatial inference outputs, validation and calibration bundles, longitudinal tracking tables, report artifacts, and provenance suitable for reviewable study workflows.

The current platform supports:
- automated segmentation with `cellpose`, `stardist`, or `sam`
- explicit custom-model selection, benchmarking, and Cellpose fine-tuning helpers
- replayable review sidecars and a local napari workflow
- ONH-centered retina registration with region-aware summaries
- rigorous spatial inference with exact domain clipping and CSR envelopes
- study-mode statistics with mixed-effects fallback logic and design-audit artifacts
- atlas reference comparison plus opt-in probabilistic atlas subtype priors
- validation against manual references, calibration sweeps, and longitudinal tracking
- HTML reports, methods appendices, provenance, tracked examples, and CI-tested packaging

## What The Project Does

At a high level, the pipeline turns raw retinal images into:
- per-object tables with centroids, areas, phenotypes, marker features, uncertainty summaries, and optional atlas subtype priors
- per-region summaries in biologically meaningful registered coordinates
- study-level cohort tables, validation summaries, calibration outputs, and figures
- reviewable artifact bundles with provenance, methods text, and HTML reports

This is the current scope of the repo today:
- single-image analysis
- cohort and manifest-driven study analysis
- schema-validated outputs
- retina registration and regional biology overlays
- spatial, atlas, and tracking analysis
- model benchmarking and fine-tuning utilities
- local UI support through napari

## Current Status

The codebase is feature-complete for the planned v1.0 path and the two post-v1.0 frontier phases are also implemented:
- reviewer-ready study statistics
- overlap-aware tiled inference
- model registry, evaluation, and training helpers
- rigorous spatial inference
- researcher-facing docs and release plumbing
- atlas subtype priors
- registration-aware longitudinal tracking

The package metadata and Trusted Publishing workflows are in the repo. The package and primary CLI name are `retinal-phenotyper`.

## Installation

### From a checkout

```bash
git clone https://github.com/marouane53/retinal-phenotyper.git
cd retinal-phenotyper
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install -e .[dev]
```

Napari users:

```bash
pip install -e ".[dev,ui]"
```

### After PyPI publication

```bash
pip install retinal-phenotyper
```

Napari users:

```bash
pip install "retinal-phenotyper[ui]"
```

Notes:
- Custom checkpoints and weights are trusted local assets only.
- Do not load untrusted model files.
- SAM is not pinned as a default dependency path; install its requirements separately if you plan to use that backend.
- Optional optic-nerve analysis depends on an external `AxonDeepSeg` CLI.

## Quick Start

If you are running from a clone, use `python main.py ...`. If you installed the package, replace `python main.py` with `retinal-phenotyper`.

### Canonical tracked-example cohort

```bash
python main.py \
  --manifest examples/manifests/example_study_manifest.csv \
  --study_output_dir Outputs_example \
  --focus_none \
  --register_retina \
  --region_schema mouse_flatmount_v1 \
  --spatial_stats --spatial_mode rigorous --spatial_envelope_sims 8 \
  --write_object_table \
  --write_provenance \
  --write_html_report
```

This writes:
- `Outputs_example/study_summary.csv`
- `Outputs_example/study_regions.csv`
- `Outputs_example/stats/`
- `Outputs_example/stats_mixed/`
- `Outputs_example/validation/`
- `Outputs_example/figures/`
- `Outputs_example/methods_appendix.md`
- `Outputs_example/report.html`
- per-sample `objects/`, `regions/`, `retina_frames/`, `registered_maps/`, and `spatial/` bundles

For production spatial analyses, omit `--spatial_envelope_sims 8` and use the default rigorous simulation budget.

### Single-image artifact bundle

```bash
python main.py \
  --input_dir input \
  --output_dir Outputs \
  --focus_qc \
  --register_retina \
  --spatial_stats --spatial_mode rigorous \
  --write_object_table \
  --write_provenance \
  --write_html_report
```

### Registration-aware longitudinal tracking

```bash
python main.py \
  --manifest your_study_manifest.csv \
  --study_output_dir Outputs_tracking \
  --track_longitudinal \
  --tracking_mode registered \
  --write_object_table \
  --write_provenance \
  --write_html_report
```

This adds:
- `tracking/track_observations.csv`
- `tracking/track_pair_qc.csv`
- `tracking/track_summary.csv`

### Opt-in atlas subtype priors

```bash
python main.py \
  --manifest examples/manifests/example_study_manifest.csv \
  --study_output_dir Outputs_atlas_subtypes \
  --register_retina \
  --region_schema mouse_flatmount_v1 \
  --atlas_subtype_priors examples/atlas/atlas_subtype_priors.example.yaml \
  --write_object_table \
  --write_provenance \
  --write_html_report
```

Atlas subtype outputs are explicitly probabilistic priors, not validated subtype truth.

## Key Capabilities

### Segmentation and model platform

- Built-in backends: Cellpose, StarDist, SAM
- Custom model selection with explicit CLI flags
- Model provenance stored in outputs and reports
- Benchmark harness for comparing multiple models on labeled data
- Cellpose fine-tuning helper and tracked model manifests

### Review and correction loop

- Replayable edit sidecars for delete, merge, relabel, and landmark edits
- Local napari dock for single-image review
- CLI reruns that apply saved edits deterministically

### Retina registration and biology-aware summaries

- ONH-centered coordinate registration
- CLI, sidecar, and auto-ONH modes
- region schemas such as `mouse_flatmount_v1`
- ring, quadrant, sector, and peripapillary summaries
- tissue-aware region areas based on actual tissue coverage

### Measurements and phenotype outputs

- per-object morphology and density metrics
- phenotype engine integration
- marker feature summaries
- interaction metrics
- optional uncertainty columns from foreground-probability maps

### Spatial inference

- legacy descriptive spatial metrics
- rigorous mode with exact Voronoi clipping
- border-corrected Ripley `L`
- pair-correlation `g`
- Monte Carlo CSR envelopes
- region-wise spatial summaries when registration is available

### Validation, calibration, and study statistics

- validation against manual counts or labels
- agreement plots and validation summaries
- calibration sweeps with saved best-parameter bundles
- study statistics with `auto`, `simple`, and `mixed` modes
- design-audit outputs for nested study designs

### Atlas and tracking layers

- region-level atlas comparison against expected densities
- opt-in atlas subtype priors that fuse region and marker evidence
- longitudinal tracking in centroid or registration-aware mode
- pairwise tracking QC and common-frame trajectory coordinates

## Outputs

### Single-image runs can write

- `results.csv`
- `objects/*_objects.parquet|csv`
- `regions/*_region_summary.parquet|csv`
- `retina_frames/*_retina_frame.json`
- `registered_maps/*_registered_density_map.png|svg`
- `spatial/*_spatial_summary.csv`
- `spatial/*_spatial_curves.csv`
- `spatial/*_ripley_l_global.png`
- `spatial/*_pair_correlation_global.png`
- `uncertainty/*_fgprob.tif`
- `qc_maps/*_focus_score.tif`
- `atlas_subtypes/*_atlas_subtype_summary.csv`
- `atlas_subtypes/*_atlas_subtype_region_summary.csv`
- `report.html`
- `provenance.json`

### Study-mode runs can additionally write

- `study_summary.csv|parquet`
- `study_regions.csv|parquet`
- `stats/`
- `stats_mixed/`
- `validation/`
- `figures/`
- `calibration/`
- `atlas/`
- `atlas_subtypes/`
- `tracking/`
- `methods_appendix.md`

## Main CLI Surface

Use `python main.py -h` for the full list. Common options:
- `--backend {cellpose,stardist,sam}`
- `--cellpose_model`, `--stardist_weights`, `--model_alias`
- `--focus_none|--focus_bbox|--focus_auto|--focus_qc`
- `--tta`, `--tiling --tile_size --tile_overlap`
- `--spatial_stats --spatial_mode legacy|rigorous`
- `--spatial_envelope_sims`, `--spatial_random_seed`
- `--register_retina --region_schema --onh_mode --onh_xy --dorsal_xy`
- `--phenotype_engine legacy|v2`, `--marker_metrics`, `--interaction_metrics`
- `--atlas_reference`
- `--atlas_subtype_priors`
- `--manifest`, `--study_output_dir`, `--manual_annotations`
- `--stats_mode auto|simple|mixed`
- `--calibration_grid`
- `--track_longitudinal --tracking_mode centroid|registered --track_max_disp_px`
- `--write_object_table --write_provenance --write_html_report`
- `--version`

## Guides And Reference Material

- [Researcher Guide](docs/researcher-guide.md): task-driven entrypoint from raw TIFFs to study outputs
- [Paper Workflow](docs/paper-workflow.md): the dominant reviewer-facing cohort path
- [Model Training](docs/model-training.md): model evaluation and Cellpose fine-tuning
- [Stats Designs](docs/stats-designs.md): how the statistics layer selects simple vs mixed-effects paths
- [Testing](TESTING.md): engineering smoke checks and release validation
- [Examples](examples/README.md): tracked example assets and what each one is for

## Local UI

Launch the napari dock with:

```bash
python scripts/run_napari_ui.py
```

The dock reuses the same runtime as the CLI and can export replayable single-image review artifacts.

## Development And Release Notes

- Current package version: `1.0.0`
- CI runs pytest and packaging checks
- Trusted Publishing workflows for TestPyPI and PyPI are included in `.github/workflows/`
- The local roadmap is archived under `planning/archive/`
- The current roadmap has no active remaining phases in `planning/`

## In Short

retinal-phenotyper is a reproducible retinal analysis workbench: segmentation, review, retina registration, spatial inference, atlas integration, validation, calibration, longitudinal tracking, and publication-oriented outputs, all from a local, scriptable pipeline.
