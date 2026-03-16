# RGC Counter

Local-first retinal spatial phenotyping suite for automated Retinal Ganglion Cell (RGC) analysis.

The current platform supports:
- per-object measurement tables with schema validation
- ONH-centered retina registration and biology-native region summaries
- rigorous spatial outputs, validation, calibration, and study-mode statistics
- replayable review sidecars and a local napari workflow
- model benchmarking and Cellpose fine-tuning helpers
- HTML reports, methods appendices, provenance, and tracked example datasets

## Install

Public package install after the first PyPI publication:

```bash
pip install rgc-counter
```

Napari users:

```bash
pip install "rgc-counter[ui]"
```

Developer install from a checkout:

```bash
git clone https://github.com/marouane53/rgc-counter.git
cd rgc-counter
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install -e .[dev]
```

Notes:
- Custom checkpoints and weights are trusted local assets only.
- SAM is not pinned in the package dependencies; install it separately if you want that backend.
- Optic-nerve analysis depends on an external `AxonDeepSeg` CLI.

## Quickstart

Tracked example cohort:

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
- `Outputs_example/figures/`
- `Outputs_example/methods_appendix.md`
- `Outputs_example/report.html`
- per-sample `spatial/`, `objects/`, `regions/`, and `retina_frames/` bundles

For production analyses, omit `--spatial_envelope_sims 8` and use the default rigorous simulation budget.

Frontier atlas subtype priors stay opt-in. Example:

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

## Guides

- [Researcher Guide](docs/researcher-guide.md): the main task-driven entrypoint
- [Paper Workflow](docs/paper-workflow.md): the dominant cohort-to-figures path
- [Model Training](docs/model-training.md): evaluate and fine-tune segmentation models
- [Stats Designs](docs/stats-designs.md): how study statistics are selected and reported
- [Testing](TESTING.md): engineering smoke checks and release verification
- [Examples](examples/README.md): what each tracked example asset is for

## Outputs

Single-image runs can write:
- `results.csv`
- `objects/*_objects.parquet`
- `regions/*_region_summary.csv`
- `retina_frames/*_retina_frame.json`
- `registered_maps/*_registered_density_map.png|svg`
- `spatial/*_spatial_summary.csv`
- `spatial/*_spatial_curves.csv`
- `spatial/*_ripley_l_global.png`
- `spatial/*_pair_correlation_global.png`
- `uncertainty/*_fgprob.tif`
- `qc_maps/*_focus_score.tif`
- `report.html`
- `provenance.json`

Study-mode runs can additionally write:
- `study_summary.csv|parquet`
- `study_regions.csv|parquet`
- `stats/`
- `stats_mixed/`
- `validation/`
- `figures/`
- `calibration/`
- `atlas/`
- `tracking/`
- `methods_appendix.md`

## Main CLI Surface

Use `python main.py -h` for the full list. Common options:
- `--backend {cellpose,stardist,sam}`
- `--cellpose_model`, `--stardist_weights`, `--model_alias`
- `--atlas_subtype_priors`
- `--focus_none|--focus_bbox|--focus_auto|--focus_qc`
- `--tta`, `--tiling --tile_size --tile_overlap`
- `--spatial_stats --spatial_mode legacy|rigorous`
- `--spatial_envelope_sims`, `--spatial_random_seed`
- `--register_retina --region_schema --onh_mode --onh_xy --dorsal_xy`
- `--phenotype_engine legacy|v2`, `--marker_metrics`, `--interaction_metrics`
- `--manifest`, `--study_output_dir`, `--manual_annotations`
- `--calibration_grid`
- `--track_longitudinal --track_max_disp_px`
- `--write_object_table --write_provenance --write_html_report`
- `--version`

## Local UI

Launch the napari dock with:

```bash
python scripts/run_napari_ui.py
```

The dock reuses the same runtime as the CLI and can export a replayable single-image artifact bundle.

## Developer Notes

- The package version is `1.0.0`.
- CI runs pytest plus packaging checks.
- TestPyPI and PyPI publishing are handled by GitHub Actions Trusted Publishing workflows.
- Live publication still requires reserving `rgc-counter` on TestPyPI/PyPI and configuring this repository as a Trusted Publisher.
- The active roadmap lives in `planning/`, and completed plans move to `planning/archive/`.
