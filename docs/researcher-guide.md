# Researcher Guide

This is the main task-driven entrypoint for using RGC Counter as a researcher.

## I have single-image flat-mount TIFFs

Use the CLI when you want a local artifact bundle for one image or a folder of images.

Example:

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

Expected outputs:
- `results.csv`
- `objects/`
- `regions/`
- `retina_frames/`
- `registered_maps/`
- `spatial/`
- `report.html`
- `provenance.json`

If you are testing quickly on the tracked examples, use the exact smoke command in [Paper Workflow](paper-workflow.md).

## I have a cohort

Use a manifest and study mode.

Canonical tracked-example workflow:

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

What to inspect after the run:
- `study_summary.csv`
- `study_regions.csv`
- `stats/`
- `figures/`
- `methods_appendix.md`
- `report.html`
- per-sample `spatial/`, `objects/`, `regions/`, and `retina_frames/`

For production analyses, omit `--spatial_envelope_sims 8` and use the default rigorous envelope budget.

For the full reviewer-facing path, go to [Paper Workflow](paper-workflow.md).

Frontier atlas subtype priors are opt-in. Minimal example:

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

## I want to review a few bad segmentations

Launch napari:

```bash
python scripts/run_napari_ui.py
```

Suggested loop:
1. Open an image.
2. Optionally add ONH and dorsal landmarks as a two-point layer.
3. Run the pipeline in the dock.
4. Queue delete, merge, relabel, or landmark edits.
5. Save the replayable sidecar.
6. Re-run the CLI with `--apply_edits path/to/image.edits.json` or leave the sidecar next to the image for auto-detection.

The sidecar is replayable; the raw image and source masks are not edited in place.

## I want to use a custom model

Custom checkpoints are trusted local assets only.

Preferred custom-model usage:

```bash
python main.py \
  --input_dir input \
  --output_dir Outputs_custom_model \
  --backend cellpose \
  --cellpose_model /absolute/path/to/rgc_cellpose_model \
  --model_alias "lab-rgc-v3" \
  --write_object_table \
  --write_provenance
```

To benchmark or fine-tune models, use [Model Training](model-training.md).

## I want a paper-ready figure

Use study mode, not ad hoc single-image runs.

The standard path is:
- run the cohort workflow
- inspect `figures/`
- use `report.html` and `methods_appendix.md`
- keep `provenance.json` with the study outputs

For the exact sequence and reviewer-facing outputs, use [Paper Workflow](paper-workflow.md).

## Where to go next

- [Paper Workflow](paper-workflow.md)
- [Model Training](model-training.md)
- [Stats Designs](stats-designs.md)
- [Testing](../TESTING.md)
- [Examples](../examples/README.md)
