# Paper Workflow

This is the dominant end-to-end path from tracked example TIFFs to figures, methods, and report artifacts.

## Canonical Command

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

This is the reproducible tracked-example path.
The tracked TIFFs in `examples/smoke_data/` are smoke/demo regression fixtures only; they are not the scientific count-validation benchmark for the paper.

For production analyses:
- omit `--spatial_envelope_sims 8`
- keep `--spatial_mode rigorous`
- point `--manifest` at your real cohort manifest

## Expected Outputs

Top-level study outputs:
- `study_summary.csv`
- `study_regions.csv`
- `stats/`
- `figures/`
- `methods_appendix.md`
- `report.html`
- `provenance.json`

Per-sample outputs under `samples/<sample_id>/`:
- `objects/`
- `regions/`
- `retina_frames/`
- `registered_maps/`
- `spatial/`

Reviewer-facing artifacts:
- `figures/`
- `stats/`
- `methods_appendix.md`
- `report.html`
- `provenance.json`

Internal/QC artifacts:
- per-sample `objects/`
- per-sample `spatial/`
- per-sample `registered_maps/`
- edit sidecars

## Reading The Bundle

Start here:
1. `report.html` for the run overview
2. `study_summary.csv` for sample-level outcomes
3. `study_regions.csv` for regional densities
4. `stats/` for selected statistics and design audit
5. `figures/` for presentation-ready plots
6. `methods_appendix.md` for run-derived methods text

## Regenerating From Saved Tables

If the image-processing run already completed, the saved tables remain the source of truth for downstream reporting:
- `study_summary.csv` and `study_regions.csv` drive cohort figures and stats
- `stats/` stores the selected statistical path and coefficient tables
- `methods_appendix.md` is derived from the resolved configuration and saved outputs

Do not rebuild figures from memory or notebooks without preserving the saved tables and `provenance.json`.

## Related Inputs

Tracked assets used by this workflow:
- `examples/manifests/example_study_manifest.csv`
- `examples/smoke_data/example_retina_a.tif`
- `examples/smoke_data/example_retina_b.tif`
- `examples/manual_annotations/example_manual_annotations.csv`

Use the tracked TIFFs and manual-count CSV to keep the workflow reproducible and regression-tested.
Do not treat these tiny smoke fixtures as matched-modality biological validation data.

For model benchmarking before a paper run, use [Model Training](model-training.md).
