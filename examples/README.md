# Examples

This folder contains the tracked assets used by the public docs, smoke tests, and release validation.

## Researcher Workflow Assets

- `smoke_data/example_retina_a.tif`
- `smoke_data/example_retina_b.tif`
- `manifests/example_study_manifest.csv`

Use these with:
- [docs/researcher-guide.md](../docs/researcher-guide.md)
- [docs/paper-workflow.md](../docs/paper-workflow.md)
- [TESTING.md](../TESTING.md)

## Validation And Manual Reference Assets

- `manual_annotations/example_manual_annotations.csv`

Use this with:
- [docs/paper-workflow.md](../docs/paper-workflow.md)
- [docs/stats-designs.md](../docs/stats-designs.md)

## Phenotype And Atlas Assets

- `phenotype_rules/phenotype_rules.v2.example.yaml`
- `atlas/atlas_reference.example.csv`
- `calibration/calibration_grid.example.yaml`

Use these with:
- [docs/researcher-guide.md](../docs/researcher-guide.md)
- [TESTING.md](../TESTING.md)

## Model Platform Assets

- `models/model_manifest.example.csv`
- `models/train_manifest.example.csv`
- `models/labels/*.tif`
- `models/assets/legacy_cellpose_model`

Use these with:
- [docs/model-training.md](../docs/model-training.md)
- [examples/models/README.md](models/README.md)

## Canonical Tracked Example Command

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
