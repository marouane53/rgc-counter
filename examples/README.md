# Examples

This folder contains a tiny tracked smoke dataset and matching inputs for a reproducible end-to-end run.

- `smoke_data/`: tiny synthetic retina-like TIFF inputs committed to the repo
- `manifests/example_study_manifest.csv`: example study manifest
- `manual_annotations/example_manual_annotations.csv`: manual-count references for validation
- `phenotype_rules/`: example phenotype configs
- `atlas/atlas_reference.example.csv`: example atlas prior table

Recommended smoke command:

```bash
python main.py \
  --input_dir examples/smoke_data \
  --output_dir Outputs_example_maps \
  --focus_qc \
  --tta \
  --register_retina \
  --onh_mode auto_combined \
  --dorsal_xy 48 14 \
  --write_object_table \
  --write_provenance \
  --write_html_report \
  --write_uncertainty_maps \
  --write_qc_maps \
  --strict_schemas \
  --no_gpu
```

Tracked study bundle:

```bash
python main.py \
  --manifest examples/manifests/example_study_manifest.csv \
  --manual_annotations examples/manual_annotations/example_manual_annotations.csv \
  --study_output_dir Outputs_example \
  --focus_none \
  --register_retina \
  --write_object_table \
  --write_provenance \
  --write_html_report \
  --strict_schemas \
  --no_gpu
```
