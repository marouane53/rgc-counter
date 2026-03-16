# Model Training

The model platform is CLI-first. Custom checkpoints are trusted local assets only.

## Trust Boundary

- Treat custom Cellpose checkpoints, StarDist weights, and SAM checkpoints as trusted local assets only.
- Do not point the pipeline at untrusted model files.
- Provenance records the resolved model identity and trust mode.

## Benchmark Built-In Vs Custom Models

Use the tracked evaluation manifest:

```bash
python scripts/evaluate_models.py \
  --model_manifest examples/models/model_manifest.example.csv \
  --output_dir Outputs_model_eval \
  --no_gpu
```

Expected outputs:
- `per_run_metrics.csv`
- `model_summary.csv`
- `best_model.json`
- `evaluation_report.md`

Use this when you want to compare built-in Cellpose models against a custom checkpoint on a labeled subset.

## Fine-Tune Cellpose

Use the tracked training manifest:

```bash
python scripts/train_cellpose.py \
  --train_manifest examples/models/train_manifest.example.csv \
  --output_dir Outputs_model_train \
  --pretrained_model cyto \
  --n_epochs 5 \
  --no_gpu
```

Expected outputs:
- `training_config.json`
- copied manifest
- `training_report.md`
- checkpoint output from the Cellpose training API

This helper is intentionally thin. It validates the manifest, records the configuration, and calls the existing Cellpose training path.

## Use The Winning Checkpoint In Main CLI

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

Legacy custom Cellpose paths through `--model_type /path/to/model` still work, but they are recorded as compatibility usage.

## Related Files

- `examples/models/model_manifest.example.csv`
- `examples/models/train_manifest.example.csv`
- `examples/models/labels/`
- `examples/models/assets/legacy_cellpose_model`
