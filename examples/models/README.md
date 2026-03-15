# Model Platform Examples

This folder contains tracked manifests for the Phase 10 model workflow.

## Trust Boundary

- Custom Cellpose checkpoints, StarDist weights, and SAM checkpoints are treated as trusted local assets only.
- Do not point these manifests at untrusted model files.

## Evaluate Built-In vs Custom Models

```bash
python scripts/evaluate_models.py \
  --model_manifest examples/models/model_manifest.example.csv \
  --output_dir Outputs_model_eval \
  --no_gpu
```

The evaluation manifest supports one row per image/model run. Repeating the same image across different model rows is expected.

## Fine-Tune Cellpose

```bash
python scripts/train_cellpose.py \
  --train_manifest examples/models/train_manifest.example.csv \
  --output_dir Outputs_model_train \
  --pretrained_model cyto \
  --n_epochs 5 \
  --no_gpu
```

The training helper is intentionally thin. It validates the manifest, records the training configuration, and calls the Cellpose training API.

## Use a Winning Model in the Main CLI

```bash
python main.py \
  --input_dir input \
  --output_dir Outputs_custom_model \
  --backend cellpose \
  --cellpose_model /absolute/path/to/rgc_cellpose_model \
  --model_alias "lab-rgc-v3" \
  --write_object_table --write_provenance
```

Legacy custom Cellpose paths through `--model_type /path/to/model` still work, but they are recorded as compatibility usage in provenance.
