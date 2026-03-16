# Testing

This document is for engineering validation, smoke checks, and release verification. Researcher-facing usage lives in [docs/researcher-guide.md](docs/researcher-guide.md).

## Local Test Suite

Run the tracked pytest suite:

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./.venv/bin/python -m pytest -q
```

Current coverage includes:
- schema validation
- review sidecars
- retina registration and region outputs
- rigorous spatial inference artifacts
- model registry, evaluation, and training helpers
- study statistics and provenance
- napari runtime export

## Tracked Example Smoke Runs

Canonical cohort smoke:

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

Expected artifacts:
- `Outputs_example/study_summary.csv`
- `Outputs_example/study_regions.csv`
- `Outputs_example/stats/`
- `Outputs_example/figures/`
- `Outputs_example/methods_appendix.md`
- `Outputs_example/report.html`
- per-sample `spatial/`, `objects/`, `regions/`, and `retina_frames/`

Single-image example bundle:

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

Model-evaluation smoke:

```bash
python scripts/evaluate_models.py \
  --model_manifest examples/models/model_manifest.example.csv \
  --output_dir Outputs_model_eval \
  --no_gpu
```

## Real-Data Smoke Notes

Ignored local data under `test_subjects/` is still the place for larger smoke checks:
- BBBC039 fluorescence microscopy
- retina-like tracked example panels
- local study-mode bundles under `test_subjects/runs/`

Recommended checks for real-data runs:
- object tables exist and contain model identity fields
- provenance records `model_spec`, `study_statistics`, and `spatial_analysis` when relevant
- HTML report links resolve to the saved artifacts
- rigorous spatial runs write `spatial_summary.csv`, `spatial_curves.csv`, and global plots

## Package Verification

Build the package locally:

```bash
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
```

Quick local version check:

```bash
python main.py --version
```

Wheel install smoke in a clean virtualenv:

```bash
python -m venv /tmp/retinal-phenotyper-wheel
/tmp/retinal-phenotyper-wheel/bin/pip install dist/*.whl
/tmp/retinal-phenotyper-wheel/bin/retinal-phenotyper --version
/tmp/retinal-phenotyper-wheel/bin/retinal-phenotyper --help
```

## Release Verification

TestPyPI-first flow:

1. Reserve `retinal-phenotyper` on TestPyPI and PyPI.
2. Configure GitHub Trusted Publishing for both services.
3. Trigger `.github/workflows/publish-testpypi.yml`.
4. Verify a clean install from TestPyPI.

Example TestPyPI install:

```bash
python -m venv /tmp/retinal-phenotyper-testpypi
/tmp/retinal-phenotyper-testpypi/bin/pip install -i https://test.pypi.org/simple/ retinal-phenotyper
/tmp/retinal-phenotyper-testpypi/bin/retinal-phenotyper --version
/tmp/retinal-phenotyper-testpypi/bin/retinal-phenotyper --help
```

Production PyPI release:

1. Publish GitHub release `v1.0.0`.
2. Let `.github/workflows/publish.yml` publish to PyPI.
3. Verify a clean install from PyPI.

```bash
python -m venv /tmp/retinal-phenotyper-pypi
/tmp/retinal-phenotyper-pypi/bin/pip install retinal-phenotyper
/tmp/retinal-phenotyper-pypi/bin/retinal-phenotyper --version
/tmp/retinal-phenotyper-pypi/bin/retinal-phenotyper --help
```
