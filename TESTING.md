# Testing RGC Counter

This repo now uses a two-track testing setup:

- tracked synthetic tests in `tests/` for fast, deterministic validation
- ignored local image assets in `test_subjects/` for end-to-end smoke tests

## 1. Install dev dependencies

```bash
pip install -e .[dev]
```

Fallback:

```bash
pip install -r requirements-dev.txt
```

## 2. Prepare local test subjects

Generate synthetic images:

```bash
python scripts/prepare_test_subjects.py synthetic
```

Optional public dataset download:

```bash
python scripts/prepare_test_subjects.py bbbc039
python scripts/prepare_test_subjects.py pmc_rgc_figures
python scripts/prepare_test_subjects.py oir_flatmount --no-extract
python scripts/prepare_test_subjects.py oir_sample
```

The `test_subjects/` folder is ignored by Git on purpose. It is your local sandbox for real image assets and generated outputs.

## 3. Run the tracked unit tests

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest
```

Current tracked coverage focuses on:

- count, area, and density calculations
- mask post-processing
- schema validation and strict/non-strict contracts
- phenotype rule filtering
- spatial metric sanity checks
- object-table emission
- uncertainty/QC map persistence
- edit-log replay
- automatic landmark detection
- tiled inference
- calibration ranking
- provenance serialization
- staged pipeline integration

## 4. Run pipeline smoke tests on local images

Simple counting sample:

```bash
python main.py \
  --input_dir test_subjects/synthetic/simple_counting/images \
  --output_dir test_subjects/runs/simple_counting \
  --focus_none \
  --save_debug
```

Multichannel phenotype sample:

```bash
python main.py \
  --input_dir test_subjects/synthetic/multichannel/images \
  --output_dir test_subjects/runs/multichannel \
  --focus_none \
  --phenotype_config test_subjects/synthetic/multichannel/phenotype_rules.synthetic.yaml \
  --save_debug
```

Wholemount spatial sample:

```bash
python main.py \
  --input_dir test_subjects/synthetic/wholemount/images \
  --output_dir test_subjects/runs/wholemount \
  --focus_none \
  --spatial_stats \
  --save_debug \
  --write_html_report
```

Phase 0 outputs:

```bash
python main.py \
  --input_dir test_subjects/synthetic/simple_counting/images \
  --output_dir test_subjects/runs/phase0 \
  --focus_none \
  --write_object_table \
  --write_provenance \
  --no_gpu
```

Phase 1 retina registration:

```bash
python main.py \
  --input_dir test_subjects/synthetic/wholemount/images \
  --output_dir test_subjects/runs/phase1 \
  --focus_none \
  --register_retina \
  --onh_mode sidecar \
  --retina_frame_path test_subjects/synthetic/wholemount/retina_frame.reference.json \
  --region_schema mouse_flatmount_v1 \
  --write_object_table \
  --write_provenance \
  --write_html_report \
  --no_gpu
```

## 5. What each dataset is for

### Synthetic test subjects

- `simple_counting/images`: validates counting, density, and debug overlays
- `multichannel/images`: validates phenotype rules and multichannel loading
- `wholemount/images`: validates ONH-style geometry and future registration work
- `cohort/cohort_manifest.csv`: gives us a ready-made manifest for future cohort and stats features

### BBBC039

Use this for real-image segmentation regression testing. It is not retina-specific, but it is useful for:

- loader validation
- object-table smoke tests
- mask/count stability
- regression snapshots when segmentation code changes

Source:
- [BBBC039 official page](https://bbbc.broadinstitute.org/BBBC039/)

### PMC RGC figures

Use this for fast retina-specific qualitative smoke tests. It is useful for:

- checking that the loader works on non-TIFF microscopy panels
- visually confirming that overlays follow obvious ganglion-cell somata
- validating report/debug artifact generation on retina-like imagery

Limits:

- this is not a benchmark or a clean raw acquisition set
- these figure panels should only be used for qualitative inspection

### OIR flat-mount dataset

This is the real retina flat-mount dataset we use for retina-native smoke testing.

- Archive source: [Figshare DOI 10.6084/m9.figshare.23690973.v3](https://doi.org/10.6084/m9.figshare.23690973.v3)
- The zip is large, so the recommended flow is to download the archive without full extraction, then extract a small `512` sample set.
- `python scripts/prepare_test_subjects.py oir_sample` extracts a limited local sample into `test_subjects/public/oir_flatmount/sample_512`.

### Real microscopy versus everyday photos

- Real microscopy images: yes, these are useful and already supported.
- Everyday photos from phones or cameras: no, they are not a meaningful test target for this pipeline.
- We already have a real microscopy smoke target in `test_subjects/public/bbbc039_smoke/images`.
- For a more retina-native public dataset later, the OIR flat-mount dataset from the 2024 paper and Figshare DOI is a good next candidate:
  [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11639695/)
  [dataset DOI](https://doi.org/10.6084/m9.figshare.23690973)

## 6. Pass/fail criteria by phase

### Phase 0

- `pytest` passes
- synthetic images load through the current CLI
- results rows and debug overlays are generated

### Phase 1

- wholemount synthetic data can emit stable ONH-relative coordinates
- region summaries are reproducible from saved tables
- registered retina-frame JSON and registered density-map artifacts are written

### Phase 2

- multichannel synthetic sample classifies expected RGC versus microglia-overlap objects
- phenotype tables preserve per-object measurements
- real microscopy grayscale data can emit geometry and channel metrics even without phenotype labels

### Phase 3

- cohort manifest drives a study-level run
- paired-eye and region-aware summary tables are reproducible
- validation artifacts are produced from the synthetic label references

Phase 3 study-mode smoke test:

```bash
python main.py \
  --manifest test_subjects/synthetic/cohort/cohort_manifest.csv \
  --study_output_dir test_subjects/runs/phase3 \
  --focus_none \
  --register_retina \
  --write_object_table \
  --write_provenance \
  --write_html_report \
  --no_gpu
```

### Phase 4

- methods appendix is generated from the actual runtime config
- report HTML includes stats and validation sections
- study figure bundle is produced from saved tables only

### Phase 5

- the napari dock imports without breaking the test suite
- the shared runtime can execute a single image and export the same artifact types as the CLI
- landmark capture from a napari points layer maps `(y, x)` viewer coordinates into `(x, y)` retina inputs correctly

Phase 5 napari launch:

```bash
python scripts/run_napari_ui.py
```

Recommended manual smoke check:

1. Open `test_subjects/public/pmc_rgc_figures/ganglion_cell_panel.jpg` in napari.
2. Run the dock with `cellpose`, `focus_mode=none`, and export enabled.
3. Confirm that labels and centroid points appear in the viewer.
4. Export and verify that `results.csv`, `report.html`, `provenance.json`, and the debug overlay are written.

### Phase 6

- atlas comparison writes observed-vs-expected region tables from a registered retina
- at least one volumetric modality adapter can reduce data into the shared analysis path
- study mode can emit opt-in longitudinal tracking outputs

Phase 6 study smoke test:

```bash
python main.py \
  --manifest test_subjects/synthetic/cohort/cohort_manifest.csv \
  --study_output_dir test_subjects/runs/phase6 \
  --focus_none \
  --register_retina \
  --atlas_reference atlas_reference.example.csv \
  --track_longitudinal \
  --write_object_table \
  --write_provenance \
  --write_html_report \
  --no_gpu
```

Phase 6 modality smoke test:

1. Create or load a small 3D stack.
2. Run `python main.py --input_dir ... --output_dir ... --modality oct --modality_projection max --focus_none`.
3. Confirm that the run completes and produces the normal single-image output bundle.

## 7. Trust-layer smoke path

Tracked example trust-layer run:

```bash
python main.py \
  --input_dir examples/smoke_data \
  --output_dir test_subjects/runs/example_trust_maps \
  --focus_qc \
  --tta \
  --register_retina \
  --onh_mode auto_combined \
  --dorsal_xy 48 14 \
  --strict_schemas \
  --write_object_table \
  --write_provenance \
  --write_html_report \
  --write_uncertainty_maps \
  --write_qc_maps \
  --no_gpu
```

Tracked example study bundle:

```bash
python main.py \
  --manifest examples/manifests/example_study_manifest.csv \
  --manual_annotations examples/manual_annotations/example_manual_annotations.csv \
  --study_output_dir test_subjects/runs/example_trust_study \
  --focus_none \
  --register_retina \
  --strict_schemas \
  --write_object_table \
  --write_provenance \
  --write_html_report \
  --no_gpu
```

Calibration smoke run:

```bash
python main.py \
  --manifest examples/manifests/example_study_manifest.csv \
  --manual_annotations examples/manual_annotations/example_manual_annotations.csv \
  --study_output_dir test_subjects/runs/example_calibration \
  --focus_none \
  --register_retina \
  --calibration_grid examples/calibration/calibration_grid.example.yaml \
  --no_gpu
```

## 8. Recommended working loop

1. Add or change one module.
2. Run `env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest`.
3. Run the relevant synthetic smoke test in `test_subjects/synthetic`.
4. Only then validate on the downloaded public dataset.
