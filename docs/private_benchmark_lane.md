# Private RBPMS `.ims` Benchmark Lane

This document covers the local-only benchmark lane for the private Sana El Hajji / IL-OHT RBPMS `.ims` files.

## Purpose

- This lane answers a narrow engineering question: does `retinal-phenotyper` behave plausibly on real private RBPMS flatmount confocal data after a private `.ims -> TIFF` preprocessing bridge?
- This lane does not promote `.ims` to a supported public input format for the default CLI.
- This lane does not validate 3D stereological whole-retina counting claims.

## Local-Only Workspace

Use:

```text
test_subjects/private/sana_el_hajji_il_oht_4w_mouse1/
  raw_ims/
  metadata/
  thumbnails/
  projected/
  rois/
  truth_search/
  pipeline_probe/
  benchmark/
  paper_sanity_only/
```

Raw `.ims` files, private projections, thumbnails, crops, overlays, and benchmark outputs must stay under ignored local paths.

## Execution Order

1. Bootstrap the repo venv with `./.venv/bin/python`.
2. Inspect `.ims` metadata with `h5py`.
3. Recover any existing truth from `.ims` scene objects or local companion files.
4. Extract all-channel max projections and review them before selecting a benchmark channel.
5. Cut ROI manifests and run ROI QC.
6. Run the real-data probe on extracted TIFFs.
7. Annotate fallback truth only if same-image truth was not recovered.
8. Run the ROI benchmark and audit it.

## Claim Boundaries

- Safe claim: narrow 2D max-projected RBPMS flatmount ROI benchmark on this specific private preparation.
- Unsafe claim: equivalence to published 3D stereological counts, treatment-effect claims, or general modality-agnostic performance.

## Public Export Guard

Advisor-packet export must refuse private ROI benchmark directories unless an explicit private override flag is passed. The default export path is public-safe only.
