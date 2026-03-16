# Stats Designs

RGC Counter now has two study-mode statistics paths: simple tests and mixed-effects models.

## When Simple Tests Are Used

Simple tests remain the compatibility path.

They are used when:
- you request `--stats_mode simple`
- or `--stats_mode auto` does not detect a nested/repeated design that qualifies for mixed-effects

Current simple outputs live in:
- `stats/study_stats.csv`
- `stats/region_stats.csv`

## When Mixed-Effects Are Used

Mixed-effects are used when `--stats_mode auto` detects repeated structure, such as:
- repeated eyes per animal
- repeated samples per animal
- repeated timepoints
- repeated regional observations nested within samples

Current mixed-effects outputs live in:
- `stats_mixed/sample_mixed_effects.csv`
- `stats_mixed/region_mixed_effects.csv`

## What The Design Audit Means

The design audit is the quick summary of what the study structure actually looks like.

It records:
- animals per condition
- eyes per animal
- samples per eye
- regions per sample
- repeated timepoints per animal-eye
- missingness in the key analysis columns

Inspect:
- `stats/design_audit.csv`
- `stats/design_audit.md`
- `stats/statistics_decision.json`

## What To Cite In Methods Or Supplements

The main study statistics record is:
- `stats/statistics_decision.json`

Use it with:
- `methods_appendix.md`
- `stats/study_stats.csv`
- `stats/region_stats.csv`
- `stats_mixed/*.csv` when mixed-effects were selected

If the design is nested, cite the mixed-effects path and keep the design audit with the submission bundle.
