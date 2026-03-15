# AGENTS.md

## Planning Workflow

- Keep active phase plans in `planning/`.
- When a plan is completed, move it into `planning/archive/`.
- Completed implementation notes that belong to a finished plan should live in `planning/archive/` as well.

## Delivery Notes

- Preserve backward compatibility for the default CLI path unless a new flag or mode is explicitly enabled.
- Prefer additive outputs such as tables and provenance files over silent changes to existing outputs.
