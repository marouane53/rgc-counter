from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_ome import load_any_image
from src.roi_benchmark import default_config_manifest_for_marker
from src.run_service import RuntimeOptions, build_runtime, export_context, run_one_image


def _log_progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.to_dict("records"):
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def runtime_options_from_row(row: pd.Series, *, use_gpu: bool, tiling: bool, tile_size: int, tile_overlap: int) -> RuntimeOptions:
    return RuntimeOptions(
        backend=None if pd.isna(row.get("backend")) or not str(row.get("backend")).strip() else str(row.get("backend")),
        segmentation_preset=None if pd.isna(row.get("segmentation_preset")) or not str(row.get("segmentation_preset")).strip() else str(row.get("segmentation_preset")),
        diameter=float(row["diameter"]) if pd.notna(row.get("diameter")) else None,
        min_size=int(row["min_size"]) if pd.notna(row.get("min_size")) else None,
        max_size=int(row["max_size"]) if pd.notna(row.get("max_size")) else None,
        apply_clahe=bool(row.get("apply_clahe", False)) if not isinstance(row.get("apply_clahe"), str) else str(row.get("apply_clahe")).strip().lower() in {"1", "true", "yes", "y"},
        modality=str(row.get("modality") or "flatmount"),
        modality_channel_index=int(row["modality_channel_index"]) if pd.notna(row.get("modality_channel_index")) else 0,
        segmenter_config=json.loads(str(row["segmenter_config_json"])) if pd.notna(row.get("segmenter_config_json")) and str(row.get("segmenter_config_json")).strip() else None,
        object_filters=json.loads(str(row["object_filters_json"])) if pd.notna(row.get("object_filters_json")) and str(row.get("object_filters_json")).strip() else None,
        focus_mode="none",
        save_debug=True,
        write_html_report=True,
        write_object_table=True,
        write_provenance=True,
        use_gpu=bool(use_gpu),
        tiling=bool(tiling),
        tile_size=int(tile_size),
        tile_overlap=int(tile_overlap),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the private real-data RBPMS probe across projected TIFFs.")
    parser.add_argument("--projected-dir", required=True, type=Path)
    parser.add_argument("--config-manifest", default=None, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tile-size", default=1024, type=int)
    parser.add_argument("--tile-overlap", default=128, type=int)
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument("--use-gpu", action="store_true")
    gpu_group.add_argument("--no-gpu", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    started_at = time.perf_counter()
    args = parse_args(argv)
    projected_paths = sorted(args.projected_dir.rglob("*_max.tif"))
    if not projected_paths:
        raise SystemExit(f"No projected TIFFs found under {args.projected_dir}")
    config_frame = pd.read_csv(args.config_manifest) if args.config_manifest is not None else default_config_manifest_for_marker("RBPMS")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_frame.to_csv(output_dir / "resolved_config_manifest.csv", index=False)

    summary_rows: list[dict[str, object]] = []
    total_runs = int(len(projected_paths) * len(config_frame))
    completed_runs = 0
    _log_progress(
        f"[run_real_rbpms_probe] Starting probe for {len(projected_paths)} projected image(s) across {len(config_frame)} config(s) = {total_runs} run(s)"
    )
    for image_index, projected_path in enumerate(projected_paths, start=1):
        image_started_at = time.perf_counter()
        image, _ = load_any_image(str(projected_path))
        tiling = bool(image.ndim >= 2 and (int(image.shape[0]) > int(args.tile_size) or int(image.shape[1]) > int(args.tile_size)))
        _log_progress(
            f"[run_real_rbpms_probe] [{image_index}/{len(projected_paths)}] {projected_path.name}: tiling={'on' if tiling else 'off'}"
        )
        for config_index, (_, row) in enumerate(config_frame.iterrows(), start=1):
            config_id = str(row["config_id"])
            run_dir = output_dir / config_id / projected_path.stem
            run_started_at = time.perf_counter()
            _log_progress(
                f"[run_real_rbpms_probe] [{image_index}/{len(projected_paths)}] {projected_path.name} config {config_id} ({config_index}/{len(config_frame)})"
            )
            runtime = build_runtime(
                runtime_options_from_row(
                    row,
                    use_gpu=bool(args.use_gpu and not args.no_gpu),
                    tiling=tiling,
                    tile_size=int(args.tile_size),
                    tile_overlap=int(args.tile_overlap),
                )
            )
            ctx = run_one_image(runtime, image_path=projected_path, modality_override="flatmount")
            artifacts = export_context(runtime, ctx, run_dir)
            summary_rows.append(
                {
                    "config_id": config_id,
                    "projected_tiff_path": str(projected_path),
                    "run_dir": str(run_dir),
                    "cell_count": ctx.summary_row.get("cell_count"),
                    "warning_count": len(ctx.warnings),
                    "warnings": "; ".join(ctx.warnings),
                    "backend": runtime.backend,
                    "model_label": runtime.model_spec.model_label,
                    "tiling": tiling,
                    "debug_overlay": str(artifacts.get("debug_overlay", "")),
                    "object_table": str(artifacts.get("object_table", "")),
                    "provenance": str(artifacts.get("provenance", "")),
                    "report": str(artifacts.get("html_report", "")),
                }
            )
            completed_runs += 1
            _log_progress(
                f"[run_real_rbpms_probe] completed {completed_runs}/{total_runs}: {projected_path.name} config {config_id} in {time.perf_counter() - run_started_at:.1f}s"
            )
        _log_progress(
            f"[run_real_rbpms_probe] [{image_index}/{len(projected_paths)}] {projected_path.name}: finished in {time.perf_counter() - image_started_at:.1f}s"
        )

    summary = pd.DataFrame(summary_rows).sort_values(["projected_tiff_path", "config_id"]).reset_index(drop=True)
    summary.to_csv(output_dir / "probe_summary.csv", index=False)
    (output_dir / "review_index.md").write_text(
        "# Real RBPMS Probe Review Index\n\n"
        "- Review debug overlays and object tables before treating this lane as benchmark-ready.\n"
        "- Stop the workflow if every configuration is obviously implausible on multiple tiles.\n\n"
        + _markdown_table(summary)
        + "\n",
        encoding="utf-8",
    )
    _log_progress(
        f"[run_real_rbpms_probe] Finished {total_runs} run(s) in {time.perf_counter() - started_at:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
