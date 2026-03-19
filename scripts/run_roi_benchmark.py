from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.roi_benchmark import (
    RoiBenchmarkConfig,
    build_roi_benchmark_report as build_roi_benchmark_report_text,
    run_roi_benchmark_config as _run_roi_benchmark_config,
)


def _json_arg(value: str | None) -> dict | None:
    if value is None or not str(value).strip():
        return None
    return json.loads(value)


def run_roi_benchmark_config(
    *,
    roi_manifest: str | Path,
    output_dir: str | Path,
    config_id: str = "roi_benchmark",
    segmentation_preset: str | None = None,
    backend: str | None = None,
    diameter: float | None = None,
    min_size: int | None = None,
    max_size: int | None = None,
    apply_clahe: bool = False,
    segmenter_config: dict | None = None,
    object_filters: dict | None = None,
    modality: str = "flatmount",
    modality_channel_index: int | None = 0,
    use_gpu: bool = False,
    save_overlays: bool = False,
    runtime_builder=None,
    runtime_runner=None,
    include_splits: list[str] | None = None,
    exclude_splits: list[str] | None = None,
) -> dict:
    config = RoiBenchmarkConfig(
        config_id=config_id,
        backend=backend,
        segmentation_preset=segmentation_preset,
        diameter=diameter,
        min_size=min_size,
        max_size=max_size,
        apply_clahe=apply_clahe,
        modality=modality,
        modality_channel_index=modality_channel_index,
        segmenter_config=segmenter_config,
        object_filters=object_filters,
        use_gpu=use_gpu,
    )
    kwargs = {
        "roi_manifest": roi_manifest,
        "output_dir": output_dir,
        "config": config,
        "save_overlays": save_overlays,
        "include_splits": include_splits,
        "exclude_splits": exclude_splits,
    }
    if runtime_builder is not None:
        kwargs["runtime_builder"] = runtime_builder
    if runtime_runner is not None:
        kwargs["runtime_runner"] = runtime_runner
    return _run_roi_benchmark_config(**kwargs)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one real ROI benchmark config.")
    parser.add_argument("--roi_manifest", "--manifest", dest="roi_manifest", required=True, help="CSV manifest for manually annotated ROIs")
    parser.add_argument("--output_dir", required=True, help="Output directory for ROI benchmark results")
    parser.add_argument("--config_id", default="roi_benchmark")
    parser.add_argument("--backend", default=None)
    parser.add_argument("--segmentation_preset", default=None)
    parser.add_argument("--diameter", default=None, type=float)
    parser.add_argument("--min_size", default=None, type=int)
    parser.add_argument("--max_size", default=None, type=int)
    parser.add_argument("--apply_clahe", action="store_true")
    parser.add_argument("--segmenter_config_json", default=None)
    parser.add_argument("--object_filters_json", default=None)
    parser.add_argument("--modality", default="flatmount")
    parser.add_argument("--modality_channel_index", default=0, type=int)
    parser.add_argument("--save_overlays", action="store_true")
    parser.add_argument("--include-splits", nargs="*", default=None)
    parser.add_argument("--exclude-splits", nargs="*", default=None)
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument("--use_gpu", action="store_true")
    gpu_group.add_argument("--no_gpu", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_roi_benchmark_config(
        roi_manifest=args.roi_manifest,
        output_dir=args.output_dir,
        config_id=args.config_id,
        backend=args.backend,
        segmentation_preset=args.segmentation_preset,
        diameter=args.diameter,
        min_size=args.min_size,
        max_size=args.max_size,
        apply_clahe=bool(args.apply_clahe),
        segmenter_config=_json_arg(args.segmenter_config_json),
        object_filters=_json_arg(args.object_filters_json),
        modality=args.modality,
        modality_channel_index=args.modality_channel_index,
        use_gpu=bool(args.use_gpu and not args.no_gpu),
        save_overlays=bool(args.save_overlays),
        include_splits=args.include_splits,
        exclude_splits=args.exclude_splits,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
