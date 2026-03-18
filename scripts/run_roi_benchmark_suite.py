from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.roi_benchmark import (
    default_config_manifest_for_marker,
    run_benchmark_suite as _run_benchmark_suite,
)
from src.roi_data import load_roi_manifest


def _load_config_manifest(path: str | Path | None, *, marker: str) -> pd.DataFrame:
    if path is None:
        return default_config_manifest_for_marker(marker)
    return pd.read_csv(path)


def run_benchmark_suite(
    *,
    roi_manifest: str | Path,
    output_dir: str | Path,
    config_manifest: str | Path | None = None,
    save_overlays: bool = False,
    runtime_builder=None,
    runtime_runner=None,
    use_gpu: bool = False,
) -> dict:
    manifest = load_roi_manifest(roi_manifest)
    marker = str(manifest.iloc[0]["marker"]).strip() if not manifest.empty else "RBPMS"
    config_frame = _load_config_manifest(config_manifest, marker=marker)

    kwargs = {
        "roi_manifest": roi_manifest,
        "config_manifest": config_frame,
        "output_dir": output_dir,
        "save_overlays": save_overlays,
        "use_gpu": use_gpu,
    }
    if runtime_builder is not None:
        kwargs["runtime_builder"] = runtime_builder
    if runtime_runner is not None:
        kwargs["runtime_runner"] = runtime_runner
    result = _run_benchmark_suite(**kwargs)

    if result["best_payload"] and not bool(result["best_payload"].get("pass_threshold")):
        result["exit_code"] = 2
    else:
        result["exit_code"] = 0
    result["config_manifest_path"] = str(config_manifest) if config_manifest is not None else None
    result["use_gpu"] = bool(use_gpu)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the real ROI benchmark comparison suite.")
    parser.add_argument("--roi_manifest", required=True, help="CSV manifest for manually annotated ROIs")
    parser.add_argument("--config_manifest", default=None, help="Optional config manifest CSV")
    parser.add_argument("--output_dir", required=True, help="Output directory for benchmark-suite results")
    parser.add_argument("--save_overlays", action="store_true")
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument("--use_gpu", action="store_true")
    gpu_group.add_argument("--no_gpu", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_benchmark_suite(
        roi_manifest=args.roi_manifest,
        config_manifest=args.config_manifest,
        output_dir=args.output_dir,
        save_overlays=bool(args.save_overlays),
        use_gpu=bool(args.use_gpu and not args.no_gpu),
    )
    return int(result["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
