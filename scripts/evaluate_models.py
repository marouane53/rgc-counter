from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model_evaluation import evaluate_model_manifest, load_model_manifest, write_evaluation_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate multiple segmentation models from a manifest.")
    parser.add_argument("--model_manifest", required=True, help="CSV manifest describing model/image evaluation runs")
    parser.add_argument("--output_dir", required=True, help="Folder for evaluation outputs")
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument("--use_gpu", action="store_true", help="Enable GPU evaluation where available")
    gpu_group.add_argument("--no_gpu", action="store_true", help="Force CPU evaluation")
    parser.add_argument("--strict_schemas", action="store_true", help="Fail on missing manifest or reference files")
    args = parser.parse_args()

    manifest = load_model_manifest(args.model_manifest)
    use_gpu = bool(args.use_gpu and not args.no_gpu)
    per_run_frame, ranked_summary, metadata = evaluate_model_manifest(
        manifest,
        use_gpu=use_gpu,
        strict_schemas=args.strict_schemas,
    )
    outputs = write_evaluation_outputs(
        output_dir=args.output_dir,
        per_run_frame=per_run_frame,
        ranked_summary=ranked_summary,
        metadata=metadata,
    )
    print(f"Wrote evaluation outputs to {args.output_dir}")
    for name, path in outputs.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
