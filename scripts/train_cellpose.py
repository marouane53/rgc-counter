from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model_training import load_train_manifest, train_cellpose_from_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a Cellpose model from a training manifest.")
    parser.add_argument("--train_manifest", required=True, help="CSV manifest describing train/val image and label paths")
    parser.add_argument("--output_dir", required=True, help="Folder for training outputs")
    parser.add_argument("--pretrained_model", default="cyto", help="Built-in Cellpose model used as the fine-tuning base")
    parser.add_argument("--diameter", type=float, default=None, help="Optional recorded diameter metadata")
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument("--use_gpu", action="store_true", help="Enable GPU training where available")
    gpu_group.add_argument("--no_gpu", action="store_true", help="Force CPU training")
    parser.add_argument("--n_epochs", type=int, default=None, help="Optional epoch override for Cellpose training")
    args = parser.parse_args()

    manifest = load_train_manifest(args.train_manifest)
    use_gpu = bool(args.use_gpu and not args.no_gpu)
    result, artifacts = train_cellpose_from_manifest(
        manifest,
        output_dir=args.output_dir,
        pretrained_model=args.pretrained_model,
        diameter=args.diameter,
        use_gpu=use_gpu,
        n_epochs=args.n_epochs,
    )
    print(f"Wrote training outputs to {args.output_dir}")
    for name, path in artifacts.items():
        print(f"- {name}: {path}")
    if result.get("checkpoint_path"):
        print(f"- checkpoint_path: {result['checkpoint_path']}")


if __name__ == "__main__":
    main()
