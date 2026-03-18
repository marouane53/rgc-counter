from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.roi_data import load_roi_manifest, qc_roi_manifest


def build_qc_markdown(qc: pd.DataFrame) -> str:
    lines = [
        "# ROI Source QC",
        "",
        f"- Rows: `{len(qc)}`",
        f"- Missing images: `{int((~qc['image_exists']).sum()) if 'image_exists' in qc.columns else 0}`",
        f"- Out-of-bounds ROIs: `{int((~qc['bounds_ok']).sum()) if 'bounds_ok' in qc.columns else 0}`",
        f"- Duplicate source files: `{int(qc['duplicate_image'].sum()) if 'duplicate_image' in qc.columns else 0}`",
        f"- Duplicate crops: `{int(qc['duplicate_crop'].sum()) if 'duplicate_crop' in qc.columns else 0}`",
        "",
    ]
    bad = qc[
        (~qc["image_exists"])
        | (~qc["bounds_ok"])
        | (~qc["crop_nonempty"])
        | (qc["duplicate_image"])
        | (qc["duplicate_crop"])
        | (~qc["marker_consistent"])
        | (~qc["modality_consistent"])
        | (qc["error"].fillna("").astype(str) != "")
    ].copy()
    if bad.empty:
        lines.extend(["## Result", "", "QC passed with no blocking issues.", ""])
    else:
        lines.extend(["## Blocking issues", "", bad.to_markdown(index=False), ""])
    return "\n".join(lines)


def qc_has_blockers(qc: pd.DataFrame) -> bool:
    if qc.empty:
        return True
    return bool(
        (~qc["image_exists"]).any()
        or (~qc["bounds_ok"]).any()
        or (~qc["crop_nonempty"]).any()
        or qc["duplicate_image"].any()
        or qc["duplicate_crop"].any()
        or (~qc["marker_consistent"]).any()
        or (~qc["modality_consistent"]).any()
        or (qc["error"].fillna("").astype(str) != "").any()
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="QC a real ROI benchmark manifest.")
    parser.add_argument("--roi_manifest", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        manifest = load_roi_manifest(args.roi_manifest)
        qc = qc_roi_manifest(manifest, manifest_path=args.roi_manifest)
    except Exception as exc:
        qc = pd.DataFrame(
            [
                {
                    "roi_id": "",
                    "image_path": str(args.roi_manifest),
                    "marker": "",
                    "modality": "",
                    "split": "",
                    "image_exists": False,
                    "manual_points_exists": False,
                    "image_sha256": None,
                    "crop_sha256": None,
                    "image_shape": None,
                    "bounds_ok": False,
                    "crop_nonempty": False,
                    "duplicate_image": False,
                    "duplicate_crop": False,
                    "marker_consistent": False,
                    "modality_consistent": False,
                    "error": str(exc),
                }
            ]
        )

    csv_path = args.output_dir / "source_qc.csv"
    md_path = args.output_dir / "source_qc.md"
    qc.to_csv(csv_path, index=False)
    md_path.write_text(build_qc_markdown(qc) + "\n", encoding="utf-8")

    return 1 if qc_has_blockers(qc) else 0


if __name__ == "__main__":
    raise SystemExit(main())
