from __future__ import annotations

import argparse
import csv
import json
import fnmatch
import sys
import textwrap
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import tifffile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "test_subjects"

PUBLIC_DATASETS = {
    "bbbc039": {
        "description": "Broad Bioimage Benchmark Collection BBBC039 fluorescence nuclei set.",
        "files": {
            "images.zip": "https://data.broadinstitute.org/bbbc/BBBC039/images.zip",
            "masks.zip": "https://data.broadinstitute.org/bbbc/BBBC039/masks.zip",
            "metadata.zip": "https://data.broadinstitute.org/bbbc/BBBC039/metadata.zip",
        },
    },
    "pmc_rgc_figures": {
        "description": "Small open-access retinal ganglion cell microscopy figures from PMC for qualitative smoke tests.",
        "files": {
            "layer_panel.jpg": "https://cdn.ncbi.nlm.nih.gov/pmc/blobs/78fc/3959221/8423841f3d81/nihms548668f2.jpg",
            "ganglion_cell_panel.jpg": "https://cdn.ncbi.nlm.nih.gov/pmc/blobs/78fc/3959221/9c428f969501/nihms548668f3.jpg",
        },
    },
    "oir_flatmount": {
        "description": "OIR flat-mount retinal image archive from Figshare (large download).",
        "files": {
            "Healthsheet.docx": "https://ndownloader.figshare.com/files/41681784",
            "OIR_Flat_Mount_Dataset.zip": "https://ndownloader.figshare.com/files/41691582",
            "Final Dataset.xls": "https://ndownloader.figshare.com/files/41696418",
        },
    },
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def paint_disk(canvas: np.ndarray, center_yx: tuple[int, int], radius: int, value: int) -> np.ndarray:
    yy, xx = np.ogrid[: canvas.shape[0], : canvas.shape[1]]
    cy, cx = center_yx
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
    canvas[mask] = np.maximum(canvas[mask], value)
    return mask


def sample_centers(
    rng: np.random.Generator,
    count: int,
    shape: tuple[int, int],
    radius: int,
    min_gap: int,
    tissue_mask: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    height, width = shape
    centers: list[tuple[int, int]] = []
    attempts = 0
    while len(centers) < count and attempts < count * 200:
        attempts += 1
        cy = int(rng.integers(radius + 4, height - radius - 4))
        cx = int(rng.integers(radius + 4, width - radius - 4))
        if tissue_mask is not None and not bool(tissue_mask[cy, cx]):
            continue
        if any((cy - py) ** 2 + (cx - px) ** 2 < min_gap ** 2 for py, px in centers):
            continue
        centers.append((cy, cx))
    if len(centers) != count:
        raise RuntimeError(f"Could not place {count} non-overlapping objects in shape={shape}.")
    return centers


def write_expected_counts_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_simple_counting_sample(base_dir: Path, rng: np.random.Generator) -> dict[str, object]:
    sample_dir = ensure_dir(base_dir / "simple_counting")
    image_dir = ensure_dir(sample_dir / "images")
    label_dir = ensure_dir(sample_dir / "labels")
    image = rng.normal(loc=300, scale=10, size=(256, 256)).astype(np.float32)
    labels = np.zeros((256, 256), dtype=np.uint16)
    centers = sample_centers(rng, count=12, shape=image.shape, radius=8, min_gap=24)

    for idx, center in enumerate(centers, start=1):
        mask = paint_disk(labels, center, radius=8, value=idx)
        image[mask] += 500

    image = np.clip(image, 0, 65535).astype(np.uint16)
    image_path = image_dir / "simple_counting_12cells.tif"
    label_path = label_dir / "simple_counting_12cells_labels.tif"
    tifffile.imwrite(image_path, image)
    tifffile.imwrite(label_path, labels)

    return {
        "sample_id": "simple_counting_12cells",
        "image_path": str(image_path),
        "label_path": str(label_path),
        "expected_total_objects": 12,
        "expected_rgc_objects": 12,
        "notes": "Single-channel smoke-test image for counting and density.",
    }


def generate_multichannel_sample(base_dir: Path, rng: np.random.Generator) -> dict[str, object]:
    sample_dir = ensure_dir(base_dir / "multichannel")
    image_dir = ensure_dir(sample_dir / "images")
    label_dir = ensure_dir(sample_dir / "labels")
    shape = (256, 256)
    image = np.zeros((shape[0], shape[1], 3), dtype=np.uint16)
    labels = np.zeros(shape, dtype=np.uint16)

    rgc_centers = sample_centers(rng, count=8, shape=shape, radius=9, min_gap=28)
    confound_centers = sample_centers(rng, count=3, shape=shape, radius=9, min_gap=28)

    for idx, center in enumerate(rgc_centers, start=1):
        mask = paint_disk(labels, center, radius=9, value=idx)
        image[..., 0][mask] = 230
        image[..., 2][mask] = 40

    start_idx = len(rgc_centers) + 1
    for offset, center in enumerate(confound_centers):
        oid = start_idx + offset
        mask = paint_disk(labels, center, radius=9, value=oid)
        image[..., 0][mask] = 230
        image[..., 1][mask] = 255
        image[..., 2][mask] = 60

    image[..., 0] += rng.integers(5, 18, size=shape, dtype=np.uint16)
    image[..., 1] += rng.integers(5, 20, size=shape, dtype=np.uint16)
    image[..., 2] += rng.integers(2, 12, size=shape, dtype=np.uint16)

    image_path = image_dir / "multichannel_rgc_markers.ome.tif"
    label_path = label_dir / "multichannel_rgc_markers_labels.tif"
    tifffile.imwrite(
        image_path,
        image,
        ome=True,
        metadata={"axes": "YXC", "Channel": {"Name": ["RBPMS", "IBA1", "GFAP"]}},
    )
    tifffile.imwrite(label_path, labels)

    rules_path = sample_dir / "phenotype_rules.synthetic.yaml"
    write_text(
        rules_path,
        textwrap.dedent(
            """\
            channels:
              rgc_channel: 0
              microglia_channel: 1

            thresholds:
              rgc_min_intensity: 180
              microglia_min_intensity: 180

            logic:
              require_rgc_positive: true
              exclude_microglia_overlap: true

            morphology_priors:
              min_area_px: 100
              max_area_px: 1000
              min_circularity: 0.2
            """
        ),
    )

    rules_v2_path = sample_dir / "phenotype_rules.v2.synthetic.yaml"
    write_text(
        rules_v2_path,
        textwrap.dedent(
            """\
            schema_version: 2

            channels:
              RBPMS: 0
              MICROGLIA: 1
              GFAP: 2

            masks:
              rgc_positive:
                channel: RBPMS
                min_intensity: 180
              microglia:
                channel: MICROGLIA
                min_intensity: 180

            classes:
              rgc:
                priority: 100
                include:
                  - {feature: relation.overlap_fraction, target: rgc_positive, op: gt, value: 0.0}
                  - {feature: geometry.area_px, op: ge, value: 100}
                  - {feature: geometry.area_px, op: le, value: 1000}
                exclude:
                  - {feature: relation.overlap_fraction, target: microglia, op: gt, value: 0.0}

              microglia:
                priority: 90
                include:
                  - {feature: relation.overlap_fraction, target: microglia, op: gt, value: 0.0}
            """
        ),
    )

    return {
        "sample_id": "multichannel_rgc_markers",
        "image_path": str(image_path),
        "label_path": str(label_path),
        "expected_total_objects": len(rgc_centers) + len(confound_centers),
        "expected_rgc_objects": len(rgc_centers),
        "notes": "Multichannel sample for phenotype filtering; three objects should be excluded as microglia overlaps.",
    }


def build_tissue_mask(shape: tuple[int, int], onh_center: tuple[int, int], outer_radius: int, onh_radius: int) -> np.ndarray:
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    cy, cx = onh_center
    tissue = (yy - cy) ** 2 + (xx - cx) ** 2 <= outer_radius ** 2
    onh = (yy - cy) ** 2 + (xx - cx) ** 2 <= onh_radius ** 2
    return tissue & ~onh


def generate_wholemount_sample(base_dir: Path, rng: np.random.Generator) -> dict[str, object]:
    sample_dir = ensure_dir(base_dir / "wholemount")
    image_dir = ensure_dir(sample_dir / "images")
    label_dir = ensure_dir(sample_dir / "labels")
    shape = (512, 512)
    onh_center = (256, 256)
    tissue_mask = build_tissue_mask(shape, onh_center, outer_radius=220, onh_radius=28)
    image = np.zeros((shape[0], shape[1], 3), dtype=np.uint16)
    labels = np.zeros(shape, dtype=np.uint16)

    image[..., 2][tissue_mask] = 25
    image[..., 2] += rng.integers(0, 8, size=shape, dtype=np.uint16)

    vessel_angles = np.linspace(0, np.pi, num=6, endpoint=False)
    yy, xx = np.indices(shape)
    for angle in vessel_angles:
        distance = np.abs((xx - onh_center[1]) * np.sin(angle) - (yy - onh_center[0]) * np.cos(angle))
        vessel_mask = (distance < 2.0) & tissue_mask
        image[..., 1][vessel_mask] = 120

    centers = sample_centers(rng, count=42, shape=shape, radius=8, min_gap=20, tissue_mask=tissue_mask)
    for idx, center in enumerate(centers, start=1):
        mask = paint_disk(labels, center, radius=8, value=idx)
        image[..., 0][mask] = 240

    image[..., 0] += rng.integers(4, 20, size=shape, dtype=np.uint16)
    image_path = image_dir / "wholemount_registered_reference.ome.tif"
    label_path = label_dir / "wholemount_registered_reference_labels.tif"
    tifffile.imwrite(
        image_path,
        image,
        ome=True,
        metadata={"axes": "YXC", "Channel": {"Name": ["RBPMS", "VESSEL", "TISSUE"]}},
    )
    tifffile.imwrite(label_path, labels)

    frame = {
        "path": str(image_path),
        "onh_x_px": onh_center[1],
        "onh_y_px": onh_center[0],
        "dorsal_x_px": onh_center[1],
        "dorsal_y_px": 40,
        "um_per_px": 1.0,
        "expected_total_objects": 42,
    }
    frame_path = sample_dir / "retina_frame.reference.json"
    frame_path.write_text(json.dumps(frame, indent=2), encoding="utf-8")

    return {
        "sample_id": "wholemount_registered_reference",
        "image_path": str(image_path),
        "label_path": str(label_path),
        "expected_total_objects": 42,
        "expected_rgc_objects": 42,
        "notes": "Wholemount-like image with ONH hole and vessel-like structure for retina registration work.",
    }


def generate_cohort_manifest(base_dir: Path, rng: np.random.Generator) -> Path:
    cohort_dir = ensure_dir(base_dir / "cohort")
    image_dir = ensure_dir(cohort_dir / "images")
    label_dir = ensure_dir(cohort_dir / "labels")
    manifest_path = cohort_dir / "cohort_manifest.csv"
    rows: list[dict[str, object]] = []
    sample_specs = [
        ("M01_OD", "M01", "OD", "treated", 38),
        ("M01_OS", "M01", "OS", "control", 50),
        ("M02_OD", "M02", "OD", "treated", 35),
        ("M02_OS", "M02", "OS", "control", 47),
    ]

    for sample_id, animal_id, eye, condition, cell_count in sample_specs:
        shape = (256, 256)
        onh_center = (128, 128)
        tissue_mask = build_tissue_mask(shape, onh_center, outer_radius=105, onh_radius=14)
        image = np.zeros((shape[0], shape[1], 3), dtype=np.uint16)
        labels = np.zeros(shape, dtype=np.uint16)
        centers = sample_centers(rng, count=cell_count, shape=shape, radius=5, min_gap=14, tissue_mask=tissue_mask)
        for idx, center in enumerate(centers, start=1):
            mask = paint_disk(labels, center, radius=5, value=idx)
            image[..., 0][mask] = 230
        image[..., 0] += rng.integers(4, 16, size=shape, dtype=np.uint16)
        image[..., 1][tissue_mask] = 18

        image_path = image_dir / f"{sample_id}.ome.tif"
        label_path = label_dir / f"{sample_id}_labels.tif"
        tifffile.imwrite(
            image_path,
            image,
            ome=True,
            metadata={"axes": "YXC", "Channel": {"Name": ["RBPMS", "TISSUE", "EMPTY"]}},
        )
        tifffile.imwrite(label_path, labels)
        rows.append(
            {
                "sample_id": sample_id,
                "animal_id": animal_id,
                "eye": eye,
                "condition": condition,
                "genotype": "WT",
                "timepoint_dpi": 14,
                "modality": "flatmount",
                "stain_panel": "RBPMS",
                "path": str(image_path),
                "label_path": str(label_path),
                "onh_x_px": onh_center[1],
                "onh_y_px": onh_center[0],
                "dorsal_x_px": onh_center[1],
                "dorsal_y_px": 12,
                "expected_total_objects": cell_count,
            }
        )

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def generate_synthetic_subjects(output_dir: Path) -> None:
    rng = np.random.default_rng(42)
    synthetic_dir = ensure_dir(output_dir / "synthetic")
    rows = [
        generate_simple_counting_sample(synthetic_dir, rng),
        generate_multichannel_sample(synthetic_dir, rng),
        generate_wholemount_sample(synthetic_dir, rng),
    ]
    manifest_path = generate_cohort_manifest(synthetic_dir, rng)
    write_expected_counts_csv(synthetic_dir / "expected_counts.csv", rows)
    write_text(
        synthetic_dir / "README.txt",
        textwrap.dedent(
            f"""\
            Synthetic test subjects for local RGC Counter development.

            Contents:
            - simple_counting/: single-channel counting smoke test
            - multichannel/: phenotype-filtering smoke test
            - wholemount/: retina-registration smoke test
            - cohort/: four-sample study manifest for cohort/statistics work

            Cohort manifest:
            {manifest_path}
            """
        ),
    )
    print(f"[OK] Generated synthetic test subjects in {synthetic_dir}")


def download_file(url: str, destination: Path) -> None:
    if destination.exists():
        print(f"[SKIP] {destination.name} already exists")
        return
    print(f"[GET] {url}")
    with urllib.request.urlopen(url) as response, destination.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    print(f"[OK] Downloaded {destination}")


def extract_zip(path: Path, destination: Path) -> None:
    marker = destination / f".extracted_{path.stem}"
    if marker.exists():
        print(f"[SKIP] {path.name} already extracted")
        return
    with zipfile.ZipFile(path, "r") as archive:
        archive.extractall(destination)
    marker.write_text("ok\n", encoding="utf-8")
    print(f"[OK] Extracted {path.name} to {destination}")


def extract_zip_members(path: Path, destination: Path, patterns: list[str], limit: int = 12) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    extracted = 0
    with zipfile.ZipFile(path, "r") as archive:
        names = archive.namelist()
        for pattern in patterns:
            for name in names:
                if extracted >= limit:
                    break
                if name.endswith("/") or not fnmatch.fnmatch(name, pattern):
                    continue
                target = destination / Path(name).name
                if target.exists():
                    extracted += 1
                    continue
                with archive.open(name) as src, target.open("wb") as dst:
                    dst.write(src.read())
                extracted += 1
            if extracted >= limit:
                break
    print(f"[OK] Extracted {extracted} member(s) from {path.name} into {destination}")


def download_public_dataset(dataset_name: str, output_dir: Path, extract: bool) -> None:
    if dataset_name not in PUBLIC_DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    dataset = PUBLIC_DATASETS[dataset_name]
    dataset_dir = ensure_dir(output_dir / "public" / dataset_name)
    for filename, url in dataset["files"].items():
        destination = dataset_dir / filename
        download_file(url, destination)
        if extract and destination.suffix == ".zip":
            extract_zip(destination, dataset_dir)
    write_text(
        dataset_dir / "SOURCE.txt",
        f"{dataset_name}: {dataset['description']}\n",
    )


def prepare_oir_sample(output_dir: Path) -> None:
    dataset_dir = ensure_dir(output_dir / "public" / "oir_flatmount")
    archive = dataset_dir / "OIR_Flat_Mount_Dataset.zip"
    if not archive.exists():
        raise FileNotFoundError(f"Expected {archive} to exist. Download the dataset first.")
    sample_dir = ensure_dir(dataset_dir / "sample_512")
    patterns = [
        "*512/*.png",
        "*512/*.jpg",
        "*512/*.jpeg",
        "*512/*.tif",
        "*512/*.tiff",
    ]
    extract_zip_members(archive, sample_dir, patterns=patterns, limit=12)


def list_datasets() -> None:
    print("Synthetic:")
    print("- synthetic: locally generated images for counting, phenotype, registration, and cohort smoke tests.")
    print("\nPublic:")
    for name, cfg in PUBLIC_DATASETS.items():
        print(f"- {name}: {cfg['description']}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ignored local test subjects for RGC Counter.")
    parser.add_argument(
        "target",
        choices=["list", "synthetic", "oir_sample", *PUBLIC_DATASETS.keys()],
        help="Dataset target to prepare.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Root folder for ignored local test subjects.",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Download archives without extracting them.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    ensure_dir(args.output_dir)

    if args.target == "list":
        list_datasets()
        return 0
    if args.target == "synthetic":
        generate_synthetic_subjects(args.output_dir)
        return 0
    if args.target == "oir_sample":
        prepare_oir_sample(args.output_dir)
        return 0

    download_public_dataset(args.target, args.output_dir, extract=not args.no_extract)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
