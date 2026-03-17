from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import textwrap
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_ome import load_any_image
from src.landmarks import build_tissue_mask, detect_onh_hole
from src.report import copy_report_bundle
from src.advisor_packet import (
    TRACKED_FIGURE_MAPPINGS,
    audit_advisor_packet,
    build_tracked_lane_comparison_md,
    export_hash_rows,
    extract_pytest_count,
    file_sha256,
)


MOEMIL_REPO = "https://github.com/MOEMIL/Intelligent-quantifying-RGCs.git"
SIMPLERGC_REPO = "https://github.com/sonjoonho/SimpleRGC.git"
RGCODE_URL = "https://bio.kuleuven.be/df/lm/downloads"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def command_display(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def git_output(*args: str) -> str:
    completed = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True)
    return completed.stdout.strip()


def git_status_lines() -> list[str]:
    output = git_output("status", "--short")
    if not output:
        return []
    return [line for line in output.splitlines() if line.strip()]


def run_logged(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    env: dict[str, str] | None = None,
    check: bool = False,
    append: bool = False,
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - started
    log_text = textwrap.dedent(
        f"""\
        cwd: {cwd}
        command: {command_display(command)}
        returncode: {completed.returncode}
        elapsed_seconds: {elapsed:.2f}

        --- stdout ---
        {completed.stdout}

        --- stderr ---
        {completed.stderr}
        """
    )
    if append and log_path.exists():
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write("\n\n")
            handle.write(log_text)
    else:
        write_text(log_path, log_text)
    if check and completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode,
            command,
            output=completed.stdout,
            stderr=completed.stderr,
        )
    return {
        "returncode": completed.returncode,
        "elapsed_seconds": elapsed,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "log_path": log_path,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        seen: list[str] = []
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.append(key)
        fieldnames = seen
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_image_gray(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    image, meta = load_any_image(str(path))
    arr = np.asarray(image)
    arr = np.squeeze(arr)
    while arr.ndim > 3:
        arr = arr[0]
    if arr.ndim == 3:
        if arr.shape[-1] <= 4:
            arr = arr[..., 0]
        else:
            arr = arr[0]
    return np.asarray(arr, dtype=np.float32), meta


def image_properties(path: Path) -> dict[str, Any]:
    gray, _ = load_image_gray(path)
    channels = 1
    try:
        raw, _ = load_any_image(str(path))
        raw_arr = np.asarray(raw)
        if raw_arr.ndim == 3 and raw_arr.shape[-1] <= 8:
            channels = int(raw_arr.shape[-1])
        elif raw_arr.ndim >= 3:
            channels = int(raw_arr.shape[0])
    except Exception:
        pass
    return {
        "height_px": int(gray.shape[0]),
        "width_px": int(gray.shape[1]),
        "channels": channels,
        "dtype": str(gray.dtype),
    }


def list_artifacts(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "relative_path": str(path.relative_to(root)),
                    "size_bytes": path.stat().st_size,
                }
            )
    return rows


def first_match(root: Path, patterns: list[str]) -> Path | None:
    for pattern in patterns:
        matches = sorted(root.rglob(pattern))
        if matches:
            return matches[0]
    return None


def copy_if_found(root: Path, patterns: list[str], destination_dir: Path, name: str | None = None) -> Path | None:
    match = first_match(root, patterns)
    if match is None:
        return None
    destination_name = name or match.name
    destination = destination_dir / destination_name
    copy_file(match, destination)
    return destination


def archive_report_bundle(source_dir: Path, destination_root: Path, bundle_name: str) -> Path | None:
    report_path = source_dir / "report.html"
    if not report_path.exists():
        return None
    return copy_report_bundle(report_path, destination_root / bundle_name)


def parse_pytest_nodes(output: str) -> list[str]:
    nodes: list[str] = []
    for line in output.splitlines():
        match = re.match(r"^(FAILED|ERROR)\s+(\S+::\S+)", line.strip())
        if match and match.group(2) not in nodes:
            nodes.append(match.group(2))
    return nodes


def repo_relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def extract_logged_command(log_path: Path) -> str | None:
    if not log_path.exists():
        return None
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("command: "):
            return line.split("command: ", 1)[1].strip()
    return None


def extract_stage_commands(runtime_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in runtime_rows:
        notes = row.get("notes")
        if not isinstance(notes, str):
            continue
        log_path = ROOT / notes
        command = extract_logged_command(log_path)
        if command is None and notes and not notes.endswith(".txt"):
            command = notes
        if command:
            rows.append({"stage": str(row.get("stage", "")), "command": command, "log_path": notes})
    return rows


def copy_exact_file(source_root: Path, source_rel: str, destination_root: Path, destination_rel: str) -> Path:
    source_path = source_root / source_rel
    if not source_path.exists():
        raise FileNotFoundError(f"Missing required source artifact: {source_path}")
    destination_path = destination_root / destination_rel
    copy_file(source_path, destination_path)
    return destination_path


def read_pytest_pass_count(pytest_log_path: Path) -> int | None:
    if not pytest_log_path.exists():
        return None
    return extract_pytest_count(pytest_log_path.read_text(encoding="utf-8"))


def require_clean_git_worktree() -> None:
    status = git_status_lines()
    if status:
        raise RuntimeError(
            "Refusing to build the advisor packet from a dirty checkout. Commit the current changes first."
        )


def build_repo_snapshot(snapshot_path: Path) -> Path | None:
    snapshot_script = Path("/Users/marouane/.codex/skills/codebase-snapshot/scripts/build_codebase_txt.py")
    if not snapshot_script.exists():
        return None
    result = subprocess.run(
        [
            str(snapshot_script),
            "--root",
            str(ROOT),
            "--output",
            str(snapshot_path),
            "--force",
            "--no-gitignore-update",
            "--quiet",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not snapshot_path.exists():
        return None
    return snapshot_path


def write_dataset_inventory(path: Path) -> None:
    rows = [
        {
            "dataset_name": "OIR Flat-Mount Dataset",
            "source_type": "public_dataset",
            "public_url_or_doi": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11620014/",
            "license": "Check Figshare record during acquisition; treat as external public data with source citation.",
            "modality": "murine retinal flatmount",
            "stain_or_marker": "vascular flat-mount imagery",
            "channels": "varies",
            "whole_retina_or_roi": "whole_retina_or_near_whole_retina",
            "annotation_type": "vessel segmentation",
            "suitable_for_count_accuracy": "no",
            "suitable_for_registration": "yes",
            "suitable_for_report_demo": "yes",
            "notes": "Use for external retinal workflow and registration/report artifacts, not RGC counting accuracy.",
        },
        {
            "dataset_name": "MOEMIL improved-YOLOv5 public samples",
            "source_type": "public_repo_samples",
            "public_url_or_doi": "https://github.com/MOEMIL/Intelligent-quantifying-RGCs",
            "license": "Inspect repo license during acquisition; do not assume permissive redistribution.",
            "modality": "whole-retina or ROI microscopy samples",
            "stain_or_marker": "BRN3A-oriented workflow",
            "channels": "multi-channel compatibility must be checked per sample",
            "whole_retina_or_roi": "mixed_or_unknown",
            "annotation_type": "sample images only",
            "suitable_for_count_accuracy": "maybe/limited",
            "suitable_for_registration": "maybe",
            "suitable_for_report_demo": "yes",
            "notes": "Public comparator/compatibility target; not a universal apples-to-apples benchmark.",
        },
        {
            "dataset_name": "BBBC039",
            "source_type": "public_dataset",
            "public_url_or_doi": "https://bbbc.broadinstitute.org/BBBC039",
            "license": "Per BBBC source page.",
            "modality": "fluorescence microscopy nuclei",
            "stain_or_marker": "nuclear fluorescence",
            "channels": "single-channel",
            "whole_retina_or_roi": "roi_tiles",
            "annotation_type": "segmentation masks",
            "suitable_for_count_accuracy": "no for retinal claims",
            "suitable_for_registration": "no",
            "suitable_for_report_demo": "segmentation smoke only",
            "notes": "Use only as a non-retina engineering smoke test.",
        },
    ]
    write_csv(path, rows)


def write_licenses_and_sources(path: Path) -> None:
    content = textwrap.dedent(
        """\
        # Licenses And Sources

        ## OIR Flat-Mount Dataset
        - Primary paper/source: <https://pmc.ncbi.nlm.nih.gov/articles/PMC11620014/>
        - Download path used by this repo helper: Figshare files referenced in `scripts/prepare_test_subjects.py`
        - Evidence stance: external retinal workflow dataset for registration/report validation, not RGC count ground truth
        - License note: confirm the Figshare record before any redistribution outside this local-only bundle

        ## MOEMIL Public Samples
        - Source repo: <https://github.com/MOEMIL/Intelligent-quantifying-RGCs>
        - Evidence stance: public compatibility/comparator samples for BRN3A-oriented workflow checks
        - License note: use the repo's declared license if present; otherwise treat outputs as local evaluation artifacts only

        ## BBBC039
        - Dataset page: <https://bbbc.broadinstitute.org/BBBC039>
        - Evidence stance: non-retina segmentation smoke test only
        - License note: follow the BBBC dataset terms from the source page

        ## Comparator References
        - SimpleRGC: <https://github.com/sonjoonho/SimpleRGC>
        - RGCode: <https://bio.kuleuven.be/df/lm/downloads>
        """
    )
    write_text(path, content)


def write_comparator_inventory(path: Path) -> None:
    content = textwrap.dedent(
        """\
        # Comparator Inventory

        | Comparator | Source | Scope In This Pass | Status |
        | --- | --- | --- | --- |
        | MOEMIL | https://github.com/MOEMIL/Intelligent-quantifying-RGCs | Mandatory execution attempt on public samples | pending |
        | SimpleRGC | https://github.com/sonjoonho/SimpleRGC | Installation/compatibility attempt only | pending |
        | RGCode | https://bio.kuleuven.be/df/lm/downloads | Inventory and compatibility notes only | documented_only |
        """
    )
    write_text(path, content)


def gather_environment_info(env_dir: Path, venv_python: Path) -> None:
    uname = platform.uname()
    system_info = textwrap.dedent(
        f"""\
        platform: {platform.platform()}
        system: {uname.system}
        node: {uname.node}
        release: {uname.release}
        version: {uname.version}
        machine: {uname.machine}
        processor: {uname.processor}
        cwd: {ROOT}
        """
    )
    write_text(env_dir / "system_info.txt", system_info)

    version = subprocess.run([str(venv_python), "--version"], capture_output=True, text=True, check=True)
    write_text(env_dir / "python_version.txt", (version.stdout or version.stderr).strip() + "\n")

    freeze = subprocess.run(
        [str(venv_python), "-m", "pip", "freeze"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    write_text(env_dir / "pip_freeze.txt", freeze.stdout)

    git_rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
    write_text(env_dir / "git_rev.txt", git_rev.stdout)

    git_status = subprocess.run(["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True, check=True)
    write_text(env_dir / "git_status.txt", git_status.stdout)


def describe_pytest_failure(stdout: str, stderr: str) -> str:
    combined = "\n".join(part for part in [stdout.strip(), stderr.strip()] if part)
    if "ModuleNotFoundError" in combined or "ImportError" in combined:
        return "Failure looks environment-related (missing import or optional dependency mismatch)."
    if "AssertionError" in combined:
        return "Failure looks behavioral/regression-related rather than install-related."
    if "RuntimeError" in combined:
        return "Failure looks runtime/data-path related; inspect the first RuntimeError in the pytest logs."
    return "Failure cause was not cleanly classifiable from stdout/stderr; inspect the captured logs."


def create_venv_and_install(evidence_root: Path, runtime_rows: list[dict[str, Any]], failure_rows: list[dict[str, Any]]) -> tuple[Path, bool]:
    env_dir = evidence_root / "00_environment"
    install_log = env_dir / "install_log.txt"
    venv_dir = ROOT / ".venv-paper-evidence"
    if venv_dir.exists():
        remove_path(venv_dir)

    create_result = run_logged(
        [sys.executable, "-m", "venv", str(venv_dir)],
        cwd=ROOT,
        log_path=install_log,
    )
    runtime_rows.append(
        {
            "stage": "venv_create",
            "dataset": "environment",
            "elapsed_seconds": round(create_result["elapsed_seconds"], 2),
            "status": "ok" if create_result["returncode"] == 0 else "failed",
            "notes": repo_relative(install_log),
        }
    )
    if create_result["returncode"] != 0:
        failure_rows.append(
            {
                "dataset": "environment",
                "stage": "venv_create",
                "symptom": "Failed to create .venv-paper-evidence",
                "suspected_cause": "Python venv creation failed.",
                "mitigation": "Inspect install_log.txt and retry before any paper-evidence execution.",
                "claim_impact": "Blocks all evidence collection.",
            }
        )
        return venv_dir / "bin/python", False

    venv_python = venv_dir / "bin" / "python"
    venv_pip = venv_dir / "bin" / "pip"
    for command in (
        [str(venv_python), "-m", "pip", "install", "--upgrade", "pip"],
        [str(venv_pip), "install", "-e", ".[dev,ui]"],
    ):
        result = run_logged(command, cwd=ROOT, log_path=install_log, append=True)
        runtime_rows.append(
            {
                "stage": "environment_install",
                "dataset": "environment",
                "elapsed_seconds": round(result["elapsed_seconds"], 2),
                "status": "ok" if result["returncode"] == 0 else "failed",
                "notes": command_display(command),
            }
        )
        if result["returncode"] != 0:
            failure_rows.append(
                {
                    "dataset": "environment",
                    "stage": "environment_install",
                    "symptom": "Editable install failed in the fresh evidence venv.",
                    "suspected_cause": "Dependency resolution or build failure during pip install.",
                    "mitigation": "Inspect install_log.txt and repair environment setup before evidence runs.",
                    "claim_impact": "Blocks reproducibility claim.",
                }
            )
            return venv_python, False

    gather_environment_info(env_dir, venv_python)
    return venv_python, True


def run_phase_a_pytest(
    evidence_root: Path,
    venv_python: Path,
    runtime_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
) -> bool:
    env_dir = evidence_root / "00_environment"
    pytest_log = env_dir / "pytest_log.txt"
    env = os.environ.copy()
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    result = run_logged(
        [str(venv_python), "-m", "pytest", "-q"],
        cwd=ROOT,
        log_path=pytest_log,
        env=env,
    )
    runtime_rows.append(
        {
            "stage": "pytest",
            "dataset": "environment",
            "elapsed_seconds": round(result["elapsed_seconds"], 2),
            "status": "ok" if result["returncode"] == 0 else "failed",
            "notes": repo_relative(pytest_log),
        }
    )
    if result["returncode"] == 0:
        return True

    nodes = parse_pytest_nodes(result["stdout"] + "\n" + result["stderr"])
    rerun_log = env_dir / "pytest_rerun_log.txt"
    rerun_status = "not_run"
    if nodes:
        rerun_result = run_logged(
            [str(venv_python), "-m", "pytest", "-q", *nodes],
            cwd=ROOT,
            log_path=rerun_log,
            env=env,
        )
        rerun_status = "ok" if rerun_result["returncode"] == 0 else "failed"
        runtime_rows.append(
            {
                "stage": "pytest_rerun",
                "dataset": "environment",
                "elapsed_seconds": round(rerun_result["elapsed_seconds"], 2),
                "status": rerun_status,
                "notes": repo_relative(rerun_log),
            }
        )
    hypothesis = describe_pytest_failure(result["stdout"], result["stderr"])
    failure_rows.append(
        {
            "dataset": "environment",
            "stage": "pytest",
            "symptom": "Clean-environment pytest failed.",
            "suspected_cause": hypothesis,
            "mitigation": "Fix the failing tests before any external-dataset or comparator work.",
            "claim_impact": "Blocks the reproducibility claim and triggers early stop.",
        }
    )
    failure_note = textwrap.dedent(
        f"""\
        # Pytest Failure Hypothesis

        - Initial run log: `{repo_relative(pytest_log)}`
        - Rerun log: `{repo_relative(rerun_log) if rerun_log.exists() else 'not_run'}`
        - Parsed failing nodes: `{', '.join(nodes) if nodes else 'none parsed'}`
        - Initial diagnosis: {hypothesis}
        - Rerun status: {rerun_status}
        """
    )
    write_text(env_dir / "pytest_failure_hypothesis.md", failure_note)
    return False


def validate_paths_exist(paths: list[Path]) -> list[str]:
    missing = [str(path) for path in paths if not path.exists()]
    return missing


def write_artifact_manifest(output_dir: Path, manifest_path: Path) -> None:
    rows = list_artifacts(output_dir)
    if not rows:
        write_text(manifest_path, "No artifacts found.\n")
        return
    lines = [f"{row['relative_path']}\t{row['size_bytes']}" for row in rows]
    write_text(manifest_path, "\n".join(lines) + "\n")


def build_tracked_manual_files(
    benchmark_root: Path,
    validation_dir: Path,
    failure_rows: list[dict[str, Any]],
) -> None:
    ensure_dir(benchmark_root / "manual_annotations")
    copy_file(
        ROOT / "examples" / "manual_annotations" / "example_manual_annotations.csv",
        benchmark_root / "manual_annotations" / "example_manual_annotations.csv",
    )
    write_text(
        benchmark_root / "annotation_protocol.md",
        textwrap.dedent(
            """\
            # Annotation Protocol

            This first-pass benchmark uses the tracked repository manual reference asset
            `examples/manual_annotations/example_manual_annotations.csv`.

            - Benchmark unit: whole tracked example image
            - Samples: `EX01_OD`, `EX01_OS`
            - Reference type: manual whole-image counts already stored in the repo
            - Intended use: conservative count-agreement check for the tracked example workflow
            - Not covered: centroid matching, mask overlap, or external manual re-annotation
            """
        ),
    )
    roi_rows = [
        {
            "sample_id": "EX01_OD",
            "roi_type": "whole_image",
            "reference_source": "examples/manual_annotations/example_manual_annotations.csv",
        },
        {
            "sample_id": "EX01_OS",
            "roi_type": "whole_image",
            "reference_source": "examples/manual_annotations/example_manual_annotations.csv",
        },
    ]
    write_csv(benchmark_root / "roi_manifest.csv", roi_rows)
    write_text(
        benchmark_root / "inter_annotator_notes.md",
        textwrap.dedent(
            """\
            # Inter-Annotator Notes

            A second human annotation pass was not performed in this first evidence bundle.
            The tracked example manual counts are used as the only reference for the v1 benchmark.
            """
        ),
    )

    details_path = validation_dir / "validation_details.csv"
    summary_path = validation_dir / "validation_summary.csv"
    if details_path.exists():
        copy_file(details_path, benchmark_root / "benchmark_results.csv")
    else:
        failure_rows.append(
            {
                "dataset": "tracked_example",
                "stage": "manual_benchmark",
                "symptom": "validation_details.csv was not produced.",
                "suspected_cause": "The manual-annotation study pass did not emit validation outputs.",
                "mitigation": "Inspect the manual benchmark run log and validation directory.",
                "claim_impact": "Blocks quantitative count-agreement claim.",
            }
        )
        write_text(benchmark_root / "benchmark_results.csv", "")
    if summary_path.exists():
        copy_file(summary_path, benchmark_root / "validation_summary.csv")


def run_repo_validation(
    evidence_root: Path,
    venv_python: Path,
    runtime_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
) -> dict[str, Path] | None:
    repo_root = evidence_root / "01_repo_validation"

    outputs_example = ROOT / "Outputs_example"
    outputs_maps = ROOT / "Outputs_example_maps"
    outputs_model_eval = ROOT / "Outputs_model_eval"
    outputs_example_postfix = ROOT / "Outputs_example_postfix"
    outputs_maps_final = ROOT / "Outputs_example_maps_final"
    outputs_model_eval_final = ROOT / "Outputs_model_eval_final"
    manual_benchmark_root = evidence_root / "04_manual_benchmark" / "tracked_example_validation"
    for path in [
        outputs_example,
        outputs_maps,
        outputs_model_eval,
        outputs_example_postfix,
        outputs_maps_final,
        outputs_model_eval_final,
        manual_benchmark_root,
    ]:
        if path.exists():
            remove_path(path)

    tracked_log = repo_root / "tracked_example_run_log.txt"
    tracked_result = run_logged(
        [
            str(venv_python),
            "main.py",
            "--manifest",
            "examples/manifests/example_study_manifest.csv",
            "--study_output_dir",
            str(outputs_example),
            "--focus_none",
            "--register_retina",
            "--region_schema",
            "mouse_flatmount_v1",
            "--spatial_stats",
            "--spatial_mode",
            "rigorous",
            "--spatial_envelope_sims",
            "8",
            "--write_object_table",
            "--write_provenance",
            "--write_html_report",
        ],
        cwd=ROOT,
        log_path=tracked_log,
    )
    runtime_rows.append(
        {
            "stage": "tracked_example",
            "dataset": "tracked_example",
            "elapsed_seconds": round(tracked_result["elapsed_seconds"], 2),
            "status": "ok" if tracked_result["returncode"] == 0 else "failed",
            "notes": repo_relative(tracked_log),
        }
    )
    expected = [
        outputs_example / "study_summary.csv",
        outputs_example / "study_regions.csv",
        outputs_example / "stats",
        outputs_example / "figures",
        outputs_example / "methods_appendix.md",
        outputs_example / "report.html",
        outputs_example / "provenance.json",
        outputs_example / "samples" / "EX01_OD" / "spatial",
        outputs_example / "samples" / "EX01_OD" / "objects",
        outputs_example / "samples" / "EX01_OD" / "regions",
        outputs_example / "samples" / "EX01_OD" / "retina_frames",
    ]
    missing = validate_paths_exist(expected)
    write_artifact_manifest(outputs_example, repo_root / "tracked_example_outputs_manifest.txt")
    if tracked_result["returncode"] != 0 or missing:
        failure_rows.append(
            {
                "dataset": "tracked_example",
                "stage": "canonical_example",
                "symptom": "Canonical tracked example failed or missed expected outputs.",
                "suspected_cause": "; ".join(missing) if missing else "main.py returned non-zero",
                "mitigation": "Inspect the tracked example log and output manifest before any external-dataset work.",
                "claim_impact": "Blocks documented end-to-end reproducibility claim.",
            }
        )
        return None

    qc_dir = repo_root / "tracked_example_qc"
    ensure_dir(qc_dir)
    maps_log = repo_root / "single_image_qc_run_log.txt"
    maps_result = run_logged(
        [
            str(venv_python),
            "main.py",
            "--input_dir",
            "examples/smoke_data",
            "--output_dir",
            str(outputs_maps),
            "--focus_qc",
            "--tta",
            "--register_retina",
            "--onh_mode",
            "cli",
            "--onh_xy",
            "48",
            "48",
            "--dorsal_xy",
            "48",
            "14",
            "--write_object_table",
            "--write_provenance",
            "--write_html_report",
            "--write_uncertainty_maps",
            "--write_qc_maps",
            "--strict_schemas",
            "--no_gpu",
        ],
        cwd=ROOT,
        log_path=maps_log,
    )
    runtime_rows.append(
        {
            "stage": "single_image_qc",
            "dataset": "tracked_example",
            "elapsed_seconds": round(maps_result["elapsed_seconds"], 2),
            "status": "ok" if maps_result["returncode"] == 0 else "failed",
            "notes": repo_relative(maps_log),
        }
    )
    for path in sorted(outputs_maps.rglob("*")):
        if path.is_file() and (
            "uncertainty" in path.name.lower()
            or "qc" in path.name.lower()
            or path.name in {"report.html", "provenance.json"}
        ):
            copy_file(path, qc_dir / path.name)

    model_eval_log = repo_root / "model_eval_run_log.txt"
    model_eval_result = run_logged(
        [
            str(venv_python),
            "scripts/evaluate_models.py",
            "--model_manifest",
            "examples/models/model_manifest.example.csv",
            "--output_dir",
            str(outputs_model_eval),
            "--no_gpu",
        ],
        cwd=ROOT,
        log_path=model_eval_log,
    )
    runtime_rows.append(
        {
            "stage": "model_eval",
            "dataset": "tracked_example",
            "elapsed_seconds": round(model_eval_result["elapsed_seconds"], 2),
            "status": "ok" if model_eval_result["returncode"] == 0 else "failed",
            "notes": repo_relative(model_eval_log),
        }
    )
    segmentation_metrics = evidence_root / "05_quantitative_results" / "segmentation_metrics.csv"
    if (outputs_model_eval / "per_run_metrics.csv").exists():
        copy_file(outputs_model_eval / "per_run_metrics.csv", segmentation_metrics)

    manual_log = repo_root / "manual_benchmark_run_log.txt"
    manual_result = run_logged(
        [
            str(venv_python),
            "main.py",
            "--manifest",
            "examples/manifests/example_study_manifest.csv",
            "--manual_annotations",
            "examples/manual_annotations/example_manual_annotations.csv",
            "--study_output_dir",
            str(manual_benchmark_root),
            "--focus_none",
            "--register_retina",
            "--region_schema",
            "mouse_flatmount_v1",
            "--spatial_stats",
            "--spatial_mode",
            "rigorous",
            "--spatial_envelope_sims",
            "8",
            "--write_object_table",
            "--write_provenance",
            "--write_html_report",
        ],
        cwd=ROOT,
        log_path=manual_log,
    )
    runtime_rows.append(
        {
            "stage": "manual_benchmark",
            "dataset": "tracked_example",
            "elapsed_seconds": round(manual_result["elapsed_seconds"], 2),
            "status": "ok" if manual_result["returncode"] == 0 else "failed",
            "notes": repo_relative(manual_log),
        }
    )
    validation_dir = manual_benchmark_root / "validation"
    if manual_result["returncode"] != 0 or not validation_dir.exists():
        failure_rows.append(
            {
                "dataset": "tracked_example",
                "stage": "manual_benchmark",
                "symptom": "Tracked manual benchmark did not complete cleanly.",
                "suspected_cause": "main.py returned non-zero or validation outputs were missing.",
                "mitigation": "Inspect manual_benchmark_run_log.txt before using any count-agreement claims.",
                "claim_impact": "Weakens quantitative validation claim.",
            }
        )
    build_tracked_manual_files(evidence_root / "04_manual_benchmark", validation_dir, failure_rows)
    if (validation_dir / "validation_details.csv").exists():
        copy_file(validation_dir / "validation_details.csv", evidence_root / "05_quantitative_results" / "count_error_metrics.csv")

    report_dir = evidence_root / "07_report_artifacts"
    html_dir = ensure_dir(report_dir / "html_reports")
    methods_dir = ensure_dir(report_dir / "methods_appendices")
    prov_dir = ensure_dir(report_dir / "provenance_json")
    for src, base_name in [
        (outputs_example, "tracked_example"),
        (manual_benchmark_root, "tracked_example_manual_validation"),
        (outputs_maps, "tracked_example_single_image"),
    ]:
        archive_report_bundle(src, html_dir, base_name)
        if (src / "methods_appendix.md").exists():
            copy_file(src / "methods_appendix.md", methods_dir / f"{base_name}.md")
        if (src / "provenance.json").exists():
            copy_file(src / "provenance.json", prov_dir / f"{base_name}.json")

    fig1_dir = ensure_dir(evidence_root / "06_figures" / "fig1_pipeline_overview_inputs_outputs")
    copy_if_found(outputs_example, ["figures/cell_count_by_condition.png"], fig1_dir, "tracked_example_summary.png")
    copy_if_found(outputs_example, ["samples/*/registered_maps/*.png"], fig1_dir, "tracked_example_registered_map.png")
    copy_if_found(outputs_example, ["samples/*/spatial/*.png"], fig1_dir, "tracked_example_spatial.png")
    archive_report_bundle(outputs_example, fig1_dir, "tracked_example_report")
    return {
        "tracked_output_dir": outputs_example,
        "single_image_output_dir": outputs_maps,
        "model_eval_output_dir": outputs_model_eval,
        "manual_validation_dir": manual_benchmark_root,
    }


def classify_oir_condition(path: Path) -> tuple[str, str]:
    lower = str(path).lower()
    if "norm" in lower or "normoxia" in lower or "healthy" in lower:
        return "normoxic", "path_token"
    if "oir" in lower:
        return "oir", "path_token"
    return "unknown", "unresolved"


def discover_image_files(root: Path) -> list[Path]:
    suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".ome.tif", ".ome.tiff"}
    paths: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.name.startswith(".") or path.name.startswith("._"):
            continue
        lower = path.name.lower()
        if any(lower.endswith(suffix) for suffix in suffixes):
            paths.append(path)
    return sorted(paths)


def remove_if_invalid_zip(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with zipfile.ZipFile(path, "r") as archive:
            archive.infolist()
        return False
    except zipfile.BadZipFile:
        path.unlink()
        return True


def select_oir_subset(
    oir_root: Path,
    evidence_dir: Path,
    failure_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    image_paths = [
        path
        for path in discover_image_files(oir_root)
        if "subset_24" not in str(path)
    ]
    raw_rows: list[dict[str, Any]] = []
    for path in image_paths:
        condition, condition_source = classify_oir_condition(path)
        raw_rows.append(
            {
                "relative_path": str(path.relative_to(oir_root)),
                "absolute_path": str(path.resolve()),
                "condition": condition,
                "condition_source": condition_source,
            }
        )
    write_csv(evidence_dir / "raw_manifest.csv", raw_rows)

    preferred = [row for row in raw_rows if "/512/" in row["absolute_path"].replace("\\", "/")]
    if not preferred:
        preferred = [row for row in raw_rows if row["condition"] in {"normoxic", "oir"}]

    selected: list[dict[str, Any]] = []
    counts = {"normoxic": 0, "oir": 0}
    curated_root = ensure_dir(oir_root / "subset_24")
    for row in preferred:
        condition = row["condition"]
        if condition not in counts or counts[condition] >= 12:
            continue
        image_path = Path(row["absolute_path"])
        try:
            gray, _ = load_image_gray(image_path)
            onh_xy, info = detect_onh_hole(gray)
            if onh_xy is None:
                continue
            tissue_mask = build_tissue_mask(gray)
        except Exception:
            continue
        height, width = gray.shape[:2]
        dorsal_y = max(0.0, float(onh_xy[1]) - max(min(height, width) * 0.25, 32.0))
        dorsal_xy = (float(onh_xy[0]), dorsal_y)
        copied = curated_root / f"{condition}_{image_path.name}"
        copy_file(image_path, copied)
        sidecar = copied.with_name(f"{copied.name.rsplit('.', 1)[0]}.retina_frame.json")
        sidecar_payload = {
            "onh_x_px": float(onh_xy[0]),
            "onh_y_px": float(onh_xy[1]),
            "dorsal_x_px": float(dorsal_xy[0]),
            "dorsal_y_px": float(dorsal_xy[1]),
            "um_per_px": 1.0,
            "source": "sidecar:auto_onh_plus_top_frame_dorsal",
            "onh_source": str(info.get("method", "auto_hole")),
            "onh_confidence": float(info.get("confidence", 0.0)),
            "tissue_coverage_fraction": float(tissue_mask.mean()),
        }
        write_text(sidecar, json.dumps(sidecar_payload, indent=2) + "\n")
        sample_id = f"{condition[:3].upper()}_{counts[condition] + 1:02d}"
        selected_row = {
            "sample_id": sample_id,
            "animal_id": sample_id,
            "eye": "UNK",
            "condition": condition,
            "genotype": "unknown",
            "timepoint_dpi": 0,
            "modality": "flatmount",
            "stain_panel": "vascular_flatmount",
            "path": str(copied.resolve()),
            "onh_x_px": float(onh_xy[0]),
            "onh_y_px": float(onh_xy[1]),
            "dorsal_x_px": float(dorsal_xy[0]),
            "dorsal_y_px": float(dorsal_xy[1]),
            "landmark_strategy": "auto_onh_plus_top_frame_dorsal",
            "manual_landmark_assistance": False,
            "onh_confidence": float(info.get("confidence", 0.0)),
            "height_px": int(height),
            "width_px": int(width),
        }
        selected.append(selected_row)
        counts[condition] += 1
        if counts["normoxic"] >= 12 and counts["oir"] >= 12:
            break

    if counts["normoxic"] < 12 or counts["oir"] < 12:
        failure_rows.append(
            {
                "dataset": "oir_flatmount",
                "stage": "subset_selection",
                "symptom": f"Selected {counts['normoxic']} normoxic and {counts['oir']} OIR images instead of 12/12.",
                "suspected_cause": "Not enough qualifying images with resolvable ONH landmarks were found with the automated heuristic.",
                "mitigation": "Treat the OIR run as partial and note the subset shortfall explicitly.",
                "claim_impact": "Limits breadth of external retinal validation but does not invalidate end-to-end evidence if some images ran cleanly.",
            }
        )
    write_csv(evidence_dir / "subset_manifest.csv", selected)
    return raw_rows, selected


def build_registration_metrics(
    subset_manifest: Path,
    output_dir: Path,
    destination: Path,
) -> None:
    subset = pd.read_csv(subset_manifest)
    rows: list[dict[str, Any]] = []
    for row in subset.to_dict("records"):
        sample_dir = output_dir / "samples" / str(row["sample_id"])
        rows.append(
            {
                "sample_id": row["sample_id"],
                "condition": row["condition"],
                "manual_landmark_assistance": bool(row.get("manual_landmark_assistance", False)),
                "has_objects": (sample_dir / "objects").exists(),
                "has_regions": (sample_dir / "regions").exists(),
                "has_spatial": (sample_dir / "spatial").exists(),
                "has_retina_frames": (sample_dir / "retina_frames").exists(),
                "output_complete": all((sample_dir / name).exists() for name in ["objects", "regions", "spatial", "retina_frames"]),
            }
        )
    if not rows:
        write_text(destination, "")
        return
    summary = {
        "sample_id": "__summary__",
        "condition": "all",
        "manual_landmark_assistance": float(pd.Series([row["manual_landmark_assistance"] for row in rows]).mean()),
        "has_objects": float(pd.Series([row["has_objects"] for row in rows]).mean()),
        "has_regions": float(pd.Series([row["has_regions"] for row in rows]).mean()),
        "has_spatial": float(pd.Series([row["has_spatial"] for row in rows]).mean()),
        "has_retina_frames": float(pd.Series([row["has_retina_frames"] for row in rows]).mean()),
        "output_complete": float(pd.Series([row["output_complete"] for row in rows]).mean()),
    }
    write_csv(destination, rows + [summary])


def create_study_manifest(rows: list[dict[str, Any]], path: Path) -> None:
    write_csv(path, rows)


def build_moemil_manifest_rows(sample_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidate_paths = [path for path in discover_image_files(sample_dir) if path.suffix.lower() in {".tif", ".tiff"}]
    for index, path in enumerate(candidate_paths, start=1):
        try:
            props = image_properties(path)
        except Exception:
            continue
        whole_retina_or_roi = "whole_retina_candidate" if min(props["height_px"], props["width_px"]) >= 1500 else "roi_or_unknown"
        rows.append(
            {
                "sample_id": f"MOE_{index:02d}",
                "animal_id": f"MOE_{index:02d}",
                "eye": "UNK",
                "condition": "public_sample",
                "genotype": "unknown",
                "timepoint_dpi": 0,
                "modality": "flatmount",
                "stain_panel": "brn3a_or_unknown",
                "path": str(path.resolve()),
                "whole_retina_or_roi": whole_retina_or_roi,
                "channels": props["channels"],
                "height_px": props["height_px"],
                "width_px": props["width_px"],
                "dtype": props["dtype"],
            }
        )
    return rows


def clone_or_refresh(repo_url: str, destination: Path, log_path: Path) -> dict[str, Any]:
    if destination.exists():
        remove_path(destination)
    return run_logged(
        ["git", "clone", "--depth", "1", repo_url, str(destination)],
        cwd=ROOT,
        log_path=log_path,
    )


def capture_simple_repo_summary(repo_dir: Path) -> str:
    readme = next(iter(sorted(repo_dir.glob("README*"))), None)
    if readme is None:
        return "No README found."
    text = readme.read_text(encoding="utf-8", errors="ignore")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines[:20])


def run_external_datasets_and_comparators(
    evidence_root: Path,
    venv_python: Path,
    runtime_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
) -> dict[str, Path]:
    datasets_dir = evidence_root / "02_public_datasets"
    oir_dir = datasets_dir / "oir_flatmount"
    moemil_dir = datasets_dir / "moemil_samples"
    bbbc_dir = datasets_dir / "bbbc039"
    comparators_dir = evidence_root / "03_comparators"
    state: dict[str, Path] = {}
    write_dataset_inventory(datasets_dir / "dataset_inventory.csv")
    write_licenses_and_sources(datasets_dir / "licenses_and_sources.md")
    write_comparator_inventory(comparators_dir / "comparator_inventory.md")

    oir_archive = ROOT / "test_subjects" / "public" / "oir_flatmount" / "OIR_Flat_Mount_Dataset.zip"
    if remove_if_invalid_zip(oir_archive):
        failure_rows.append(
            {
                "dataset": "oir_flatmount",
                "stage": "archive_cache",
                "symptom": "Removed a stale invalid cached OIR zip before re-download.",
                "suspected_cause": "Previous interrupted or non-zip download left a corrupt local cache file.",
                "mitigation": "Re-download through the existing helper.",
                "claim_impact": "None if re-download succeeds.",
            }
        )
    oir_download_log = oir_dir / "run_log.txt"
    oir_fetch = run_logged(
        [str(venv_python), "scripts/prepare_test_subjects.py", "oir_flatmount", "--output-dir", "test_subjects"],
        cwd=ROOT,
        log_path=oir_download_log,
    )
    runtime_rows.append(
        {
            "stage": "oir_download",
            "dataset": "oir_flatmount",
            "elapsed_seconds": round(oir_fetch["elapsed_seconds"], 2),
            "status": "ok" if oir_fetch["returncode"] == 0 else "failed",
            "notes": repo_relative(oir_download_log),
        }
    )
    if oir_fetch["returncode"] != 0:
        failure_rows.append(
            {
                "dataset": "oir_flatmount",
                "stage": "download",
                "symptom": "OIR acquisition helper failed.",
                "suspected_cause": "Download or extraction error from remote data or cached local files.",
                "mitigation": "Inspect the OIR run log, clear bad cached artifacts if needed, and rerun.",
                "claim_impact": "Blocks the external retinal dataset claim until repaired.",
            }
        )
    if oir_fetch["returncode"] == 0:
        oir_root = ROOT / "test_subjects" / "public" / "oir_flatmount"
        raw_rows, subset_rows = select_oir_subset(oir_root, oir_dir, failure_rows)
        write_text(
            oir_dir / "acquisition_notes.md",
            textwrap.dedent(
                f"""\
                # OIR Acquisition Notes

                - Download helper: `scripts/prepare_test_subjects.py oir_flatmount --output-dir test_subjects`
                - Raw discovered image files: `{len(raw_rows)}`
                - Selected subset rows: `{len(subset_rows)}`
                - Selection method: prefer paths containing `/512/`, require successful auto ONH detection, use a top-of-frame dorsal heuristic for consistent registration inputs
                - Important limit: dorsal orientation in this first pass is heuristic rather than biologically validated
                """
            ),
        )
        oir_output_dir = ensure_dir(oir_dir / "outputs")
        if oir_output_dir.exists():
            remove_path(oir_output_dir)
        oir_run = run_logged(
            [
                str(venv_python),
                "main.py",
                "--manifest",
                str((oir_dir / "subset_manifest.csv").resolve()),
                "--study_output_dir",
                str(oir_output_dir.resolve()),
                "--focus_none",
                "--register_retina",
                "--region_schema",
                "mouse_flatmount_v1",
                "--onh_mode",
                "sidecar",
                "--spatial_stats",
                "--spatial_mode",
                "rigorous",
                "--spatial_envelope_sims",
                "32",
                "--write_object_table",
                "--write_provenance",
                "--write_html_report",
                "--strict_schemas",
                "--no_gpu",
            ],
            cwd=ROOT,
            log_path=oir_dir / "study_run_log.txt",
        )
        runtime_rows.append(
            {
                "stage": "oir_study_run",
                "dataset": "oir_flatmount",
                "elapsed_seconds": round(oir_run["elapsed_seconds"], 2),
                "status": "ok" if oir_run["returncode"] == 0 else "failed",
                "notes": repo_relative(oir_dir / "study_run_log.txt"),
            }
        )
        if oir_run["returncode"] != 0:
            failure_rows.append(
                {
                    "dataset": "oir_flatmount",
                    "stage": "study_run",
                    "symptom": "OIR study-mode execution failed.",
                    "suspected_cause": "main.py returned non-zero on the curated OIR subset.",
                    "mitigation": "Inspect OIR study_run_log.txt and treat the external-retinal claim as partial or blocked.",
                    "claim_impact": "Can block the external real-data claim if no images complete end-to-end.",
                }
            )
        if (oir_dir / "subset_manifest.csv").exists():
            build_registration_metrics(
                oir_dir / "subset_manifest.csv",
                oir_output_dir,
                evidence_root / "05_quantitative_results" / "registration_success_metrics.csv",
            )
        report_dir = evidence_root / "07_report_artifacts"
        archive_report_bundle(oir_output_dir, report_dir / "html_reports", "oir_flatmount")
        if (oir_output_dir / "methods_appendix.md").exists():
            copy_file(oir_output_dir / "methods_appendix.md", report_dir / "methods_appendices" / "oir_flatmount.md")
        if (oir_output_dir / "provenance.json").exists():
            copy_file(oir_output_dir / "provenance.json", report_dir / "provenance_json" / "oir_flatmount.json")
        fig2_dir = ensure_dir(evidence_root / "06_figures" / "fig2_external_real_data_examples")
        copy_if_found(oir_output_dir, ["samples/*/registered_maps/*.png"], fig2_dir, "oir_registered_map.png")
        copy_if_found(oir_output_dir, ["samples/*/spatial/*.png"], fig2_dir, "oir_spatial.png")
        copy_if_found(oir_output_dir, ["figures/cell_count_by_condition.png"], fig2_dir, "oir_summary.png")
        state["oir_output_dir"] = oir_output_dir

    moemil_repo_dir = ROOT / "test_subjects" / "public" / "moemil_repo"
    moemil_clone_log = moemil_dir / "clone_log.txt"
    moemil_clone = clone_or_refresh(MOEMIL_REPO, moemil_repo_dir, moemil_clone_log)
    runtime_rows.append(
        {
            "stage": "moemil_clone",
            "dataset": "moemil_samples",
            "elapsed_seconds": round(moemil_clone["elapsed_seconds"], 2),
            "status": "ok" if moemil_clone["returncode"] == 0 else "failed",
            "notes": repo_relative(moemil_clone_log),
        }
    )
    moemil_sample_dir = ROOT / "test_subjects" / "public" / "moemil_samples"
    ensure_dir(moemil_sample_dir)
    if moemil_clone["returncode"] == 0:
        for path in sorted((moemil_repo_dir / "sample").glob("*.tif*")):
            copy_file(path, moemil_sample_dir / path.name)
        moemil_rows = build_moemil_manifest_rows(moemil_sample_dir)
        write_csv(moemil_dir / "raw_manifest.csv", moemil_rows)
        create_study_manifest(moemil_rows, moemil_dir / "manifest.csv")
        write_text(
            moemil_dir / "acquisition_notes.md",
            textwrap.dedent(
                f"""\
                # MOEMIL Acquisition Notes

                - Source repo: `{MOEMIL_REPO}`
                - Sample TIFFs copied into: `{repo_relative(moemil_sample_dir)}`
                - Sample count: `{len(moemil_rows)}`
                - Whole-retina classification is heuristic and based on image size/file naming only.
                """
            ),
        )
        moemil_output_dir = ensure_dir(moemil_dir / "outputs")
        if moemil_output_dir.exists():
            remove_path(moemil_output_dir)
        moemil_run = run_logged(
            [
                str(venv_python),
                "main.py",
                "--manifest",
                str((moemil_dir / "manifest.csv").resolve()),
                "--study_output_dir",
                str(moemil_output_dir.resolve()),
                "--focus_none",
                "--write_object_table",
                "--write_provenance",
                "--write_html_report",
                "--no_gpu",
            ],
            cwd=ROOT,
            log_path=moemil_dir / "run_log.txt",
        )
        runtime_rows.append(
            {
                "stage": "moemil_pipeline_run",
                "dataset": "moemil_samples",
                "elapsed_seconds": round(moemil_run["elapsed_seconds"], 2),
                "status": "ok" if moemil_run["returncode"] == 0 else "failed",
                "notes": repo_relative(moemil_dir / "run_log.txt"),
            }
        )
        if moemil_run["returncode"] != 0:
            failure_rows.append(
                {
                    "dataset": "moemil_samples",
                    "stage": "pipeline_run",
                    "symptom": "MOEMIL compatibility pass failed.",
                    "suspected_cause": "Channel arrangement or modality mismatch with the current pipeline.",
                    "mitigation": "Treat MOEMIL as compatibility evidence only and inspect the run log for exact mismatches.",
                    "claim_impact": "Does not block the paper if other evidence lanes succeed.",
                }
            )
        report_dir = evidence_root / "07_report_artifacts"
        archive_report_bundle(moemil_output_dir, report_dir / "html_reports", "moemil_samples")
        if (moemil_output_dir / "provenance.json").exists():
            copy_file(moemil_output_dir / "provenance.json", report_dir / "provenance_json" / "moemil_samples.json")
        state["moemil_output_dir"] = moemil_output_dir

        comparator_moemil_dir = ensure_dir(comparators_dir / "moemil")
        repo_summary = capture_simple_repo_summary(moemil_repo_dir)
        write_text(
            comparator_moemil_dir / "install_notes.md",
            textwrap.dedent(
                f"""\
                # MOEMIL Install Notes

                - Clone log: `{repo_relative(moemil_clone_log)}`
                - Repo root: `{repo_relative(moemil_repo_dir)}`
                - README summary:

                ```
                {repo_summary}
                ```
                """
            ),
        )
        entrypoint = None
        for candidate in ["detect.py", "inference.py", "predict.py", "main.py"]:
            if (moemil_repo_dir / candidate).exists():
                entrypoint = moemil_repo_dir / candidate
                break
        if entrypoint is not None:
            attempt = run_logged(
                [str(venv_python), str(entrypoint), "--help"],
                cwd=moemil_repo_dir,
                log_path=comparator_moemil_dir / "execution_attempt_log.txt",
            )
            status = "ran_help" if attempt["returncode"] == 0 else "blocked_or_failed"
        else:
            status = "no_entrypoint_found"
            write_text(comparator_moemil_dir / "execution_attempt_log.txt", "No obvious CLI entrypoint found.\n")
        write_text(
            comparator_moemil_dir / "compatibility_notes.md",
            textwrap.dedent(
                f"""\
                # MOEMIL Compatibility Notes

                - Comparator execution attempt status: `{status}`
                - This evidence bundle keeps MOEMIL claims separate from retinal-phenotyper claims.
                - Any reported MOEMIL paper metrics belong to their own dataset and method, not this repo.
                """
            ),
        )

    simplergc_repo_dir = ROOT / "test_subjects" / "public" / "simplergc_repo"
    simplergc_clone_log = comparators_dir / "simplergc" / "clone_log.txt"
    ensure_dir(comparators_dir / "simplergc" / "outputs")
    simplergc_clone = clone_or_refresh(SIMPLERGC_REPO, simplergc_repo_dir, simplergc_clone_log)
    runtime_rows.append(
        {
            "stage": "simplergc_clone",
            "dataset": "simplergc",
            "elapsed_seconds": round(simplergc_clone["elapsed_seconds"], 2),
            "status": "ok" if simplergc_clone["returncode"] == 0 else "failed",
            "notes": repo_relative(simplergc_clone_log),
        }
    )
    fiji_candidates = [
        Path("/Applications/Fiji.app"),
        Path.home() / "Applications" / "Fiji.app",
    ]
    fiji_path = next((path for path in fiji_candidates if path.exists()), None)
    write_text(
        comparators_dir / "simplergc" / "install_notes.md",
        textwrap.dedent(
            f"""\
            # SimpleRGC Install Notes

            - Clone log: `{repo_relative(simplergc_clone_log)}`
            - Fiji detected: `{str(fiji_path) if fiji_path is not None else 'not_found'}`
            - Repo summary:

            ```
            {capture_simple_repo_summary(simplergc_repo_dir) if simplergc_repo_dir.exists() else 'clone_failed'}
            ```
            """
        ),
    )
    if fiji_path is None:
        compatibility = "Fiji was not available on this machine, so the install attempt stopped after repo acquisition and compatibility review."
    else:
        compatibility = f"Fiji exists at `{fiji_path}`, but automated plugin execution was not wired in this first focused pass."
    write_text(
        comparators_dir / "simplergc" / "compatibility_notes.md",
        textwrap.dedent(
            f"""\
            # SimpleRGC Compatibility Notes

            {compatibility}
            """
        ),
    )

    write_text(
        comparators_dir / "rgcode" / "install_notes.md",
        textwrap.dedent(
            f"""\
            # RGCode Install Notes

            - Public download page: `{RGCODE_URL}`
            - This focused pass documented RGCode as a conceptual comparator only.
            - No execution attempt was made in this round.
            """
        ),
    )
    write_text(
        comparators_dir / "rgcode" / "compatibility_notes.md",
        textwrap.dedent(
            """\
            # RGCode Compatibility Notes

            RGCode remained inventory-only in this first focused evidence pass.
            Do not claim execution or apples-to-apples benchmark parity from this bundle.
            """
        ),
    )

    bbbc_log = bbbc_dir / "run_log.txt"
    bbbc_fetch = run_logged(
        [str(venv_python), "scripts/prepare_test_subjects.py", "bbbc039", "--output-dir", "test_subjects"],
        cwd=ROOT,
        log_path=bbbc_log,
    )
    runtime_rows.append(
        {
            "stage": "bbbc039_download",
            "dataset": "bbbc039",
            "elapsed_seconds": round(bbbc_fetch["elapsed_seconds"], 2),
            "status": "ok" if bbbc_fetch["returncode"] == 0 else "failed",
            "notes": repo_relative(bbbc_log),
        }
    )
    if bbbc_fetch["returncode"] == 0:
        bbbc_root = ROOT / "test_subjects" / "public" / "bbbc039"
        image_paths = [path for path in discover_image_files(bbbc_root) if "masks" not in str(path).lower()]
        subset_paths = image_paths[:3]
        subset_dir = ensure_dir(bbbc_root / "subset_3")
        rows: list[dict[str, Any]] = []
        for index, path in enumerate(subset_paths, start=1):
            copied = subset_dir / path.name
            copy_file(path, copied)
            try:
                props = image_properties(copied)
            except Exception:
                failure_rows.append(
                    {
                        "dataset": "bbbc039",
                        "stage": "subset_selection",
                        "symptom": f"Skipped unreadable BBBC039 file: {copied.name}",
                        "suspected_cause": "Unreadable sidecar or hidden macOS metadata file in the extracted dataset tree.",
                        "mitigation": "Skip unreadable files and continue with readable smoke-test images.",
                        "claim_impact": "None if at least one readable image remains.",
                    }
                )
                continue
            rows.append(
                {
                    "sample_id": f"BBBC_{index:02d}",
                    "animal_id": f"BBBC_{index:02d}",
                    "eye": "NA",
                    "condition": "bbbc039_smoke",
                    "genotype": "na",
                    "timepoint_dpi": 0,
                    "modality": "flatmount",
                    "stain_panel": "nuclei_smoke",
                    "path": str(copied.resolve()),
                    **props,
                }
            )
        write_csv(bbbc_dir / "raw_manifest.csv", rows)
        create_study_manifest(rows, bbbc_dir / "manifest.csv")
        write_text(
            bbbc_dir / "acquisition_notes.md",
            textwrap.dedent(
                f"""\
                # BBBC039 Acquisition Notes

                - Download helper: `scripts/prepare_test_subjects.py bbbc039 --output-dir test_subjects`
                - Selected subset size: `{len(rows)}`
                - This run is engineering-only and must not be used for retinal biology claims.
                """
            ),
        )
        bbbc_output_dir = ensure_dir(bbbc_dir / "outputs")
        if bbbc_output_dir.exists():
            remove_path(bbbc_output_dir)
        bbbc_run = run_logged(
            [
                str(venv_python),
                "main.py",
                "--manifest",
                str((bbbc_dir / "manifest.csv").resolve()),
                "--study_output_dir",
                str(bbbc_output_dir.resolve()),
                "--focus_none",
                "--write_object_table",
                "--write_provenance",
                "--write_html_report",
                "--no_gpu",
            ],
            cwd=ROOT,
            log_path=bbbc_dir / "study_run_log.txt",
        )
        runtime_rows.append(
            {
                "stage": "bbbc039_smoke_run",
                "dataset": "bbbc039",
                "elapsed_seconds": round(bbbc_run["elapsed_seconds"], 2),
                "status": "ok" if bbbc_run["returncode"] == 0 else "failed",
                "notes": repo_relative(bbbc_dir / "study_run_log.txt"),
            }
        )
        if bbbc_run["returncode"] != 0:
            failure_rows.append(
                {
                    "dataset": "bbbc039",
                    "stage": "smoke_run",
                    "symptom": "BBBC039 smoke run failed.",
                    "suspected_cause": "Non-retina fluorescence sample incompatibility with the default pipeline assumptions.",
                    "mitigation": "Keep BBBC039 in the engineering appendix only.",
                    "claim_impact": "Does not affect retinal claims.",
                }
            )
        report_dir = evidence_root / "07_report_artifacts"
        archive_report_bundle(bbbc_output_dir, report_dir / "html_reports", "bbbc039")
        if (bbbc_output_dir / "provenance.json").exists():
            copy_file(bbbc_output_dir / "provenance.json", report_dir / "provenance_json" / "bbbc039.json")
        state["bbbc_output_dir"] = bbbc_output_dir

    fig4_dir = ensure_dir(evidence_root / "06_figures" / "fig4_registration_spatial_outputs")
    oir_output_dir = oir_dir / "outputs"
    copy_if_found(oir_output_dir, ["samples/*/retina_frames/*.json"], fig4_dir, "oir_retina_frame.json")
    copy_if_found(oir_output_dir, ["figures/*.png"], fig4_dir, "oir_summary_plot.png")

    fig5_dir = ensure_dir(evidence_root / "06_figures" / "fig5_failure_modes")
    write_text(
        fig5_dir / "README.md",
        textwrap.dedent(
            """\
            # Failure Modes

            See `paper_evidence/05_quantitative_results/failure_modes.csv` for the authoritative failure table.
            This folder is reserved for curated failure visuals; add screenshots manually if a later pass needs them.
            """
        ),
    )
    return state


def write_summary_files(
    evidence_root: Path,
    runtime_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    build_state: dict[str, Path],
) -> dict[str, Any]:
    runtime_path = evidence_root / "05_quantitative_results" / "runtime_summary.csv"
    failures_path = evidence_root / "05_quantitative_results" / "failure_modes.csv"
    write_csv(runtime_path, runtime_rows)
    write_csv(
        failures_path,
        failure_rows,
        fieldnames=["dataset", "stage", "symptom", "suspected_cause", "mitigation", "claim_impact"],
    )

    env_dir = evidence_root / "00_environment"
    pytest_log = env_dir / "pytest_log.txt"
    pytest_row = next((row for row in runtime_rows if row["stage"] == "pytest"), None)
    pytest_passed = bool(pytest_row and pytest_row["status"] == "ok" and pytest_log.exists())
    pytest_pass_count = read_pytest_pass_count(pytest_log)
    report_dir = evidence_root / "07_report_artifacts"
    tracked_report = report_dir / "html_reports" / "tracked_example" / "report.html"
    tracked_summary = evidence_root / "01_tables" / "tracked_example_study_summary.csv"
    if build_state.get("tracked_output_dir") and (build_state["tracked_output_dir"] / "study_summary.csv").exists():
        copy_file(build_state["tracked_output_dir"] / "study_summary.csv", tracked_summary)
    tracked_ok = tracked_summary.exists() and tracked_report.exists()
    manual_report = report_dir / "html_reports" / "tracked_example_manual_validation" / "report.html"
    single_report = report_dir / "html_reports" / "tracked_example_single_image" / "report.html"
    oir_report = report_dir / "html_reports" / "oir_flatmount" / "report.html"
    oir_ok = oir_report.exists()
    benchmark_ok = (evidence_root / "05_quantitative_results" / "count_error_metrics.csv").exists() and manual_report.exists()

    supported_rows = [
        {
            "claim": "The tracked engineering validation can be reproduced on a clean machine.",
            "status": "supported" if pytest_passed else "unsupported",
            "evidence_path": repo_relative(pytest_log),
            "notes": "Grounded in the clean evidence venv pytest run.",
        },
        {
            "claim": "The documented canonical tracked example runs end-to-end and emits reviewer-facing artifacts.",
            "status": "supported" if tracked_ok else "unsupported",
            "evidence_path": repo_relative(tracked_report),
            "notes": "Uses the exact documented command path and archives the report as a portable bundle.",
        },
        {
            "claim": "At least one real external public retinal dataset ran end-to-end through the workflow.",
            "status": "supported" if oir_ok else "unsupported",
            "evidence_path": repo_relative(oir_report),
            "notes": "This claim is about workflow execution and artifacts, not RGC count accuracy.",
        },
        {
            "claim": "The repo has at least one quantitative benchmark tied to tracked manual references, plus a separate labeled model-evaluation lane.",
            "status": "supported" if benchmark_ok else "unsupported",
            "evidence_path": repo_relative(evidence_root / "05_quantitative_results" / "count_error_metrics.csv"),
            "notes": "The paper-facing tracked benchmark uses manual counts; labeled TIFFs remain a separate model-evaluation fixture lane.",
        },
        {
            "claim": "Universal RGC counting superiority across modalities.",
            "status": "unsupported",
            "evidence_path": repo_relative(failures_path),
            "notes": "Do not claim this from the current evidence bundle.",
        },
        {
            "claim": "OIR-based RGC count accuracy.",
            "status": "unsupported due to no public ground truth",
            "evidence_path": repo_relative(evidence_root / "02_public_datasets" / "dataset_inventory.csv"),
            "notes": "OIR is used for external retinal workflow evidence only.",
        },
        {
            "claim": "Modality-agnostic performance across incompatible stains and channel layouts.",
            "status": "unsupported due to modality mismatch",
            "evidence_path": repo_relative(evidence_root / "02_public_datasets" / "moemil_samples" / "raw_manifest.csv"),
            "notes": "Compatibility must be stated per dataset and per stain/channel setup.",
        },
    ]
    supported_md = ["| claim | status | evidence_path | notes |", "| --- | --- | --- | --- |"]
    for row in supported_rows:
        supported_md.append(
            f"| {row['claim']} | {row['status']} | `{row['evidence_path']}` | {row['notes']} |"
        )
    write_text(evidence_root / "08_summary" / "paper_claims_supported.md", "\n".join(supported_md) + "\n")

    unsupported = textwrap.dedent(
        """\
        # Unsupported Claims

        - Universal RGC counting superiority over all comparator tools and modalities.
        - OIR-derived RGC count accuracy or state-of-the-art vessel benchmark claims.
        - Biologically validated dorsal/temporal regional interpretation for the OIR subset in this first pass, because dorsal orientation was heuristic.
        - Cross-modality claims that blur BRN3A-oriented public samples, vessel datasets, and BBBC039 nuclei tiles into one benchmark bucket.
        """
    )
    write_text(evidence_root / "08_summary" / "unsupported_claims.md", unsupported)

    failure_table = pd.read_csv(failures_path) if failures_path.exists() and failures_path.stat().st_size > 0 else pd.DataFrame()
    top_failures = []
    if not failure_table.empty:
        top_failures = [f"- {row['dataset']}: {row['symptom']}" for row in failure_table.head(5).to_dict("records")]
    else:
        top_failures = ["- No failures were recorded in `failure_modes.csv`."]

    figure_lines = []
    for label, folder in [
        ("Figure 1", evidence_root / "06_figures" / "fig1_pipeline_overview_inputs_outputs"),
        ("Figure 2", evidence_root / "06_figures" / "fig2_external_real_data_examples"),
        ("Figure 3", evidence_root / "06_figures" / "fig3_quantitative_validation"),
        ("Figure 4", evidence_root / "06_figures" / "fig4_registration_spatial_outputs"),
        ("Figure 5", evidence_root / "06_figures" / "fig5_failure_modes"),
    ]:
        if folder.exists():
            figure_lines.append(f"- {label}: `{repo_relative(folder)}`")

    quantitative_dir = evidence_root / "05_quantitative_results"
    fig3_dir = ensure_dir(evidence_root / "06_figures" / "fig3_quantitative_validation")
    copy_exact_file(report_dir / "html_reports" / "tracked_example_manual_validation", "validation/bland_altman.png", fig3_dir, "tracked_example_bland_altman.png")
    copy_exact_file(report_dir / "html_reports" / "tracked_example_manual_validation", "validation/agreement_scatter.png", fig3_dir, "tracked_example_agreement_scatter.png")
    if (quantitative_dir / "segmentation_metrics.csv").exists():
        copy_file(quantitative_dir / "segmentation_metrics.csv", fig3_dir / "segmentation_metrics.csv")

    top_failure_text = "\n".join(top_failures)
    figure_text = "\n".join(figure_lines) if figure_lines else "- Figure folders were created, but visual curation still needs a later pass."

    exec_summary = textwrap.dedent(
        f"""\
        # Executive Summary

        1. Did the pytest suite pass?
        {'Yes. The fresh evidence environment pytest run passed with ' + str(pytest_pass_count) + ' tests.' if pytest_pass_count is not None and pytest_passed else 'No. See `00_environment/pytest_log.txt` and `00_environment/pytest_failure_hypothesis.md`.'}

        2. Did the canonical tracked example run successfully?
        {'Yes.' if tracked_ok else 'No. Inspect `01_repo_validation/tracked_example_run_log.txt`.'}

        3. Did at least one real external public retinal dataset run end-to-end?
        {'Yes, the OIR external retinal pass produced report/provenance outputs.' if oir_ok else 'No, the OIR external retinal pass did not complete end-to-end.'}

        4. Did we obtain at least one quantitative benchmark with manual or reference data?
        {'Yes, the tracked manual-count benchmark produced benchmark files, and the model-evaluation helper remains a separate labeled-fixture lane.' if benchmark_ok else 'No, the benchmark files were not fully produced.'}

        5. Which claims are supported right now?
        - Reproducible engineering validation on a fresh venv, if pytest passed
        - Canonical tracked-example workflow with report, methods, provenance, and study outputs
        - External real-data workflow execution on OIR if the OIR report exists
        - Conservative quantitative validation from tracked manual references, plus a separate labeled model-evaluation lane
        - Single-image tracked output retained only as a QC/demo lane, not as a count-comparable validation surface

        6. Which claims are not yet supported and must not appear in the paper?
        - Universal RGC counting superiority
        - OIR-based RGC count accuracy
        - Modality-agnostic performance claims across incompatible stains/channels
        - Biologically validated dorsal interpretation for the heuristic OIR orientation pass

        7. What are the top 5 failure modes?
        {top_failure_text}

        8. Which 4 to 6 figures are already publication-ready?
        {figure_text}
        """
    )
    write_text(evidence_root / "08_summary" / "executive_summary.md", exec_summary)
    return {
        "pytest_passed": pytest_passed,
        "pytest_pass_count": pytest_pass_count,
        "tracked_report": tracked_report,
        "tracked_summary": tracked_summary,
        "manual_report": manual_report,
        "single_report": single_report,
        "oir_report": oir_report,
        "benchmark_ok": benchmark_ok,
    }


def build_run_manifest(
    *,
    evidence_root: Path,
    packet_root: Path,
    runtime_rows: list[dict[str, Any]],
    summary_state: dict[str, Any],
    repo_snapshot_path: Path | None,
    consistency: dict[str, Any] | None = None,
) -> dict[str, Any]:
    export_hashes = export_hash_rows(packet_root)
    payload: dict[str, Any] = {
        "repo_root": str(ROOT),
        "commit_hash": git_output("rev-parse", "HEAD"),
        "git_status_clean": not git_status_lines(),
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "commands": extract_stage_commands(runtime_rows),
        "pytest": {
            "passed": bool(summary_state.get("pytest_passed")),
            "passed_count": summary_state.get("pytest_pass_count"),
            "log_path": repo_relative(evidence_root / "00_environment" / "pytest_log.txt"),
        },
        "export_hashes": export_hashes,
    }
    if repo_snapshot_path is not None and repo_snapshot_path.exists():
        payload["repo_snapshot"] = {
            "relative_path": str(repo_snapshot_path.relative_to(packet_root)),
            "sha256": file_sha256(repo_snapshot_path),
        }
    else:
        payload["repo_snapshot"] = None
    if consistency is not None:
        payload["consistency"] = {
            "passed": bool(consistency.get("passed")),
            "issues": list(consistency.get("issues", [])),
        }
    return payload


def render_codex_report(summary_state: dict[str, Any], packet_root: Path) -> str:
    tracked_summary = pd.read_csv(summary_state["tracked_summary"])
    lines = [
        "# Codex Report For AI Advisor",
        "",
        "## What changed",
        "",
        "- Rebuilt the evidence bundle and advisor packet from one clean local commit and one full rerun.",
        "- The advisor packet is now exported directly from fresh archived report bundles and tables, not by copying older packet folders.",
        "- The single-image tracked lane is explicitly retained as a QC/demo lane and is not used for count-coherence claims.",
        "- The packet includes `run_manifest.json` plus an in-repo consistency check so stale mixed artifacts fail the export.",
        "",
        "## Verification I ran",
        "",
        f"- Clean evidence pytest: `{summary_state.get('pytest_pass_count')}` passed." if summary_state.get("pytest_pass_count") is not None else "- Clean evidence pytest count was not available.",
        "- Full tracked study, manual benchmark, model evaluation, OIR, and packet export rerun from the same checkout.",
        "- The exported advisor packet includes an in-repo consistency audit in `00_summary/consistency_report.json`.",
        "",
        "## Tracked study snapshot",
        "",
    ]
    for row in tracked_summary[["sample_id", "filename", "cell_count", "warning_count"]].to_dict("records"):
        lines.append(
            f"- `{row['sample_id']}` / `{row['filename']}`: `cell_count={int(row['cell_count'])}`, `warning_count={int(row['warning_count'])}`"
        )
    lines.extend(
        [
            "",
            "## Packet reading order",
            "",
            "1. `00_summary/codex_report.md`",
            "2. `00_summary/executive_summary.md`",
            "3. `00_summary/tracked_lane_comparison.md`",
            "4. `01_tables/tracked_example_study_summary.csv`",
            "5. `03_reports/tracked_example/report.html`",
            f"6. `{repo_relative(packet_root / 'run_manifest.json')}`",
            "",
        ]
    )
    return "\n".join(lines)


def export_advisor_packet(
    evidence_root: Path,
    runtime_rows: list[dict[str, Any]],
    summary_state: dict[str, Any],
    *,
    packet_root: Path,
) -> dict[str, Any]:
    if packet_root.exists():
        remove_path(packet_root)
    ensure_dir(packet_root / "00_summary")
    ensure_dir(packet_root / "01_tables")
    ensure_dir(packet_root / "02_images")
    ensure_dir(packet_root / "03_reports")

    for summary_name in ["executive_summary.md", "paper_claims_supported.md", "unsupported_claims.md"]:
        copy_file(evidence_root / "08_summary" / summary_name, packet_root / "00_summary" / summary_name)

    report_root = evidence_root / "07_report_artifacts" / "html_reports"
    for bundle_name in ["tracked_example", "tracked_example_manual_validation", "tracked_example_single_image", "oir_flatmount", "moemil_samples", "bbbc039"]:
        source_bundle = report_root / bundle_name
        if source_bundle.exists():
            copy_tree(source_bundle, packet_root / "03_reports" / bundle_name)

    table_sources = {
        "tracked_example_study_summary.csv": summary_state["tracked_summary"],
        "count_error_metrics.csv": evidence_root / "05_quantitative_results" / "count_error_metrics.csv",
        "runtime_summary.csv": evidence_root / "05_quantitative_results" / "runtime_summary.csv",
        "segmentation_metrics.csv": evidence_root / "05_quantitative_results" / "segmentation_metrics.csv",
        "registration_success_metrics.csv": evidence_root / "05_quantitative_results" / "registration_success_metrics.csv",
        "failure_modes.csv": evidence_root / "05_quantitative_results" / "failure_modes.csv",
        "dataset_inventory.csv": evidence_root / "02_public_datasets" / "dataset_inventory.csv",
    }
    for name, source in table_sources.items():
        if source.exists():
            copy_file(source, packet_root / "01_tables" / name)

    for target_rel, source_rel in TRACKED_FIGURE_MAPPINGS:
        copy_exact_file(packet_root, source_rel, packet_root, target_rel)

    for source_rel, destination_name in [
        ("03_reports/oir_flatmount/figures/cell_count_by_condition.png", "oir_summary.png"),
        ("03_reports/oir_flatmount/samples/OIR_01/registered_maps/example_retina_registered_density_map.png", "oir_registered_map.png"),
    ]:
        source_path = packet_root / source_rel
        if source_path.exists():
            copy_file(source_path, packet_root / "02_images" / destination_name)

    tracked_lane_md = build_tracked_lane_comparison_md(
        evidence_root / "07_report_artifacts" / "provenance_json" / "tracked_example.json",
        evidence_root / "07_report_artifacts" / "provenance_json" / "tracked_example_single_image.json",
        summary_state["tracked_summary"],
        summary_state["single_report"],
    )
    write_text(packet_root / "00_summary" / "tracked_lane_comparison.md", tracked_lane_md)
    write_text(packet_root / "00_summary" / "codex_report.md", render_codex_report(summary_state, packet_root))
    write_text(
        packet_root / "README.md",
        textwrap.dedent(
            """\
            # Send This Folder To The AI Advisor

            Open these first:
            - `00_summary/codex_report.md`
            - `00_summary/executive_summary.md`
            - `00_summary/tracked_lane_comparison.md`
            - `01_tables/tracked_example_study_summary.csv`
            - `03_reports/tracked_example/report.html`

            The `tracked_example_single_image` bundle is retained as a QC/demo lane only and is not count-comparable to the tracked study lane.
            """
        ),
    )

    repo_snapshot_path = build_repo_snapshot(ROOT / "codebase" / "retinal-phenotyper.txt")
    packet_snapshot_path: Path | None = None
    if repo_snapshot_path is not None and repo_snapshot_path.exists():
        packet_snapshot_path = packet_root / "04_repo_snapshot" / "retinal-phenotyper.txt"
        copy_file(repo_snapshot_path, packet_snapshot_path)

    preliminary_manifest = build_run_manifest(
        evidence_root=evidence_root,
        packet_root=packet_root,
        runtime_rows=runtime_rows,
        summary_state=summary_state,
        repo_snapshot_path=packet_snapshot_path,
        consistency=None,
    )
    write_json(packet_root / "run_manifest.json", preliminary_manifest)
    write_json(evidence_root / "run_manifest.json", preliminary_manifest)

    consistency = audit_advisor_packet(packet_root)
    final_manifest = build_run_manifest(
        evidence_root=evidence_root,
        packet_root=packet_root,
        runtime_rows=runtime_rows,
        summary_state=summary_state,
        repo_snapshot_path=packet_snapshot_path,
        consistency=consistency,
    )
    write_json(packet_root / "run_manifest.json", final_manifest)
    write_json(evidence_root / "run_manifest.json", final_manifest)
    write_json(packet_root / "00_summary" / "consistency_report.json", consistency)

    if not consistency["passed"]:
        return consistency

    archive_base = packet_root.parent / packet_root.name
    archive_path = shutil.make_archive(str(archive_base), "zip", root_dir=packet_root.parent, base_dir=packet_root.name)
    consistency["archive_path"] = archive_path
    return consistency


def create_tree(root: Path) -> None:
    for path in [
        root / "00_environment",
        root / "01_repo_validation" / "tracked_example_qc",
        root / "02_public_datasets" / "oir_flatmount" / "outputs",
        root / "02_public_datasets" / "moemil_samples" / "outputs",
        root / "02_public_datasets" / "bbbc039" / "outputs",
        root / "03_comparators" / "simplergc" / "outputs",
        root / "03_comparators" / "rgcode" / "outputs",
        root / "03_comparators" / "moemil" / "outputs",
        root / "04_manual_benchmark" / "manual_annotations",
        root / "05_quantitative_results",
        root / "06_figures" / "fig1_pipeline_overview_inputs_outputs",
        root / "06_figures" / "fig2_external_real_data_examples",
        root / "06_figures" / "fig3_quantitative_validation",
        root / "06_figures" / "fig4_registration_spatial_outputs",
        root / "06_figures" / "fig5_failure_modes",
        root / "07_report_artifacts" / "html_reports",
        root / "07_report_artifacts" / "methods_appendices",
        root / "07_report_artifacts" / "provenance_json",
        root / "08_summary",
    ]:
        ensure_dir(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the local-only paper evidence bundle.")
    parser.add_argument("--evidence-root", type=Path, default=ROOT / "paper_evidence")
    parser.add_argument("--advisor-packet-dir", type=Path, default=None)
    parser.add_argument("--stop-after-phase-a", action="store_true", help="Exit after clean-env pytest for debugging.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        require_clean_git_worktree()
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    evidence_root = args.evidence_root.resolve()
    packet_root = (args.advisor_packet_dir or (evidence_root / "10_ai_advisor_packet")).resolve()
    if evidence_root.exists():
        remove_path(evidence_root)
    create_tree(evidence_root)

    runtime_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    venv_python, install_ok = create_venv_and_install(evidence_root, runtime_rows, failure_rows)
    if not install_ok:
        write_summary_files(evidence_root, runtime_rows, failure_rows)
        return 1

    pytest_ok = run_phase_a_pytest(evidence_root, venv_python, runtime_rows, failure_rows)
    if not pytest_ok or args.stop_after_phase_a:
        write_summary_files(evidence_root, runtime_rows, failure_rows, {})
        return 1 if not pytest_ok else 0

    build_state: dict[str, Path] = {}
    tracked_state = run_repo_validation(evidence_root, venv_python, runtime_rows, failure_rows)
    if not tracked_state:
        write_summary_files(evidence_root, runtime_rows, failure_rows, build_state)
        return 1
    build_state.update(tracked_state)

    build_state.update(run_external_datasets_and_comparators(evidence_root, venv_python, runtime_rows, failure_rows))
    summary_state = write_summary_files(evidence_root, runtime_rows, failure_rows, build_state)
    consistency = export_advisor_packet(
        evidence_root,
        runtime_rows,
        summary_state,
        packet_root=packet_root,
    )
    return 0 if consistency.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
