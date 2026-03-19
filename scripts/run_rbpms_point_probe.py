from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.micro_roi_benchmark import load_or_build_projection_lab_config_manifest
from src.point_detection import detect_dog_peaks, detect_hmax_peaks, detect_log_peaks


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _load_detector_input(path: Path) -> np.ndarray:
    return np.asarray(tifffile.imread(str(path)))


def _load_exclude_mask(path: str | Path | None) -> np.ndarray | None:
    if path is None or (isinstance(path, float) and np.isnan(path)) or not str(path).strip():
        return None
    resolved = Path(path)
    if not resolved.exists():
        return None
    return np.asarray(tifffile.imread(str(resolved))).astype(bool)


def _detect_points(image: np.ndarray, exclude_mask: np.ndarray | None, backend: str, config: dict) -> pd.DataFrame:
    normalized = str(backend).strip().lower()
    if normalized == "log":
        return detect_log_peaks(image, exclude_mask=exclude_mask, **config)
    if normalized == "dog":
        return detect_dog_peaks(image, exclude_mask=exclude_mask, **config)
    if normalized == "hmax":
        return detect_hmax_peaks(image, exclude_mask=exclude_mask, **config)
    raise ValueError(f"Unsupported predictor backend: {backend}")


def _save_overlay(image: np.ndarray, points: pd.DataFrame, destination: Path, title: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image, cmap="gray")
    if not points.empty:
        ax.scatter(points["x_px"], points["y_px"], s=24, marker="+", c="yellow")
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the private point-detector probe on projection-lab views.")
    parser.add_argument("--view-manifest", required=True, type=Path)
    parser.add_argument("--config-manifest", default=None, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    views = pd.read_csv(args.view_manifest)
    configs = load_or_build_projection_lab_config_manifest(str(args.config_manifest) if args.config_manifest is not None else None)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configs.to_csv(output_dir / "resolved_config_manifest.csv", index=False)

    _log(
        f"[run_rbpms_point_probe] Starting point probe with {len(configs)} config(s) over {len(views)} view row(s)"
    )
    rows: list[dict[str, object]] = []
    completed_runs = 0
    total_runs = 0
    matching_views_by_config: list[tuple[dict[str, object], pd.DataFrame]] = []
    for config in configs.to_dict("records"):
        projection_recipe_json = str(config["projection_recipe_json"])
        preprocess_json = str(config["preprocess_json"])
        matching_views = views.loc[
            (views["projection_recipe_json"].astype(str) == projection_recipe_json)
            & (views["preprocess_json"].astype(str) == preprocess_json)
        ].copy()
        matching_views_by_config.append((config, matching_views))
        total_runs += len(matching_views)
    _log(f"[run_rbpms_point_probe] Planned detector runs: {total_runs}")

    for index, (config, matching_views) in enumerate(matching_views_by_config, start=1):
        if matching_views.empty:
            _log(f"[run_rbpms_point_probe] [{index}/{len(configs)}] {config['config_id']}: 0 matching views, skipping")
            continue
        predictor_backend = str(config["predictor_backend"])
        predictor_config = json.loads(str(config["predictor_config_json"]))
        _log(f"[run_rbpms_point_probe] [{index}/{len(configs)}] {config['config_id']}: {len(matching_views)} view(s)")
        for view_index, (_, view) in enumerate(matching_views.iterrows(), start=1):
            roi_id = str(view["roi_id"])
            _log(
                f"[run_rbpms_point_probe] run {completed_runs + 1}/{total_runs}: "
                f"{config['config_id']} on ROI {roi_id} ({view_index}/{len(matching_views)} in config)"
            )
            image = _load_detector_input(Path(str(view["detector_input_path"])))
            exclude_mask = _load_exclude_mask(view.get("exclude_mask_path"))
            points = _detect_points(image, exclude_mask, predictor_backend, predictor_config)
            run_dir = output_dir / "runs" / str(config["config_id"]) / roi_id
            run_dir.mkdir(parents=True, exist_ok=True)
            points_path = run_dir / f"{roi_id}__points.csv"
            overlay_path = run_dir / f"{roi_id}__overlay.png"
            points.to_csv(points_path, index=False)
            _save_overlay(image, points, overlay_path, title=f"{roi_id} | {config['config_id']}")
            score_mean = float(points["score"].mean()) if not points.empty else float("nan")
            score_p90 = float(np.percentile(points["score"], 90.0)) if not points.empty else float("nan")
            rows.append(
                {
                    "config_id": str(config["config_id"]),
                    "roi_id": roi_id,
                    "split": str(view.get("split", "")),
                    "projection_id": str(view["projection_id"]),
                    "preprocess_id": str(view["preprocess_id"]),
                    "predictor_backend": predictor_backend,
                    "point_count": int(len(points)),
                    "score_mean": score_mean,
                    "score_p90": score_p90,
                    "points_csv_path": str(points_path),
                    "overlay_png_path": str(overlay_path),
                    "detector_input_path": str(view["detector_input_path"]),
                }
            )
            completed_runs += 1
            _log(
                f"[run_rbpms_point_probe] completed {completed_runs}/{total_runs}: "
                f"{config['config_id']} ROI {roi_id} -> {len(points)} point(s)"
            )

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(["roi_id", "score_p90", "score_mean", "point_count", "config_id"], ascending=[True, False, False, True, True]).reset_index(drop=True)
        summary["rank_within_roi"] = summary.groupby("roi_id", sort=False).cumcount() + 1
    summary.to_csv(output_dir / "probe_summary.csv", index=False)
    if summary.empty:
        aggregate = pd.DataFrame(columns=["config_id", "predictor_backend", "n_rois", "point_count_mean", "score_mean_mean", "score_p90_mean"])
    else:
        aggregate = (
            summary.groupby(["config_id", "predictor_backend"], dropna=False)
            .agg(
                n_rois=("roi_id", "nunique"),
                point_count_mean=("point_count", "mean"),
                score_mean_mean=("score_mean", "mean"),
                score_p90_mean=("score_p90", "mean"),
            )
            .reset_index()
            .sort_values(["score_p90_mean", "score_mean_mean", "point_count_mean", "config_id"], ascending=[False, False, True, True])
            .reset_index(drop=True)
        )
        aggregate["rank"] = np.arange(1, len(aggregate) + 1)
    aggregate.to_csv(output_dir / "config_summary.csv", index=False)
    (output_dir / "review_index.md").write_text(
        "# RBPMS Point Probe Review Index\n\n"
        f"- Configs: `{len(configs)}`\n"
        f"- Runs: `{len(summary)}`\n\n"
        + ("_No runs._\n" if summary.empty else summary.to_markdown(index=False))
        + "\n",
        encoding="utf-8",
    )
    _log(
        f"[run_rbpms_point_probe] Finished: wrote {len(summary)} run row(s) to {output_dir / 'probe_summary.csv'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
