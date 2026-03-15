from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import MICRONS_PER_PIXEL
from src.landmarks import build_tissue_mask, detect_onh_hole


@dataclass
class RetinaFrame:
    onh_xy_px: np.ndarray
    dorsal_xy_px: np.ndarray
    dorsal_xy_unit: np.ndarray
    temporal_xy_unit: np.ndarray
    um_per_px: float
    source: str
    onh_source: str
    onh_confidence: float
    tissue_coverage_fraction: float = 0.0

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "onh_x_px": float(self.onh_xy_px[0]),
            "onh_y_px": float(self.onh_xy_px[1]),
            "dorsal_x_px": float(self.dorsal_xy_px[0]),
            "dorsal_y_px": float(self.dorsal_xy_px[1]),
            "dorsal_unit_x": float(self.dorsal_xy_unit[0]),
            "dorsal_unit_y": float(self.dorsal_xy_unit[1]),
            "temporal_unit_x": float(self.temporal_xy_unit[0]),
            "temporal_unit_y": float(self.temporal_xy_unit[1]),
            "um_per_px": float(self.um_per_px),
            "source": self.source,
            "onh_source": self.onh_source,
            "onh_confidence": float(self.onh_confidence),
            "tissue_coverage_fraction": float(self.tissue_coverage_fraction),
        }


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        raise ValueError("Retina orientation vector cannot be zero.")
    return vec / norm


def _infer_um_per_px(meta: dict[str, Any], fallback: float = MICRONS_PER_PIXEL) -> float:
    x = meta.get("microns_per_pixel_x")
    y = meta.get("microns_per_pixel_y")
    if x and y:
        return float((float(x) + float(y)) / 2.0)
    if x:
        return float(x)
    if y:
        return float(y)
    return float(fallback)


def retina_frame_from_points(
    *,
    onh_xy_px: tuple[float, float],
    dorsal_xy_px: tuple[float, float],
    um_per_px: float,
    source: str,
    onh_source: str | None = None,
    onh_confidence: float = 1.0,
    tissue_coverage_fraction: float = 0.0,
) -> RetinaFrame:
    onh = np.asarray(onh_xy_px, dtype=float)
    dorsal_point = np.asarray(dorsal_xy_px, dtype=float)
    dorsal_unit = _normalize(dorsal_point - onh)
    temporal_unit = _normalize(np.array([-dorsal_unit[1], dorsal_unit[0]], dtype=float))
    return RetinaFrame(
        onh_xy_px=onh,
        dorsal_xy_px=dorsal_point,
        dorsal_xy_unit=dorsal_unit,
        temporal_xy_unit=temporal_unit,
        um_per_px=float(um_per_px),
        source=source,
        onh_source=onh_source or source,
        onh_confidence=float(onh_confidence),
        tissue_coverage_fraction=float(tissue_coverage_fraction),
    )


def _frame_with_tissue_metrics(frame: RetinaFrame, gray_image: np.ndarray | None) -> RetinaFrame:
    if gray_image is None:
        return frame
    tissue_mask = build_tissue_mask(gray_image)
    ys, xs = np.where(tissue_mask)
    if len(xs) == 0:
        frame.tissue_coverage_fraction = 0.0
        return frame
    distances = np.sqrt((xs.astype(float) - frame.onh_xy_px[0]) ** 2 + (ys.astype(float) - frame.onh_xy_px[1]) ** 2)
    max_radius_px = float(distances.max()) if len(distances) else 0.0
    ideal_area = np.pi * (max_radius_px ** 2)
    frame.tissue_coverage_fraction = float(tissue_mask.sum() / ideal_area) if ideal_area > 0 else 0.0
    return frame


def load_retina_frame_json(path: str | Path) -> RetinaFrame:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if {"onh_x_px", "onh_y_px", "dorsal_x_px", "dorsal_y_px"}.issubset(payload):
        return retina_frame_from_points(
            onh_xy_px=(payload["onh_x_px"], payload["onh_y_px"]),
            dorsal_xy_px=(payload["dorsal_x_px"], payload["dorsal_y_px"]),
            um_per_px=float(payload.get("um_per_px", MICRONS_PER_PIXEL)),
            source=str(payload.get("source", f"sidecar:{Path(path)}")),
            onh_source=str(payload.get("onh_source", f"sidecar:{Path(path)}")),
            onh_confidence=float(payload.get("onh_confidence", 1.0)),
            tissue_coverage_fraction=float(payload.get("tissue_coverage_fraction", 0.0)),
        )
    if {"onh_x_px", "onh_y_px", "dorsal_unit_x", "dorsal_unit_y"}.issubset(payload):
        onh = np.asarray([payload["onh_x_px"], payload["onh_y_px"]], dtype=float)
        dorsal_point = onh + np.asarray([payload["dorsal_unit_x"], payload["dorsal_unit_y"]], dtype=float) * 100.0
        return retina_frame_from_points(
            onh_xy_px=(float(onh[0]), float(onh[1])),
            dorsal_xy_px=(float(dorsal_point[0]), float(dorsal_point[1])),
            um_per_px=float(payload.get("um_per_px", MICRONS_PER_PIXEL)),
            source=str(payload.get("source", f"sidecar:{Path(path)}")),
            onh_source=str(payload.get("onh_source", f"sidecar:{Path(path)}")),
            onh_confidence=float(payload.get("onh_confidence", 1.0)),
            tissue_coverage_fraction=float(payload.get("tissue_coverage_fraction", 0.0)),
        )
    raise ValueError("Retina frame JSON is missing required dorsal landmark information.")


def default_retina_frame_candidates(image_path: str | Path) -> list[Path]:
    image_path = Path(image_path)
    stem = image_path.name.rsplit(".", 1)[0]
    return [
        image_path.with_name(f"{stem}.retina_frame.json"),
        image_path.parent / "retina_frame.json",
        image_path.parent.parent / "retina_frame.json",
        image_path.parent.parent / "retina_frame.reference.json",
    ]


def _load_sidecar_landmarks(image_path: str | Path, retina_frame_path: str | None) -> tuple[tuple[float, float], tuple[float, float], float, str] | None:
    candidates = [Path(retina_frame_path)] if retina_frame_path else default_retina_frame_candidates(image_path)
    for candidate in candidates:
        if candidate.exists():
            frame = load_retina_frame_json(candidate)
            return (
                (float(frame.onh_xy_px[0]), float(frame.onh_xy_px[1])),
                (float(frame.dorsal_xy_px[0]), float(frame.dorsal_xy_px[1])),
                float(frame.um_per_px),
                str(candidate),
            )
    return None


def resolve_retina_frame(
    *,
    image_path: str | Path,
    gray_image: np.ndarray | None,
    meta: dict[str, Any],
    onh_mode: str,
    onh_xy: tuple[float, float] | None,
    dorsal_xy: tuple[float, float] | None,
    retina_frame_path: str | None,
) -> RetinaFrame:
    um_per_px = _infer_um_per_px(meta)
    if onh_mode == "cli":
        if onh_xy is None or dorsal_xy is None:
            raise ValueError("CLI retina registration requires both --onh_xy and --dorsal_xy.")
        return _frame_with_tissue_metrics(
            retina_frame_from_points(
                onh_xy_px=onh_xy,
                dorsal_xy_px=dorsal_xy,
                um_per_px=um_per_px,
                source="cli",
                onh_source="cli",
            ),
            gray_image,
        )

    if onh_mode == "sidecar":
        sidecar = _load_sidecar_landmarks(image_path, retina_frame_path)
        if sidecar is None:
            raise FileNotFoundError(
                "Could not find a retina frame sidecar. Provide --retina_frame_path or use --onh_mode cli."
            )
        sidecar_onh, sidecar_dorsal, sidecar_um_per_px, sidecar_source = sidecar
        return _frame_with_tissue_metrics(
            retina_frame_from_points(
                onh_xy_px=sidecar_onh,
                dorsal_xy_px=sidecar_dorsal,
                um_per_px=sidecar_um_per_px,
                source=f"sidecar:{sidecar_source}",
                onh_source=f"sidecar:{sidecar_source}",
            ),
            gray_image,
        )

    if onh_mode in {"auto_hole", "auto_combined"}:
        if gray_image is None:
            raise ValueError(f"{onh_mode} requires a grayscale image for ONH detection.")
        detected_onh, info = detect_onh_hole(gray_image)
        if detected_onh is None:
            raise RuntimeError("Failed auto_hole ONH detection.")

        resolved_dorsal = dorsal_xy
        dorsal_source = "cli"
        if resolved_dorsal is None and onh_mode == "auto_combined":
            sidecar = _load_sidecar_landmarks(image_path, retina_frame_path)
            if sidecar is not None:
                _, sidecar_dorsal, _, sidecar_source = sidecar
                resolved_dorsal = sidecar_dorsal
                dorsal_source = f"sidecar:{sidecar_source}"
        if resolved_dorsal is None:
            raise ValueError(f"{onh_mode} requires dorsal orientation via --dorsal_xy or sidecar fallback.")

        return _frame_with_tissue_metrics(
            retina_frame_from_points(
                onh_xy_px=detected_onh,
                dorsal_xy_px=resolved_dorsal,
                um_per_px=um_per_px,
                source=f"{onh_mode}:{dorsal_source}",
                onh_source=str(info.get("method", onh_mode)),
                onh_confidence=float(info.get("confidence", 0.0)),
            ),
            gray_image,
        )

    raise ValueError(f"Unsupported onh_mode: {onh_mode}")


def register_cells(cell_df: pd.DataFrame, frame: RetinaFrame) -> pd.DataFrame:
    if cell_df.empty:
        out = cell_df.copy()
        for column in ["ret_x_um", "ret_y_um", "ecc_um", "theta_deg"]:
            out[column] = pd.Series(dtype=float)
        return out

    xy = cell_df[["centroid_x_px", "centroid_y_px"]].to_numpy(dtype=float)
    rel = xy - frame.onh_xy_px[None, :]
    temporal = rel @ frame.temporal_xy_unit
    dorsal = rel @ frame.dorsal_xy_unit

    out = cell_df.copy()
    out["ret_x_um"] = temporal * frame.um_per_px
    out["ret_y_um"] = dorsal * frame.um_per_px
    out["ecc_um"] = np.hypot(out["ret_x_um"], out["ret_y_um"])
    out["theta_deg"] = (np.degrees(np.arctan2(out["ret_y_um"], out["ret_x_um"])) + 360.0) % 360.0
    return out


def register_focus_mask_pixels(focus_mask: np.ndarray, frame: RetinaFrame) -> pd.DataFrame:
    ys, xs = np.where(focus_mask)
    if len(xs) == 0:
        return pd.DataFrame(columns=["x_px", "y_px", "ret_x_um", "ret_y_um", "ecc_um", "theta_deg"])

    xy = np.column_stack([xs.astype(float), ys.astype(float)])
    rel = xy - frame.onh_xy_px[None, :]
    temporal = rel @ frame.temporal_xy_unit
    dorsal = rel @ frame.dorsal_xy_unit
    ret_x_um = temporal * frame.um_per_px
    ret_y_um = dorsal * frame.um_per_px
    ecc_um = np.hypot(ret_x_um, ret_y_um)
    theta_deg = (np.degrees(np.arctan2(ret_y_um, ret_x_um)) + 360.0) % 360.0
    return pd.DataFrame(
        {
            "x_px": xs,
            "y_px": ys,
            "ret_x_um": ret_x_um,
            "ret_y_um": ret_y_um,
            "ecc_um": ecc_um,
            "theta_deg": theta_deg,
        }
    )


def retina_frame_output_path(output_dir: str | Path, source_path: str | Path) -> Path:
    filename = Path(source_path).name.rsplit(".", 1)[0] + "_retina_frame.json"
    return Path(output_dir) / "retina_frames" / filename


def write_retina_frame_json(frame: RetinaFrame, destination: str | Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(frame.to_json_dict(), indent=2), encoding="utf-8")
    return destination


def registered_density_plot_path(output_dir: str | Path, source_path: str | Path, suffix: str = ".png") -> Path:
    filename = Path(source_path).name.rsplit(".", 1)[0] + f"_registered_density_map{suffix}"
    return Path(output_dir) / "registered_maps" / filename


def save_registered_density_plot(object_table: pd.DataFrame, destination: str | Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    if not object_table.empty:
        hb = ax.hexbin(
            object_table["ret_x_um"],
            object_table["ret_y_um"],
            gridsize=28,
            mincnt=1,
            cmap="viridis",
        )
        fig.colorbar(hb, ax=ax, label="Cell count")
    ax.scatter([0], [0], color="crimson", s=40, label="ONH")
    ax.axhline(0, color="#888", linewidth=0.8)
    ax.axvline(0, color="#888", linewidth=0.8)
    ax.set_xlabel("Temporal-Nasal (um)")
    ax.set_ylabel("Dorsal-Ventral (um)")
    ax.set_title("Registered Retina Density Map")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(destination, dpi=200)
    plt.close(fig)
    return destination
