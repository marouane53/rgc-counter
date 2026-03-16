from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import unary_union

from src.retina_coords import RetinaFrame
from src.schema import REGION_TABLE_COLUMNS, order_columns, validate_region_table


@dataclass(frozen=True)
class RegionSchema:
    name: str
    ring_edges_norm: tuple[float, ...]
    ring_labels: tuple[str, ...]
    peripapillary_norm: float
    sector_edges_deg: tuple[float, ...]
    sector_labels: tuple[str, ...]


SCHEMAS: dict[str, RegionSchema] = {
    "mouse_flatmount_v1": RegionSchema(
        name="mouse_flatmount_v1",
        ring_edges_norm=(0.33, 0.66, 1.0),
        ring_labels=("central", "pericentral", "peripheral"),
        peripapillary_norm=0.15,
        sector_edges_deg=(22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5, 360.0),
        sector_labels=(
            "temporal",
            "dorsal_temporal",
            "dorsal",
            "dorsal_nasal",
            "nasal",
            "ventral_nasal",
            "ventral",
            "ventral_temporal",
        ),
    ),
    "rat_flatmount_v1": RegionSchema(
        name="rat_flatmount_v1",
        ring_edges_norm=(0.4, 0.72, 1.0),
        ring_labels=("central", "midperipheral", "peripheral"),
        peripapillary_norm=0.18,
        sector_edges_deg=(22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5, 360.0),
        sector_labels=(
            "temporal",
            "dorsal_temporal",
            "dorsal",
            "dorsal_nasal",
            "nasal",
            "ventral_nasal",
            "ventral",
            "ventral_temporal",
        ),
    ),
}


def get_region_schema(name: str) -> RegionSchema:
    if name not in SCHEMAS:
        raise ValueError(f"Unknown region schema: {name}")
    return SCHEMAS[name]


def _normalized_eccentricity(ecc_um: pd.Series | np.ndarray, max_ecc_um: float) -> np.ndarray:
    if max_ecc_um <= 1e-12:
        return np.zeros(len(ecc_um), dtype=float)
    return np.clip(np.asarray(ecc_um, dtype=float) / max_ecc_um, 0.0, 1.0)


def _quadrant_labels(ret_x_um: pd.Series, ret_y_um: pd.Series) -> np.ndarray:
    x = np.asarray(ret_x_um, dtype=float)
    y = np.asarray(ret_y_um, dtype=float)
    labels = np.empty(len(x), dtype=object)
    labels[(x >= 0) & (y >= 0)] = "dorsal_temporal"
    labels[(x < 0) & (y >= 0)] = "dorsal_nasal"
    labels[(x < 0) & (y < 0)] = "ventral_nasal"
    labels[(x >= 0) & (y < 0)] = "ventral_temporal"
    return labels


def _sector_labels(theta_deg: pd.Series, schema: RegionSchema) -> np.ndarray:
    theta = np.asarray(theta_deg, dtype=float)
    idx = (((theta + 22.5) % 360.0) // 45.0).astype(int)
    idx = np.clip(idx, 0, len(schema.sector_labels) - 1)
    return np.asarray(schema.sector_labels, dtype=object)[idx]


def assign_regions(object_table: pd.DataFrame, *, schema_name: str, max_ecc_um: float) -> pd.DataFrame:
    schema = get_region_schema(schema_name)
    if object_table.empty:
        out = object_table.copy()
        for column in ["retina_region_schema", "region_schema", "normalized_ecc", "ring", "quadrant", "sector", "peripapillary_bin"]:
            out[column] = pd.Series(dtype="object")
        return out

    normalized_ecc = _normalized_eccentricity(object_table["ecc_um"], max_ecc_um)
    ring = pd.cut(
        normalized_ecc,
        bins=[-np.inf, *schema.ring_edges_norm],
        labels=schema.ring_labels,
        include_lowest=True,
    ).astype(str)
    quadrant = _quadrant_labels(object_table["ret_x_um"], object_table["ret_y_um"])
    sector = _sector_labels(object_table["theta_deg"], schema)
    peripapillary = np.where(normalized_ecc <= schema.peripapillary_norm, "peripapillary", "non_peripapillary")

    out = object_table.copy()
    out["retina_region_schema"] = schema.name
    out["region_schema"] = schema.name
    out["normalized_ecc"] = normalized_ecc
    out["ring"] = ring
    out["quadrant"] = quadrant
    out["sector"] = sector
    out["peripapillary_bin"] = peripapillary
    return out


def mask_to_polygon(mask: np.ndarray) -> Polygon | MultiPolygon:
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None or not contours:
        return Polygon()

    polygons: list[Polygon] = []
    hierarchy = hierarchy[0]
    for index, contour in enumerate(contours):
        if hierarchy[index][3] != -1:
            continue
        shell = contour[:, 0, :].astype(float)
        if len(shell) < 3:
            continue
        holes: list[list[tuple[float, float]]] = []
        child = hierarchy[index][2]
        while child != -1:
            hole = contours[child][:, 0, :].astype(float)
            if len(hole) >= 3:
                holes.append([(float(x), float(y)) for x, y in hole])
            child = hierarchy[child][0]
        polygon = Polygon([(float(x), float(y)) for x, y in shell], holes)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if not polygon.is_empty:
            polygons.append(polygon)
    if not polygons:
        return Polygon()
    merged = unary_union(polygons)
    return merged


def build_registered_region_masks(
    *,
    tissue_pixels: pd.DataFrame,
    tissue_mask: np.ndarray,
    schema_name: str,
    max_ecc_um: float,
) -> dict[tuple[str, str], np.ndarray]:
    if tissue_pixels.empty:
        return {}

    assigned = assign_regions(tissue_pixels, schema_name=schema_name, max_ecc_um=max_ecc_um)
    masks: dict[tuple[str, str], np.ndarray] = {}
    shape = tissue_mask.shape
    for axis in ("ring", "quadrant", "sector", "peripapillary_bin"):
        if axis not in assigned.columns:
            continue
        for label, frame in assigned.groupby(axis, dropna=False):
            domain_mask = np.zeros(shape, dtype=bool)
            ys = np.clip(frame["y_px"].astype(int).to_numpy(), 0, shape[0] - 1)
            xs = np.clip(frame["x_px"].astype(int).to_numpy(), 0, shape[1] - 1)
            domain_mask[ys, xs] = True
            domain_mask &= tissue_mask.astype(bool)
            masks[(axis, str(label))] = domain_mask
    return masks


def _polar_point(frame: RetinaFrame, radius_px: float, theta_deg: float) -> tuple[float, float]:
    theta = np.deg2rad(theta_deg)
    direction = np.cos(theta) * frame.temporal_xy_unit + np.sin(theta) * frame.dorsal_xy_unit
    point = frame.onh_xy_px + direction * float(radius_px)
    return float(point[0]), float(point[1])


def _sector_polygon(
    frame: RetinaFrame,
    *,
    inner_radius_px: float,
    outer_radius_px: float,
    start_deg: float,
    end_deg: float,
    resolution: int = 96,
) -> Polygon:
    span = end_deg - start_deg
    while span <= 0:
        span += 360.0
    if span >= 359.5:
        outer = Point(float(frame.onh_xy_px[0]), float(frame.onh_xy_px[1])).buffer(outer_radius_px, resolution=resolution)
        if inner_radius_px <= 0:
            return outer
        inner = Point(float(frame.onh_xy_px[0]), float(frame.onh_xy_px[1])).buffer(inner_radius_px, resolution=resolution)
        return outer.difference(inner)

    angles = np.linspace(start_deg, start_deg + span, max(8, int(resolution * span / 360.0)))
    outer_arc = [_polar_point(frame, outer_radius_px, angle) for angle in angles]
    inner_arc = [_polar_point(frame, inner_radius_px, angle) for angle in angles[::-1]] if inner_radius_px > 0 else [tuple(frame.onh_xy_px)]
    polygon = Polygon(outer_arc + inner_arc)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    return polygon


def _axis_polygons(schema: RegionSchema, frame: RetinaFrame, max_radius_px: float) -> dict[str, dict[str, Polygon]]:
    ring_polygons: dict[str, Polygon] = {}
    inner_norm = 0.0
    for outer_norm, label in zip(schema.ring_edges_norm, schema.ring_labels):
        ring_polygons[label] = _sector_polygon(
            frame,
            inner_radius_px=inner_norm * max_radius_px,
            outer_radius_px=outer_norm * max_radius_px,
            start_deg=0.0,
            end_deg=360.0,
        )
        inner_norm = outer_norm

    quadrant_polygons = {
        "dorsal_temporal": _sector_polygon(frame, inner_radius_px=0.0, outer_radius_px=max_radius_px, start_deg=0.0, end_deg=90.0),
        "dorsal_nasal": _sector_polygon(frame, inner_radius_px=0.0, outer_radius_px=max_radius_px, start_deg=90.0, end_deg=180.0),
        "ventral_nasal": _sector_polygon(frame, inner_radius_px=0.0, outer_radius_px=max_radius_px, start_deg=180.0, end_deg=270.0),
        "ventral_temporal": _sector_polygon(frame, inner_radius_px=0.0, outer_radius_px=max_radius_px, start_deg=270.0, end_deg=360.0),
    }

    sector_polygons: dict[str, Polygon] = {}
    start = -22.5
    for label in schema.sector_labels:
        sector_polygons[label] = _sector_polygon(
            frame,
            inner_radius_px=0.0,
            outer_radius_px=max_radius_px,
            start_deg=start,
            end_deg=start + 45.0,
        )
        start += 45.0

    peripapillary_polygons = {
        "peripapillary": _sector_polygon(
            frame,
            inner_radius_px=0.0,
            outer_radius_px=schema.peripapillary_norm * max_radius_px,
            start_deg=0.0,
            end_deg=360.0,
        ),
        "non_peripapillary": _sector_polygon(
            frame,
            inner_radius_px=schema.peripapillary_norm * max_radius_px,
            outer_radius_px=max_radius_px,
            start_deg=0.0,
            end_deg=360.0,
        ),
    }

    return {
        "ring": ring_polygons,
        "quadrant": quadrant_polygons,
        "sector": sector_polygons,
        "peripapillary_bin": peripapillary_polygons,
    }


def _area_px_from_polygons(region_polygon: Polygon, tissue_polygon: Polygon | MultiPolygon) -> float:
    if region_polygon.is_empty or tissue_polygon.is_empty:
        return 0.0
    return float(region_polygon.intersection(tissue_polygon).area)


def summarize_regions(
    *,
    object_table: pd.DataFrame,
    focus_pixels: pd.DataFrame,
    tissue_pixels: pd.DataFrame,
    tissue_mask: np.ndarray | None,
    frame: RetinaFrame,
    schema_name: str,
    source_path: str | Path,
) -> pd.DataFrame:
    schema = get_region_schema(schema_name)
    source_path = str(source_path)
    image_id = Path(source_path).name.rsplit(".", 1)[0]
    max_ecc_um = float(tissue_pixels["ecc_um"].max()) if not tissue_pixels.empty else float(focus_pixels["ecc_um"].max()) if not focus_pixels.empty else 0.0
    max_radius_px = max_ecc_um / frame.um_per_px if frame.um_per_px > 0 else 0.0
    tissue_polygon = mask_to_polygon(tissue_mask.astype(bool)) if tissue_mask is not None else Polygon()
    axis_polygons = _axis_polygons(schema, frame, max_radius_px)

    rows: list[dict[str, object]] = []
    for axis, label_polygons in axis_polygons.items():
        counts = object_table.groupby(axis).size().to_dict() if axis in object_table.columns else {}
        for label, polygon in label_polygons.items():
            area_px = _area_px_from_polygons(polygon, tissue_polygon) if tissue_mask is not None else 0.0
            area_mm2 = area_px * (frame.um_per_px ** 2) / 1e6
            object_count = int(counts.get(label, 0))
            density = float(object_count / area_mm2) if area_mm2 > 0 else 0.0
            rows.append(
                {
                    "image_id": image_id,
                    "source_path": source_path,
                    "retina_region_schema": schema.name,
                    "region_schema": schema.name,
                    "region_axis": axis,
                    "region_label": label,
                    "area_px": int(round(area_px)),
                    "area_mm2": float(area_mm2),
                    "object_count": object_count,
                    "density_cells_per_mm2": density,
                    "max_ecc_um": max_ecc_um,
                }
            )

    return order_columns(pd.DataFrame(rows), REGION_TABLE_COLUMNS)


def region_table_path_for(output_dir: str | Path, source_path: str | Path) -> Path:
    filename = Path(source_path).name.rsplit(".", 1)[0] + "_region_summary.csv"
    return Path(output_dir) / "regions" / filename


def write_region_table(frame: pd.DataFrame, destination: str | Path, *, strict: bool = False) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame = validate_region_table(frame, strict=strict)
    frame.to_csv(destination, index=False)
    return destination
