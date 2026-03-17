from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.spatial import Voronoi, cKDTree
from shapely.geometry import MultiPolygon, Polygon, box

from src.config import MICRONS_PER_PIXEL
from src.regions import build_registered_region_masks, mask_to_polygon


DEFAULT_RIGOROUS_RADII_PX: tuple[float, ...] = (25.0, 50.0, 75.0, 100.0, 150.0, 200.0)
MIN_RIGOROUS_POINTS = 5


@dataclass(frozen=True)
class SpatialDomain:
    image_id: str
    analysis_level: str
    region_axis: str
    region_label: str
    mask: np.ndarray
    polygon: Polygon | MultiPolygon
    area_px: float
    area_mm2: float
    domain_source: str
    random_seed: int


@dataclass(frozen=True)
class RigorousSpatialResult:
    summary_row: dict[str, Any]
    curve_frame: pd.DataFrame


def centroids_from_masks(masks: np.ndarray) -> np.ndarray:
    """Return Nx2 array of (y, x) centroids for labeled mask."""
    ids = np.unique(masks)
    ids = ids[ids != 0]
    cents = []
    for oid in ids:
        ys, xs = np.where(masks == oid)
        if len(ys) == 0:
            continue
        cents.append([float(ys.mean()), float(xs.mean())])
    return np.array(cents, dtype=np.float32)


def nn_regularity_index(centroids: np.ndarray) -> dict[str, float]:
    """Nearest-neighbor distances mean and CV. NNRI defined as mean / std."""
    if len(centroids) < 2:
        return {"mean": 0.0, "std": 0.0, "nnri": 0.0}

    tree = cKDTree(centroids)
    dists, _ = tree.query(centroids, k=2)
    nn = dists[:, 1]
    mean = float(nn.mean())
    std = float(nn.std()) if nn.std() > 1e-12 else 1e-12
    return {"mean": mean, "std": std, "nnri": mean / std}


def voronoi_regulariry_index(centroids: np.ndarray, shape: tuple[int, int]) -> dict[str, float]:
    """
    Legacy Voronoi regularity measure.

    This intentionally preserves the previous simplified vertex-clamping behavior
    for backward-compatible `legacy` mode outputs.
    """
    if len(centroids) < 3:
        return {"cv": 0.0, "vdri": 0.0}

    points = centroids[:, ::-1]
    v = Voronoi(points)
    areas = []
    w, h = shape[1], shape[0]

    def poly_area(poly: np.ndarray) -> float:
        x = poly[:, 0]
        y = poly[:, 1]
        return float(0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))

    for region_index in v.point_region:
        region = v.regions[region_index]
        if -1 in region or len(region) == 0:
            continue
        poly = np.array([v.vertices[i] for i in region], dtype=np.float64)
        poly[:, 0] = np.clip(poly[:, 0], 0, w)
        poly[:, 1] = np.clip(poly[:, 1], 0, h)
        if len(poly) >= 3:
            areas.append(poly_area(poly))

    areas = np.array(areas, dtype=np.float64)
    if len(areas) == 0:
        return {"cv": 0.0, "vdri": 0.0}

    mean = areas.mean()
    std = areas.std() if areas.std() > 1e-12 else 1e-12
    cv = float(std / mean) if mean > 1e-12 else 0.0
    return {"cv": cv, "vdri": float(1.0 / cv) if cv > 0 else 0.0}


def ripley_k(centroids: np.ndarray, radii: Sequence[float], area_px: float) -> dict[float, float]:
    """Legacy K-function without edge correction."""
    n = len(centroids)
    if n < 2 or area_px <= 0:
        return {float(r): 0.0 for r in radii}
    tree = cKDTree(centroids)
    intensity = n / area_px
    results: dict[float, float] = {}
    for radius in radii:
        counts = np.asarray(tree.query_ball_point(centroids, radius, return_length=True), dtype=float) - 1.0
        k = counts.sum() / (n * intensity)
        results[float(radius)] = float(k)
    return results


def isodensity_map(centroids: np.ndarray, shape: tuple[int, int], sigma_px: float = 50.0) -> np.ndarray:
    """Build a smoothed density map by splatting centroids and applying Gaussian blur."""
    m = np.zeros(shape, dtype=np.float32)
    if len(centroids) == 0:
        return m
    yi = np.clip(np.round(centroids[:, 0]).astype(int), 0, shape[0] - 1)
    xi = np.clip(np.round(centroids[:, 1]).astype(int), 0, shape[1] - 1)
    m[yi, xi] = 1.0
    return gaussian_filter(m, sigma=sigma_px)


def _stable_hash(*parts: object) -> int:
    joined = "|".join(str(part) for part in parts)
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def _coerce_polygon(geometry: Polygon | MultiPolygon | None) -> Polygon | MultiPolygon:
    if geometry is None:
        return Polygon()
    if not geometry.is_valid:
        geometry = geometry.buffer(0)
    return geometry


def _points_in_mask(points_yx: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if len(points_yx) == 0:
        return np.zeros(0, dtype=bool)
    ys = np.clip(np.round(points_yx[:, 0]).astype(int), 0, mask.shape[0] - 1)
    xs = np.clip(np.round(points_yx[:, 1]).astype(int), 0, mask.shape[1] - 1)
    return mask[ys, xs].astype(bool)


def kept_object_table(object_table: pd.DataFrame) -> pd.DataFrame:
    if object_table.empty:
        return object_table.copy()
    if "kept" not in object_table.columns:
        return object_table.copy()
    kept_mask = object_table["kept"].fillna(True).astype(bool)
    return object_table.loc[kept_mask].copy()


def rigorous_points_from_object_table(
    object_table: pd.DataFrame,
    *,
    level: str,
    axis: str | None = None,
    label: str | None = None,
) -> np.ndarray:
    frame = kept_object_table(object_table)
    if frame.empty:
        return np.empty((0, 2), dtype=float)
    if level == "global":
        return frame[["centroid_y_px", "centroid_x_px"]].to_numpy(dtype=float)
    if axis is None or label is None or axis not in frame.columns:
        return np.empty((0, 2), dtype=float)
    subset = frame.loc[frame[axis].astype(str) == str(label)]
    if subset.empty:
        return np.empty((0, 2), dtype=float)
    return subset[["centroid_y_px", "centroid_x_px"]].to_numpy(dtype=float)


def _sample_points_from_mask(mask: np.ndarray, count: int, seed: int) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(xs) < count or count <= 0:
        return np.empty((0, 2), dtype=np.float64)
    rng = np.random.default_rng(seed)
    take = rng.choice(len(xs), size=count, replace=False)
    return np.column_stack((ys[take].astype(float), xs[take].astype(float)))


def _domain_seed(image_id: str, region_axis: str, region_label: str, base_seed: int) -> int:
    return int(base_seed + (_stable_hash(image_id, region_axis, region_label) % 1_000_000))


def _voronoi_finite_polygons_2d(vor: Voronoi, radius: float | None = None) -> tuple[list[list[int]], np.ndarray]:
    if vor.points.shape[1] != 2:
        raise ValueError("Voronoi input must be 2D.")

    new_regions: list[list[int]] = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    if radius is None:
        radius = float(vor.points.ptp().max() * 2.0) if len(vor.points) else 1.0

    all_ridges: dict[int, list[tuple[int, int, int]]] = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region_index in enumerate(vor.point_region):
        vertices = vor.regions[region_index]
        if all(vertex >= 0 for vertex in vertices):
            new_regions.append(vertices)
            continue

        ridges = all_ridges.get(p1, [])
        new_region = [vertex for vertex in vertices if vertex >= 0]
        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue
            tangent = vor.points[p2] - vor.points[p1]
            tangent /= np.linalg.norm(tangent)
            normal = np.array([-tangent[1], tangent[0]])
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, normal)) * normal
            far_point = vor.vertices[v2] + direction * radius
            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        region_vertices = np.asarray([new_vertices[vertex] for vertex in new_region])
        region_center = region_vertices.mean(axis=0)
        angles = np.arctan2(region_vertices[:, 1] - region_center[1], region_vertices[:, 0] - region_center[0])
        new_region = [vertex for _, vertex in sorted(zip(angles, new_region))]
        new_regions.append(new_region)

    return new_regions, np.asarray(new_vertices, dtype=float)


def exact_voronoi_clipped_areas(
    centroids: np.ndarray,
    domain_polygon: Polygon | MultiPolygon,
    *,
    image_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    if len(centroids) < 3:
        return np.array([], dtype=float)

    polygon = _coerce_polygon(domain_polygon)
    if polygon.is_empty and image_shape is not None:
        polygon = box(0, 0, image_shape[1], image_shape[0])
    if polygon.is_empty:
        return np.array([], dtype=float)

    points_xy = np.asarray(centroids[:, ::-1], dtype=float)
    vor = Voronoi(points_xy)
    regions, vertices = _voronoi_finite_polygons_2d(vor)

    areas: list[float] = []
    for region in regions:
        poly = Polygon(vertices[region])
        if not poly.is_valid:
            poly = poly.buffer(0)
        clipped = poly.intersection(polygon)
        if clipped.is_empty:
            continue
        area = float(clipped.area)
        if area > 0:
            areas.append(area)
    return np.asarray(areas, dtype=float)


def rigorous_voronoi_metrics(
    centroids: np.ndarray,
    domain_polygon: Polygon | MultiPolygon,
    *,
    image_shape: tuple[int, int] | None = None,
) -> dict[str, float]:
    areas = exact_voronoi_clipped_areas(centroids, domain_polygon, image_shape=image_shape)
    if len(areas) < 2:
        return {"cv": float("nan"), "vdri": float("nan")}
    mean = float(np.mean(areas))
    if mean <= 1e-12:
        return {"cv": float("nan"), "vdri": float("nan")}
    std = float(np.std(areas, ddof=1)) if len(areas) > 1 else 0.0
    cv = std / mean if mean > 0 else float("nan")
    return {
        "cv": float(cv),
        "vdri": float(1.0 / cv) if np.isfinite(cv) and cv > 0 else float("nan"),
    }


def exact_voronoi_clip_area(poly_xy: np.ndarray, *, width: int, height: int, domain_polygon: Polygon | MultiPolygon | None = None) -> float:
    polygon = Polygon(poly_xy)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    clip_domain = _coerce_polygon(domain_polygon) if domain_polygon is not None else box(0, 0, width, height)
    if clip_domain.is_empty:
        clip_domain = box(0, 0, width, height)
    clipped = polygon.intersection(clip_domain)
    return float(clipped.area) if not clipped.is_empty else 0.0


def _border_corrected_l_values(points_yx: np.ndarray, domain_mask: np.ndarray, radii_px: Sequence[float]) -> np.ndarray:
    radii = np.asarray(radii_px, dtype=float)
    values = np.full(len(radii), np.nan, dtype=float)
    n_points = len(points_yx)
    if n_points < MIN_RIGOROUS_POINTS:
        return values

    area_px = float(domain_mask.sum())
    if area_px <= 0:
        return values

    distance_map = distance_transform_edt(domain_mask.astype(bool))
    tree = cKDTree(points_yx)
    ys = np.clip(np.round(points_yx[:, 0]).astype(int), 0, domain_mask.shape[0] - 1)
    xs = np.clip(np.round(points_yx[:, 1]).astype(int), 0, domain_mask.shape[1] - 1)
    boundary_distances = distance_map[ys, xs]

    for index, radius in enumerate(radii):
        eligible = boundary_distances >= radius
        eligible_points = points_yx[eligible]
        n_eligible = len(eligible_points)
        if n_eligible == 0:
            continue
        counts = np.asarray(tree.query_ball_point(eligible_points, radius, return_length=True), dtype=float) - 1.0
        k_border = area_px * counts.sum() / (n_points * n_eligible)
        if k_border < 0:
            continue
        values[index] = float(np.sqrt(k_border / np.pi) - radius)
    return values


def _pair_correlation_values(points_yx: np.ndarray, domain_mask: np.ndarray, radii_px: Sequence[float]) -> np.ndarray:
    radii = np.asarray(radii_px, dtype=float)
    values = np.full(len(radii), np.nan, dtype=float)
    n_points = len(points_yx)
    if n_points < MIN_RIGOROUS_POINTS:
        return values

    area_px = float(domain_mask.sum())
    if area_px <= 0:
        return values

    distance_map = distance_transform_edt(domain_mask.astype(bool))
    tree = cKDTree(points_yx)
    ys = np.clip(np.round(points_yx[:, 0]).astype(int), 0, domain_mask.shape[0] - 1)
    xs = np.clip(np.round(points_yx[:, 1]).astype(int), 0, domain_mask.shape[1] - 1)
    boundary_distances = distance_map[ys, xs]

    prev_radius = 0.0
    for index, radius in enumerate(radii):
        eligible = boundary_distances >= radius
        eligible_points = points_yx[eligible]
        n_eligible = len(eligible_points)
        if n_eligible == 0:
            prev_radius = float(radius)
            continue
        outer = np.asarray(tree.query_ball_point(eligible_points, radius, return_length=True), dtype=float) - 1.0
        inner = np.asarray(tree.query_ball_point(eligible_points, prev_radius, return_length=True), dtype=float) - 1.0
        annulus_counts = np.clip(outer - inner, 0.0, None)
        annulus_area = np.pi * ((float(radius) ** 2) - (float(prev_radius) ** 2))
        if annulus_area <= 0:
            prev_radius = float(radius)
            continue
        values[index] = float(area_px * annulus_counts.sum() / (n_points * n_eligible * annulus_area))
        prev_radius = float(radius)
    return values


def simulate_csr_points(domain_mask: np.ndarray, n_points: int, *, seed: int) -> np.ndarray:
    return _sample_points_from_mask(domain_mask.astype(bool), n_points, seed)


def compute_csr_envelopes(
    points_yx: np.ndarray,
    domain_mask: np.ndarray,
    *,
    radii_px: Sequence[float],
    simulation_count: int,
    seed: int,
) -> dict[str, Any]:
    radii = np.asarray(radii_px, dtype=float)
    observed_l = _border_corrected_l_values(points_yx, domain_mask, radii)
    observed_g = _pair_correlation_values(points_yx, domain_mask, radii)

    if len(points_yx) < MIN_RIGOROUS_POINTS or domain_mask.sum() <= 0 or simulation_count <= 0:
        return {
            "l_obs": observed_l,
            "g_obs": observed_g,
            "l_env_low": np.full(len(radii), np.nan, dtype=float),
            "l_env_high": np.full(len(radii), np.nan, dtype=float),
            "g_env_low": np.full(len(radii), np.nan, dtype=float),
            "g_env_high": np.full(len(radii), np.nan, dtype=float),
            "l_outside_envelope_any": False,
            "l_global_p_value": float("nan"),
            "l_max_abs_deviation": float("nan"),
            "g_peak_value": float("nan"),
        }

    l_sims: list[np.ndarray] = []
    g_sims: list[np.ndarray] = []
    sim_stats: list[float] = []

    for sim_index in range(simulation_count):
        sim_points = simulate_csr_points(domain_mask, len(points_yx), seed=seed + sim_index)
        sim_l = _border_corrected_l_values(sim_points, domain_mask, radii)
        sim_g = _pair_correlation_values(sim_points, domain_mask, radii)
        l_sims.append(sim_l)
        g_sims.append(sim_g)
        if np.isfinite(sim_l).any():
            sim_stats.append(float(np.nanmax(np.abs(sim_l))))

    l_stack = np.vstack(l_sims) if l_sims else np.empty((0, len(radii)))
    g_stack = np.vstack(g_sims) if g_sims else np.empty((0, len(radii)))
    l_env_low = _safe_percentile(l_stack, 2.5) if l_sims else np.full(len(radii), np.nan, dtype=float)
    l_env_high = _safe_percentile(l_stack, 97.5) if l_sims else np.full(len(radii), np.nan, dtype=float)
    g_env_low = _safe_percentile(g_stack, 2.5) if g_sims else np.full(len(radii), np.nan, dtype=float)
    g_env_high = _safe_percentile(g_stack, 97.5) if g_sims else np.full(len(radii), np.nan, dtype=float)

    outside = bool(
        np.any(
            np.isfinite(observed_l)
            & np.isfinite(l_env_low)
            & np.isfinite(l_env_high)
            & ((observed_l < l_env_low) | (observed_l > l_env_high))
        )
    )
    observed_stat = float(np.nanmax(np.abs(observed_l))) if np.isfinite(observed_l).any() else float("nan")
    if np.isfinite(observed_stat) and sim_stats:
        exceed = sum(stat >= observed_stat for stat in sim_stats)
        p_value = float((1 + exceed) / (simulation_count + 1))
    else:
        p_value = float("nan")

    return {
        "l_obs": observed_l,
        "g_obs": observed_g,
        "l_env_low": l_env_low,
        "l_env_high": l_env_high,
        "g_env_low": g_env_low,
        "g_env_high": g_env_high,
        "l_outside_envelope_any": outside,
        "l_global_p_value": p_value,
        "l_max_abs_deviation": observed_stat,
        "g_peak_value": float(np.nanmax(observed_g)) if np.isfinite(observed_g).any() else float("nan"),
    }


def _safe_percentile(stack: np.ndarray, percentile: float) -> np.ndarray:
    if stack.size == 0:
        return np.array([], dtype=float)
    rows, cols = stack.shape
    out = np.full(cols, np.nan, dtype=float)
    for idx in range(cols):
        column = stack[:, idx]
        if np.isfinite(column).any():
            out[idx] = float(np.nanpercentile(column, percentile))
    return out


def _domain_status(domain_mask: np.ndarray, point_count: int) -> str:
    if float(domain_mask.sum()) <= 0:
        return "empty_domain"
    if point_count < MIN_RIGOROUS_POINTS:
        return "insufficient_points"
    return "ok"


def _coerce_bool_mask(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if mask is None:
        return np.zeros(shape, dtype=bool)
    return np.asarray(mask, dtype=bool)


def build_spatial_domains(
    *,
    image_id: str,
    tissue_mask: np.ndarray,
    um_per_px: float,
    base_seed: int,
    registered_tissue_pixels: pd.DataFrame | None = None,
    schema_name: str | None = None,
    max_ecc_um: float | None = None,
) -> list[SpatialDomain]:
    domains: list[SpatialDomain] = []
    global_mask = _coerce_bool_mask(tissue_mask, tissue_mask.shape)
    global_polygon = _coerce_polygon(mask_to_polygon(global_mask))
    domains.append(
        SpatialDomain(
            image_id=image_id,
            analysis_level="global",
            region_axis="global",
            region_label="global",
            mask=global_mask,
            polygon=global_polygon,
            area_px=float(global_mask.sum()),
            area_mm2=float(global_mask.sum()) * (float(um_per_px) ** 2) / 1e6,
            domain_source="tissue_mask",
            random_seed=_domain_seed(image_id, "global", "global", base_seed),
        )
    )

    if registered_tissue_pixels is None or registered_tissue_pixels.empty or not schema_name or max_ecc_um is None:
        return domains

    region_masks = build_registered_region_masks(
        tissue_pixels=registered_tissue_pixels,
        tissue_mask=global_mask,
        schema_name=schema_name,
        max_ecc_um=float(max_ecc_um),
    )
    for (axis, label), region_mask in sorted(region_masks.items()):
        polygon = _coerce_polygon(mask_to_polygon(region_mask))
        domains.append(
            SpatialDomain(
                image_id=image_id,
                analysis_level="region",
                region_axis=str(axis),
                region_label=str(label),
                mask=np.asarray(region_mask, dtype=bool),
                polygon=polygon,
                area_px=float(np.asarray(region_mask, dtype=bool).sum()),
                area_mm2=float(np.asarray(region_mask, dtype=bool).sum()) * (float(um_per_px) ** 2) / 1e6,
                domain_source="registered_region",
                random_seed=_domain_seed(image_id, axis, label, base_seed),
            )
        )
    return domains


def analyze_rigorous_domain(
    domain: SpatialDomain,
    *,
    points_yx: np.ndarray,
    image_shape: tuple[int, int],
    radii_px: Sequence[float],
    simulation_count: int,
) -> RigorousSpatialResult:
    points = np.asarray(points_yx, dtype=float)
    status = _domain_status(domain.mask, len(points))
    legacy_nn = nn_regularity_index(points)
    legacy_vd = voronoi_regulariry_index(points, image_shape)
    rigorous_voronoi = rigorous_voronoi_metrics(points, domain.polygon, image_shape=image_shape)
    envelopes = compute_csr_envelopes(
        points,
        domain.mask,
        radii_px=radii_px,
        simulation_count=simulation_count,
        seed=domain.random_seed,
    )

    curve_rows: list[dict[str, Any]] = []
    for radius, l_obs, l_low, l_high, g_obs, g_low, g_high in zip(
        radii_px,
        envelopes["l_obs"],
        envelopes["l_env_low"],
        envelopes["l_env_high"],
        envelopes["g_obs"],
        envelopes["g_env_low"],
        envelopes["g_env_high"],
    ):
        curve_rows.append(
            {
                "image_id": domain.image_id,
                "analysis_level": domain.analysis_level,
                "region_axis": domain.region_axis,
                "region_label": domain.region_label,
                "radius_px": float(radius),
                "l_obs": float(l_obs) if np.isfinite(l_obs) else float("nan"),
                "l_env_low": float(l_low) if np.isfinite(l_low) else float("nan"),
                "l_env_high": float(l_high) if np.isfinite(l_high) else float("nan"),
                "g_obs": float(g_obs) if np.isfinite(g_obs) else float("nan"),
                "g_env_low": float(g_low) if np.isfinite(g_low) else float("nan"),
                "g_env_high": float(g_high) if np.isfinite(g_high) else float("nan"),
                "simulation_count": int(simulation_count),
                "random_seed": int(domain.random_seed),
            }
        )

    summary_row = {
        "image_id": domain.image_id,
        "spatial_mode": "rigorous",
        "analysis_level": domain.analysis_level,
        "region_axis": domain.region_axis,
        "region_label": domain.region_label,
        "n_points": int(len(points)),
        "domain_area_px": float(domain.area_px),
        "domain_area_mm2": float(domain.area_mm2),
        "domain_source": domain.domain_source,
        "status": status,
        "legacy_nnri": float(legacy_nn["nnri"]),
        "legacy_vdri": float(legacy_vd["vdri"]),
        "rigorous_voronoi_cv": float(rigorous_voronoi["cv"]) if np.isfinite(rigorous_voronoi["cv"]) else float("nan"),
        "rigorous_vdri": float(rigorous_voronoi["vdri"]) if np.isfinite(rigorous_voronoi["vdri"]) else float("nan"),
        "l_outside_envelope_any": bool(envelopes["l_outside_envelope_any"]),
        "l_global_p_value": float(envelopes["l_global_p_value"]) if np.isfinite(envelopes["l_global_p_value"]) else float("nan"),
        "l_max_abs_deviation": float(envelopes["l_max_abs_deviation"]) if np.isfinite(envelopes["l_max_abs_deviation"]) else float("nan"),
        "g_peak_value": float(envelopes["g_peak_value"]) if np.isfinite(envelopes["g_peak_value"]) else float("nan"),
        "simulation_count": int(simulation_count),
        "random_seed": int(domain.random_seed),
    }
    return RigorousSpatialResult(summary_row=summary_row, curve_frame=pd.DataFrame(curve_rows))


def compute_rigorous_spatial_bundle(
    *,
    image_id: str,
    object_table: pd.DataFrame,
    image_shape: tuple[int, int],
    tissue_mask: np.ndarray,
    um_per_px: float | None = None,
    registered_tissue_pixels: pd.DataFrame | None = None,
    schema_name: str | None = None,
    max_ecc_um: float | None = None,
    radii_px: Sequence[float] = DEFAULT_RIGOROUS_RADII_PX,
    simulation_count: int = 999,
    base_seed: int = 1337,
) -> dict[str, Any]:
    resolved_um_per_px = float(um_per_px if um_per_px is not None else MICRONS_PER_PIXEL)

    domains = build_spatial_domains(
        image_id=image_id,
        tissue_mask=tissue_mask,
        um_per_px=resolved_um_per_px,
        base_seed=base_seed,
        registered_tissue_pixels=registered_tissue_pixels,
        schema_name=schema_name,
        max_ecc_um=max_ecc_um,
    )

    results = [
        analyze_rigorous_domain(
            domain,
            points_yx=rigorous_points_from_object_table(
                object_table,
                level="global" if domain.analysis_level == "global" else "region",
                axis=None if domain.analysis_level == "global" else domain.region_axis,
                label=None if domain.analysis_level == "global" else domain.region_label,
            ),
            image_shape=image_shape,
            radii_px=radii_px,
            simulation_count=simulation_count,
        )
        for domain in domains
    ]
    summary = pd.DataFrame([result.summary_row for result in results])
    curves = pd.concat([result.curve_frame for result in results], ignore_index=True) if results else pd.DataFrame()
    global_row = summary[(summary["analysis_level"] == "global") & (summary["region_axis"] == "global")].head(1)
    global_summary = {}
    if not global_row.empty:
        row = global_row.iloc[0]
        global_summary = {
            "spatial_mode": "rigorous",
            "rigorous_global_point_count": int(row["n_points"]),
            "rigorous_global_vdri": float(row["rigorous_vdri"]) if pd.notna(row["rigorous_vdri"]) else float("nan"),
            "rigorous_global_l_outside_envelope": bool(row["l_outside_envelope_any"]),
            "rigorous_global_l_global_p_value": float(row["l_global_p_value"]) if pd.notna(row["l_global_p_value"]) else float("nan"),
            "rigorous_global_l_max_abs_deviation": float(row["l_max_abs_deviation"]) if pd.notna(row["l_max_abs_deviation"]) else float("nan"),
            "rigorous_global_g_peak_value": float(row["g_peak_value"]) if pd.notna(row["g_peak_value"]) else float("nan"),
        }

    return {
        "summary": summary,
        "curves": curves,
        "global_summary": global_summary,
        "spatial_analysis": {
            "mode": "rigorous",
            "radii_px": [float(radius) for radius in radii_px],
            "simulation_count": int(simulation_count),
            "random_seed": int(base_seed),
            "regionwise_analysis_run": bool((summary["analysis_level"] == "region").any()) if not summary.empty else False,
        },
    }


def _plot_curve_frame(
    curve_frame: pd.DataFrame,
    destination: str | Path,
    *,
    value_column: str,
    low_column: str,
    high_column: str,
    ylabel: str,
    title: str,
) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    if not curve_frame.empty:
        radii = curve_frame["radius_px"].to_numpy(dtype=float)
        observed = curve_frame[value_column].to_numpy(dtype=float)
        low = curve_frame[low_column].to_numpy(dtype=float)
        high = curve_frame[high_column].to_numpy(dtype=float)
        plt.plot(radii, observed, label="Observed", color="#1f77b4")
        plt.fill_between(radii, low, high, alpha=0.25, color="#ff7f0e", label="CSR envelope")
    plt.axhline(0.0, color="#999999", linewidth=1.0, linestyle="--")
    plt.xlabel("Radius (px)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(destination, dpi=180)
    plt.close()
    return destination


def save_ripley_l_plot(curve_frame: pd.DataFrame, destination: str | Path) -> Path:
    return _plot_curve_frame(
        curve_frame,
        destination,
        value_column="l_obs",
        low_column="l_env_low",
        high_column="l_env_high",
        ylabel="Ripley L(r)",
        title="Global Ripley L",
    )


def save_pair_correlation_plot(curve_frame: pd.DataFrame, destination: str | Path) -> Path:
    return _plot_curve_frame(
        curve_frame,
        destination,
        value_column="g_obs",
        low_column="g_env_low",
        high_column="g_env_high",
        ylabel="Pair correlation g(r)",
        title="Global Pair Correlation",
    )


def spatial_summary_output_path(output_dir: str | Path, source_path: str | Path) -> Path:
    return Path(output_dir) / "spatial" / f"{Path(source_path).name.rsplit('.', 1)[0]}_spatial_summary.csv"


def spatial_curves_output_path(output_dir: str | Path, source_path: str | Path) -> Path:
    return Path(output_dir) / "spatial" / f"{Path(source_path).name.rsplit('.', 1)[0]}_spatial_curves.csv"


def ripley_l_plot_output_path(output_dir: str | Path, source_path: str | Path) -> Path:
    return Path(output_dir) / "spatial" / f"{Path(source_path).name.rsplit('.', 1)[0]}_ripley_l_global.png"


def pair_correlation_plot_output_path(output_dir: str | Path, source_path: str | Path) -> Path:
    return Path(output_dir) / "spatial" / f"{Path(source_path).name.rsplit('.', 1)[0]}_pair_correlation_global.png"


def write_spatial_summary(frame: pd.DataFrame, destination: str | Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return destination


def write_spatial_curves(frame: pd.DataFrame, destination: str | Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return destination
