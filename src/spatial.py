# src/spatial.py

from __future__ import annotations
from typing import Tuple, Dict, Any, List
import numpy as np
from scipy.spatial import cKDTree, Voronoi
from scipy.ndimage import gaussian_filter


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


def nn_regularity_index(centroids: np.ndarray) -> Dict[str, float]:
    """
    Nearest neighbor distances mean and CV. NNRI defined as mean / std.
    """
    if len(centroids) < 2:
        return {"mean": 0.0, "std": 0.0, "nnri": 0.0}

    tree = cKDTree(centroids)
    dists, _ = tree.query(centroids, k=2)
    nn = dists[:, 1]  # first is zero to self
    mean = float(nn.mean())
    std = float(nn.std()) if nn.std() > 1e-12 else 1e-12
    return {"mean": mean, "std": std, "nnri": mean / std}


def voronoi_regulariry_index(centroids: np.ndarray, shape: Tuple[int, int]) -> Dict[str, float]:
    """
    Compute the CV of Voronoi cell areas as a regularity measure.
    Lower CV = more regular mosaic. Report 1/CV as 'vdri'.
    """
    if len(centroids) < 3:
        return {"cv": 0.0, "vdri": 0.0}

    # Build Voronoi. Beware infinite regions; clip to image bounds.
    points = centroids[:, ::-1]  # Voronoi expects (x, y)
    v = Voronoi(points)
    # Compute finite region areas clipped to image rectangle
    areas = []
    w, h = shape[1], shape[0]
    rect = np.array([[0, 0], [w, 0], [w, h], [0, h]])

    # Utility to polygon area
    def poly_area(poly):
        x = poly[:, 0]
        y = poly[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    # Clip each finite region to the rectangle by a simple discard of infs
    for region_index in v.point_region:
        region = v.regions[region_index]
        if -1 in region or len(region) == 0:
            continue  # skip infinite or empty
        poly = np.array([v.vertices[i] for i in region], dtype=np.float64)
        # Clip by image bounds by discarding out-of-bounds vertices.
        # Note: this is a simplification. For precise clipping, use shapely.
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


def ripley_k(centroids: np.ndarray,
             radii: List[float],
             area_px: float) -> Dict[float, float]:
    """
    Very simple K-function without sophisticated edge correction.
    For production use, consider proper edge-correction.
    """
    n = len(centroids)
    if n < 2 or area_px <= 0:
        return {r: 0.0 for r in radii}
    tree = cKDTree(centroids)
    lamb = n / area_px  # intensity per pixel^2
    results = {}
    for r in radii:
        counts = tree.query_ball_point(centroids, r, return_length=True)
        # subtract self
        counts = np.array(counts) - 1
        k = counts.sum() / (n * lamb)
        results[r] = float(k)
    return results


def isodensity_map(centroids: np.ndarray,
                   shape: Tuple[int, int],
                   sigma_px: float = 50.0) -> np.ndarray:
    """
    Build a smoothed density map by splatting centroids and applying Gaussian blur.
    """
    m = np.zeros(shape, dtype=np.float32)
    yi = np.clip(centroids[:, 0].round().astype(int), 0, shape[0] - 1)
    xi = np.clip(centroids[:, 1].round().astype(int), 0, shape[1] - 1)
    m[yi, xi] = 1.0
    return gaussian_filter(m, sigma=sigma_px)

