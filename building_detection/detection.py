"""
Core building detection algorithm.

Detects ancient building foundations as man-made ground irregularities in
LiDAR-derived DEMs, in two shape families that share one pass over the data:

    rectangular  edges meeting at right angles (the original detector)
    circular     edges with no preferred direction, at a constant radius --
                 round platforms, ring walls, chultun mouths, altars

Both families read the same per-candidate edge sample, from opposite ends:
the rectangular path wants two sharp perpendicular peaks in the direction
distribution, the circular path wants that distribution to be flat.  Circular
detection is opt-in via BuildingConfig.detect_circular; with it off the
pipeline behaves exactly as before.  See circular.py for the round-shape
metrics.

Pipeline:
    1. Build fine-resolution DEM (1m, min Z per cell)
    2. Compute local relative elevation (DEM - smoothed DEM)
    3. Threshold and extract connected components (candidate regions)
    4. Compute gradient field (Sobel on relative elevation)
    5. Per-candidate edge sample -> angle score (rectangular) and
       isotropy (circular)
    6. Rectangle and annulus template scoring
    7. Per-family filtering, then a shape label from the two template scores
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    find_objects,
    gaussian_filter,
    label as ndimage_label,
    sobel,
)

from .circular import CircularMetrics, compute_circular_metrics
from .metrics import BuildingCandidate, BuildingDetectionResult


@dataclass
class BuildingConfig:
    """All tunable parameters for building detection."""
    # DEM construction
    dem_cell_size: float = 2.0

    # Local relative elevation
    smoothing_radius_m: float = 25.0        # Gaussian sigma for background

    # Elevation thresholding
    min_relative_elevation: float = 0.3     # metres above background
    max_relative_elevation: float = 3.0     # metres; taller = tree not ruin

    # Connected component size filters (m²)
    min_cluster_area_m2: float = 25.0       # 5x5m minimum
    max_cluster_area_m2: float = 900.0      # 30x30m maximum

    # Rectangularity (relaxed -- angle score is now primary)
    min_rectangularity: float = 0.20

    # Wall signal
    min_wall_score: float = 0.10

    # Height
    min_peak_height: float = 0.4
    max_peak_height: float = 3.0

    # Aspect ratio of bounding rectangle
    max_aspect_ratio: float = 5.0

    # Point density filter (edges vs interior)
    min_edge_density_ratio: float = 1.0

    # Anti-crop-field: mean elevation of cluster above background
    min_mean_relative_elevation: float = 0.2

    # Gradient / angle analysis (NEW)
    min_gradient_strength: float = 0.05     # min Sobel magnitude for edge pixel
    gradient_angle_bins: int = 90           # 2-degree resolution
    angle_tolerance_deg: float = 30.0       # tolerance for "aligned" edges
    min_angle_score: float = 0.05            # min right-angle signal
    min_template_score: float = 0.0         # min template match

    # Subsurface mode (for detecting buried structures)
    subsurface_mode: bool = False
    subsurface_max_depth: float = 2.0       # max depth below ground to search (m)

    # Shape families to report ------------------------------------------------
    detect_rectangular: bool = True         # right-angle structures (default)
    detect_circular: bool = False           # round platforms / ring structures

    # Circular filters (only used when detect_circular is on)
    min_circular_score: float = 0.45        # combined circular evidence (0-1)
    min_radial_symmetry: float = 0.35       # constancy of boundary radius
    min_directional_isotropy: float = 0.30  # flat gradient-direction histogram
    min_circ_template_score: float = 0.0    # annular template match
    max_circular_aspect_ratio: float = 1.6  # round means roughly as wide as tall
    min_circular_diameter_m: float = 8.0    # ~4 cells at the default 2m DEM;
                                            # below this a circle and a blob
                                            # are indistinguishable

    # When a region satisfies both families, the circular template correlation
    # must beat the rectangular one by this relative margin to win the label.
    circular_preference_margin: float = 0.10


# ---------------------------------------------------------------------------
# DEM construction
# ---------------------------------------------------------------------------

def build_fine_dem(
    points: np.ndarray,
    cell_size: float = 1.0,
) -> Tuple[np.ndarray, float, float]:
    """
    Build a raster DEM using minimum Z per cell (ground surface proxy).

    Returns (dem, x_origin, y_origin).
    NaN marks cells with no points.
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    x_origin = float(x.min())
    y_origin = float(y.min())

    cols = ((x - x_origin) / cell_size).astype(np.intp)
    rows = ((y - y_origin) / cell_size).astype(np.intp)

    n_rows = int(rows.max()) + 1
    n_cols = int(cols.max()) + 1

    dem = np.full((n_rows, n_cols), np.inf, dtype=np.float64)
    np.minimum.at(dem, (rows, cols), z)
    dem[dem == np.inf] = np.nan

    return dem, x_origin, y_origin


def gaussian_smooth_dem(dem: np.ndarray, sigma_cells: float) -> np.ndarray:
    """
    NaN-aware Gaussian smoothing of a DEM.

    Uses weighted averaging: smooth(values * weights) / smooth(weights)
    so NaN cells contribute zero weight.
    """
    nan_mask = np.isnan(dem)
    dem_filled = dem.copy()
    dem_filled[nan_mask] = 0.0
    weights = (~nan_mask).astype(np.float64)

    smoothed_num = gaussian_filter(dem_filled, sigma=sigma_cells)
    smoothed_den = gaussian_filter(weights, sigma=sigma_cells)
    smoothed_den = np.maximum(smoothed_den, 1e-10)

    background = smoothed_num / smoothed_den
    background[nan_mask] = np.nan
    return background


# ---------------------------------------------------------------------------
# Subsurface depth map
# ---------------------------------------------------------------------------

def _build_subsurface_depth_map(
    points: np.ndarray,
    ground_surface: np.ndarray,
    x_origin: float,
    y_origin: float,
    cell_size: float,
    max_depth: float,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Filter LiDAR points to those below the ground surface and build a depth map.

    Returns (depth_map, subsurface_dem, n_subsurface_points).
    depth_map has positive values = depth below ground surface.
    Returns all-NaN arrays if too few subsurface points.
    """
    n_rows, n_cols = ground_surface.shape

    # Look up smoothed ground Z at each point's XY location
    pt_cols = ((points[:, 0] - x_origin) / cell_size).astype(np.intp)
    pt_rows = ((points[:, 1] - y_origin) / cell_size).astype(np.intp)
    pt_cols = np.clip(pt_cols, 0, n_cols - 1)
    pt_rows = np.clip(pt_rows, 0, n_rows - 1)
    ground_z = ground_surface[pt_rows, pt_cols]

    # Keep points that are 0 to max_depth metres below ground
    valid = ~np.isnan(ground_z)
    depth_per_point = ground_z - points[:, 2]   # positive = below ground
    subsurface_mask = valid & (depth_per_point > 0) & (depth_per_point <= max_depth)

    n_sub = int(subsurface_mask.sum())
    empty = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
    if n_sub < 100:
        return empty, empty, n_sub

    # Build DEM from subsurface points (min Z = deepest return per cell)
    sub_z = points[subsurface_mask, 2]
    sub_cols = pt_cols[subsurface_mask]
    sub_rows = pt_rows[subsurface_mask]

    sub_dem = np.full((n_rows, n_cols), np.inf, dtype=np.float64)
    np.minimum.at(sub_dem, (sub_rows, sub_cols), sub_z)
    sub_dem[sub_dem == np.inf] = np.nan

    # Depth = ground_surface - subsurface_dem  (positive = deeper)
    depth_map = ground_surface - sub_dem
    depth_map[np.isnan(sub_dem)] = np.nan

    return depth_map, sub_dem, n_sub


# ---------------------------------------------------------------------------
# Gradient and angle analysis (NEW)
# ---------------------------------------------------------------------------

# How far the edge sample reaches past a candidate's own cells, in DEM cells.
DILATION_ITERATIONS = 3


def compute_gradient_field(rel_elev: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute gradient magnitude and direction from relative elevation.

    Returns (magnitude, direction) where direction is in radians [0, pi).
    NaN pixels produce zero magnitude and NaN direction.
    """
    clean = rel_elev.copy()
    nan_mask = np.isnan(clean)
    clean[nan_mask] = 0.0

    gx = sobel(clean, axis=1)  # horizontal gradient
    gy = sobel(clean, axis=0)  # vertical gradient

    magnitude = np.sqrt(gx**2 + gy**2)
    # Map to [0, pi) since edge direction is ambiguous by 180 degrees
    direction = np.arctan2(gy, gx) % np.pi

    magnitude[nan_mask] = 0.0
    direction[nan_mask] = np.nan

    return magnitude, direction


def extract_strong_gradients(
    magnitude: np.ndarray,
    direction: np.ndarray,
    component_mask: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Collect the edge pixels that carry a candidate region's shape signal.

    Dilates the component to reach its boundary, then keeps the pixels whose
    gradient magnitude is in the top half for that region.

    Returns (directions, magnitudes) or None when there is too little signal.
    Both shape families read this same sample, so it is extracted once per
    candidate.

    The work is done inside the component's bounding box, padded by the
    dilation radius. The dilation can reach at most that far, so everything
    outside the padded box is False regardless -- but a tile raster runs to
    millions of cells and a candidate to a few dozen, and dilating the whole
    raster once per candidate costs more than the rest of the pipeline
    combined. Results are identical to operating on the full array.
    """
    rows, cols = np.where(component_mask)
    if rows.size == 0:
        return None

    pad = DILATION_ITERATIONS
    r0 = max(0, int(rows.min()) - pad)
    r1 = min(component_mask.shape[0], int(rows.max()) + 1 + pad)
    c0 = max(0, int(cols.min()) - pad)
    c1 = min(component_mask.shape[1], int(cols.max()) + 1 + pad)

    sub_mask = component_mask[r0:r1, c0:c1]
    sub_magnitude = magnitude[r0:r1, c0:c1]
    sub_direction = direction[r0:r1, c0:c1]

    # Dilate to capture edges at boundary of the component
    analysis_mask = binary_dilation(sub_mask, iterations=pad)

    # Adaptive threshold: use top 50% of gradients in this region
    region_mags = sub_magnitude[analysis_mask]
    region_mags = region_mags[region_mags > 0]
    if len(region_mags) < 10:
        return None

    adaptive_threshold = float(np.percentile(region_mags, 50))
    strong = (sub_magnitude > adaptive_threshold) & analysis_mask
    if strong.sum() < 10:
        return None

    return sub_direction[strong], sub_magnitude[strong]


def build_direction_histogram(
    dirs: np.ndarray,
    mags: np.ndarray,
    n_bins: int = 36,
) -> np.ndarray:
    """Magnitude-weighted, circularly smoothed histogram of directions in [0, pi)."""
    bin_edges = np.linspace(0, np.pi, n_bins + 1)
    hist, _ = np.histogram(dirs, bins=bin_edges, weights=mags)

    # Circular smoothing with wider kernel for 36 bins
    kernel_width = 2
    kernel = np.exp(-0.5 * np.arange(-kernel_width, kernel_width + 1)**2)
    kernel /= kernel.sum()
    padded = np.concatenate([hist[-kernel_width:], hist, hist[:kernel_width]])
    return np.convolve(padded, kernel, mode='valid')[:n_bins]


def compute_direction_histogram(
    magnitude: np.ndarray,
    direction: np.ndarray,
    component_mask: np.ndarray,
    n_bins: int = 36,
) -> Optional[np.ndarray]:
    """
    Build the magnitude-weighted, circularly smoothed histogram of gradient
    directions in [0, pi) for one candidate region.

    Returns None when the region carries too little edge signal to analyse.
    """
    sample = extract_strong_gradients(magnitude, direction, component_mask)
    if sample is None:
        return None
    return build_direction_histogram(sample[0], sample[1], n_bins)


def compute_angle_score(
    magnitude: np.ndarray,
    direction: np.ndarray,
    component_mask: np.ndarray,
    min_gradient_strength: float = 0.05,
    n_bins: int = 36,
    smoothed_hist: Optional[np.ndarray] = None,
) -> Tuple[float, float, float]:
    """
    Analyse gradient directions within a candidate region.

    Buildings produce two sharp peaks separated by ~90 degrees in the
    direction histogram; natural features produce diffuse histograms.

    Pass ``smoothed_hist`` from compute_direction_histogram() to reuse an
    already-built histogram; otherwise it is built here.

    Returns (angle_score, dominant_angle_rad, secondary_angle_rad).
    angle_score: 0-1, how strongly edges align to two perpendicular directions.
    """
    if smoothed_hist is None:
        smoothed_hist = compute_direction_histogram(
            magnitude, direction, component_mask, n_bins,
        )
    if smoothed_hist is None:
        return 0.0, 0.0, np.pi / 2
    smoothed = smoothed_hist

    # Find dominant peak (sum of peak + 1 neighbor on each side)
    peak1_bin = int(np.argmax(smoothed))
    peak1_strength = sum(smoothed[(peak1_bin + d) % n_bins] for d in range(-1, 2))
    peak1_angle = (peak1_bin + 0.5) * np.pi / n_bins

    # Look for secondary peak near 90 degrees away
    # In [0, pi) space with 36 bins, 90 degrees = 18 bins
    target_bin = (peak1_bin + n_bins // 2) % n_bins
    search_range = max(1, n_bins // 6)  # +/- 30 degrees tolerance

    best_secondary = 0.0
    best_secondary_bin = target_bin
    for offset in range(-search_range, search_range + 1):
        b = (target_bin + offset) % n_bins
        # Avoid picking a bin near the primary peak
        dist_to_peak1 = min(abs(b - peak1_bin), n_bins - abs(b - peak1_bin))
        if dist_to_peak1 < 3:
            continue
        s = sum(smoothed[(b + d) % n_bins] for d in range(-1, 2))
        if s > best_secondary:
            best_secondary = s
            best_secondary_bin = b

    peak2_angle = (best_secondary_bin + 0.5) * np.pi / n_bins

    # Total is sum of 3-bin windows (normalized)
    total = smoothed.sum()
    if total < 1e-10:
        return 0.0, 0.0, np.pi / 2

    # What fraction of total gradient is in the two peaks?
    peak_frac_1 = peak1_strength / (total + 1e-10)
    peak_frac_2 = best_secondary / (total + 1e-10)

    # Angular accuracy: how close to 90 degrees apart?
    angle_diff = abs(peak2_angle - peak1_angle)
    # In [0, pi), the distance wraps around
    if angle_diff > np.pi / 2:
        angle_diff = np.pi - angle_diff
    angle_error = abs(angle_diff - np.pi / 2)
    angle_accuracy = max(0, 1.0 - angle_error / (np.pi / 6))  # 30-deg tolerance

    # Concentration: how much of the signal is in the two perpendicular peaks
    # For a perfect rectangle with 36 bins, each 3-bin peak holds ~1/12 of signal
    # (3 bins out of 36), so peak_frac ~= 0.083 each. Normalize by this baseline.
    # Two peaks together: peak_frac_1 + peak_frac_2 vs baseline of 6/36 = 0.167
    combined_frac = peak_frac_1 + peak_frac_2
    baseline = 6.0 / n_bins  # expected for uniform distribution
    concentration = np.clip((combined_frac - baseline) / (1.0 - baseline), 0, 1)

    angle_score = concentration * angle_accuracy

    return float(np.clip(angle_score, 0, 1)), float(peak1_angle), float(peak2_angle)


def compute_template_score(
    rel_elev: np.ndarray,
    center_row: int,
    center_col: int,
    width_cells: int,
    height_cells: int,
    angle_rad: float,
) -> float:
    """
    Score how well a rectangular ring template matches the relative elevation.

    Creates a ring-shaped template (elevated edges, zero interior)
    at the given orientation and measures normalized cross-correlation.
    """
    # Cap template size for performance
    width_cells = min(width_cells, 40)
    height_cells = min(height_cells, 40)
    max_dim = max(width_cells, height_cells) + 8
    half = max_dim // 2
    template = np.zeros((max_dim, max_dim), dtype=np.float64)

    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    hw, hh = width_cells / 2.0, height_cells / 2.0

    # Vectorized template construction
    y_range = np.arange(max_dim) - half
    x_range = np.arange(max_dim) - half
    dx, dy = np.meshgrid(x_range, y_range)
    rx = dx * cos_a + dy * sin_a
    ry = -dx * sin_a + dy * cos_a

    inside = (np.abs(rx) <= hw) & (np.abs(ry) <= hh)
    dx_edge = np.abs(np.abs(rx) - hw)
    dy_edge = np.abs(np.abs(ry) - hh)
    near_edge = np.minimum(dx_edge, dy_edge) <= 1.5

    template[inside & near_edge] = 1.0
    template[inside & ~near_edge] = -0.3

    # Normalize template
    template -= template.mean()
    t_norm = np.sqrt((template**2).sum())
    if t_norm < 1e-10:
        return 0.0
    template /= t_norm

    # Extract patch from relative elevation (same size as template)
    r_lo = max(0, center_row - half)
    r_hi = min(rel_elev.shape[0], center_row - half + max_dim)
    c_lo = max(0, center_col - half)
    c_hi = min(rel_elev.shape[1], center_col - half + max_dim)

    patch = rel_elev[r_lo:r_hi, c_lo:c_hi].copy()
    patch[np.isnan(patch)] = 0.0

    # Crop template to match patch dimensions
    t_r_lo = r_lo - (center_row - half)
    t_c_lo = c_lo - (center_col - half)
    t_crop = template[t_r_lo:t_r_lo + patch.shape[0],
                      t_c_lo:t_c_lo + patch.shape[1]]

    if t_crop.shape != patch.shape:
        return 0.0

    # Normalized cross-correlation
    patch_centered = patch - patch.mean()
    p_norm = np.sqrt((patch_centered**2).sum())
    if p_norm < 1e-10:
        return 0.0

    ncc = (t_crop * patch_centered).sum() / p_norm
    return float(np.clip(ncc, 0, 1))


# ---------------------------------------------------------------------------
# Rectangle fitting and scoring (kept from v1)
# ---------------------------------------------------------------------------

def fit_bounding_rectangle(
    cells_xy: np.ndarray,
) -> Tuple[np.ndarray, float, float, float, np.ndarray]:
    """
    Fit a PCA-based minimum bounding rectangle to cell coordinates.

    Returns (center_xy, width, height, angle_deg, eigenvectors).
    width >= height by convention.
    """
    mean = cells_xy.mean(axis=0)
    centered = cells_xy - mean
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    order = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, order]

    projected = centered @ eigenvectors
    p_min = projected.min(axis=0)
    p_max = projected.max(axis=0)
    width = p_max[0] - p_min[0]
    height = p_max[1] - p_min[1]

    if height > width:
        width, height = height, width
        eigenvectors = eigenvectors[:, ::-1]

    angle = float(np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])))

    return mean, width, height, angle, eigenvectors


def compute_rectangularity(cluster_area_m2: float, rect_width: float, rect_height: float) -> float:
    """Ratio of cluster area to bounding rectangle area. 1.0 = perfect fill."""
    rect_area = rect_width * rect_height
    if rect_area < 1e-6:
        return 0.0
    return min(cluster_area_m2 / rect_area, 1.0)


def compute_wall_score(
    cells_xy: np.ndarray,
    center: np.ndarray,
    width: float,
    height: float,
    eigenvectors: np.ndarray,
    edge_fraction: float = 0.25,
) -> float:
    """
    Fraction of elevated cells near rectangle edges vs uniform expectation.

    Returns 0-1 where higher = more edge-concentrated.
    """
    centered = cells_xy - center
    projected = centered @ eigenvectors

    proj_norm_x = projected[:, 0] / max(width, 1e-6)
    proj_norm_y = projected[:, 1] / max(height, 1e-6)

    edge_threshold = 0.5 - edge_fraction
    near_edge_x = np.abs(proj_norm_x) > edge_threshold
    near_edge_y = np.abs(proj_norm_y) > edge_threshold
    near_any_edge = near_edge_x | near_edge_y

    expected_edge_frac = 1.0 - (1.0 - 2.0 * edge_fraction) ** 2
    actual_edge_frac = near_any_edge.sum() / max(len(cells_xy), 1)

    ratio = actual_edge_frac / max(expected_edge_frac, 1e-6)
    return float(np.clip((ratio - 0.8) / 0.8, 0.0, 1.0))


class PointIndex:
    """Spatial grid index for fast point queries."""

    def __init__(self, points: np.ndarray, grid_size: float = 50.0):
        self.points = points
        self.grid_size = grid_size
        self.x_origin = float(points[:, 0].min())
        self.y_origin = float(points[:, 1].min())

        cols = ((points[:, 0] - self.x_origin) / grid_size).astype(np.intp)
        rows = ((points[:, 1] - self.y_origin) / grid_size).astype(np.intp)
        self.n_cols = int(cols.max()) + 1

        cell_ids = rows * self.n_cols + cols
        order = np.argsort(cell_ids)
        self.sorted_points = points[order]
        sorted_ids = cell_ids[order]

        # Build lookup: cell_id -> (start, end) index
        changes = np.where(np.diff(sorted_ids))[0] + 1
        starts = np.concatenate([[0], changes])
        ends = np.concatenate([changes, [len(sorted_ids)]])
        self._cells = {}
        for i in range(len(starts)):
            self._cells[sorted_ids[starts[i]]] = (starts[i], ends[i])

    def query_box(self, cx: float, cy: float, half_w: float, half_h: float) -> np.ndarray:
        """Return points within the axis-aligned bounding box."""
        gs = self.grid_size
        c0 = max(0, int((cx - half_w - self.x_origin) / gs))
        c1 = int((cx + half_w - self.x_origin) / gs) + 1
        r0 = max(0, int((cy - half_h - self.y_origin) / gs))
        r1 = int((cy + half_h - self.y_origin) / gs) + 1

        chunks = []
        for r in range(r0, r1):
            for c in range(c0, c1):
                cell_id = r * self.n_cols + c
                if cell_id in self._cells:
                    s, e = self._cells[cell_id]
                    chunks.append(self.sorted_points[s:e])

        if not chunks:
            return np.empty((0, 3))
        all_pts = np.concatenate(chunks)

        # Refine to exact box
        mask = ((np.abs(all_pts[:, 0] - cx) <= half_w) &
                (np.abs(all_pts[:, 1] - cy) <= half_h))
        return all_pts[mask]


def compute_edge_density_ratio(
    point_index: PointIndex,
    center_xy: np.ndarray,
    width: float,
    height: float,
    eigenvectors: np.ndarray,
    dem: np.ndarray,
    x_origin: float,
    y_origin: float,
    cell_size: float,
    edge_fraction: float = 0.25,
) -> Tuple[float, int, int]:
    """
    Compare raw LiDAR point density at rectangle edges vs interior.

    Returns (edge_density_ratio, num_ground_points, num_elevated_points).
    """
    buffer = 2.0
    half_w, half_h = width / 2 + buffer, height / 2 + buffer
    box_pts = point_index.query_box(
        float(center_xy[0]), float(center_xy[1]), half_w, half_h
    )

    if len(box_pts) < 10:
        return 1.0, 0, 0

    pt_cols = ((box_pts[:, 0] - x_origin) / cell_size).astype(np.intp)
    pt_rows = ((box_pts[:, 1] - y_origin) / cell_size).astype(np.intp)
    pt_cols = np.clip(pt_cols, 0, dem.shape[1] - 1)
    pt_rows = np.clip(pt_rows, 0, dem.shape[0] - 1)
    dem_z = dem[pt_rows, pt_cols]
    valid = ~np.isnan(dem_z)
    near_ground = valid & (box_pts[:, 2] - dem_z < 3.0)
    ground_pts = box_pts[near_ground]
    num_ground = len(ground_pts)

    if num_ground < 10:
        return 1.0, num_ground, 0

    ground_z = dem_z[near_ground]
    elevated_mask = (ground_pts[:, 2] - ground_z) >= 0.2
    num_elevated = int(elevated_mask.sum())

    centered = ground_pts[:, :2] - center_xy
    projected = centered @ eigenvectors
    proj_norm_x = projected[:, 0] / max(width, 1e-6)
    proj_norm_y = projected[:, 1] / max(height, 1e-6)

    inside = (np.abs(proj_norm_x) <= 0.5) & (np.abs(proj_norm_y) <= 0.5)
    inside_pts = inside.sum()
    if inside_pts < 5:
        return 1.0, num_ground, num_elevated

    edge_threshold = 0.5 - edge_fraction
    near_edge = (np.abs(proj_norm_x) > edge_threshold) | (np.abs(proj_norm_y) > edge_threshold)
    edge_inside = inside & near_edge
    interior_inside = inside & ~near_edge

    edge_area_frac = 1.0 - (1.0 - 2.0 * edge_fraction) ** 2
    interior_area_frac = 1.0 - edge_area_frac

    edge_count = edge_inside.sum()
    interior_count = interior_inside.sum()

    if interior_count < 1 or interior_area_frac < 1e-6:
        return 1.0, num_ground, num_elevated

    edge_density = edge_count / edge_area_frac
    interior_density = interior_count / interior_area_frac
    ratio = edge_density / max(interior_density, 1e-6)

    return float(ratio), num_ground, num_elevated


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def detect_buildings(
    points: np.ndarray,
    config: BuildingConfig = None,
) -> BuildingDetectionResult:
    """
    Full building detection pipeline.

    Args:
        points: Nx3 array of (x, y, z) in UTM metres.
        config: Detection parameters.

    Returns:
        BuildingDetectionResult with filtered and raw candidates.
    """
    if config is None:
        config = BuildingConfig()

    cs = config.dem_cell_size

    # Stage 1: Build DEM
    dem, x_origin, y_origin = build_fine_dem(points, cell_size=cs)

    # Stage 2: Local relative elevation (or depth below ground)
    sigma_cells = config.smoothing_radius_m / cs
    background = gaussian_smooth_dem(dem, sigma_cells)

    if config.subsurface_mode:
        # Subsurface: filter points below ground, build depth map
        depth_map, sub_dem, n_sub = _build_subsurface_depth_map(
            points, background, x_origin, y_origin, cs,
            config.subsurface_max_depth,
        )
        print(f"  Subsurface: {n_sub:,} points below ground "
              f"({n_sub/len(points)*100:.1f}%)")
        if n_sub < 100:
            return BuildingDetectionResult(
                candidates=[], raw_candidates=[],
                dem=dem, relative_elevation=np.zeros_like(dem),
                x_origin=x_origin, y_origin=y_origin, cell_size=cs,
            )
        n_depth_cells = int((~np.isnan(depth_map)).sum())
        print(f"  Depth map: {n_depth_cells:,} cells with data, "
              f"max depth {np.nanmax(depth_map):.2f}m")
        rel_elev = depth_map   # positive values = depth below ground
    else:
        rel_elev = dem - background

    # Stage 3: Threshold and connected components
    mask = (rel_elev >= config.min_relative_elevation) & \
           (rel_elev <= config.max_relative_elevation)
    mask[np.isnan(rel_elev)] = False
    labeled, n_features = ndimage_label(mask)

    min_cells = int(config.min_cluster_area_m2 / (cs * cs))
    max_cells = int(config.max_cluster_area_m2 / (cs * cs))

    # Stage 4: Compute gradient field (once for entire DEM)
    grad_mag, grad_dir = compute_gradient_field(rel_elev)

    # Pre-compute component sizes with bincount for fast filtering
    component_sizes = np.bincount(labeled.ravel())
    valid_labels = set(np.where(
        (component_sizes >= min_cells) & (component_sizes <= max_cells)
    )[0]) - {0}

    # Bounding box of every component, in one pass. A tile raster runs to
    # millions of cells while a candidate covers a few dozen, so every
    # per-candidate step below works inside its own padded window instead of
    # scanning the whole raster. The padding is the gradient sample's reach,
    # so the window holds everything those steps can see.
    label_slices = find_objects(labeled)

    # Build spatial index for fast point queries
    pt_index = PointIndex(points)

    raw_candidates: List[BuildingCandidate] = []
    filtered_candidates: List[BuildingCandidate] = []

    for label_id in valid_labels:
        bbox = label_slices[label_id - 1]
        if bbox is None:
            continue
        n_cells = int(component_sizes[label_id])

        cluster_area = n_cells * cs * cs

        # Padded window around this component
        pad = DILATION_ITERATIONS
        row_lo = max(0, bbox[0].start - pad)
        row_hi = min(labeled.shape[0], bbox[0].stop + pad)
        col_lo = max(0, bbox[1].start - pad)
        col_hi = min(labeled.shape[1], bbox[1].stop + pad)
        window = (slice(row_lo, row_hi), slice(col_lo, col_hi))

        component = labeled[window] == label_id
        window_rel = rel_elev[window]

        # Get cell coordinates in UTM
        local_rows, local_cols = np.where(component)
        rows = local_rows + row_lo
        cols = local_cols + col_lo
        cells_x = x_origin + (cols + 0.5) * cs
        cells_y = y_origin + (rows + 0.5) * cs
        cells_xy = np.column_stack([cells_x, cells_y])

        # Fit bounding rectangle (PCA)
        if len(cells_xy) < 3:
            continue
        center, width, height, angle, eigvecs = fit_bounding_rectangle(cells_xy)

        if height < 1e-6:
            continue
        aspect_ratio = width / height

        # Rectangularity and wall score (secondary metrics now)
        rectangularity = compute_rectangularity(cluster_area, width, height)
        wall_score = compute_wall_score(cells_xy, center, width, height, eigvecs)

        # Elevation stats
        cluster_rel = window_rel[component]
        peak_rel = float(np.nanmax(cluster_rel))
        mean_rel = float(np.nanmean(cluster_rel))

        # Stage 5: Angle analysis (primary rectangular discriminator).
        # The edge sample is extracted once and read by both shape families,
        # which look at the same directions from opposite ends: the
        # rectangular path wants two sharp perpendicular peaks, the circular
        # path wants no preferred direction at all.
        grad_sample = extract_strong_gradients(
            grad_mag[window], grad_dir[window], component,
        )
        if grad_sample is None:
            dir_hist = None
        else:
            dir_hist = build_direction_histogram(
                grad_sample[0], grad_sample[1], config.gradient_angle_bins,
            )
        a_score, dom_angle, sec_angle = compute_angle_score(
            grad_mag[window], grad_dir[window], component,
            min_gradient_strength=config.min_gradient_strength,
            n_bins=config.gradient_angle_bins,
            smoothed_hist=dir_hist,
        )

        # Use gradient-derived orientation when angle score is good
        if a_score > 0.1:
            orientation_deg = float(np.degrees(dom_angle))
        else:
            orientation_deg = angle  # fall back to PCA

        # Stage 6: Template scoring. These read the full raster on purpose --
        # a rectangle or ring template can reach tens of cells past the
        # component, well beyond the window's padding. Both only slice out
        # the patch they need, so neither scans the whole array.
        center_row = int(np.mean(rows))
        center_col = int(np.mean(cols))
        t_score = compute_template_score(
            rel_elev, center_row, center_col,
            int(width / cs), int(height / cs),
            dom_angle,
        )

        # Stage 6b: Circular hypothesis (same region, opposite signature)
        if config.detect_circular:
            circ = compute_circular_metrics(
                rel_elev, component, grad_sample,
                center_row, center_col,
                width, height, aspect_ratio, cs,
            )
        else:
            circ = CircularMetrics()

        # Point density analysis (secondary)
        edge_ratio, n_ground, n_elevated = compute_edge_density_ratio(
            pt_index, center, width, height, eigvecs,
            dem, x_origin, y_origin, cs,
        )

        candidate = BuildingCandidate(
            label_id=label_id,
            center_xy=center,
            width_m=width,
            height_m=height,
            area_m2=cluster_area,
            orientation_deg=orientation_deg,
            peak_relative_elevation=peak_rel,
            mean_relative_elevation=mean_rel,
            rectangularity=rectangularity,
            wall_score=wall_score,
            edge_density_ratio=edge_ratio,
            aspect_ratio=aspect_ratio,
            num_ground_points=n_ground,
            num_elevated_points=n_elevated,
            eigenvectors=eigvecs,
            angle_score=a_score,
            template_score=t_score,
            dominant_orientation_deg=orientation_deg,
            circular_score=circ.circular_score,
            circ_template_score=circ.circ_template_score,
            radial_symmetry=circ.radial_symmetry,
            directional_isotropy=circ.directional_isotropy,
            roundness=circ.roundness,
            diameter_m=circ.diameter_m,
        )
        raw_candidates.append(candidate)

        # Stage 7: Apply filters -- shared geometry gates first
        if peak_rel < config.min_peak_height or peak_rel > config.max_peak_height:
            continue
        if mean_rel < config.min_mean_relative_elevation:
            continue

        # Rectangular family: right-angle signature
        passes_rect = config.detect_rectangular
        if passes_rect:
            if aspect_ratio > config.max_aspect_ratio:
                passes_rect = False
            elif a_score < config.min_angle_score:
                passes_rect = False
            elif t_score < config.min_template_score:
                passes_rect = False
            elif rectangularity < config.min_rectangularity:
                passes_rect = False
            elif wall_score < config.min_wall_score:
                passes_rect = False
            elif edge_ratio < config.min_edge_density_ratio:
                passes_rect = False

        # Circular family: isotropic edges at a constant radius
        passes_circ = config.detect_circular
        if passes_circ:
            if aspect_ratio > config.max_circular_aspect_ratio:
                passes_circ = False
            elif circ.diameter_m < config.min_circular_diameter_m:
                passes_circ = False
            elif circ.circular_score < config.min_circular_score:
                passes_circ = False
            elif circ.radial_symmetry < config.min_radial_symmetry:
                passes_circ = False
            elif circ.directional_isotropy < config.min_directional_isotropy:
                passes_circ = False
            elif circ.circ_template_score < config.min_circ_template_score:
                passes_circ = False

        if not passes_rect and not passes_circ:
            continue

        if passes_rect and passes_circ:
            # Both fit: the two template correlations are on the same scale,
            # so let them decide. The margin keeps borderline regions in the
            # rectangular family they were historically assigned to.
            margin = 1.0 + config.circular_preference_margin
            is_circular = circ.circ_template_score > t_score * margin
        else:
            is_circular = passes_circ

        candidate.shape_type = 'circular' if is_circular else 'rectangular'
        filtered_candidates.append(candidate)

    # Sort by the quality metric of each candidate's own shape family
    filtered_candidates.sort(
        key=lambda c: (c.circular_score if c.shape_type == 'circular'
                       else c.angle_score * c.template_score),
        reverse=True,
    )

    return BuildingDetectionResult(
        candidates=filtered_candidates,
        raw_candidates=raw_candidates,
        dem=dem,
        relative_elevation=rel_elev,
        x_origin=x_origin,
        y_origin=y_origin,
        cell_size=cs,
    )
