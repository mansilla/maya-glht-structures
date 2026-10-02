"""
Circular structure analysis.

The rectangular detector keys on gradient normals that concentrate into two
perpendicular peaks.  Round structures -- circular platforms, round house
foundations, chultun mouths, ring walls, altars -- produce the opposite
signature: the gradient normals spread evenly over every direction, the
boundary cells sit at a near-constant radius from the centroid, and the
depth field matches an annular template instead of a rectangular one.

Four independent measurements are combined into ``circular_score``:

    directional_isotropy  absence of a preferred edge direction, from the
                          fourth-order resultant (1 = uniform, 0 = four-fold)
    radial_symmetry       1 - normalised spread of the boundary-cell radii
    roundness             how close the bounding-rectangle aspect ratio is to 1
    circ_template_score   cross-correlation with an annular ring template

Everything here works on the same ``rel_elev`` array the rectangular path
uses, which in subsurface mode is the *depth below ground* map -- so a
circular detection is, like a rectangular one, a below-ground feature.

The rectangular and circular families are told apart by comparing the two
template correlations (``template_score`` vs ``circ_template_score``), the
only two metrics measured on the same scale.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import binary_erosion, binary_fill_holes


@dataclass
class CircularMetrics:
    """Shape metrics for the circular hypothesis of one candidate region."""
    circular_score: float = 0.0
    directional_isotropy: float = 0.0
    radial_symmetry: float = 0.0
    roundness: float = 0.0
    circ_template_score: float = 0.0
    diameter_m: float = 0.0


# ---------------------------------------------------------------------------
# Individual measurements
# ---------------------------------------------------------------------------

def compute_directional_isotropy(
    grad_sample: Optional[Tuple[np.ndarray, np.ndarray]],
) -> float:
    """
    How uniformly the gradient normals cover [0, pi).

    Measured with the magnitude-weighted fourth-order resultant

        R4 = |sum(w * exp(4i*theta))| / sum(w)

    which is the natural statistic for four-fold symmetry: a rectangle's
    normals sit at theta and theta+90 degrees, and multiplying by four maps
    both onto the same phase, so R4 approaches 1.  A ring spreads its normals
    over every direction, so the terms cancel and R4 approaches 0.  Unlike a
    binned histogram this needs no resolution choice and stays meaningful for
    the small regions (a few dozen edge cells) that dominate the data.

    R4 is debiased for sample size -- under a genuinely uniform distribution
    the raw resultant of n effective samples still averages ~1/sqrt(n).

    Returns 1 for no preferred direction, 0 for a strongly four-fold or
    single-direction edge set.
    """
    if grad_sample is None:
        return 0.0
    dirs, mags = grad_sample
    if dirs.size == 0:
        return 0.0

    valid = np.isfinite(dirs) & np.isfinite(mags) & (mags > 0)
    if valid.sum() < 8:
        return 0.0
    dirs = dirs[valid]
    mags = mags[valid]

    weight_sum = float(mags.sum())
    if weight_sum <= 1e-12:
        return 0.0

    resultant = np.abs(np.sum(mags * np.exp(4j * dirs))) / weight_sum

    # Effective sample size for weighted data, used to remove the small-sample
    # bias that would otherwise make every tiny region look anisotropic.
    n_eff = weight_sum ** 2 / float((mags ** 2).sum())
    debiased = np.sqrt(max(resultant ** 2 - 1.0 / n_eff, 0.0))

    return float(np.clip(1.0 - debiased, 0.0, 1.0))


def compute_radial_symmetry(
    component_mask: np.ndarray,
    cell_size: float,
) -> Tuple[float, float]:
    """
    Constancy of the distance from the region centroid to its boundary cells.

    Interior holes are filled first, so a ring-shaped component is measured
    on its outer rim only -- otherwise its inner boundary would drag the
    radius distribution into two modes and make a textbook circle score zero.

    Returns (radial_symmetry, mean_radius_m).  The CV is mapped through a
    0.25 cutoff, so anything whose boundary wanders by a quarter of its own
    radius scores 0.  At the 2 m DEM resolution a 20 m disc lands near 0.68
    and a 20 m square near 0.55, so this metric ranks shapes rather than
    separating them outright -- directional isotropy does the separating.
    """
    # Work inside the component's bounding box with a one-cell margin. The
    # margin keeps the exterior connected to the crop border, so filling
    # holes and eroding give exactly what the full raster would -- at a
    # fraction of the cost, since this runs once per candidate and a tile
    # raster is millions of cells.
    box_rows, box_cols = np.where(component_mask)
    if box_rows.size == 0:
        return 0.0, 0.0
    r0, r1 = int(box_rows.min()), int(box_rows.max()) + 1
    c0, c1 = int(box_cols.min()), int(box_cols.max()) + 1

    local = np.zeros((r1 - r0 + 2, c1 - c0 + 2), dtype=bool)
    local[1:-1, 1:-1] = component_mask[r0:r1, c0:c1]

    solid = binary_fill_holes(local)
    if solid is None:
        solid = local

    boundary = solid & ~binary_erosion(solid)
    rows, cols = np.where(boundary)
    if len(rows) < 8:
        return 0.0, 0.0

    all_rows, all_cols = np.where(solid)
    centre_row = all_rows.mean()
    centre_col = all_cols.mean()

    radii = np.hypot(rows - centre_row, cols - centre_col) * cell_size
    mean_r = float(radii.mean())
    if mean_r < 1e-6:
        return 0.0, 0.0

    cv = float(radii.std()) / mean_r
    symmetry = float(np.clip(1.0 - cv / 0.25, 0.0, 1.0))
    return symmetry, mean_r


def compute_roundness(aspect_ratio: float, tolerance: float = 0.8) -> float:
    """1.0 for a square bounding box, falling to 0 at aspect_ratio 1+tolerance."""
    return float(np.clip(1.0 - (aspect_ratio - 1.0) / tolerance, 0.0, 1.0))


def _normalised_cross_correlation(
    field: np.ndarray,
    template: np.ndarray,
    centre_row: int,
    centre_col: int,
) -> float:
    """
    Zero-mean normalised cross-correlation of a template against a patch of
    ``field`` centred on (centre_row, centre_col).  NaN cells count as zero.
    """
    size = template.shape[0]
    half = size // 2

    template = template - template.mean()
    t_norm = np.sqrt((template ** 2).sum())
    if t_norm < 1e-10:
        return 0.0
    template = template / t_norm

    r_lo = max(0, centre_row - half)
    r_hi = min(field.shape[0], centre_row - half + size)
    c_lo = max(0, centre_col - half)
    c_hi = min(field.shape[1], centre_col - half + size)

    patch = field[r_lo:r_hi, c_lo:c_hi].copy()
    if patch.size == 0:
        return 0.0
    patch[np.isnan(patch)] = 0.0

    t_r_lo = r_lo - (centre_row - half)
    t_c_lo = c_lo - (centre_col - half)
    t_crop = template[t_r_lo:t_r_lo + patch.shape[0],
                      t_c_lo:t_c_lo + patch.shape[1]]
    if t_crop.shape != patch.shape:
        return 0.0

    patch_centered = patch - patch.mean()
    p_norm = np.sqrt((patch_centered ** 2).sum())
    if p_norm < 1e-10:
        return 0.0

    ncc = float((t_crop * patch_centered).sum() / p_norm)
    return float(np.clip(ncc, 0.0, 1.0))


def compute_circular_template_score(
    rel_elev: np.ndarray,
    centre_row: int,
    centre_col: int,
    radius_cells: float,
    ring_thickness_cells: float = 1.5,
) -> float:
    """
    Match an annular ring template (raised rim, hollow interior) against the
    relative-elevation / depth field.  This is the circular counterpart of
    ``detection.compute_template_score`` and uses the same +1 rim / -0.3
    interior weighting so the two scores are directly comparable.
    """
    radius_cells = float(np.clip(radius_cells, 1.5, 20.0))
    size = int(2 * radius_cells) + 8
    half = size // 2

    offsets = np.arange(size) - half
    dx, dy = np.meshgrid(offsets, offsets)
    rr = np.hypot(dx, dy)

    template = np.zeros((size, size), dtype=np.float64)
    template[rr < radius_cells - ring_thickness_cells] = -0.3
    template[np.abs(rr - radius_cells) <= ring_thickness_cells] = 1.0

    return _normalised_cross_correlation(rel_elev, template, centre_row, centre_col)


# ---------------------------------------------------------------------------
# Combined score
# ---------------------------------------------------------------------------

def compute_circular_metrics(
    rel_elev: np.ndarray,
    component_mask: np.ndarray,
    grad_sample: Optional[Tuple[np.ndarray, np.ndarray]],
    centre_row: int,
    centre_col: int,
    width_m: float,
    height_m: float,
    aspect_ratio: float,
    cell_size: float,
    template_norm: float = 0.4,
) -> CircularMetrics:
    """
    Evaluate the circular hypothesis for one candidate region.

    ``grad_sample`` is the (directions, magnitudes) edge sample already
    extracted for the rectangular path -- reused here so the extra shape
    family costs no additional gradient work.
    """
    isotropy = compute_directional_isotropy(grad_sample)
    symmetry, mean_radius_m = compute_radial_symmetry(component_mask, cell_size)
    roundness = compute_roundness(aspect_ratio)

    # Prefer the boundary-derived radius; fall back to the bounding box.
    if mean_radius_m > 1e-6:
        radius_m = mean_radius_m
    else:
        radius_m = (width_m + height_m) / 4.0

    template_score = compute_circular_template_score(
        rel_elev, centre_row, centre_col, radius_m / cell_size,
    )
    template_term = float(np.clip(template_score / max(template_norm, 1e-6), 0.0, 1.0))

    circular_score = (0.30 * template_term +
                      0.25 * symmetry +
                      0.25 * isotropy +
                      0.20 * roundness)

    return CircularMetrics(
        circular_score=float(np.clip(circular_score, 0.0, 1.0)),
        directional_isotropy=isotropy,
        radial_symmetry=symmetry,
        roundness=roundness,
        circ_template_score=template_score,
        diameter_m=2.0 * radius_m,
    )
