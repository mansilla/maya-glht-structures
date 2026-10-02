"""
Dataclasses and metric computation for building detection.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# Coordinate conversion shared with the LAS loader
from building_detection.lasio import utm_to_latlon


@dataclass
class BuildingCandidate:
    """A single detected building candidate."""
    label_id: int
    center_xy: np.ndarray            # UTM (easting, northing)
    width_m: float                    # long side of bounding rectangle
    height_m: float                   # short side
    area_m2: float                    # actual cluster area
    orientation_deg: float            # angle of long axis from east
    peak_relative_elevation: float    # max height above background
    mean_relative_elevation: float    # mean height above background
    rectangularity: float             # cluster_area / rect_area (0-1)
    wall_score: float                 # edge concentration score (0-1)
    edge_density_ratio: float         # point density at edges vs interior
    aspect_ratio: float               # width / height
    num_ground_points: int = 0        # LiDAR points near ground in bbox
    num_elevated_points: int = 0      # LiDAR points in the elevated mound
    eigenvectors: np.ndarray = field(default_factory=lambda: np.eye(2))
    angle_score: float = 0.0         # right-angle signal strength (0-1)
    template_score: float = 0.0      # rectangular template match (0-1)
    dominant_orientation_deg: float = 0.0  # from gradient histogram

    # Shape family -- 'rectangular' or 'circular'
    shape_type: str = 'rectangular'
    # Circular evidence (all 0 when circular detection is off)
    circular_score: float = 0.0        # combined circular evidence (0-1)
    circ_template_score: float = 0.0   # annular template match (0-1)
    radial_symmetry: float = 0.0       # constancy of boundary radius (0-1)
    directional_isotropy: float = 0.0  # flatness of direction histogram (0-1)
    roundness: float = 0.0             # aspect ratio closeness to 1 (0-1)
    diameter_m: float = 0.0            # mean boundary diameter, circular fit


@dataclass
class BuildingDetectionResult:
    """Full result from detect_buildings()."""
    candidates: List[BuildingCandidate]      # after filtering
    raw_candidates: List[BuildingCandidate]  # before filtering
    dem: np.ndarray
    relative_elevation: np.ndarray
    x_origin: float
    y_origin: float
    cell_size: float


def _height_score(h: float) -> float:
    """Height plausibility: peaks around 1.0m, drops off at 0.3 and 3.0."""
    if h < 0.3:
        return 0.0
    if h <= 1.5:
        return float(np.clip((h - 0.3) / 1.2, 0.0, 1.0))
    return float(np.clip(1.0 - (h - 1.5) / 1.5, 0.0, 1.0))


# Quantile calibration mapping raw circular confidence onto the rectangular
# scale. Measured over the full 528-tile run (190,789 rectangular and 1,005
# circular detections, 2026-09-16): each entry pairs a percentile of the raw
# circular distribution with the same percentile of the rectangular one.
#
# This is a CONVENTION, not a probability. Neither family's score is
# calibrated against ground truth -- both are evidence rankings with
# hand-tuned weights. What the mapping buys is that one threshold selects the
# same *fraction* of each family, so "confidence > 0.4" means "top ~3% of its
# kind" either way and a single colour scale can serve both on a map. It does
# NOT assert that a circular and a rectangular detection at 0.45 are equally
# likely to be real; that would need field validation.
#
# Re-measure these with tools/recalibrate_confidence.py if the detection
# thresholds change, since the distributions move with them.
_CIRCULAR_CALIBRATION_RAW = (
    0.1760, 0.1902, 0.2204, 0.2420, 0.2990, 0.4230,
    0.5860, 0.6940, 0.7448, 0.8030, 0.8969, 0.9420,
)
_CIRCULAR_CALIBRATION_MAPPED = (
    0.0710, 0.1200, 0.1510, 0.1760, 0.2180, 0.2620,
    0.3070, 0.3510, 0.3800, 0.4370, 0.4980, 0.5960,
)


def calibrate_circular_confidence(raw: float) -> float:
    """
    Map a raw circular confidence onto the rectangular scale.

    Monotone piecewise-linear interpolation through matched quantiles, so
    ordering within the circular family is untouched -- only the scale moves.
    Values outside the measured range clamp to its ends.
    """
    return float(np.interp(
        raw, _CIRCULAR_CALIBRATION_RAW, _CIRCULAR_CALIBRATION_MAPPED,
    ))


def compute_raw_circular_confidence(c: BuildingCandidate) -> float:
    """
    Composite 0-1 evidence score for a circular candidate, before calibration.

    Each term is re-centred on its filter floor so the score spans its usable
    range. The result lives on the circular family's own scale, which sits
    well above the rectangular one -- calibrate_circular_confidence() maps it
    across. Use this when comparing circular candidates to each other.

    - 0.40 * circular_score, re-centred on the 0.45 acceptance floor
    - 0.20 * annular template match (normalised by a typical NCC of 0.4)
    - 0.15 * radial symmetry of the boundary
    - 0.10 * directional isotropy
    - 0.10 * height (depth) plausibility
    - 0.05 * roundness
    """
    combined = np.clip((c.circular_score - 0.45) / 0.45, 0.0, 1.0)
    template = np.clip(c.circ_template_score / 0.4, 0.0, 1.0)
    symmetry = np.clip(c.radial_symmetry, 0.0, 1.0)
    isotropy = np.clip(c.directional_isotropy, 0.0, 1.0)
    roundness = np.clip(c.roundness, 0.0, 1.0)

    confidence = (0.40 * combined +
                  0.20 * template +
                  0.15 * symmetry +
                  0.10 * isotropy +
                  0.10 * _height_score(c.mean_relative_elevation) +
                  0.05 * roundness)
    return float(np.clip(confidence, 0.0, 1.0))


def compute_circular_confidence(c: BuildingCandidate) -> float:
    """Calibrated circular confidence, on the same scale as the rectangular one."""
    return calibrate_circular_confidence(compute_raw_circular_confidence(c))


def compute_shape_confidence(c: BuildingCandidate) -> float:
    """
    Confidence for a candidate on one scale shared by both shape families.

    A threshold applied to this value selects the same fraction of each
    family, so maps and counts can mix the two. See the calibration note
    above for what that does and does not claim.
    """
    if c.shape_type == 'circular':
        return compute_circular_confidence(c)
    return compute_building_confidence(c)


def compute_raw_shape_confidence(c: BuildingCandidate) -> float:
    """Uncalibrated evidence score, on each shape family's own native scale."""
    if c.shape_type == 'circular':
        return compute_raw_circular_confidence(c)
    return compute_building_confidence(c)


def compute_building_confidence(c: BuildingCandidate) -> float:
    """
    Composite 0-1 confidence score for a rectangular building candidate.

    Weights emphasize angle and template scores (right-angle detection):
    - 0.30 * angle_score (right-angle signal, primary)
    - 0.20 * template_score (rectangular template match)
    - 0.15 * wall_score (edge concentration)
    - 0.10 * rectangularity (fill ratio)
    - 0.10 * edge_density_ratio (point density at edges)
    - 0.10 * height score (peaks around 0.5-1.5m)
    - 0.05 * aspect ratio score (1.0-3.0 ideal)
    """
    angle = np.clip(c.angle_score, 0.0, 1.0)
    template = np.clip(c.template_score, 0.0, 1.0)
    rect_score = np.clip((c.rectangularity - 0.2) / 0.8, 0.0, 1.0)
    wall = np.clip(c.wall_score, 0.0, 1.0)
    edge = np.clip((c.edge_density_ratio - 1.0) / 1.0, 0.0, 1.0)

    height_score = _height_score(c.mean_relative_elevation)

    # Aspect ratio: 1.0-3.0 ideal for buildings
    ar = c.aspect_ratio
    if ar <= 3.0:
        ar_score = 1.0
    else:
        ar_score = np.clip(1.0 - (ar - 3.0) / 2.0, 0.0, 1.0)

    confidence = (0.30 * angle +
                  0.20 * template +
                  0.15 * wall +
                  0.10 * rect_score +
                  0.10 * edge +
                  0.10 * height_score +
                  0.05 * ar_score)
    return float(np.clip(confidence, 0.0, 1.0))


def compute_all_building_metrics(
    candidate: BuildingCandidate,
    points: np.ndarray,
    utm_zone: int = 15,
) -> Dict:
    """Return a flat dict suitable for CSV export."""
    lat, lon = utm_to_latlon(
        float(candidate.center_xy[0]),
        float(candidate.center_xy[1]),
        utm_zone,
    )
    confidence = compute_shape_confidence(candidate)
    confidence_raw = compute_raw_shape_confidence(candidate)

    return {
        'latitude': round(lat, 6),
        'longitude': round(lon, 6),
        'easting': round(float(candidate.center_xy[0]), 1),
        'northing': round(float(candidate.center_xy[1]), 1),
        'width_m': round(candidate.width_m, 1),
        'height_m': round(candidate.height_m, 1),
        'area_m2': round(candidate.area_m2, 1),
        'orientation_deg': round(candidate.orientation_deg, 1),
        'aspect_ratio': round(candidate.aspect_ratio, 2),
        'rectangularity': round(candidate.rectangularity, 3),
        'wall_score': round(candidate.wall_score, 3),
        'edge_density_ratio': round(candidate.edge_density_ratio, 3),
        'peak_height_m': round(candidate.peak_relative_elevation, 2),
        'mean_height_m': round(candidate.mean_relative_elevation, 2),
        'building_confidence': round(confidence, 3),
        'confidence_raw': round(confidence_raw, 3),
        'num_ground_points': candidate.num_ground_points,
        'num_elevated_points': candidate.num_elevated_points,
        'angle_score': round(candidate.angle_score, 3),
        'template_score': round(candidate.template_score, 3),
        'dominant_orientation_deg': round(candidate.dominant_orientation_deg, 1),
        'shape_type': candidate.shape_type,
        'circular_score': round(candidate.circular_score, 3),
        'circ_template_score': round(candidate.circ_template_score, 3),
        'radial_symmetry': round(candidate.radial_symmetry, 3),
        'directional_isotropy': round(candidate.directional_isotropy, 3),
        'roundness': round(candidate.roundness, 3),
        'diameter_m': round(candidate.diameter_m, 1),
    }
