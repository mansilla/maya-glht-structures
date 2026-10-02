"""
Ancient building detection from airborne LiDAR point clouds.

Detects two families of buried stone accumulations (collapsed Maya
foundations) by analysing local relative elevation and point density:

  rectangular  right-angle structures -- gradient normals concentrate into
               two perpendicular directions
  circular     round platforms, ring walls, chultun mouths -- gradient
               normals spread isotropically at a constant radius

Set ``BuildingConfig.detect_circular = True`` to report the circular family
alongside the rectangular one; each detection carries a ``shape_type``.

Usage:
    from building_detection import detect_buildings, BuildingConfig
    from building_detection.metrics import compute_all_building_metrics
    from building_detection.batch import run_batch
"""
from .detection import detect_buildings, BuildingConfig
from .circular import CircularMetrics, compute_circular_metrics
from .metrics import (
    BuildingCandidate,
    BuildingDetectionResult,
    compute_all_building_metrics,
    compute_shape_confidence,
    compute_raw_shape_confidence,
    calibrate_circular_confidence,
)
from .batch import run_batch
