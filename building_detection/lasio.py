"""LAS loading and UTM -> WGS84 conversion shared by the detector and its tools."""
from __future__ import annotations

import gzip
import functools
from typing import Tuple

import numpy as np
import laspy


def load_las(path: str) -> np.ndarray:
    """Load LAS/LAZ file (handles .gz). Returns Nx3 array in scaled UTM (metres)."""
    path = str(path)
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            las = laspy.read(f)
    else:
        las = laspy.read(path)
    return np.stack([las.x, las.y, las.z], axis=0).T.copy()


@functools.lru_cache(maxsize=4)
def _get_transformer(utm_zone: int, south: bool = False):
    """Cached pyproj transformer for UTM -> WGS84."""
    import pyproj
    utm_proj = pyproj.Proj(proj='utm', zone=utm_zone, ellps='WGS84', south=south)
    wgs84_proj = pyproj.Proj(proj='latlong', datum='WGS84')
    return pyproj.Transformer.from_proj(utm_proj, wgs84_proj, always_xy=True)


def utm_to_latlon(easting: float, northing: float, utm_zone: int = 15) -> Tuple[float, float]:
    """Convert UTM coordinates to lat/lon. Returns (lat, lon)."""
    transformer = _get_transformer(utm_zone)
    lon, lat = transformer.transform(easting, northing)
    return lat, lon


