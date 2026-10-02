#!/usr/bin/env python3
"""
Canopy cover over each published detection, from the raw returns.

For every detection in the chosen tiles: take the returns inside a disc of
radius max(width_m, height_m)/2 + 2 m around the centroid, build the ground as
the minimum Z per 2 m cell (the detector's own ground proxy), and measure

    canopy_cover  = first returns with height above ground > 2 m
                    / all first returns in the disc            (standard LiDAR
                                                                canopy cover)
    canopy_p95_m  = 95th percentile of height above ground in the disc
    n_first       = first returns in the disc

A detection is called "under canopy" when canopy_cover >= 0.5 and "open" when
canopy_cover < 0.2; the band between is partial cover. Thresholds are
reported alongside so they can be changed.

Usage:
    python tools/canopy_cover.py --per-dataset 1        # one tile per dataset
    python tools/canopy_cover.py --all                   # every tile (slow)
"""
from __future__ import annotations

import argparse
import gzip
import io
import sys
import time
from pathlib import Path

import laspy
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
CAT = ROOT / "catalogue/detections_tau016.csv"
DATA = ROOT / "data"
OUT = ROOT / "catalogue/canopy_cover.csv"

CANOPY_H = 2.0      # metres above ground that count as canopy
CELL = 2.0


def read_tile(path: Path):
    with gzip.open(path, "rb") as g:
        las = laspy.read(io.BytesIO(g.read()))
    xyz = np.column_stack([np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)])
    rn = np.asarray(las.return_number).astype(np.int16)
    return xyz, rn


def measure(xyz, rn, det: pd.DataFrame) -> pd.DataFrame:
    tree = cKDTree(xyz[:, :2])
    rows = []
    for r in det.itertuples():
        rad = max(r.width_m, r.height_m) / 2.0 + 2.0
        idx = tree.query_ball_point([r.easting, r.northing], rad)
        if len(idx) < 5:
            rows.append((r.filename, r.building_id, np.nan, np.nan, len(idx), 0))
            continue
        p = xyz[idx]
        first = rn[idx] == 1
        # ground = min Z per 2 m cell, then each point's height above its cell
        cx = np.floor((p[:, 0] - p[:, 0].min()) / CELL).astype(int)
        cy = np.floor((p[:, 1] - p[:, 1].min()) / CELL).astype(int)
        key = cx * (cy.max() + 1) + cy
        gmin = pd.Series(p[:, 2]).groupby(key).transform("min").to_numpy()
        h = p[:, 2] - gmin
        nf = int(first.sum())
        cover = float((h[first] > CANOPY_H).mean()) if nf else np.nan
        rows.append((r.filename, r.building_id, cover,
                     float(np.percentile(h, 95)), len(idx), nf))
    return pd.DataFrame(rows, columns=["filename", "building_id", "canopy_cover",
                                       "canopy_p95_m", "n_points", "n_first"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-dataset", type=int, default=1,
                    help="tiles per dataset, chosen by detection count")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()

    d = pd.read_csv(CAT)
    counts = d.groupby(["source_folder", "filename"]).size().reset_index(name="n")
    if a.all:
        sel = counts
    else:
        sel = (counts.sort_values("n", ascending=False)
               .groupby("source_folder").head(a.per_dataset))
    print(f"{len(sel)} tiles, {sel.n.sum()} detections", flush=True)

    out = []
    for t in sel.itertuples():
        path = DATA / t.source_folder / t.filename
        t0 = time.time()
        try:
            xyz, rn = read_tile(path)
        except Exception as e:      # noqa: BLE001
            print(f"  {t.filename}: {e}", flush=True)
            continue
        m = measure(xyz, rn, d[d.filename == t.filename])
        m.insert(0, "source_folder", t.source_folder)
        out.append(m)
        print(f"  {t.filename}: {len(m)} det, {len(xyz)/1e6:.1f} M pts, "
              f"cover>=0.5 {np.nanmean(m.canopy_cover >= 0.5):.2f}, {time.time()-t0:.0f}s", flush=True)
        pd.concat(out).to_csv(a.out, index=False)

    res = pd.concat(out)
    res.to_csv(a.out, index=False)
    c = res.canopy_cover.dropna()
    print(f"\nmeasured {len(c)} detections in {len(out)} tiles")
    print(f"under canopy (cover >= 0.5): {(c >= 0.5).mean():.1%}")
    print(f"partial (0.2-0.5):           {((c >= 0.2) & (c < 0.5)).mean():.1%}")
    print(f"open (cover < 0.2):          {(c < 0.2).mean():.1%}")
    print(f"cover quantiles: {c.quantile([.1, .25, .5, .75, .9]).round(2).to_dict()}")
    print(res.groupby("source_folder").canopy_cover
          .agg(n="size", median="median", under=lambda s: (s >= 0.5).mean()).round(2).to_string())


if __name__ == "__main__":
    main()
