#!/usr/bin/env python3
"""
Inventory the LiDAR archive from the tiles themselves, not from the download list.

Reads only each tile's LAS public header block, which sits at the start of the
gzip stream, so the whole 528-tile / ~167 GB archive is summarised in under a
minute without decompressing a single point record.

Per tile it records the point count, the bounding box and its area, the point
density, the elevation range, the sensor and processing software strings, the
LAS version and point format, and the EPSG code from the GeoTIFF VLR. Per
regional dataset it aggregates those, and joins the detection counts from the
catalogue so acquisition properties can be read against yield.

Usage:
    python tools/data_inventory.py
    python tools/data_inventory.py --out-dir intermediate
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
from pathlib import Path

import laspy
import numpy as np
import pandas as pd

HEADER_BYTES = 1 << 16


def read_header(path: Path) -> dict | None:
    try:
        with gzip.open(path, "rb") as g:
            chunk = g.read(HEADER_BYTES)
        h = laspy.open(io.BytesIO(chunk)).header
    except Exception as exc:                      # noqa: BLE001
        print(f"  ! {path.name}: {exc}")
        return None

    epsg = None
    for vlr in h.vlrs:
        if getattr(vlr, "record_id", None) == 34735:
            for gk in getattr(vlr, "geo_keys", []):
                if gk.id == 3072:
                    epsg = int(gk.value_offset)
    w = float(h.maxs[0] - h.mins[0])
    ht = float(h.maxs[1] - h.mins[1])
    return {
        "source_folder": path.parent.name,
        "filename": path.name,
        "bytes_gz": path.stat().st_size,
        "point_count": int(h.point_count),
        "las_version": str(h.version),
        "point_format": int(h.point_format.id),
        "creation_date": str(h.creation_date),
        "system_identifier": h.system_identifier.strip(),
        "generating_software": h.generating_software.strip(),
        "epsg": epsg,
        "scale_xy": float(h.scales[0]),
        "min_x": float(h.mins[0]), "max_x": float(h.maxs[0]),
        "min_y": float(h.mins[1]), "max_y": float(h.maxs[1]),
        "min_z": float(h.mins[2]), "max_z": float(h.maxs[2]),
        "extent_x_m": w, "extent_y_m": ht,
        "bbox_area_km2": w * ht / 1e6,
        "density_per_m2": h.point_count / max(w * ht, 1.0),
        "relief_m": float(h.maxs[2] - h.mins[2]),
    }


def sample_classifications(data_root: Path, n_records: int = 300_000) -> pd.DataFrame:
    """
    Check whether the tiles carry a ground classification.

    The manuscript draft states that no return classification is available in
    these files and that the ground surface must therefore be estimated from
    geometry alone. That is worth testing rather than assuming, so one tile per
    regional dataset is sampled: the LAS point records are parsed directly out
    of the decompressed prefix (point format 1 is a fixed 28-byte record, with
    the classification in the low five bits of byte 15), which avoids reading a
    multi-gigabyte tile to look at a one-byte field.
    """
    rows = []
    for folder in sorted(p for p in data_root.iterdir() if p.is_dir()):
        files = sorted(folder.glob("*.las.gz"))
        if not files:
            continue
        path = files[len(files) // 2]
        try:
            with gzip.open(path, "rb") as g:
                chunk = g.read(12_000_000)
            h = laspy.open(io.BytesIO(chunk)).header
            off, rl = h.offset_to_point_data, h.point_format.size
            n = min(n_records, (len(chunk) - off) // rl)
            buf = np.frombuffer(chunk, dtype=np.uint8, count=n * rl,
                                offset=off).reshape(n, rl)
            cls = buf[:, 15] & 0x1F
        except Exception as exc:                  # noqa: BLE001
            print(f"  ! {path.name}: {exc}")
            continue
        counts = np.bincount(cls, minlength=3)
        rows.append({
            "source_folder": folder.name, "tile": path.name,
            "records_sampled": int(n),
            "class_1_unclassified": int(counts[1]),
            "class_2_ground": int(counts[2]),
            "ground_fraction": round(float(counts[2] / n), 4),
            "other_classes": int(n - counts[1] - counts[2]),
        })
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--out-dir", type=Path, default=Path("intermediate"))
    p.add_argument("--catalogue", type=Path,
                   default=Path("intermediate/gated_candidates.csv"))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(args.data_root.glob("*/*.las.gz"))
    print(f"Reading {len(files)} LAS headers...")
    rows = [r for f in files if (r := read_header(f)) is not None]
    tiles = pd.DataFrame(rows)
    tiles.to_csv(args.out_dir / "tile_inventory.csv", index=False)

    agg = tiles.groupby("source_folder").agg(
        tiles=("filename", "size"),
        gb=("bytes_gz", lambda s: s.sum() / 1e9),
        points_M=("point_count", lambda s: s.sum() / 1e6),
        area_km2=("bbox_area_km2", "sum"),
        median_density=("density_per_m2", "median"),
        median_tile_x_m=("extent_x_m", "median"),
        median_tile_y_m=("extent_y_m", "median"),
        min_z=("min_z", "min"), max_z=("max_z", "max"),
        median_relief=("relief_m", "median"),
        epsg=("epsg", lambda s: "/".join(sorted({str(v) for v in s}))),
    ).reset_index()

    if args.catalogue.exists():
        cat = pd.read_csv(args.catalogue)
        cat = cat[cat["shape_type"].fillna("rectangular") == "rectangular"]
        counts = cat.groupby("source_folder").size().rename("detections")
        agg = agg.merge(counts, on="source_folder", how="left")
        agg["detections"] = agg["detections"].fillna(0).astype(int)
        agg["detections_per_km2"] = agg["detections"] / agg["area_km2"]

    print("\nSampling point classifications, one tile per dataset...")
    cls = sample_classifications(args.data_root)
    cls.to_csv(args.out_dir / "classification_sample.csv", index=False)
    print(cls.to_string(index=False))

    agg = agg.sort_values("tiles", ascending=False)
    agg.to_csv(args.out_dir / "dataset_inventory.csv", index=False)

    summary = {
        "n_tiles": int(len(tiles)),
        "total_gb_compressed": round(float(tiles["bytes_gz"].sum() / 1e9), 1),
        "total_points": int(tiles["point_count"].sum()),
        "total_bbox_area_km2": round(float(tiles["bbox_area_km2"].sum()), 1),
        "las_versions": sorted(tiles["las_version"].unique().tolist()),
        "point_formats": sorted(int(v) for v in tiles["point_format"].unique()),
        "sensors": sorted(tiles["system_identifier"].unique().tolist()),
        "software": sorted(tiles["generating_software"].unique().tolist()),
        "epsg_codes": sorted(int(v) for v in tiles["epsg"].dropna().unique()),
        "creation_dates": [str(tiles["creation_date"].min()),
                           str(tiles["creation_date"].max())],
        "density_per_m2": {
            "min": round(float(tiles["density_per_m2"].min()), 2),
            "p25": round(float(tiles["density_per_m2"].quantile(.25)), 2),
            "median": round(float(tiles["density_per_m2"].median()), 2),
            "p75": round(float(tiles["density_per_m2"].quantile(.75)), 2),
            "max": round(float(tiles["density_per_m2"].max()), 2),
        },
        "scale_xy": sorted(float(v) for v in tiles["scale_xy"].unique()),
        "ground_classification_present": bool((cls["class_2_ground"] > 0).all()),
        "ground_fraction_range": [round(float(cls["ground_fraction"].min()), 4),
                                  round(float(cls["ground_fraction"].max()), 4)],
        "classes_observed": sorted(
            set([1] * int(cls["class_1_unclassified"].sum() > 0)
                + [2] * int(cls["class_2_ground"].sum() > 0))),
    }
    with open(args.out_dir / "data_inventory.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print()
    cols = ["source_folder", "tiles", "gb", "points_M", "area_km2",
            "median_density", "median_tile_x_m", "median_tile_y_m", "epsg"]
    if "detections" in agg:
        cols += ["detections", "detections_per_km2"]
    print(agg[cols].round(2).to_string(index=False))
    print(f"\nWrote {args.out_dir}/tile_inventory.csv, dataset_inventory.csv, "
          f"data_inventory.json")


if __name__ == "__main__":
    main()
