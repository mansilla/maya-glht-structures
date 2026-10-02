#!/usr/bin/env python3
"""
Batch process all LiDAR files to detect ancient buildings and export to CSV.

Usage:
    python -m building_detection.batch --data-root ./data
    python -m building_detection.batch --data-root ./data --max-files 5
"""
from __future__ import annotations

import argparse
import csv
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import numpy as np

from building_detection.lasio import load_las
from .detection import BuildingConfig, detect_buildings
from .metrics import compute_all_building_metrics


def find_all_las_files(root_dir: Path) -> List[Path]:
    """Find all .las, .las.gz, and .laz files recursively."""
    las_files = []
    for path in root_dir.rglob("*"):
        if path.suffix == '.las' or path.name.endswith('.las.gz') or path.suffix == '.laz':
            las_files.append(path)
    return sorted(las_files)


def _detect_utm_zone(points) -> int:
    """Detect UTM zone from easting values."""
    median_easting = float(np.median(points[:, 0]))
    return 16 if median_easting < 600_000 else 15


def save_cropped_points(
    points: np.ndarray,
    candidate,
    output_dir: Path,
    base_name: str,
    building_idx: int,
    buffer_m: float = 10.0,
) -> str:
    """Save LiDAR points around a detected building as compressed .npy."""
    center = candidate.center_xy
    half_w = candidate.width_m / 2 + buffer_m
    half_h = candidate.height_m / 2 + buffer_m

    dx = points[:, 0] - center[0]
    dy = points[:, 1] - center[1]
    mask = (np.abs(dx) <= half_w) & (np.abs(dy) <= half_h)
    cropped = points[mask]

    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{base_name}_building{building_idx}.npy"
    out_path = crops_dir / filename
    np.save(str(out_path), cropped)
    return str(out_path)


def process_las_file(
    las_path: Path,
    config: BuildingConfig,
    utm_zone: int = None,
    save_crops: bool = True,
    output_dir: Path = None,
) -> List[Dict]:
    """Process a single LAS file. Returns list of building metric dicts."""
    results = []

    points = load_las(str(las_path))
    if utm_zone is None:
        utm_zone = _detect_utm_zone(points)
    print(f"  {len(points):,} points (UTM zone {utm_zone})", end=" ", flush=True)

    detection = detect_buildings(points, config)
    n_circ = sum(1 for c in detection.candidates if c.shape_type == 'circular')
    n_rect = len(detection.candidates) - n_circ
    print(f"-> {len(detection.raw_candidates)} raw, "
          f"{len(detection.candidates)} filtered "
          f"({n_rect} rectangular, {n_circ} circular)")

    if len(detection.candidates) == 0:
        return results

    base_name = las_path.stem.replace('.las', '')

    for idx, candidate in enumerate(detection.candidates):
        metrics = compute_all_building_metrics(candidate, points, utm_zone=utm_zone)
        metrics['source_folder'] = las_path.parent.name
        metrics['filename'] = las_path.name
        metrics['building_id'] = idx + 1

        if candidate.shape_type == 'circular':
            print(f"    Circular {idx + 1}: "
                  f"d={candidate.diameter_m:.0f}m, "
                  f"area={candidate.area_m2:.0f}m², "
                  f"circ={candidate.circular_score:.2f}, "
                  f"tmpl={candidate.circ_template_score:.2f}, "
                  f"sym={candidate.radial_symmetry:.2f}, "
                  f"iso={candidate.directional_isotropy:.2f}, "
                  f"conf={metrics['building_confidence']:.2f}")
        else:
            print(f"    Building {idx + 1}: "
                  f"{candidate.width_m:.0f}x{candidate.height_m:.0f}m, "
                  f"area={candidate.area_m2:.0f}m², "
                  f"angle={candidate.angle_score:.2f}, "
                  f"tmpl={candidate.template_score:.2f}, "
                  f"rect={candidate.rectangularity:.2f}, "
                  f"wall={candidate.wall_score:.2f}, "
                  f"conf={metrics['building_confidence']:.2f}")

        if save_crops and output_dir:
            crop_path = save_cropped_points(
                points, candidate, output_dir, base_name, idx + 1,
            )
            metrics['crop_file'] = crop_path

        results.append(metrics)

    return results


FIELDNAMES = [
    'source_folder', 'filename', 'building_id',
    'latitude', 'longitude', 'easting', 'northing',
    'width_m', 'height_m', 'area_m2', 'orientation_deg', 'aspect_ratio',
    'rectangularity', 'wall_score', 'edge_density_ratio',
    'angle_score', 'template_score', 'dominant_orientation_deg',
    'shape_type', 'circular_score', 'circ_template_score',
    'radial_symmetry', 'directional_isotropy', 'roundness', 'diameter_m',
    'peak_height_m', 'mean_height_m', 'building_confidence', 'confidence_raw',
    'num_ground_points', 'num_elevated_points',
]


def run_batch(
    data_root: str = "./data",
    output_dir: str = "outputs/detector",
    csv_name: str = "buildings.csv",
    save_crops: bool = True,
    max_files: int = None,
    utm_zone: int = None,
    config: BuildingConfig = None,
) -> str:
    """
    Run batch building detection on all LAS files.
    Returns path to the output CSV.
    """
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config is None:
        config = BuildingConfig()

    print("=" * 70)
    print("LiDAR Ancient Building Detection - Batch Processing")
    print("=" * 70)
    print(f"Data root:  {data_root}")
    print(f"Output dir: {output_dir}")
    print(f"Config:     cell={config.dem_cell_size}m, "
          f"smooth={config.smoothing_radius_m}m, "
          f"rel_elev=[{config.min_relative_elevation},{config.max_relative_elevation}]m, "
          f"area=[{config.min_cluster_area_m2},{config.max_cluster_area_m2}]m², "
          f"min_rect={config.min_rectangularity}, "
          f"min_wall={config.min_wall_score}")
    shapes = [n for n, on in (("rectangular", config.detect_rectangular),
                              ("circular", config.detect_circular)) if on]
    print(f"Shapes:     {', '.join(shapes) if shapes else 'NONE'}"
          + (f"  (min_circular_score={config.min_circular_score})"
             if config.detect_circular else ""))
    print(f"Started:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    print("\nScanning for LiDAR files...")
    las_files = find_all_las_files(data_root)
    print(f"Found {len(las_files)} LAS files")

    if not las_files:
        print("No files found!")
        return ""

    if max_files:
        las_files = las_files[:max_files]
        print(f"Processing first {max_files} files")

    # CSV setup -- write incrementally so results survive crashes
    csv_path = output_dir / csv_name
    csv_file = open(csv_path, 'w', newline='')
    writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES, extrasaction='ignore')
    writer.writeheader()
    csv_file.flush()

    all_results = []
    error_count = 0

    try:
        for i, las_path in enumerate(las_files):
            print(f"\n[{i+1}/{len(las_files)}] {las_path.parent.name}/{las_path.name}")
            try:
                results = process_las_file(
                    las_path, config,
                    utm_zone=utm_zone,
                    save_crops=save_crops,
                    output_dir=output_dir,
                )
                all_results.extend(results)
                if results:
                    writer.writerows(results)
                    csv_file.flush()
            except Exception as e:
                print(f"  ERROR: {e}")
                traceback.print_exc()
                error_count += 1

            if (i + 1) % 20 == 0:
                print(f"\n--- Progress: {i+1}/{len(las_files)} files, "
                      f"{len(all_results)} buildings found, {error_count} errors ---")
    finally:
        csv_file.close()

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"Files processed:      {len(las_files)}")
    print(f"Errors:               {error_count}")
    print(f"Buildings detected:   {len(all_results)}")
    n_circ_total = sum(1 for r in all_results if r.get('shape_type') == 'circular')
    print(f"  rectangular:        {len(all_results) - n_circ_total}")
    print(f"  circular:           {n_circ_total}")
    print(f"CSV file:             {csv_path}")
    if save_crops:
        print(f"Cropped point clouds: {output_dir / 'crops'}")
    print(f"Finished:             {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 70}")

    if all_results:
        print(f"\nTop detections by confidence:")
        sorted_results = sorted(all_results, key=lambda r: r['building_confidence'], reverse=True)
        print(f"  {'File':<40} {'Size':>10} {'Area':>7} {'Angle':>6} {'Tmpl':>5} {'Rect':>5} {'Wall':>5} {'Conf':>5}")
        print(f"  {'-'*85}")
        for r in sorted_results[:15]:
            print(f"  {r['filename'][:38]:<40} "
                  f"{r['width_m']:>4.0f}x{r['height_m']:<4.0f}m "
                  f"{r['area_m2']:>6.0f} "
                  f"{r['angle_score']:>6.2f} "
                  f"{r['template_score']:>5.2f} "
                  f"{r['rectangularity']:>5.2f} "
                  f"{r['wall_score']:>5.2f} "
                  f"{r['building_confidence']:>5.2f}")

    return str(csv_path)


def main():
    parser = argparse.ArgumentParser(
        description="Batch detect ancient buildings in LiDAR data and export CSV",
    )
    parser.add_argument("--data-root", type=str,
                        default="data",
                        help="Root directory containing LAS files")
    parser.add_argument("--output-dir", type=str,
                        default="outputs/detector",
                        help="Output directory for CSV and crops")
    parser.add_argument("--csv-name", type=str, default="buildings.csv",
                        help="Name of the output CSV file")
    parser.add_argument("--no-crops", action="store_true",
                        help="Skip saving cropped point clouds")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Process at most N files (for testing)")
    parser.add_argument("--utm-zone", type=int, default=None,
                        help="UTM zone (auto-detect if not set)")

    # Detection parameter overrides
    parser.add_argument("--cell-size", type=float, default=None)
    parser.add_argument("--smoothing-radius", type=float, default=None)
    parser.add_argument("--min-rel-elev", type=float, default=None)
    parser.add_argument("--max-rel-elev", type=float, default=None)
    parser.add_argument("--min-area", type=float, default=None)
    parser.add_argument("--max-area", type=float, default=None)
    parser.add_argument("--min-rectangularity", type=float, default=None)
    parser.add_argument("--min-wall-score", type=float, default=None)
    parser.add_argument("--max-aspect-ratio", type=float, default=None)
    parser.add_argument("--min-edge-density", type=float, default=None)

    # Shape families
    parser.add_argument("--circular", action="store_true",
                        help="Also detect circular structures")
    parser.add_argument("--circular-only", action="store_true",
                        help="Detect only circular structures")
    parser.add_argument("--min-circular-score", type=float, default=None)
    parser.add_argument("--min-radial-symmetry", type=float, default=None)
    parser.add_argument("--max-circular-aspect-ratio", type=float, default=None)

    args = parser.parse_args()

    config = BuildingConfig()
    if args.cell_size is not None:
        config.dem_cell_size = args.cell_size
    if args.smoothing_radius is not None:
        config.smoothing_radius_m = args.smoothing_radius
    if args.min_rel_elev is not None:
        config.min_relative_elevation = args.min_rel_elev
    if args.max_rel_elev is not None:
        config.max_relative_elevation = args.max_rel_elev
    if args.min_area is not None:
        config.min_cluster_area_m2 = args.min_area
    if args.max_area is not None:
        config.max_cluster_area_m2 = args.max_area
    if args.min_rectangularity is not None:
        config.min_rectangularity = args.min_rectangularity
    if args.min_wall_score is not None:
        config.min_wall_score = args.min_wall_score
    if args.max_aspect_ratio is not None:
        config.max_aspect_ratio = args.max_aspect_ratio
    if args.min_edge_density is not None:
        config.min_edge_density_ratio = args.min_edge_density
    if args.circular or args.circular_only:
        config.detect_circular = True
    if args.circular_only:
        config.detect_rectangular = False
    if args.min_circular_score is not None:
        config.min_circular_score = args.min_circular_score
    if args.min_radial_symmetry is not None:
        config.min_radial_symmetry = args.min_radial_symmetry
    if args.max_circular_aspect_ratio is not None:
        config.max_circular_aspect_ratio = args.max_circular_aspect_ratio

    run_batch(
        data_root=args.data_root,
        output_dir=args.output_dir,
        csv_name=args.csv_name,
        save_crops=not args.no_crops,
        max_files=args.max_files,
        utm_zone=args.utm_zone,
        config=config,
    )


if __name__ == "__main__":
    main()
