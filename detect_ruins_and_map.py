#!/usr/bin/env python3
"""
Detect ancient buried ruins in LiDAR data and generate interactive HTML maps.

Processes ALL data folders, auto-detecting the UTM zone from each LAS file's
GeoTIFF VLR header.

Approach: detect man-made shapes in the subsurface layer (0 to -2m below
ground level). Ancient Maya buildings buried under centuries of soil and
vegetation create subtle below-ground signatures that LiDAR can detect.

Two shape families are reported, both strictly below ground:

  rectangular  right-angle foundations -- edge normals concentrate into two
               perpendicular directions
  circular     round platforms, ring walls, chultun mouths and altars -- edge
               normals show no preferred direction and sit at a constant
               radius from the centre

Pipeline:
  1. Estimate ground surface (smoothed DEM)
  2. Filter LiDAR points to 0-2m below ground
  3. Build depth map from subsurface points
  4. Detect rectangular AND circular patterns in the depth map
  5. Filter each family by its own shape evidence and geometry
  6. Map both families, rectangles in blue and circles in magenta
"""
import csv
import gzip
import io
import sys
import traceback
from datetime import datetime
from pathlib import Path

import laspy
import numpy as np
from pyproj import Transformer

from building_detection.lasio import load_las
from building_detection.detection import BuildingConfig, detect_buildings
from building_detection.metrics import compute_all_building_metrics
from building_detection.batch import FIELDNAMES
from building_detection.visualization import (
    SHAPE_COLORS, SHAPE_LABELS, base_map, shape_color, shape_legend_html,
)

# ── Configuration ──────────────────────────────────────────────────────────
DATA_ROOT = Path("data")
OUTPUT_DIR = Path("outputs/detector")
MIN_SIDE_TO_HEIGHT_RATIO = 2.0      # side / depth (relaxed for buried features)

config = BuildingConfig(
    # Subsurface mode: look below ground instead of above
    subsurface_mode=True,
    subsurface_max_depth=2.0,           # search 0 to -2m below ground

    # Mask thresholds: deliberately wide. Do NOT narrow these to chase
    # precision -- it backfires. A broad band lets a saturated slope or a
    # noise field form ONE large connected component that max_cluster_area_m2
    # then rejects; narrowing the band shatters that same artifact into many
    # small components that each pass the area filter. Measured on a Chiapas
    # tile: narrowing to [0.15, 1.80] raised detections from 1,508 to 3,538.
    min_relative_elevation=0.05,        # min depth 5cm (was 0.3m for surface)
    max_relative_elevation=2.0,         # max depth 2m (was 3.0m)

    # Depth gates. These carry the precision, and they are set from the
    # instrument rather than by taste. The depth histogram of an unrestricted
    # run is bimodal, and both modes are artefacts of the search window:
    #
    #   below ~0.25 m   sub-noise micro-relief. G-LiHT's vertical accuracy is
    #                   ~0.10-0.15 m, so a 0.11 m "depression" is not a
    #                   measurement. 57% of an unrestricted catalogue, and
    #                   82% of the flat-karst Yucatan GLAS tiles.
    #   at/above 1.8 m  truncated, not measured. The subsurface window stops
    #                   at subsurface_max_depth, so anything deeper saturates
    #                   at that ceiling -- 9,332 detections landed on exactly
    #                   2.00 m. Concentrated in steep, dense-canopy Chiapas
    #                   (85% of Chiaps_1_GLAS), where min-Z ground modelling
    #                   breaks down. Genuinely deep closed depressions
    #                   (sinkholes) are outside this detector's scope.
    #
    # The published catalogue applies these gates AFTER this script, with
    # tools/filter_catalogue.py (peak depth 0.25-1.80 m, mean depth >= 0.15 m)
    # on the peak_height_m / mean_height_m columns of ruins.csv. They are
    # disabled here on purpose: the detector's own gate tests an internal depth
    # estimate that differs slightly from those columns (on tile
    # AMIGACarb_Yuc_North_2_GLAS_Apr2013_l3s458 it drops 4 of the 59 rows the
    # published route keeps), so gating here would not reproduce the catalogue.
    min_peak_height=0.0,
    max_peak_height=1e9,
    min_mean_relative_elevation=0.0,

    # Geometry filters (same as surface mode)
    dem_cell_size=2.0,
    smoothing_radius_m=25.0,
    min_cluster_area_m2=25.0,
    max_cluster_area_m2=900.0,
    max_aspect_ratio=5.0,

    # Relax secondary filters for weaker subsurface signal
    min_rectangularity=0.15,            # was 0.20
    min_wall_score=0.05,                # was 0.10
    min_edge_density_ratio=0.5,         # was 1.0
    min_angle_score=0.03,               # was 0.05
    min_template_score=0.0,

    # Report round structures alongside the rectangular ones
    detect_rectangular=True,
    detect_circular=True,
    min_circular_score=0.45,            # combined circular evidence
    min_radial_symmetry=0.35,           # boundary at a near-constant radius
    min_directional_isotropy=0.30,      # no preferred edge direction
    max_circular_aspect_ratio=1.6,      # roughly as wide as it is tall
    min_circular_diameter_m=8.0,        # 4 DEM cells; below this a 2m DEM
                                        # cannot tell a circle from a blob
)


# ── UTM Zone Detection ───────────────────────────────────────────────────
def detect_utm_zone(las_path: Path) -> int:
    """Read the EPSG code from a LAS.gz file's GeoTIFF VLR header.

    Returns the UTM zone number (e.g. 15 or 16).
    Falls back to 16 if the header can't be read.
    """
    try:
        with gzip.open(str(las_path), 'rb') as gz:
            data = gz.read()
        las = laspy.read(io.BytesIO(data))
        for vlr in las.header.vlrs:
            if vlr.record_id == 34735:  # GeoKeyDirectoryTag
                for gk in vlr.geo_keys:
                    if gk.id == 3072:  # ProjectedCSTypeGeoKey
                        epsg = gk.value_offset
                        # EPSG 326XX = UTM Zone XX North
                        if 32600 < epsg < 32661:
                            return epsg - 32600
        return 16
    except Exception:
        return 16


# Cache: folder_name -> utm_zone
_folder_zones = {}


def get_folder_utm_zone(folder: Path) -> int:
    """Get UTM zone for a folder (reads first file, caches result)."""
    name = folder.name
    if name not in _folder_zones:
        sample = next(folder.glob("*.las.gz"), None)
        if sample:
            _folder_zones[name] = detect_utm_zone(sample)
        else:
            _folder_zones[name] = 16
    return _folder_zones[name]


# ── Detection ──────────────────────────────────────────────────────────────
def run_detection(resume=True):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "ruins.csv"

    # Collect all LAS files across all data folders
    all_folders = sorted(
        d for d in DATA_ROOT.iterdir()
        if d.is_dir() and list(d.glob("*.las.gz"))
    )
    all_las_files = []
    for folder in all_folders:
        for f in sorted(folder.glob("*.las.gz")):
            all_las_files.append(f)

    # Check for existing results to resume from. The CSV gained shape columns
    # when circular detection was added, so a file written by the old schema
    # cannot be appended to -- it is set aside rather than silently mixed.
    done_files = set()
    prior_results = []
    if resume and csv_path.exists():
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames or []
            if header != FIELDNAMES:
                missing = [c for c in FIELDNAMES if c not in header]
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                legacy = csv_path.with_name(f"ruins_legacy_{stamp}.csv")
                print(f"WARNING: {csv_path.name} uses the previous column "
                      f"layout (missing {', '.join(missing) or 'columns'}).")
                print(f"         Moving it to {legacy.name} and starting a "
                      f"fresh run with the new schema.")
                f.close()
                csv_path.rename(legacy)
            else:
                for row in reader:
                    done_files.add(row["filename"])
                    prior_results.append(row)
                if done_files:
                    print(f"RESUME MODE: {len(done_files)} files already "
                          f"processed ({len(prior_results)} detections), "
                          f"skipping them.")

    remaining = [f for f in all_las_files if f.name not in done_files]

    print(f"Data root: {DATA_ROOT}")
    print(f"Output:    {OUTPUT_DIR}")
    print(f"Folders:   {len(all_folders)}")
    print(f"Files:     {len(all_las_files)} total, {len(remaining)} remaining")
    print(f"Mode:      SUBSURFACE (0 to -{config.subsurface_max_depth}m below ground)")
    shapes = [n for n, on in (("rectangular", config.detect_rectangular),
                              ("circular", config.detect_circular)) if on]
    print(f"Shapes:    {', '.join(shapes)}")
    print(f"Filter:    span/depth >= {MIN_SIDE_TO_HEIGHT_RATIO}")

    # Detect UTM zones per folder
    print("\nUTM zones:")
    for folder in all_folders:
        zone = get_folder_utm_zone(folder)
        n_files = len(list(folder.glob("*.las.gz")))
        print(f"  {folder.name}: Zone {zone} ({n_files} files)")

    print("=" * 70)

    # Append mode if resuming with existing data, otherwise write fresh
    if done_files:
        csv_file = open(csv_path, "a", newline="")
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES, extrasaction="ignore")
    else:
        csv_file = open(csv_path, "w", newline="")
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
    csv_file.flush()

    # Track tile bounding boxes: filename -> (e_min, e_max, n_min, n_max, utm_zone)
    tile_bounds = {}
    all_results = list(prior_results)
    folder_stats = {}  # folder_name -> (n_files_processed, n_ruins)

    for i, las_path in enumerate(remaining):
        folder_name = las_path.parent.name
        utm_zone = get_folder_utm_zone(las_path.parent)
        sz_mb = las_path.stat().st_size / 1e6
        print(f"\n[{i+1}/{len(remaining)}] {folder_name}/{las_path.name} "
              f"({sz_mb:.0f}MB, UTM{utm_zone})")

        try:
            points = load_las(str(las_path))
            print(f"  {len(points):,} points", end=" ", flush=True)

            # Store tile extent with UTM zone
            tile_bounds[las_path.name] = (
                float(points[:, 0].min()), float(points[:, 0].max()),
                float(points[:, 1].min()), float(points[:, 1].max()),
                utm_zone,
            )

            detection = detect_buildings(points, config)
            print(f"-> {len(detection.raw_candidates)} raw, "
                  f"{len(detection.candidates)} filtered")

            kept = 0
            kept_by_shape = {"rectangular": 0, "circular": 0}
            for cand in detection.candidates:
                metrics = compute_all_building_metrics(
                    cand, points, utm_zone=utm_zone
                )
                peak_h = float(metrics.get("peak_height_m", 0))
                # A buried structure must be wider than it is deep. For a
                # circle the horizontal span is its diameter, not a side.
                if cand.shape_type == "circular":
                    span = cand.diameter_m
                else:
                    span = max(cand.width_m, cand.height_m)

                if peak_h <= 0 or span / peak_h < MIN_SIDE_TO_HEIGHT_RATIO:
                    continue

                kept += 1
                kept_by_shape[cand.shape_type] = (
                    kept_by_shape.get(cand.shape_type, 0) + 1
                )
                metrics["source_folder"] = folder_name
                metrics["filename"] = las_path.name
                metrics["building_id"] = kept
                conf = float(metrics["building_confidence"])
                ratio = span / peak_h

                if cand.shape_type == "circular":
                    shape_desc = f"circular d={cand.diameter_m:.1f}m"
                else:
                    shape_desc = f"rect {cand.width_m:.1f}x{cand.height_m:.1f}m"
                print(f"    #{kept}: {shape_desc} "
                      f"depth={peak_h:.2f}m ratio={ratio:.1f} conf={conf:.3f}")

                writer.writerow(metrics)
                csv_file.flush()
                all_results.append(metrics)

            if kept:
                print(f"  Ruins: {kept} "
                      f"({kept_by_shape['rectangular']} rectangular, "
                      f"{kept_by_shape['circular']} circular)")

            # Update folder stats
            if folder_name not in folder_stats:
                folder_stats[folder_name] = [0, 0]
            folder_stats[folder_name][0] += 1
            folder_stats[folder_name][1] += kept

        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()

    csv_file.close()
    n_circ = sum(1 for r in all_results
                 if (r.get("shape_type") or "rectangular") == "circular")
    print(f"\n{'='*70}")
    print(f"Total ruins detected: {len(all_results)}")
    print(f"  rectangular:        {len(all_results) - n_circ}")
    print(f"  circular:           {n_circ}")
    print(f"\nPer-folder summary:")
    for fname in sorted(folder_stats):
        n_proc, n_ruins = folder_stats[fname]
        print(f"  {fname}: {n_proc} files, {n_ruins} ruins")
    print(f"\nCSV: {csv_path}")

    return all_results, tile_bounds


# ── HTML Map Generation ───────────────────────────────────────────────────
def generate_maps(all_results, tile_bounds):
    """Generate overview heatmap + per-folder detail maps.

    Uses lat/lon from CSV directly (already computed with correct UTM zone
    during detection) instead of re-transforming from UTM.
    """
    import folium
    from folium.plugins import HeatMap, MarkerCluster

    # Build transformers (needed only for rectangle corners in per-folder maps)
    folder_zones = {}
    for folder in sorted(DATA_ROOT.iterdir()):
        if folder.is_dir() and list(folder.glob("*.las.gz")):
            folder_zones[folder.name] = get_folder_utm_zone(folder)
    transformers = {}
    for z in set(folder_zones.values()):
        transformers[z] = Transformer.from_crs(
            f"EPSG:326{z:02d}", "EPSG:4326", always_xy=True
        )

    # Parse detections using lat/lon from CSV
    detections = []
    for r in all_results:
        lat = float(r["latitude"])
        lon = float(r["longitude"])
        w = float(r["width_m"])
        h = float(r["height_m"])
        ph = float(r["peak_height_m"])
        conf = float(r["building_confidence"])
        shape = r.get("shape_type") or "rectangular"
        diameter = float(r.get("diameter_m") or 0.0) or (w + h) / 2
        span = diameter if shape == "circular" else max(w, h)
        ratio = span / ph if ph > 0 else 0
        detections.append({
            "lat": lat, "lon": lon,
            "easting": float(r["easting"]),
            "northing": float(r["northing"]),
            "width": w, "height": h,
            "shape": shape,
            "diameter": diameter,
            "span": span,
            "peak_h": ph, "conf": conf,
            "ratio": ratio,
            "filename": r["filename"],
            "source_folder": r["source_folder"],
            "building_id": r["building_id"],
            "area": float(r["area_m2"]),
            "orientation": float(r["orientation_deg"]),
            "circular_score": float(r.get("circular_score") or 0.0),
            "circ_template_score": float(r.get("circ_template_score") or 0.0),
            "radial_symmetry": float(r.get("radial_symmetry") or 0.0),
            "directional_isotropy": float(r.get("directional_isotropy") or 0.0),
            "angle_score": float(r.get("angle_score") or 0.0),
            "template_score": float(r.get("template_score") or 0.0),
        })

    n_by_shape = {"rectangular": 0, "circular": 0}
    for d in detections:
        n_by_shape[d["shape"]] = n_by_shape.get(d["shape"], 0) + 1
    print(f"  {n_by_shape['rectangular']} rectangular, "
          f"{n_by_shape['circular']} circular")

    # ── 1. Overview map: heatmap (lightweight, no individual markers) ──
    print("\nGenerating overview heatmap...")
    center_lat = np.mean([d["lat"] for d in detections])
    center_lon = np.mean([d["lon"] for d in detections])

    m = base_map([center_lat, center_lon], 7)

    # One heatmap per shape family, each in its own hue, so density in the
    # rectangular and the circular record can be compared or viewed alone.
    heat_gradients = {
        "rectangular": {0.3: '#80d8ff', 0.6: '#00b3ff', 1.0: '#0033cc'},
        "circular":    {0.3: '#ffa3d7', 0.6: '#ff4db8', 1.0: '#99004f'},
    }
    for shape in ("rectangular", "circular"):
        shape_dets = [d for d in detections if d["shape"] == shape]
        if not shape_dets:
            continue
        HeatMap(
            [[d["lat"], d["lon"], d["conf"]] for d in shape_dets],
            name=f"{SHAPE_LABELS[shape]} heatmap ({len(shape_dets):,})",
            min_opacity=0.3, radius=12, blur=8,
            gradient=heat_gradients[shape],
        ).add_to(m)

    # High-confidence markers only (clustered) — much smaller subset
    high_conf = [d for d in detections if d["conf"] > 0.4]
    high_by_shape = {"rectangular": 0, "circular": 0}
    for shape in ("rectangular", "circular"):
        shape_high = [d for d in high_conf if d["shape"] == shape]
        high_by_shape[shape] = len(shape_high)
        if not shape_high:
            continue
        cluster = MarkerCluster(
            name=f"{SHAPE_LABELS[shape]} — high confidence ({len(shape_high):,})"
        )
        for d in shape_high:
            color = shape_color(shape, d["conf"])
            if shape == "circular":
                size_line = f"<b>Diameter:</b> {d['diameter']:.1f} m<br>"
            else:
                size_line = (f"<b>Footprint:</b> {d['width']:.1f} x "
                             f"{d['height']:.1f} m<br>")
            popup_html = f"""
            <div style='font-family:monospace; font-size:12px; min-width:220px;'>
            <b>Buried Ruin #{d['building_id']}</b><br>
            <b>Shape:</b> {SHAPE_LABELS[shape]}<br>
            <b>File:</b> {d['filename']}<br>
            <b>Dataset:</b> {d['source_folder']}<br>
            <hr style='margin:4px 0;'>
            {size_line}
            <b>Area:</b> {d['area']:.0f} m&sup2;<br>
            <b>Depth:</b> {d['peak_h']:.2f} m<br>
            <b>Confidence:</b> {d['conf']:.3f}<br>
            <b>WGS84:</b> {d['lat']:.5f}, {d['lon']:.5f}<br>
            </div>
            """
            folium.CircleMarker(
                location=[d["lat"], d["lon"]],
                radius=5, color=color, fill=True,
                fill_color=color, fill_opacity=0.7,
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"#{d['building_id']} {shape} conf={d['conf']:.2f}",
            ).add_to(cluster)
        cluster.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(
        shape_legend_html(n_by_shape, title="Buried structures")
    ))

    overview_path = OUTPUT_DIR / "ruins_overview_map.html"
    m.save(str(overview_path))
    print(f"  Saved: {overview_path} ({len(detections)} total, "
          f"{high_by_shape['rectangular']} rectangular + "
          f"{high_by_shape['circular']} circular high-conf markers)")

    # ── 2. Per-folder maps with MarkerCluster ──
    maps_dir = OUTPUT_DIR / "tile_maps"
    maps_dir.mkdir(exist_ok=True)

    dets_by_folder = {}
    for d in detections:
        dets_by_folder.setdefault(d["source_folder"], []).append(d)

    all_folder_names = set()
    for folder in sorted(DATA_ROOT.iterdir()):
        if folder.is_dir() and list(folder.glob("*.las.gz")):
            all_folder_names.add(folder.name)
    all_folder_names.update(dets_by_folder.keys())

    print(f"\nGenerating per-folder maps for {len(all_folder_names)} datasets...")

    index_rows = []
    for folder_name in sorted(all_folder_names):
        folder_dets = dets_by_folder.get(folder_name, [])

        if not folder_dets:
            n_files = len(list((DATA_ROOT / folder_name).glob("*.las.gz"))) if (DATA_ROOT / folder_name).exists() else 0
            index_rows.append((folder_name, n_files, 0, 0, 0, 0,
                               f"{folder_name}.html"))
            continue

        center_lat = np.mean([d["lat"] for d in folder_dets])
        center_lon = np.mean([d["lon"] for d in folder_dets])

        tm = base_map([center_lat, center_lon], 12)

        # One cluster per shape family so either can be toggled off
        zone = folder_zones.get(folder_name, 16)
        t = transformers.get(zone)
        folder_counts = {"rectangular": 0, "circular": 0}
        for d in folder_dets:
            folder_counts[d["shape"]] = folder_counts.get(d["shape"], 0) + 1

        clusters = {}
        for shape in ("rectangular", "circular"):
            if folder_counts.get(shape):
                clusters[shape] = MarkerCluster(
                    name=f"{SHAPE_LABELS[shape]} ({folder_counts[shape]:,})"
                )

        for d in folder_dets:
            shape = d["shape"]
            color = shape_color(shape, d["conf"])

            if shape == "circular":
                geometry_lines = (
                    f"<b>Diameter:</b> {d['diameter']:.1f} m<br>"
                    f"<b>Area:</b> {d['area']:.0f} m&sup2;<br>"
                )
                evidence_lines = (
                    f"<b>Circular score:</b> {d['circular_score']:.3f}<br>"
                    f"<b>Ring template:</b> {d['circ_template_score']:.3f}<br>"
                    f"<b>Radial symmetry:</b> {d['radial_symmetry']:.3f}<br>"
                    f"<b>Isotropy:</b> {d['directional_isotropy']:.3f}<br>"
                )
                tooltip = (f"#{d['building_id']} circular d={d['diameter']:.0f}m "
                           f"depth={d['peak_h']:.1f}m conf={d['conf']:.2f}")
            else:
                geometry_lines = (
                    f"<b>Footprint:</b> {d['width']:.1f} x {d['height']:.1f} m<br>"
                    f"<b>Area:</b> {d['area']:.0f} m&sup2;<br>"
                    f"<b>Orientation:</b> {d['orientation']:.1f}&deg;<br>"
                )
                evidence_lines = (
                    f"<b>Angle score:</b> {d['angle_score']:.3f}<br>"
                    f"<b>Template score:</b> {d['template_score']:.3f}<br>"
                )
                tooltip = (f"#{d['building_id']} "
                           f"{d['width']:.0f}x{d['height']:.0f}m "
                           f"depth={d['peak_h']:.1f}m conf={d['conf']:.2f}")

            popup_html = f"""
            <div style='font-family:monospace; font-size:12px; min-width:220px;'>
            <b>Buried Ruin #{d['building_id']}</b><br>
            <b>Shape:</b> {SHAPE_LABELS[shape]}<br>
            <hr style='margin:4px 0;'>
            {geometry_lines}
            <b>Depth below ground:</b> {d['peak_h']:.2f} m<br>
            <b>Span/depth ratio:</b> {d['ratio']:.1f}<br>
            <hr style='margin:4px 0;'>
            {evidence_lines}
            <b>Confidence:</b> {d['conf']:.3f}<br>
            <b>WGS84:</b> {d['lat']:.5f}, {d['lon']:.5f}<br>
            </div>
            """

            if shape == "circular":
                # Circle drawn as a polygon so it scales with the map in metres
                radius = d["diameter"] / 2
                angles = np.linspace(0, 2 * np.pi, 36, endpoint=False)
                ring_utm = [(d["easting"] + radius * np.cos(a),
                             d["northing"] + radius * np.sin(a)) for a in angles]
            else:
                # Oriented rectangle on zoomed-in per-folder maps
                half_w = d["width"] / 2
                half_h = d["height"] / 2
                angle_rad = np.radians(d["orientation"])
                cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
                ring_utm = []
                for dx, dy in [(-half_w, -half_h), (half_w, -half_h),
                               (half_w, half_h), (-half_w, half_h)]:
                    rx = dx * cos_a - dy * sin_a
                    ry = dx * sin_a + dy * cos_a
                    ring_utm.append((d["easting"] + rx, d["northing"] + ry))

            ring_ll = [t.transform(e, n) for e, n in ring_utm]
            ring_latlon = [(lat, lon) for lon, lat in ring_ll]
            ring_latlon.append(ring_latlon[0])

            folium.Polygon(
                locations=ring_latlon,
                color=color, weight=2, fill=True,
                fill_color=color, fill_opacity=0.25,
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=tooltip,
            ).add_to(clusters[shape])

        for cluster in clusters.values():
            cluster.add_to(tm)
        folium.LayerControl(collapsed=False).add_to(tm)
        tm.get_root().html.add_child(folium.Element(
            shape_legend_html(folder_counts, title="Buried structures")
        ))

        tile_html = maps_dir / f"{folder_name}.html"
        tm.save(str(tile_html))

        n_high = sum(1 for d in folder_dets if d["conf"] > 0.4)
        n_files = len(list((DATA_ROOT / folder_name).glob("*.las.gz"))) if (DATA_ROOT / folder_name).exists() else 0
        index_rows.append((folder_name, n_files, len(folder_dets), n_high,
                           folder_counts["rectangular"], folder_counts["circular"],
                           tile_html.name))
        print(f"  {folder_name}: {len(folder_dets)} ruins "
              f"({folder_counts['rectangular']} rectangular, "
              f"{folder_counts['circular']} circular, {n_high} high-conf)")

    # ── 3. Index HTML ──
    index_rows.sort(key=lambda x: x[2], reverse=True)
    index_html = OUTPUT_DIR / "ruins_index.html"
    total_dets = sum(r[2] for r in index_rows)
    total_high = sum(r[3] for r in index_rows)
    total_rect = sum(r[4] for r in index_rows)
    total_circ = sum(r[5] for r in index_rows)
    total_files = sum(r[1] for r in index_rows)
    rect_swatch = SHAPE_COLORS['rectangular']['high']
    circ_swatch = SHAPE_COLORS['circular']['high']
    with open(index_html, "w") as f:
        f.write(f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'>
<title>Buried Ruin Detection (Subsurface) - All Datasets</title>
<style>
body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
h1 {{ color: #333; }}
.stats {{ background: #fff; padding: 15px; border-radius: 8px; margin: 15px 0;
          box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
table {{ border-collapse: collapse; width: 100%; background: #fff;
         border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
th {{ background: #2c3e50; color: white; padding: 12px; text-align: left; }}
td {{ padding: 10px 12px; border-bottom: 1px solid #eee; }}
tr:hover {{ background: #f0f7ff; }}
a {{ color: #2980b9; text-decoration: none; font-weight: bold; }}
a:hover {{ text-decoration: underline; }}
.badge {{ display: inline-block; padding: 2px 8px; border-radius: 10px;
          font-size: 12px; font-weight: bold; }}
.high {{ background: #27ae60; color: white; }}
.med {{ background: #f39c12; color: white; }}
.low {{ background: #e74c3c; color: white; }}
.swatch {{ display: inline-block; width: 12px; height: 12px;
           border-radius: 2px; vertical-align: middle; }}
.swatch.circle {{ border-radius: 50%; }}
</style></head><body>
<h1>Buried Ruin Detection Results (Subsurface Mode)</h1>
<div class='stats'>
<b>Datasets:</b> {len(index_rows)} folders, {total_files} LAS files<br>
<b>Mode:</b> Subsurface (0 to -2m below ground)<br>
<b>Total detections:</b> {total_dets:,}<br>
<b><span class='swatch' style='background:{rect_swatch};'></span>
   Rectangular:</b> {total_rect:,} &nbsp;
<b><span class='swatch circle' style='background:{circ_swatch};'></span>
   Circular:</b> {total_circ:,}<br>
<b>High confidence (&gt;0.4):</b> {total_high:,}<br>
<b>Filter:</b> span/depth ratio &ge; {MIN_SIDE_TO_HEIGHT_RATIO}<br>
<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
<br>
<a href='ruins_overview_map.html'>Open Overview Map (all datasets)</a>
</div>
<table>
<tr><th>Dataset</th><th>Files</th><th>Detections</th>
<th><span class='swatch' style='background:{rect_swatch};'></span> Rectangular</th>
<th><span class='swatch circle' style='background:{circ_swatch};'></span> Circular</th>
<th>High Conf</th><th>Map</th></tr>
""")
        for (folder_name, n_files, n_det, n_high,
             n_rect, n_circ, html_name) in index_rows:
            badge = "high" if n_high > 0 else ("med" if n_det > 10 else "low")
            f.write(f"<tr><td><b>{folder_name}</b></td><td>{n_files}</td>"
                    f"<td>{n_det}</td>"
                    f"<td>{n_rect}</td><td>{n_circ}</td>"
                    f"<td><span class='badge {badge}'>{n_high}</span></td>"
                    f"<td><a href='tile_maps/{html_name}'>Open Map</a></td></tr>\n")
        f.write("</table></body></html>\n")

    print(f"\n{'='*70}")
    print(f"Index page: {index_html}")
    print(f"Overview map: {overview_path}")
    print(f"Per-folder maps: {maps_dir}/")
    print(f"{'='*70}")


# ── Cross-Section Plots ───────────────────────────────────────────────────
def generate_cross_sections(all_results, n_sample=30, per_shape=True):
    """
    Generate cross-section plots for the top detections by confidence.

    With per_shape (the default) the budget is split evenly between the
    shape families, so both get diagnostics. Ranking purely on confidence
    would not guarantee that: the families are far from equal in size, and
    whichever one happens to hold the top of the list can take every slot.
    """
    import matplotlib
    matplotlib.use('Agg')

    from building_detection.visualization import plot_building_cross_section
    from building_detection.detection import build_fine_dem, gaussian_smooth_dem

    cross_dir = OUTPUT_DIR / "cross_sections"
    cross_dir.mkdir(exist_ok=True)

    def by_confidence(rows):
        return sorted(rows, key=lambda r: float(r['building_confidence']),
                      reverse=True)

    if per_shape:
        shapes = ('rectangular', 'circular')
        per_family = max(1, n_sample // len(shapes))
        sample = []
        for shape in shapes:
            rows = [r for r in all_results
                    if (r.get('shape_type') or 'rectangular') == shape]
            sample.extend(by_confidence(rows)[:per_family])
        counts = {s: sum(1 for r in sample
                         if (r.get('shape_type') or 'rectangular') == s)
                  for s in shapes}
        detail = ", ".join(f"{n} {s}" for s, n in counts.items())
    else:
        sample = by_confidence(all_results)[:n_sample]
        detail = "by confidence"

    print(f"\nGenerating cross-section plots for {len(sample)} detections "
          f"({detail})...")

    # Group by (source_folder, filename) to avoid reloading the same LAS
    by_file = {}
    for r in sample:
        key = (r['source_folder'], r['filename'])
        by_file.setdefault(key, []).append(r)

    for (folder_name, fname), detections_list in by_file.items():
        las_path = DATA_ROOT / folder_name / fname
        if not las_path.exists():
            print(f"  {folder_name}/{fname}: file not found, skipping")
            continue

        print(f"  Loading {folder_name}/{fname}...")
        points = load_las(str(las_path))

        # Estimate ground surface for this tile
        dem, x_orig, y_orig = build_fine_dem(points, cell_size=config.dem_cell_size)
        sigma = config.smoothing_radius_m / config.dem_cell_size
        ground_surface = gaussian_smooth_dem(dem, sigma)

        for r in detections_list:
            bid = r['building_id']
            e, n = float(r['easting']), float(r['northing'])
            w, h = float(r['width_m']), float(r['height_m'])
            orient = float(r['orientation_deg'])
            peak_d = float(r['peak_height_m'])
            conf = float(r['building_confidence'])

            # Estimate ground Z at detection center
            col = int((e - x_orig) / config.dem_cell_size)
            row = int((n - y_orig) / config.dem_cell_size)
            col = max(0, min(col, ground_surface.shape[1] - 1))
            row = max(0, min(row, ground_surface.shape[0] - 1))
            gz = float(ground_surface[row, col])
            if np.isnan(gz):
                gz = float(np.nanmedian(dem))

            label = fname.split("_")[-1].replace(".las.gz", "")
            buf = max(max(w, h) / 2 + 15, 25.0)

            plot_building_cross_section(
                points=points,
                center_xy=np.array([e, n]),
                width_m=w,
                height_m=h,
                orientation_deg=orient,
                ground_z=gz,
                peak_depth=peak_d,
                metrics=r,
                title=f"{folder_name} / {label} - Buried Ruin #{bid} (conf={conf:.3f})",
                output_path=str(cross_dir / f"{folder_name}_{label}_ruin_{bid}_cross_section.png"),
                show=False,
                buffer_m=buf,
            )

    n_plots = len(list(cross_dir.glob("*.png")))
    print(f"\n  Cross-section plots: {n_plots} saved to {cross_dir}/")


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    all_results, tile_bounds = run_detection()
    if all_results:
        generate_maps(all_results, tile_bounds)
        generate_cross_sections(all_results, n_sample=30)
    else:
        print("No ruins detected.")
