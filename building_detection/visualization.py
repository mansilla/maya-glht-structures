"""
Visualization for building detection results.

Two output types:
  1. Interactive folium map with satellite imagery and building markers
  2. Matplotlib DEM heatmap with detected building outlines overlaid

Detections come in two shape families and are always drawn apart: a
rectangular detection is an oriented rectangle in the blue/cyan family, a
circular detection is a circle in the magenta/pink family. Within a family
the shade carries the confidence, so hue answers "what shape is it?" and
lightness answers "how sure are we?" without the two ever being confused.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

from .metrics import BuildingCandidate, BuildingDetectionResult


# ---------------------------------------------------------------------------
# Shape palette -- hue = shape family, lightness = confidence
# ---------------------------------------------------------------------------

SHAPE_COLORS = {
    'rectangular': {'high': '#0066ff', 'med': '#00b3ff', 'low': '#80d8ff'},
    'circular':    {'high': '#e6007e', 'med': '#ff4db8', 'low': '#ffa3d7'},
}

SHAPE_LABELS = {
    'rectangular': 'Rectangular structure',
    'circular': 'Circular structure',
}

# Matplotlib variants, chosen to stay readable on dark DEM colormaps
SHAPE_COLORS_MPL = {
    'rectangular': {'high': '#00b3ff', 'med': '#4dd0e1', 'low': '#b3e5fc'},
    'circular':    {'high': '#ff2d95', 'med': '#ff6fb5', 'low': '#ffc1e3'},
}

HIGH_CONFIDENCE = 0.40
MED_CONFIDENCE = 0.30


# ---------------------------------------------------------------------------
# Base map
# ---------------------------------------------------------------------------

# Every map in this project is built from these layers, in this order.
#
# OpenStreetMap is deliberately absent. Folium's "OpenStreetMap" shorthand
# points at tile.openstreetmap.org, whose usage policy does not cover bulk or
# automated clients, and their server answers one with a tile that reads
# "Access blocked -- this app is not following the tile usage policy" instead
# of map imagery. A generated archive-scale map is exactly that kind of
# client. Esri's World Street Map carries the same road-and-label content
# from a host this project already uses for the satellite layer, so the
# street view survives without asking anyone's tile server for a favour.
BASE_LAYERS = (
    ("World_Imagery",   "Satellite"),
    ("World_Street_Map", "Street map"),
    ("World_Topo_Map",  "Topographic"),
)

ESRI_TILES = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
              "{service}/MapServer/tile/{{z}}/{{y}}/{{x}}")


def add_base_layers(m, layers=BASE_LAYERS) -> None:
    """Attach the project's base layers to a folium map, first one default."""
    import folium

    for service, name in layers:
        folium.TileLayer(
            tiles=ESRI_TILES.format(service=service),
            attr="Esri", name=name, overlay=False, control=True,
        ).add_to(m)


def base_map(centre, zoom: int, layers=BASE_LAYERS):
    """A folium map with no default tiles but the project's base layers on it."""
    import folium

    m = folium.Map(location=centre, zoom_start=zoom, tiles=None)
    add_base_layers(m, layers)
    return m


def confidence_level(confidence: float) -> str:
    """Bucket a confidence into 'high' / 'med' / 'low'."""
    if confidence >= HIGH_CONFIDENCE:
        return 'high'
    if confidence >= MED_CONFIDENCE:
        return 'med'
    return 'low'


def shape_color(shape_type: str, confidence: float, matplotlib: bool = False) -> str:
    """Colour for a detection: hue from its shape family, shade from confidence."""
    palette = SHAPE_COLORS_MPL if matplotlib else SHAPE_COLORS
    family = palette.get(shape_type, palette['rectangular'])
    return family[confidence_level(confidence)]


def shape_legend_html(counts: Dict[str, int] = None, title: str = "Detections") -> str:
    """Floating HTML legend explaining the shape/confidence colour scheme."""
    counts = counts or {}
    rows = []
    for shape in ('rectangular', 'circular'):
        family = SHAPE_COLORS[shape]
        marker_radius = '50%' if shape == 'circular' else '2px'
        swatches = ''.join(
            f'<i style="background:{family[lvl]};width:12px;height:12px;'
            f'display:inline-block;border-radius:{marker_radius};'
            f'margin-right:3px;"></i>'
            for lvl in ('high', 'med', 'low')
        )
        n = counts.get(shape)
        label = SHAPE_LABELS[shape] + (f" ({n:,})" if n is not None else "")
        rows.append(f"{swatches} {label}<br>")

    return f"""
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
         background: rgba(255,255,255,0.94); padding: 10px 12px;
         border-radius: 5px; border: 2px solid #555; font-size: 13px;
         font-family: 'Segoe UI', Arial, sans-serif;">
    <b>{title}</b><br>
    {''.join(rows)}
    <span style="font-size:11px;color:#555;">
    Shape = hue &middot; shade = confidence
    (dark &ge;{HIGH_CONFIDENCE:.2f}, mid &ge;{MED_CONFIDENCE:.2f}, light below)
    </span>
    </div>
    """


def circle_latlon(
    easting: float, northing: float, radius_m: float,
    utm_zone: int = 15, n_vertices: int = 36,
):
    """Vertices of a circle in UTM metres, converted to lat/lon, closed."""
    from building_detection.lasio import utm_to_latlon

    angles = np.linspace(0, 2 * np.pi, n_vertices, endpoint=False)
    ring = [
        list(utm_to_latlon(easting + radius_m * np.cos(a),
                           northing + radius_m * np.sin(a), utm_zone))
        for a in angles
    ]
    ring.append(ring[0])
    return ring


# ---------------------------------------------------------------------------
# Interactive map (folium)
# ---------------------------------------------------------------------------

def _utm_rect_corners_to_latlon(
    easting: float, northing: float,
    width_m: float, height_m: float,
    orientation_deg: float,
    utm_zone: int = 15,
):
    """Compute 4 corners of a rotated rectangle and convert to lat/lon."""
    from building_detection.lasio import utm_to_latlon

    angle_rad = np.radians(orientation_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    hw, hh = width_m / 2.0, height_m / 2.0

    # Corner offsets in local frame, then rotate
    corners_local = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    corners_latlon = []
    for dx, dy in corners_local:
        rx = dx * cos_a - dy * sin_a
        ry = dx * sin_a + dy * cos_a
        lat, lon = utm_to_latlon(easting + rx, northing + ry, utm_zone)
        corners_latlon.append([lat, lon])
    # Close the polygon
    corners_latlon.append(corners_latlon[0])
    return corners_latlon


def create_building_map(
    csv_path: str,
    output_path: str = None,
    open_browser: bool = True,
    data_root: str = None,
    utm_zone: int = 15,
) -> str:
    """
    Create an interactive folium map from a buildings CSV file.

    Each building is shown as a rotated rectangle polygon coloured by confidence.
    If data_root is provided, tile boundaries are drawn from LAS file extents.
    Returns the path to the saved HTML file.
    """
    import folium

    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        print("No buildings in CSV.")
        return ""

    lats = [float(r['latitude']) for r in rows]
    lons = [float(r['longitude']) for r in rows]
    center_lat = np.mean(lats)
    center_lon = np.mean(lons)

    m = base_map([center_lat, center_lon], 14)

    # --- Tile boundaries from LAS files ---
    if data_root is not None:
        _add_tile_boundaries(m, rows, data_root, utm_zone)

    # --- Detections, one toggleable layer per shape family ---
    counts = {'rectangular': 0, 'circular': 0}
    groups = {
        shape: folium.FeatureGroup(name=SHAPE_LABELS[shape], show=True).add_to(m)
        for shape in ('rectangular', 'circular')
    }

    for r in rows:
        lat = float(r['latitude'])
        lon = float(r['longitude'])
        conf = float(r['building_confidence'])
        w = float(r['width_m'])
        h = float(r['height_m'])
        area = float(r['area_m2'])
        easting = float(r['easting'])
        northing = float(r['northing'])
        orientation = float(r['orientation_deg'])

        shape = r.get('shape_type') or 'rectangular'
        if shape not in groups:
            shape = 'rectangular'
        counts[shape] += 1
        color = shape_color(shape, conf)

        header = (f"<b>{r['filename']}</b> #{r['building_id']}<br>"
                  f"<b>Shape:</b> {SHAPE_LABELS[shape]}<br>")
        footer = (f"<b>Confidence:</b> {conf:.2f}<br>"
                  f"<b>Coords:</b> {lat:.6f}, {lon:.6f}")

        if shape == 'circular':
            diameter = float(r.get('diameter_m') or (w + h) / 2)
            popup_html = (
                header +
                f"<b>Diameter:</b> {diameter:.0f} m ({area:.0f} m²)<br>"
                f"<b>Circular score:</b> {float(r.get('circular_score', 0)):.3f}<br>"
                f"<b>Ring template:</b> {float(r.get('circ_template_score', 0)):.3f}<br>"
                f"<b>Radial symmetry:</b> {float(r.get('radial_symmetry', 0)):.3f}<br>"
                f"<b>Directional isotropy:</b> {float(r.get('directional_isotropy', 0)):.3f}<br>"
                + footer
            )
            geometry = folium.Polygon(
                locations=circle_latlon(easting, northing, diameter / 2, utm_zone),
                color=color, weight=2, fill=True, fill_color=color,
                fill_opacity=0.3,
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"circular d={diameter:.0f}m conf={conf:.2f}",
            )
        else:
            popup_html = (
                header +
                f"<b>Size:</b> {w:.0f} x {h:.0f} m ({area:.0f} m²)<br>"
                f"<b>Angle score:</b> {float(r.get('angle_score', 0)):.3f}<br>"
                f"<b>Template score:</b> {float(r.get('template_score', 0)):.3f}<br>"
                f"<b>Rectangularity:</b> {float(r['rectangularity']):.3f}<br>"
                f"<b>Wall score:</b> {float(r['wall_score']):.3f}<br>"
                + footer
            )
            geometry = folium.Polygon(
                locations=_utm_rect_corners_to_latlon(
                    easting, northing, w, h, orientation, utm_zone,
                ),
                color=color, weight=2, fill=True, fill_color=color,
                fill_opacity=0.3,
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"rectangular {w:.0f}x{h:.0f}m conf={conf:.2f}",
            )

        geometry.add_to(groups[shape])

    folium.LayerControl().add_to(m)
    m.get_root().html.add_child(folium.Element(shape_legend_html(counts)))

    if output_path is None:
        out_dir = Path(csv_path).parent
        output_path = str(out_dir / "buildings_map.html")

    m.save(output_path)
    print(f"Map saved to: {output_path}")

    if open_browser:
        import webbrowser
        webbrowser.open(f"file://{output_path}")

    return output_path


def _add_tile_boundaries(m, rows, data_root: str, utm_zone: int = 15):
    """Add LAS tile boundary rectangles to the folium map."""
    import folium
    from building_detection.lasio import load_las, utm_to_latlon

    data_root = Path(data_root)
    tiles_group = folium.FeatureGroup(name="Tile Boundaries").add_to(m)

    # Collect unique files from CSV
    seen = set()
    files_to_load = []
    for r in rows:
        key = (r['source_folder'], r['filename'])
        if key not in seen:
            seen.add(key)
            files_to_load.append(key)

    for folder, filename in files_to_load:
        las_path = data_root / folder / filename
        if not las_path.exists():
            # Try searching recursively
            matches = list(data_root.rglob(filename))
            if not matches:
                print(f"  Tile boundary: {filename} not found, skipping")
                continue
            las_path = matches[0]

        try:
            pts = load_las(str(las_path))
            e_min, e_max = float(pts[:, 0].min()), float(pts[:, 0].max())
            n_min, n_max = float(pts[:, 1].min()), float(pts[:, 1].max())

            sw = utm_to_latlon(e_min, n_min, utm_zone)
            se = utm_to_latlon(e_max, n_min, utm_zone)
            ne = utm_to_latlon(e_max, n_max, utm_zone)
            nw = utm_to_latlon(e_min, n_max, utm_zone)

            corners = [
                [sw[0], sw[1]], [se[0], se[1]],
                [ne[0], ne[1]], [nw[0], nw[1]],
                [sw[0], sw[1]],
            ]

            n_buildings = sum(1 for r in rows if r['filename'] == filename)
            folium.Polygon(
                locations=corners,
                color='white',
                weight=2,
                fill=False,
                dash_array='8 4',
                tooltip=f"{filename} ({n_buildings} buildings)",
            ).add_to(tiles_group)
            print(f"  Tile boundary: {filename} "
                  f"({e_max - e_min:.0f}x{n_max - n_min:.0f}m)")
        except Exception as e:
            print(f"  Tile boundary: {filename} error: {e}")


# ---------------------------------------------------------------------------
# DEM heatmap with building rectangles
# ---------------------------------------------------------------------------

def _make_rotated_rect(
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    angle_deg: float,
) -> mpatches.FancyBboxPatch:
    """Create a rotated rectangle as a matplotlib patch."""
    rect = mpatches.Rectangle(
        (-width / 2, -height / 2), width, height,
        linewidth=2, edgecolor='lime', facecolor='none',
    )
    t = (
        plt.matplotlib.transforms.Affine2D()
        .rotate_deg(angle_deg)
        .translate(center_x, center_y)
    )
    rect.set_transform(t)
    return rect, t


def _draw_candidate_outline(ax, c: BuildingCandidate) -> None:
    """Draw one detection on a matplotlib axis: circle if round, else rectangle."""
    cx, cy = float(c.center_xy[0]), float(c.center_xy[1])
    color = shape_color(c.shape_type, _quick_confidence(c), matplotlib=True)

    if c.shape_type == 'circular':
        radius = (c.diameter_m / 2.0) or (c.width_m + c.height_m) / 4.0
        ax.add_patch(mpatches.Circle(
            (cx, cy), radius,
            linewidth=1.5, edgecolor=color, facecolor='none',
        ))
        return

    rect = mpatches.Rectangle(
        (cx - c.width_m / 2, cy - c.height_m / 2), c.width_m, c.height_m,
        linewidth=1.5, edgecolor=color, facecolor='none',
    )
    rect.set_transform(
        plt.matplotlib.transforms.Affine2D()
        .rotate_deg_around(cx, cy, c.orientation_deg)
        + ax.transData
    )
    ax.add_patch(rect)


def plot_dem_with_buildings(
    result: BuildingDetectionResult,
    candidates: List[BuildingCandidate] = None,
    title: str = "Building Detection",
    output_path: str = None,
    show: bool = True,
    max_candidates: int = 50,
    figsize: tuple = (16, 10),
) -> Optional[str]:
    """
    Plot the relative elevation DEM with detected building rectangles.

    Args:
        result: BuildingDetectionResult from detect_buildings()
        candidates: list of candidates to plot (default: result.candidates)
        title: plot title
        output_path: save to file if provided
        show: call plt.show()
        max_candidates: limit number of rectangles drawn
    """
    if candidates is None:
        candidates = result.candidates

    rel_elev = result.relative_elevation
    cs = result.cell_size
    x0 = result.x_origin
    y0 = result.y_origin

    # Extent in UTM
    extent = [
        x0,
        x0 + rel_elev.shape[1] * cs,
        y0,
        y0 + rel_elev.shape[0] * cs,
    ]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: relative elevation
    ax1 = axes[0]
    im = ax1.imshow(
        rel_elev,
        origin='lower',
        extent=extent,
        cmap='terrain',
        vmin=-1, vmax=3,
        aspect='equal',
    )
    plt.colorbar(im, ax=ax1, label='Relative elevation (m)', shrink=0.7)
    ax1.set_title(f'{title} — Relative Elevation')
    ax1.set_xlabel('Easting (m)')
    ax1.set_ylabel('Northing (m)')

    # Draw detection outlines (rectangles and circles)
    for c in candidates[:max_candidates]:
        _draw_candidate_outline(ax1, c)

    # Right: gradient magnitude
    ax2 = axes[1]
    from scipy.ndimage import sobel
    clean = rel_elev.copy()
    clean[np.isnan(clean)] = 0.0
    gx = sobel(clean, axis=1)
    gy = sobel(clean, axis=0)
    grad_mag = np.sqrt(gx**2 + gy**2)

    im2 = ax2.imshow(
        grad_mag,
        origin='lower',
        extent=extent,
        cmap='hot',
        vmin=0, vmax=np.percentile(grad_mag[grad_mag > 0], 95),
        aspect='equal',
    )
    plt.colorbar(im2, ax=ax2, label='Gradient magnitude', shrink=0.7)
    ax2.set_title(f'{title} — Edge Strength')
    ax2.set_xlabel('Easting (m)')
    ax2.set_ylabel('Northing (m)')

    # Draw the same outlines on the gradient panel
    for c in candidates[:max_candidates]:
        _draw_candidate_outline(ax2, c)

    n_circ = sum(1 for c in candidates if c.shape_type == 'circular')
    n_rect = len(candidates) - n_circ
    ax1.legend(
        handles=[
            mpatches.Patch(color=SHAPE_COLORS_MPL['rectangular']['high'],
                           label=f'Rectangular ({n_rect})'),
            mpatches.Patch(color=SHAPE_COLORS_MPL['circular']['high'],
                           label=f'Circular ({n_circ})'),
        ],
        loc='upper right', fontsize=8, framealpha=0.8,
    )
    plt.suptitle(
        f'{len(candidates)} structures detected '
        f'({n_rect} rectangular, {n_circ} circular)',
        fontsize=14, fontweight='bold',
    )
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


def _quick_confidence(c: BuildingCandidate) -> float:
    """Confidence for a candidate, using the formula for its shape family."""
    from .metrics import compute_shape_confidence
    return compute_shape_confidence(c)


def plot_single_building(
    result: BuildingDetectionResult,
    candidate: BuildingCandidate,
    points: np.ndarray = None,
    output_path: str = None,
    show: bool = True,
    buffer_m: float = 15.0,
) -> Optional[str]:
    """
    Detailed plot of a single building candidate.

    Shows zoomed-in DEM, gradient, and optionally LiDAR point density.
    """
    rel_elev = result.relative_elevation
    cs = result.cell_size
    x0 = result.x_origin
    y0 = result.y_origin

    cx, cy = float(candidate.center_xy[0]), float(candidate.center_xy[1])
    w, h = candidate.width_m, candidate.height_m
    half = max(w, h) / 2 + buffer_m

    # Pixel bounds
    c_lo = max(0, int((cx - half - x0) / cs))
    c_hi = min(rel_elev.shape[1], int((cx + half - x0) / cs) + 1)
    r_lo = max(0, int((cy - half - y0) / cs))
    r_hi = min(rel_elev.shape[0], int((cy + half - y0) / cs) + 1)

    patch = rel_elev[r_lo:r_hi, c_lo:c_hi]
    ext = [
        x0 + c_lo * cs,
        x0 + c_hi * cs,
        y0 + r_lo * cs,
        y0 + r_hi * cs,
    ]

    n_panels = 3 if points is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))

    angle = candidate.orientation_deg

    def draw_outline(ax, color='lime'):
        """Circle for a circular candidate, oriented rectangle otherwise."""
        if candidate.shape_type == 'circular':
            radius = (candidate.diameter_m / 2.0) or (w + h) / 4.0
            ax.add_patch(mpatches.Circle(
                (cx, cy), radius, linewidth=2, edgecolor=color,
                facecolor='none', linestyle='--',
            ))
        else:
            rect = mpatches.Rectangle(
                (cx - w/2, cy - h/2), w, h,
                linewidth=2, edgecolor=color, facecolor='none', linestyle='--',
            )
            rect.set_transform(
                plt.matplotlib.transforms.Affine2D()
                .rotate_deg_around(cx, cy, angle)
                + ax.transData
            )
            ax.add_patch(rect)
        ax.plot(cx, cy, 'x', color=color, markersize=8, markeredgewidth=2)

    # Panel 1: Relative elevation
    ax = axes[0]
    im = ax.imshow(patch, origin='lower', extent=ext, cmap='terrain',
                   vmin=-0.5, vmax=3, aspect='equal')
    plt.colorbar(im, ax=ax, label='Rel. elevation (m)', shrink=0.8)
    draw_outline(ax, 'red')
    ax.set_title('Relative Elevation')
    ax.set_xlabel('Easting')
    ax.set_ylabel('Northing')

    # Panel 2: Gradient magnitude
    ax = axes[1]
    clean = patch.copy()
    clean[np.isnan(clean)] = 0.0
    from scipy.ndimage import sobel
    gx = sobel(clean, axis=1)
    gy = sobel(clean, axis=0)
    grad = np.sqrt(gx**2 + gy**2)
    im2 = ax.imshow(grad, origin='lower', extent=ext, cmap='hot',
                    vmin=0, vmax=np.percentile(grad[grad > 0], 95) if grad.any() else 1,
                    aspect='equal')
    plt.colorbar(im2, ax=ax, label='Gradient', shrink=0.8)
    draw_outline(ax, 'lime')
    ax.set_title('Edge Strength')
    ax.set_xlabel('Easting')

    # Panel 3: Point density (if points provided)
    if points is not None:
        ax = axes[2]
        mask = (
            (np.abs(points[:, 0] - cx) <= half) &
            (np.abs(points[:, 1] - cy) <= half)
        )
        local_pts = points[mask]
        if len(local_pts) > 0:
            h2d, xedges, yedges = np.histogram2d(
                local_pts[:, 0], local_pts[:, 1],
                bins=[int(2 * half / cs), int(2 * half / cs)],
                range=[[cx - half, cx + half], [cy - half, cy + half]],
            )
            im3 = ax.imshow(
                h2d.T, origin='lower',
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                cmap='YlOrRd', aspect='equal',
            )
            plt.colorbar(im3, ax=ax, label='Point count', shrink=0.8)
        draw_outline(ax, 'cyan')
        ax.set_title('LiDAR Point Density')
        ax.set_xlabel('Easting')

    conf = _quick_confidence(candidate)
    if candidate.shape_type == 'circular':
        summary = (
            f'circular  d={candidate.diameter_m:.0f}m  '
            f'area={candidate.area_m2:.0f}m²  '
            f'circ={candidate.circular_score:.3f}  '
            f'ring={candidate.circ_template_score:.3f}  '
            f'sym={candidate.radial_symmetry:.2f}  '
            f'iso={candidate.directional_isotropy:.2f}  '
            f'conf={conf:.2f}'
        )
    else:
        summary = (
            f'rectangular  {w:.0f}x{h:.0f}m  '
            f'area={candidate.area_m2:.0f}m²  '
            f'angle={candidate.angle_score:.3f}  '
            f'tmpl={candidate.template_score:.3f}  '
            f'rect={candidate.rectangularity:.2f}  '
            f'wall={candidate.wall_score:.2f}  '
            f'conf={conf:.2f}'
        )
    plt.suptitle(summary, fontsize=11)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


# ---------------------------------------------------------------------------
# Cross-section plots (adapted from cenote impact analysis)
# ---------------------------------------------------------------------------

def _metric(metrics: dict, key: str, default: float = 0.0) -> float:
    """
    Read a numeric metric that may arrive as a string.

    Detections reach the plots either as live dicts from the pipeline
    (numbers) or as rows re-read from the results CSV (strings, as on a
    resumed run or a regeneration). Both must format.
    """
    value = metrics.get(key, default)
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def plot_building_cross_section(
    points: np.ndarray,
    center_xy: np.ndarray,
    width_m: float,
    height_m: float,
    orientation_deg: float,
    ground_z: float,
    peak_depth: float,
    metrics: dict,
    title: str = "",
    output_path: str = None,
    show: bool = False,
    buffer_m: float = 20.0,
) -> Optional[str]:
    """
    4-panel cross-section plot for a building detection.

    Adapted from cenote impact analysis plots. Shows:
      - Top-left: top-down point cloud with rectangle overlay
      - Top-right: depth histogram profile
      - Bottom-left: E-W cross-section at ground level
      - Bottom-right: N-S cross-section at ground level
    """
    cx, cy = float(center_xy[0]), float(center_xy[1])
    longest_side = max(width_m, height_m)

    # Select points within buffer
    mask = (
        (np.abs(points[:, 0] - cx) <= buffer_m) &
        (np.abs(points[:, 1] - cy) <= buffer_m)
    )
    local_pts = points[mask]
    if len(local_pts) < 10:
        return None

    # Classify points relative to ground
    pts_depth = ground_z - local_pts[:, 2]  # positive = below ground
    below_ground = pts_depth > 0.0

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # ── Panel 1: Top-down view ──────────────────────────────────
    ax1 = axes[0, 0]
    sc = ax1.scatter(
        local_pts[:, 0], local_pts[:, 1],
        c=local_pts[:, 2], cmap='terrain', s=0.3, alpha=0.5,
    )
    plt.colorbar(sc, ax=ax1, label='Elevation (m)', shrink=0.7)

    # Highlight subsurface points
    if below_ground.any():
        ax1.scatter(
            local_pts[below_ground, 0], local_pts[below_ground, 1],
            c='red', s=1, alpha=0.4, label='Below ground',
        )

    # Draw the detected outline: circle for a circular detection, oriented
    # rectangle otherwise
    shape_type = metrics.get('shape_type', 'rectangular')
    diameter_m = _metric(metrics, 'diameter_m') or (width_m + height_m) / 2
    angle_rad = np.radians(orientation_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    hw, hh = width_m / 2, height_m / 2

    outline_color = SHAPE_COLORS_MPL[
        shape_type if shape_type in SHAPE_COLORS_MPL else 'rectangular'
    ]['high']
    if shape_type == 'circular':
        ax1.add_patch(mpatches.Circle(
            (cx, cy), diameter_m / 2,
            linewidth=2, edgecolor=outline_color, facecolor='none',
        ))
    else:
        corners = []
        for dx, dy in [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh), (-hw, -hh)]:
            rx = dx * cos_a - dy * sin_a
            ry = dx * sin_a + dy * cos_a
            corners.append((cx + rx, cy + ry))
        corners_arr = np.array(corners)
        ax1.plot(corners_arr[:, 0], corners_arr[:, 1],
                 color=outline_color, linewidth=2)
    ax1.plot(cx, cy, 'y*', markersize=10)

    ax1.set_xlim(cx - buffer_m, cx + buffer_m)
    ax1.set_ylim(cy - buffer_m, cy + buffer_m)
    ax1.set_xlabel("Easting (m)")
    ax1.set_ylabel("Northing (m)")
    ax1.set_aspect('equal')
    conf = _metric(metrics, 'building_confidence')
    ax1.set_title(f"Top-down view -- {shape_type} (conf={conf:.3f})")
    ax1.legend(fontsize=8, loc='upper right')

    # ── Panel 2: Depth profile ──────────────────────────────────
    ax2 = axes[0, 1]

    rect_mask = (
        (np.abs(local_pts[:, 0] - cx) <= longest_side / 2 + 2) &
        (np.abs(local_pts[:, 1] - cy) <= longest_side / 2 + 2)
    )
    rect_pts = local_pts[rect_mask]
    if len(rect_pts) > 0:
        rect_depth = ground_z - rect_pts[:, 2]
        depth_range = np.linspace(-3, max(3, peak_depth + 1), 50)
        ax2.hist(rect_depth, bins=depth_range, orientation='horizontal',
                 color='steelblue', alpha=0.7, edgecolor='white', linewidth=0.3)
        ax2.axhline(y=0, color='green', linewidth=2, linestyle='-',
                    label='Ground level', alpha=0.8)
        if peak_depth > 0:
            ax2.axhline(y=peak_depth, color='darkred', linewidth=1,
                        linestyle=':', label=f'Peak depth ({peak_depth:.2f}m)')

    ax2.set_ylabel("Depth below ground (m)")
    ax2.set_xlabel("Point count")
    ax2.invert_yaxis()
    ax2.set_title("Depth profile")
    ax2.legend(loc='lower right', fontsize=8)

    # ── Panel 3: E-W cross-section ──────────────────────────────
    ax3 = axes[1, 0]
    slice_width = max(longest_side * 0.3, 5.0)
    ew_mask = np.abs(local_pts[:, 1] - cy) < slice_width
    ew_pts = local_pts[ew_mask]
    if len(ew_pts) > 0:
        ew_depth = ground_z - ew_pts[:, 2]
        ew_below = ew_depth > 0.0

        if (~ew_below).any():
            ax3.scatter(ew_pts[~ew_below, 0], ew_pts[~ew_below, 2],
                       c='gray', s=0.5, alpha=0.3)
        if ew_below.any():
            ax3.scatter(ew_pts[ew_below, 0], ew_pts[ew_below, 2],
                       c='red', s=2, alpha=0.7)

        ax3.axhline(y=ground_z, color='green', linewidth=1.5,
                    linestyle='-', alpha=0.7, label='Ground level')
        if peak_depth > 0:
            ax3.axhline(y=ground_z - peak_depth, color='darkred',
                        linewidth=1, linestyle=':', alpha=0.5,
                        label='Peak depth')

        # Detected extent projected onto the E-W axis
        if shape_type == 'circular':
            half_extent_x = diameter_m / 2
        else:
            half_extent_x = max(abs(hw * cos_a) + abs(hh * sin_a), 1e-6)
        for x_edge in (cx - half_extent_x, cx + half_extent_x):
            ax3.axvline(x=x_edge, color=outline_color, linewidth=1.5,
                        linestyle='--', alpha=0.5)

    ax3.set_xlim(cx - buffer_m, cx + buffer_m)
    ax3.set_xlabel("Easting (m)")
    ax3.set_ylabel("Elevation (m)")
    ax3.set_title(f"E-W cross-section (slice width={slice_width:.0f}m)")
    ax3.legend(fontsize=8, loc='upper right')

    # ── Panel 4: N-S cross-section ──────────────────────────────
    ax4 = axes[1, 1]
    ns_mask = np.abs(local_pts[:, 0] - cx) < slice_width
    ns_pts = local_pts[ns_mask]
    if len(ns_pts) > 0:
        ns_depth = ground_z - ns_pts[:, 2]
        ns_below = ns_depth > 0.0

        if (~ns_below).any():
            ax4.scatter(ns_pts[~ns_below, 1], ns_pts[~ns_below, 2],
                       c='gray', s=0.5, alpha=0.3)
        if ns_below.any():
            ax4.scatter(ns_pts[ns_below, 1], ns_pts[ns_below, 2],
                       c='red', s=2, alpha=0.7)

        ax4.axhline(y=ground_z, color='green', linewidth=1.5,
                    linestyle='-', alpha=0.7, label='Ground level')
        if peak_depth > 0:
            ax4.axhline(y=ground_z - peak_depth, color='darkred',
                        linewidth=1, linestyle=':', alpha=0.5)

        # Detected extent projected onto the N-S axis
        if shape_type == 'circular':
            half_extent_y = diameter_m / 2
        else:
            half_extent_y = max(abs(hw * sin_a) + abs(hh * cos_a), 1e-6)
        for y_edge in (cy - half_extent_y, cy + half_extent_y):
            ax4.axvline(x=y_edge, color=outline_color, linewidth=1.5,
                        linestyle='--', alpha=0.5)

    ax4.set_xlim(cy - buffer_m, cy + buffer_m)
    ax4.set_xlabel("Northing (m)")
    ax4.set_ylabel("Elevation (m)")
    ax4.set_title(f"N-S cross-section (slice width={slice_width:.0f}m)")
    ax4.legend(fontsize=8, loc='upper right')

    # ── Metrics annotation ──────────────────────────────────────
    if shape_type == 'circular':
        shape_metrics = (
            f"Circular score: {_metric(metrics, 'circular_score'):.3f}\n"
            f"Ring template: {_metric(metrics, 'circ_template_score'):.3f}\n"
            f"Radial symmetry: {_metric(metrics, 'radial_symmetry'):.3f}\n"
            f"Isotropy: {_metric(metrics, 'directional_isotropy'):.3f}\n"
            f"Roundness: {_metric(metrics, 'roundness'):.3f}\n"
        )
        footprint = f"Diameter: {diameter_m:.1f} m\n"
    else:
        shape_metrics = (
            f"Angle score: {_metric(metrics, 'angle_score'):.3f}\n"
            f"Template score: {_metric(metrics, 'template_score'):.3f}\n"
            f"Rectangularity: {_metric(metrics, 'rectangularity'):.3f}\n"
            f"Wall score: {_metric(metrics, 'wall_score'):.3f}\n"
            f"Edge density: {_metric(metrics, 'edge_density_ratio'):.3f}\n"
        )
        footprint = f"Footprint: {width_m:.1f} x {height_m:.1f} m\n"

    # Orientation only means something for a rectangle
    orientation_line = (
        "" if shape_type == 'circular'
        else f"Orientation: {orientation_deg:.1f}\u00b0\n"
    )

    metrics_text = (
        f"Shape: {shape_type}\n"
        f"Confidence: {conf:.3f}\n"
        f"{'─' * 30}\n"
        + shape_metrics
        + f"{'─' * 30}\n"
        + footprint
        + f"Area: {_metric(metrics, 'area_m2'):.0f} m²\n"
        + f"Peak depth: {peak_depth:.2f} m\n"
        + f"Mean depth: {_metric(metrics, 'mean_height_m'):.2f} m\n"
        + orientation_line
        + f"Aspect ratio: {_metric(metrics, 'aspect_ratio'):.2f}\n"
        + f"{'─' * 30}\n"
        + f"Ground pts: {metrics.get('num_ground_points', 0)}\n"
        + f"Elevated pts: {metrics.get('num_elevated_points', 0)}"
    )

    conf_color = 'green' if conf >= 0.4 else ('orange' if conf >= 0.3 else 'red')
    fig.text(0.02, 0.02, metrics_text, fontsize=8, fontfamily='monospace',
             verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='lightyellow',
                       alpha=0.9, edgecolor=conf_color, linewidth=2))

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0.0, 0.08, 1.0, 0.96])

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Cross-section saved: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path
