#!/usr/bin/env python3
"""
Build the published catalogue (tau = 0.16) with an agglomeration id per
detection, plus one table with the properties of each agglomeration.

Every row of the published catalogue is kept, with a column that says which
agglomeration a detection belongs to, so that structures per agglomeration
and the properties of each agglomeration can be studied directly.

Definition of an agglomeration
------------------------------
DBSCAN over the centroids of the published detections, in local metres:

    eps         = 50 m   patio-group spacing; the nearest-neighbour histogram
                         peaks at 10-20 m and thins out past ~50 m
    min_samples = 3      a group, not a pair

DBSCAN is used rather than plain single-linkage because a chain of isolated
pairs should not be welded into one settlement: a detection is a *core*
point only if at least two others lie within 50 m, and clusters grow only
through core points. Detections that are not reachable from a core point
are "isolated" and get agglomeration_id = -1.

Coordinates are projected with a per-point cosine scaling of lat/lon (see
local_metres), not per-tile UTM, so an agglomeration can cross a tile edge
or a dataset boundary, and the two UTM zones of the archive do not matter.

Ids are assigned by size: agglomeration 1 is the largest, ties broken by
latitude, so the id is stable across reruns with the same input.

Outputs (default catalogue/):
    detections_tau016.csv        24,805 rows x (31 original + 4 new columns)
    agglomerations_tau016.csv    one row per agglomeration
    data_dictionary.csv          one row per column of the two tables
    catalogue_tau016.xlsx        the three tables as sheets, for convenience
    stats.json                   the numbers quoted in the document

Usage:
    python tools/agglomerations.py
    python tools/agglomerations.py --tau 0.20 --eps 40 --min-samples 4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, cKDTree
from sklearn.cluster import DBSCAN

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_CSV = Path("intermediate/gated_candidates.csv")
DEFAULT_OUT = Path("catalogue")

TAU = 0.16
EPS_M = 50.0
MIN_SAMPLES = 3

# Size classes by number of structures.
SIZE_CLASSES = (
    (3, 4, "patio group"),
    (5, 9, "hamlet"),
    (10, 19, "small village"),
    (20, 49, "village"),
    (50, 10 ** 9, "large settlement"),
)


def size_class(n: int) -> str:
    for lo, hi, name in SIZE_CLASSES:
        if lo <= n <= hi:
            return name
    return "isolated"


def local_metres(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Per-point cosine scaling; error inside a 50 m neighbourhood << 1 m."""
    return np.column_stack([
        lon * 111_320.0 * np.cos(np.radians(lat)),
        lat * 110_540.0,
    ])


def axial_coherence(theta_deg: np.ndarray) -> float:
    """
    Mean resultant length of the doubled orientation angles, in [0, 1].

    Orientations are axial (0 == 180), so they are doubled before averaging.
    1.0 means every structure shares one axis; a uniform spread gives ~0.
    """
    if len(theta_deg) < 2:
        return float("nan")
    a = np.radians(2.0 * theta_deg)
    return float(np.hypot(np.cos(a).mean(), np.sin(a).mean()))


def hull_area_ha(pts: np.ndarray) -> float:
    if len(pts) < 3:
        return 0.0
    try:
        return float(ConvexHull(pts).volume) / 10_000.0
    except Exception:          # collinear
        return 0.0


def summarise(t: pd.DataFrame, xy: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    rows = []
    for cid in sorted(set(labels) - {-1}):
        m = labels == cid
        g = t[m]
        pts = xy[m]
        n = int(m.sum())
        span = pts.max(axis=0) - pts.min(axis=0)
        # Nearest neighbour *inside* the agglomeration.
        dist, _ = cKDTree(pts).query(pts, k=2)
        nn = dist[:, 1]
        area_ha = hull_area_ha(pts)
        tiles = g.filename.unique()
        datasets = g.source_folder.unique()
        rows.append({
            "agglomeration_id": int(cid),
            "size_class": size_class(n),
            "n_structures": n,
            "latitude": round(float(g.latitude.mean()), 6),
            "longitude": round(float(g.longitude.mean()), 6),
            "extent_m": round(float(np.hypot(*span)), 1),
            "hull_area_ha": round(area_ha, 3),
            "density_per_ha": round(n / area_ha, 2) if area_ha > 0 else np.nan,
            "nn_median_m": round(float(np.median(nn)), 1),
            "score_mean": round(float(g.building_confidence.mean()), 3),
            "score_median": round(float(g.building_confidence.median()), 3),
            "score_max": round(float(g.building_confidence.max()), 3),
            "area_m2_median": round(float(g.area_m2.median()), 1),
            "area_m2_max": round(float(g.area_m2.max()), 1),
            "area_m2_sum": round(float(g.area_m2.sum()), 1),
            "n_over_200m2": int((g.area_m2 > 200).sum()),
            "peak_depth_mean_m": round(float(g.peak_height_m.mean()), 2),
            "orientation_coherence": round(axial_coherence(g.orientation_deg.to_numpy()), 3),
            "orientation_mean_deg": round(float(
                (np.degrees(np.arctan2(np.sin(np.radians(2 * g.orientation_deg)).mean(),
                                       np.cos(np.radians(2 * g.orientation_deg)).mean())) / 2.0) % 180.0), 1),
            "n_tiles": int(len(tiles)),
            "n_datasets": int(len(datasets)),
            "source_folder": g.source_folder.mode().iloc[0],
            "tiles": ";".join(sorted(f.replace(".las.gz", "") for f in tiles)),
        })
    return pd.DataFrame(rows)


def relabel_by_size(labels: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Map DBSCAN's arbitrary ids to 1..K, largest agglomeration first."""
    ids = [c for c in set(labels) if c >= 0]
    key = {c: (-(labels == c).sum(), float(lat[labels == c].mean())) for c in ids}
    order = sorted(ids, key=lambda c: key[c])
    mapping = {c: i + 1 for i, c in enumerate(order)}
    mapping[-1] = -1
    return np.array([mapping[c] for c in labels], dtype=int)


DETECTION_DICT = {
    # identity and position
    "source_folder": ("Regional G-LiHT dataset the tile belongs to (16 values).",
                      "Dataset regional G-LiHT al que pertenece el tile (16 valores)."),
    "filename": ("LAS tile the detection was found in. Tiles are processed independently.",
                 "Tile LAS en el que se encontró la detección. Los tiles se procesan por separado."),
    "building_id": ("Sequential id of the candidate inside its tile; (filename, building_id) is the unique key.",
                    "Id secuencial del candidato dentro de su tile; (filename, building_id) es la clave única."),
    "latitude": ("Centroid latitude, WGS84 decimal degrees.", "Latitud del centroide, WGS84 grados decimales."),
    "longitude": ("Centroid longitude, WGS84 decimal degrees.", "Longitud del centroide, WGS84 grados decimales."),
    "easting": ("Centroid UTM easting (m) in the tile's zone: 15N for six datasets, 16N for ten; do not mix zones.",
                "Este UTM del centroide (m) en la zona del tile: 15N en seis datasets, 16N en diez; no mezclar zonas."),
    "northing": ("Centroid UTM northing (m), same zone as easting.",
                 "Norte UTM del centroide (m), misma zona que easting."),
    # geometry
    "width_m": ("Long side of the fitted rectangle (m), from PCA of the component cells.",
                "Lado largo del rectángulo ajustado (m), por PCA de las celdas del componente."),
    "height_m": ("Short side of the fitted rectangle (m). Not a height: the name is historical.",
                 "Lado corto del rectángulo ajustado (m). No es una altura: el nombre es heredado."),
    "area_m2": ("Planimetric area of the component (m²) = number of 2 m cells × 4.",
                "Área planimétrica del componente (m²) = número de celdas de 2 m × 4."),
    "orientation_deg": ("Azimuth of the long axis, degrees from east, axial (0 ≡ 180). Quantised by the 2 m raster.",
                        "Azimut del eje largo, grados desde el este, axial (0 ≡ 180). Cuantizado por el raster de 2 m."),
    "aspect_ratio": ("width_m / height_m (≥ 1).", "width_m / height_m (≥ 1)."),
    "rectangularity": ("Component area / fitted-rectangle area, in [0, 1]. Score term (w = 0.10).",
                       "Área del componente / área del rectángulo ajustado, en [0, 1]. Término del score (w = 0.10)."),
    "wall_score": ("Mass concentrated in the outer 25 % band of the rectangle, rescaled to [0, 1]. Score term (w = 0.15).",
                   "Masa concentrada en la banda exterior del 25 % del rectángulo, reescalada a [0, 1]. Término del score (w = 0.15)."),
    "edge_density_ratio": ("Raw-return density near the rectangle edges / density inside. Score term uses max(0, x − 1) (w = 0.10).",
                           "Densidad de retornos crudos en los bordes / densidad interior. El score usa max(0, x − 1) (w = 0.10)."),
    "angle_score": ("Right-angle signal: concentration of edge normals in two perpendicular peaks, [0, 1]. Score term (w = 0.30).",
                    "Señal de ángulo recto: concentración de las normales de borde en dos picos perpendiculares, [0, 1]. Término del score (w = 0.30)."),
    "template_score": ("Normalised cross-correlation with a synthetic rectangular ring, clipped at 0. Score term (w = 0.20).",
                       "Correlación cruzada normalizada con un anillo rectangular sintético, recortada en 0. Término del score (w = 0.20)."),
    "dominant_orientation_deg": ("Orientation of the dominant gradient peak (degrees). Equals orientation_deg in this catalogue.",
                                 "Orientación del pico de gradiente dominante (grados). Igual a orientation_deg en este catálogo."),
    "shape_type": ("Shape family. Always 'rectangular' here; the circular family is not published.",
                   "Familia de forma. Siempre 'rectangular' aquí; la familia circular no se publica."),
    # circular-family columns (kept for completeness)
    "circular_score": ("Combined circular evidence, [0, 1]. Diagnostic only; not part of the building score.",
                       "Evidencia circular combinada, [0, 1]. Solo diagnóstico; no forma parte del building score."),
    "circ_template_score": ("Correlation with an annular ring template, [0, 1]. Diagnostic only.",
                            "Correlación con una plantilla anular, [0, 1]. Solo diagnóstico."),
    "radial_symmetry": ("Constancy of the boundary radius, [0, 1]. Diagnostic only (0 when not computed).",
                        "Constancia del radio del contorno, [0, 1]. Solo diagnóstico (0 si no se calculó)."),
    "directional_isotropy": ("Flatness of the gradient-direction histogram, [0, 1]. Diagnostic only.",
                             "Planitud del histograma de direcciones de gradiente, [0, 1]. Solo diagnóstico."),
    "roundness": ("Closeness of the aspect ratio to 1, [0, 1]. Diagnostic only.",
                  "Cercanía de la razón de aspecto a 1, [0, 1]. Solo diagnóstico."),
    "diameter_m": ("Mean boundary diameter of the circular fit (m). Diagnostic only.",
                   "Diámetro medio del contorno del ajuste circular (m). Solo diagnóstico."),
    # relief
    "peak_height_m": ("Peak depth of the low-relief residual inside the component (m). Gate: 0.25–1.80 m. Despite the name it is a depth below the background.",
                      "Profundidad pico del residuo de bajo relieve dentro del componente (m). Compuerta: 0.25–1.80 m. Pese al nombre es una profundidad bajo el fondo."),
    "mean_height_m": ("Mean depth of the residual inside the component (m). Gate: ≥ 0.15 m. Feeds the height term (w = 0.10).",
                      "Profundidad media del residuo dentro del componente (m). Compuerta: ≥ 0.15 m. Alimenta el término height (w = 0.10)."),
    # score
    "building_confidence": ("The building score (§2.4), in [0, 1]. The published cut is building_confidence ≥ τ = 0.16.",
                            "El building score (§2.4), en [0, 1]. El corte publicado es building_confidence ≥ τ = 0.16."),
    "confidence_raw": ("The score before any cross-family quantile matching. Identical to building_confidence for rectangular candidates.",
                       "El score antes de cualquier igualación por cuantiles entre familias. Idéntico a building_confidence en candidatos rectangulares."),
    "num_ground_points": ("Raw LiDAR returns near the modelled ground inside the bounding box.",
                          "Retornos LiDAR crudos cerca del suelo modelado dentro de la caja envolvente."),
    "num_elevated_points": ("Raw LiDAR returns inside the component's residual band.",
                            "Retornos LiDAR crudos dentro de la banda de residuo del componente."),
    # new
    "agglomeration_id": ("Id of the agglomeration (DBSCAN, 50 m, ≥ 3). 1 = largest. −1 = isolated (no agglomeration).",
                         "Id de la aglomeración (DBSCAN, 50 m, ≥ 3). 1 = la mayor. −1 = aislada (sin aglomeración)."),
    "agglomeration_n": ("Number of published detections in the same agglomeration (0 if isolated).",
                        "Número de detecciones publicadas en la misma aglomeración (0 si aislada)."),
    "agglomeration_class": ("Size class of the agglomeration: isolated, patio group (3–4), hamlet (5–9), small village (10–19), village (20–49), large settlement (≥ 50).",
                            "Clase de tamaño de la aglomeración: isolated, patio group (3–4), hamlet (5–9), small village (10–19), village (20–49), large settlement (≥ 50)."),
    "nn_dist_m": ("Distance to the nearest other published detection (m), any tile, any agglomeration.",
                  "Distancia a la detección publicada más cercana (m), en cualquier tile y aglomeración."),
}

AGGLOMERATION_DICT = {
    "agglomeration_id": ("Id, 1 = largest; joins detections_tau016.agglomeration_id.",
                         "Id, 1 = la mayor; se une con detections_tau016.agglomeration_id."),
    "size_class": ("patio group (3–4), hamlet (5–9), small village (10–19), village (20–49), large settlement (≥ 50).",
                   "patio group (3–4), hamlet (5–9), small village (10–19), village (20–49), large settlement (≥ 50)."),
    "n_structures": ("Number of published detections in the agglomeration.",
                     "Número de detecciones publicadas en la aglomeración."),
    "latitude": ("Mean latitude of the members (WGS84).", "Latitud media de los miembros (WGS84)."),
    "longitude": ("Mean longitude of the members (WGS84).", "Longitud media de los miembros (WGS84)."),
    "extent_m": ("Diagonal of the bounding box of the members (m).", "Diagonal de la caja envolvente de los miembros (m)."),
    "hull_area_ha": ("Area of the convex hull of the members (ha); 0 if collinear.",
                     "Área de la envolvente convexa de los miembros (ha); 0 si son colineales."),
    "density_per_ha": ("n_structures / hull_area_ha; empty when the hull has no area.",
                       "n_structures / hull_area_ha; vacío cuando la envolvente no tiene área."),
    "nn_median_m": ("Median nearest-neighbour distance among the members (m).",
                    "Mediana de la distancia al vecino más cercano entre los miembros (m)."),
    "score_mean": ("Mean building score of the members.", "Score medio de los miembros."),
    "score_median": ("Median building score of the members.", "Score mediano de los miembros."),
    "score_max": ("Maximum building score among the members.", "Score máximo entre los miembros."),
    "area_m2_median": ("Median footprint area of the members (m²).", "Área mediana de huella de los miembros (m²)."),
    "area_m2_max": ("Largest footprint in the agglomeration (m²).", "Huella mayor de la aglomeración (m²)."),
    "area_m2_sum": ("Sum of footprint areas (m²): built surface.", "Suma de áreas de huella (m²): superficie construida."),
    "n_over_200m2": ("Members with footprint > 200 m² (platform-sized).", "Miembros con huella > 200 m² (tamaño plataforma)."),
    "peak_depth_mean_m": ("Mean of peak_height_m over the members (m).", "Media de peak_height_m sobre los miembros (m)."),
    "orientation_coherence": ("Mean resultant length of the doubled orientations, [0, 1]: 1 = all members share one axis, ≈ 0 = no shared axis.",
                              "Longitud resultante media de las orientaciones dobladas, [0, 1]: 1 = todos comparten un eje, ≈ 0 = sin eje común."),
    "orientation_mean_deg": ("Axial mean orientation of the members (degrees from east, 0–180).",
                             "Orientación media axial de los miembros (grados desde el este, 0–180)."),
    "n_tiles": ("Number of LAS tiles the members come from (> 1 = crosses a tile edge).",
                "Número de tiles LAS de los miembros (> 1 = cruza un borde de tile)."),
    "n_datasets": ("Number of regional datasets the members come from.",
                   "Número de datasets regionales de los miembros."),
    "source_folder": ("Most frequent dataset among the members.", "Dataset más frecuente entre los miembros."),
    "tiles": ("All member tiles, ';'-separated.", "Todos los tiles de los miembros, separados por ';'."),
}


def build_dictionary(det_cols, agg_cols) -> pd.DataFrame:
    rows = []
    for c in det_cols:
        en, es = DETECTION_DICT[c]
        rows.append({"table": "detections_tau016", "column": c, "description_en": en, "descripcion_es": es})
    for c in agg_cols:
        en, es = AGGLOMERATION_DICT[c]
        rows.append({"table": "agglomerations_tau016", "column": c, "description_en": en, "descripcion_es": es})
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--tau", type=float, default=TAU)
    p.add_argument("--eps", type=float, default=EPS_M)
    p.add_argument("--min-samples", type=int, default=MIN_SAMPLES)
    args = p.parse_args()

    d = pd.read_csv(args.csv)
    r = d[d.shape_type.eq("rectangular")]
    t = r[r.building_confidence >= args.tau].copy().reset_index(drop=True)
    print(f"gated rows {len(d)} (rectangular {len(r)}); published at tau >= {args.tau}: "
          f"{len(t)} in {t.filename.nunique()} tiles, {t.source_folder.nunique()} datasets")

    lat, lon = t.latitude.to_numpy(), t.longitude.to_numpy()
    xy = local_metres(lat, lon)
    raw = DBSCAN(eps=args.eps, min_samples=args.min_samples).fit_predict(xy)
    labels = relabel_by_size(raw, lat)

    agg = summarise(t, xy, labels)
    n_by_id = dict(zip(agg.agglomeration_id, agg.n_structures))
    t["agglomeration_id"] = labels
    t["agglomeration_n"] = [n_by_id.get(c, 0) for c in labels]
    t["agglomeration_class"] = [size_class(n) for n in t.agglomeration_n]
    dist, _ = cKDTree(xy).query(xy, k=2)
    t["nn_dist_m"] = np.round(dist[:, 1], 1)

    grouped = labels >= 0
    print(f"agglomerations: {len(agg)}; detections in one: {grouped.sum()} "
          f"({grouped.mean():.1%}); isolated: {(~grouped).sum()}")
    cls = agg.groupby("size_class").agg(groups=("agglomeration_id", "size"),
                                        structures=("n_structures", "sum"))
    order = [name for _, _, name in SIZE_CLASSES]
    cls = cls.reindex(order).fillna(0).astype(int)
    print(cls.to_string())
    print(f"cross-tile agglomerations: {(agg.n_tiles > 1).sum()}; "
          f"cross-dataset: {(agg.n_datasets > 1).sum()}")
    print(f"largest: {agg.n_structures.head(5).tolist()}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    det_path = args.out_dir / "detections_tau016.csv"
    agg_path = args.out_dir / "agglomerations_tau016.csv"
    dic_path = args.out_dir / "data_dictionary.csv"
    t.to_csv(det_path, index=False)
    agg.to_csv(agg_path, index=False)
    dic = build_dictionary(list(t.columns), list(agg.columns))
    dic.to_csv(dic_path, index=False)
    print(f"wrote {det_path} ({len(t)} rows, {t.shape[1]} columns)")
    print(f"wrote {agg_path} ({len(agg)} rows, {agg.shape[1]} columns)")
    print(f"wrote {dic_path} ({len(dic)} rows)")

    xlsx = args.out_dir / "catalogue_tau016.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        t.to_excel(w, sheet_name="detections", index=False)
        agg.to_excel(w, sheet_name="agglomerations", index=False)
        dic.to_excel(w, sheet_name="dictionary", index=False)
    print(f"wrote {xlsx}")

    stats = {
        "tau": args.tau, "eps_m": args.eps, "min_samples": args.min_samples,
        "published": int(len(t)), "tiles": int(t.filename.nunique()),
        "datasets": int(t.source_folder.nunique()),
        "agglomerations": int(len(agg)),
        "grouped": int(grouped.sum()), "grouped_frac": float(grouped.mean()),
        "isolated": int((~grouped).sum()),
        "by_class": {k: {"groups": int(v.groups), "structures": int(v.structures)}
                     for k, v in cls.iterrows()},
        "cross_tile": int((agg.n_tiles > 1).sum()),
        "cross_dataset": int((agg.n_datasets > 1).sum()),
        "largest_n": agg.n_structures.head(10).tolist(),
        "n_structures_median": float(agg.n_structures.median()),
        "extent_m_median": float(agg.extent_m.median()),
        "nn_median_m_median": float(agg.nn_median_m.median()),
        "orientation_coherence_median": float(agg.orientation_coherence.median()),
        "orientation_coherence_q75": float(agg.orientation_coherence.quantile(.75)),
        "nn_dist_isolated_median_m": float(t.loc[~grouped, "nn_dist_m"].median()),
        "nn_dist_grouped_median_m": float(t.loc[grouped, "nn_dist_m"].median()),
        "score_mean_grouped": float(t.loc[grouped, "building_confidence"].mean()),
        "score_mean_isolated": float(t.loc[~grouped, "building_confidence"].mean()),
        "area_median_grouped": float(t.loc[grouped, "area_m2"].median()),
        "area_median_isolated": float(t.loc[~grouped, "area_m2"].median()),
        "by_dataset": {
            f: {"published": int(len(g)), "grouped": int((g.agglomeration_id >= 0).sum()),
                "groups": int(g.loc[g.agglomeration_id >= 0, "agglomeration_id"].nunique())}
            for f, g in t.groupby("source_folder")},
    }
    with open(args.out_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"wrote {args.out_dir / 'stats.json'}")


if __name__ == "__main__":
    main()
