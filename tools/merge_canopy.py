#!/usr/bin/env python3
"""
Add the canopy columns measured by tools/canopy_cover.py to the published
catalogue, and refresh the data dictionary and the workbook to match.

tools/agglomerations.py writes detections_tau016.csv with 35 columns;
tools/canopy_cover.py --all then measures canopy over every detection from the
raw returns (catalogue/canopy_cover.csv, ~2.5 h). This step joins the two on
(filename, building_id) and adds

    canopy_cover   first-return cover > 2 m, rounded to 3 decimals
    canopy_p95_m   95th percentile of height above ground (m), 2 decimals
    canopy_class   under canopy (>= 0.5), partial (0.2-0.5), open (< 0.2)

giving the 38-column table that is published.

Usage:
    python tools/merge_canopy.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CAT_DIR = ROOT / "catalogue"
UNDER, OPEN = 0.5, 0.2

CANOPY_DICT = {
    "canopy_cover": (
        "Canopy cover over the footprint: fraction of first returns more than 2 m above "
        "the min-Z ground, in a disc of half the long side + 2 m around the centroid. "
        "Empty when fewer than 5 returns.",
        "Cobertura de dosel sobre la huella: fracción de primeros retornos a más de 2 m "
        "del suelo (mínimo Z), en un disco de medio lado mayor + 2 m alrededor del "
        "centroide. Vacío si hay menos de 5 retornos."),
    "canopy_p95_m": (
        "95th percentile of return height above ground in that disc (m): an upper canopy height.",
        "Percentil 95 de la altura de los retornos sobre el suelo en ese disco (m): "
        "altura del dosel superior."),
    "canopy_class": (
        "under canopy (cover >= 0.5), partial (0.2-0.5), open (< 0.2).",
        "under canopy (cobertura >= 0.5), partial (0.2-0.5), open (< 0.2)."),
}


def canopy_class(c: float) -> str | float:
    if np.isnan(c):
        return np.nan
    return "under canopy" if c >= UNDER else ("partial" if c >= OPEN else "open")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=CAT_DIR)
    a = ap.parse_args()

    det_path = a.dir / "detections_tau016.csv"
    dic_path = a.dir / "data_dictionary.csv"
    d = pd.read_csv(det_path).drop(columns=list(CANOPY_DICT), errors="ignore")
    c = pd.read_csv(a.dir / "canopy_cover.csv")[["filename", "building_id",
                                                  "canopy_cover", "canopy_p95_m"]]
    m = d.merge(c, on=["filename", "building_id"], how="left", validate="one_to_one")
    assert len(m) == len(d)
    m["canopy_cover"] = m.canopy_cover.round(3)
    m["canopy_p95_m"] = m.canopy_p95_m.round(2)
    m["canopy_class"] = [canopy_class(v) for v in m.canopy_cover]
    m.to_csv(det_path, index=False)
    print(f"wrote {det_path} ({len(m)} rows, {m.shape[1]} columns)")
    print(m.canopy_class.value_counts(normalize=True, dropna=False).round(3).to_string())

    dic = pd.read_csv(dic_path)
    dic = dic[~((dic.table == "detections_tau016") & dic.column.isin(list(CANOPY_DICT)))]
    new = pd.DataFrame([{"table": "detections_tau016", "column": k,
                         "description_en": en, "descripcion_es": es}
                        for k, (en, es) in CANOPY_DICT.items()])
    det_rows = dic[dic.table == "detections_tau016"]
    dic = pd.concat([det_rows, new, dic[dic.table != "detections_tau016"]], ignore_index=True)
    dic.to_csv(dic_path, index=False)
    print(f"wrote {dic_path} ({len(dic)} rows)")

    agg = pd.read_csv(a.dir / "agglomerations_tau016.csv")
    xlsx = a.dir / "catalogue_tau016.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        m.to_excel(w, sheet_name="detections", index=False)
        agg.to_excel(w, sheet_name="agglomerations", index=False)
        dic.to_excel(w, sheet_name="dictionary", index=False)
    print(f"wrote {xlsx}")


if __name__ == "__main__":
    main()
