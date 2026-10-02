#!/usr/bin/env python3
"""
Apply the detector's depth gates to an existing catalogue, without reprocessing.

This is the step that produced intermediate/gated_candidates.csv from the
ungated detector run (detect_ruins_and_map.py, whose own depth gates are
disabled). The gates act on the peak_height_m and mean_height_m columns and
only remove rows; they never change a candidate's metrics or move it between
the rectangular and circular families. They are not identical to the
detector's internal gates, which test a slightly different depth estimate
(see the comment in detect_ruins_and_map.config).

Filtering a finished CSV works ONLY for the per-candidate gates. Changing anything
that shapes the mask or the components -- min/max_relative_elevation,
dem_cell_size, smoothing_radius_m, the cluster-area bounds,
subsurface_max_depth -- really does need a re-run, because the connected
components themselves change.

See the comment in detect_ruins_and_map.config for why 0.25 and 1.80 rather
than some other pair.

Usage:
    python tools/filter_catalogue.py
    python tools/filter_catalogue.py --min-depth 0.3 --max-depth 1.5
    python tools/filter_catalogue.py --min-confidence 0.4 --out-dir outputs/high_conf
"""
from __future__ import annotations

import argparse
import collections
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from building_detection.batch import FIELDNAMES

DEFAULT_CSV = Path("outputs/detector/ruins.csv")
DEFAULT_OUT = Path("intermediate")


def load_rows(csv_path: Path) -> list:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def apply_gates(rows, min_depth, max_depth, min_mean_depth, min_confidence):
    kept = []
    for row in rows:
        peak = float(row['peak_height_m'])
        if not (min_depth <= peak <= max_depth):
            continue
        if float(row['mean_height_m']) < min_mean_depth:
            continue
        if float(row['building_confidence']) < min_confidence:
            continue
        kept.append(row)
    return kept


def renumber(rows) -> None:
    """Re-issue building_id per source file, so ids stay 1..n within a tile."""
    counters = collections.Counter()
    for row in rows:
        key = row['filename']
        counters[key] += 1
        row['building_id'] = counters[key]


def report(rows, kept) -> None:
    def shapes(rs):
        return collections.Counter(r.get('shape_type') or 'rectangular' for r in rs)

    s_all, s_keep = shapes(rows), shapes(kept)
    print(f"input : {len(rows):>8,}")
    print(f"kept  : {len(kept):>8,}  ({len(kept) / max(len(rows), 1) * 100:.1f}%)")
    for shape in ('rectangular', 'circular'):
        if s_all[shape]:
            print(f"  {shape:12} {s_keep[shape]:>7,} / {s_all[shape]:>7,} "
                  f"= {s_keep[shape] / s_all[shape] * 100:5.1f}%")

    tot = collections.Counter(r['source_folder'] for r in rows)
    sur = collections.Counter(r['source_folder'] for r in kept)
    print("\nsurvival by dataset:")
    for dataset in sorted(tot, key=lambda d: -sur[d] / max(tot[d], 1)):
        name = dataset.replace('AMIGACarb_', '')[:38]
        print(f"  {name:40} {sur[dataset]:>6,}/{tot[dataset]:>7,} "
              f"= {sur[dataset] / tot[dataset] * 100:>5.1f}%")

    if kept:
        conf = np.array([float(r['building_confidence']) for r in kept])
        print(f"\nconfidence: median {np.median(conf):.3f}  "
              f">0.4 {(conf > 0.4).sum():,} ({(conf > 0.4).mean() * 100:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-depth", type=float, default=0.25,
                        help="min peak depth, metres (default: sensor noise floor)")
    parser.add_argument("--max-depth", type=float, default=1.80,
                        help="max peak depth, metres (default: below window saturation)")
    parser.add_argument("--min-mean-depth", type=float, default=0.15)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--maps", action="store_true",
                        help="also regenerate maps and index for the filtered set")
    parser.add_argument("--cross-sections", type=int, default=0,
                        help="also plot N cross-sections, split evenly by shape")
    args = parser.parse_args()

    if not args.csv.exists():
        parser.error(f"{args.csv} not found")

    rows = load_rows(args.csv)
    kept = apply_gates(rows, args.min_depth, args.max_depth,
                       args.min_mean_depth, args.min_confidence)
    print(f"gates: depth [{args.min_depth}, {args.max_depth}] m, "
          f"mean depth >= {args.min_mean_depth} m"
          + (f", confidence >= {args.min_confidence}" if args.min_confidence else ""))
    report(rows, kept)

    if not kept:
        print("\nNothing survived the gates; not writing an empty catalogue.")
        return

    renumber(kept)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / "gated_candidates.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(kept)
    print(f"\nWrote {out_csv}")

    if args.maps or args.cross_sections:
        import detect_ruins_and_map as m
        m.OUTPUT_DIR = args.out_dir
        if args.maps:
            m.generate_maps(kept, {})
        if args.cross_sections:
            m.generate_cross_sections(kept, n_sample=args.cross_sections,
                                      per_shape=True)


if __name__ == "__main__":
    main()
