#!/usr/bin/env python3
"""
Put both shape families' confidence scores on one scale, and re-measure the
mapping that does it.

The rectangular and circular detectors each produce an evidence score with
its own hand-tuned weights, and those scores land on different scales: over
the 528-tile run the circular median sat at 0.423 against the rectangular
0.262, so a single "> 0.4" threshold passed 53% of circular detections and
3% of rectangular ones. The calibration in building_detection.metrics maps
the circular score onto the rectangular scale through matched quantiles, so
one threshold selects the same fraction of each family.

Two modes:

    --measure   print fresh calibration anchors from the CSV's raw scores,
                to paste into metrics._CIRCULAR_CALIBRATION_*. Run this
                whenever detection thresholds change, because the two
                families' distributions move with them.

    (default)   rewrite the CSV with the calibration currently in
                metrics.py applied.

No LAS file is touched either way. The rewrite is exact: a row's stored
confidence is the raw score the detector computed, so calibrating it needs
no reconstruction from the rounded metric columns. Rectangular rows keep
their value untouched -- the rectangular scale is the target, not the thing
being moved.

Usage:
    python tools/recalibrate_confidence.py --measure
    python tools/recalibrate_confidence.py
    python tools/recalibrate_confidence.py --csv path/to/ruins.csv
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from building_detection.batch import FIELDNAMES
from building_detection.metrics import calibrate_circular_confidence

DEFAULT_CSV = Path("outputs/detector/ruins.csv")

# Percentiles anchoring the piecewise-linear calibration. Dense in the upper
# tail, where the threshold decisions actually land.
ANCHOR_PERCENTILES = (0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9, 100)


def raw_score(row: dict) -> float:
    """
    The detector's uncalibrated evidence score for a row.

    Rows written before the calibration existed carry it in
    building_confidence; rows written after also keep it in confidence_raw.
    """
    value = row.get('confidence_raw')
    if value in (None, ""):
        value = row.get('building_confidence') or 0.0
    return float(value)


def load_rows(csv_path: Path) -> list:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def split_by_shape(rows: list) -> dict:
    out = {'rectangular': [], 'circular': []}
    for row in rows:
        out.setdefault(row.get('shape_type') or 'rectangular', []).append(row)
    return out


def measure(csv_path: Path) -> None:
    """Print calibration anchors derived from the CSV's raw evidence scores."""
    by_shape = split_by_shape(load_rows(csv_path))
    rect = np.sort(np.array([raw_score(r) for r in by_shape['rectangular']]))
    circ = np.sort(np.array([raw_score(r) for r in by_shape['circular']]))

    print(f"{len(rect):,} rectangular, {len(circ):,} circular\n")
    if len(circ) < 100 or len(rect) < 100:
        print("Too few detections in one family to calibrate against.")
        return

    print(f"{'pct':>6} {'circular(raw)':>14} {'rectangular':>13}")
    for p in ANCHOR_PERCENTILES:
        print(f"{p:>6} {np.percentile(circ, p):>14.4f} {np.percentile(rect, p):>13.4f}")

    def fmt(values):
        return "\n".join(
            "    " + ", ".join(f"{v:.4f}" for v in values[i:i + 6]) + ","
            for i in range(0, len(values), 6)
        )

    print("\nPaste into building_detection/metrics.py:\n")
    print("_CIRCULAR_CALIBRATION_RAW = (")
    print(fmt([np.percentile(circ, p) for p in ANCHOR_PERCENTILES]))
    print(")")
    print("_CIRCULAR_CALIBRATION_MAPPED = (")
    print(fmt([np.percentile(rect, p) for p in ANCHOR_PERCENTILES]))
    print(")")


def rewrite(csv_path: Path, backup: bool = True) -> None:
    """Apply the current calibration to the circular rows of a CSV."""
    rows = load_rows(csv_path)
    if not rows:
        print(f"{csv_path} has no rows.")
        return

    if backup:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = csv_path.with_name(f"{csv_path.stem}_preconf_{stamp}.csv")
        shutil.copy2(csv_path, backup_path)
        print(f"Backup: {backup_path.name}")

    before = {'rectangular': [], 'circular': []}
    after = {'rectangular': [], 'circular': []}

    for row in rows:
        shape = row.get('shape_type') or 'rectangular'
        raw = raw_score(row)
        before[shape].append(float(row.get('building_confidence') or 0.0))

        # The rectangular scale is the target, so those rows keep their value.
        calibrated = calibrate_circular_confidence(raw) if shape == 'circular' else raw
        row['confidence_raw'] = round(raw, 3)
        row['building_confidence'] = round(calibrated, 3)
        after[shape].append(calibrated)

    tmp_path = csv_path.with_suffix(".tmp")
    with open(tmp_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(csv_path)

    print(f"Rewrote {len(rows):,} rows -> {csv_path}\n")
    print(f"{'':12} {'n':>8} {'>0.4 before':>12} {'>0.4 after':>11} "
          f"{'p50 before':>11} {'p50 after':>10}")
    for shape in ('rectangular', 'circular'):
        b, a = np.array(before[shape]), np.array(after[shape])
        if not len(b):
            continue
        print(f"{shape:12} {len(b):>8,} {(b > 0.4).mean() * 100:>11.2f}% "
              f"{(a > 0.4).mean() * 100:>10.2f}% "
              f"{np.median(b):>11.3f} {np.median(a):>10.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--measure", action="store_true",
                        help="Print fresh calibration anchors instead of rewriting")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    if not args.csv.exists():
        parser.error(f"{args.csv} not found")

    if args.measure:
        measure(args.csv)
    else:
        rewrite(args.csv, backup=not args.no_backup)


if __name__ == "__main__":
    main()
