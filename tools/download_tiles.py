#!/usr/bin/env python3
"""
Download the 528 G-LiHT tiles listed in data/tile_urls.txt into
data/<dataset>/<tile>.las.gz, the layout detect_ruins_and_map.py expects.
Tiles already present with the right size are skipped, so the script can be
re-run after an interruption. About 183 GB in total.

Usage:
    python tools/download_tiles.py
    python tools/download_tiles.py --dataset Kuiic1_Apr2013     # one dataset
"""
from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--urls", type=Path, default=ROOT / "data/tile_urls.txt")
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--dataset", default=None, help="only tiles of this dataset")
    a = ap.parse_args()

    urls = [u.strip() for u in a.urls.read_text().splitlines() if u.strip()]
    for i, url in enumerate(urls, 1):
        # .../G-LiHT/<dataset>/lidar/las/<tile>.las.gz
        parts = url.split("/")
        dataset, name = parts[-4], parts[-1]
        if a.dataset and dataset != a.dataset:
            continue
        out = a.data_root / dataset / name
        out.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url) as r:
            size = int(r.headers.get("Content-Length", -1))
            if out.exists() and out.stat().st_size == size:
                print(f"[{i}/{len(urls)}] {name} present")
                continue
            print(f"[{i}/{len(urls)}] {name} ({size / 1e6:.0f} MB)", flush=True)
            tmp = out.with_suffix(".part")
            with open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, length=1 << 20)
            tmp.rename(out)


if __name__ == "__main__":
    main()
