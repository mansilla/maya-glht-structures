# Candidate Maya structures and agglomerations from NASA G-LiHT airborne LiDAR

Data and code for

> R. Mansilla and R. Mansilla-Sánchez, *Scaling properties of ancient Maya
> settlement from an archive-scale LiDAR catalogue of candidate structures*
> (submitted).

The package has four parts: the list of the 528 public LiDAR tiles the work is
based on, the deterministic detector that turns them into candidate
structures, the published catalogue (24,805 candidate structures in 1,452
agglomerations), and the code for every analysis in the paper. Every
deposited table can be regenerated from the tiles with the code here (see
[Reproducing the deposit](#reproducing-the-deposit)).

The detections are **candidates**, not confirmed archaeological structures.
The detector has not been validated against ground truth, so treat the
catalogue as a list of locations to check.

## Contents

```
data/
  tile_urls.txt                 528 tile URLs (gzip-compressed LAS 1.1) at NASA Goddard's G-LiHT portal
  metadata_urls.txt             the portal's metadata PDF for each of the 16 datasets
catalogue/                      the published catalogue (score threshold tau = 0.16)
  detections_tau016.csv         24,805 candidate structures x 38 columns
  agglomerations_tau016.csv     1,452 agglomerations x 23 columns
  data_dictionary.csv           every column of both tables, in English and Spanish
  catalogue_tau016.xlsx         the three tables above as sheets of one workbook
  canopy_cover.csv              per-detection canopy measurement (input to merge_canopy.py)
  stats.json                    summary counts of the catalogue
intermediate/
  gated_candidates.csv          detector output after the depth gates, all scores, both
                                shape families (28,224 rows); input to the catalogue
                                and to the threshold-sensitivity analysis
  tile_inventory.csv            per-tile extent, point counts and CRS read from the LAS headers
analysis/                       outputs of tools/scaling_analysis.py: the paper's tables
  results.json, *.csv           (fits, null models, sensitivity grids)
  figures/*.png                 the paper's main-text figures from that analysis
building_detection/             the detector (Python package)
detect_ruins_and_map.py         runs the detector over every tile; writes outputs/detector/ruins.csv
tools/
  download_tiles.py             fetch the tiles into data/<dataset>/
  data_inventory.py             tile inventory from the LAS headers
  filter_catalogue.py           depth gates -> intermediate/gated_candidates.csv
  agglomerations.py             threshold tau = 0.16 + DBSCAN agglomerations -> catalogue/
  canopy_cover.py               canopy cover over each detection, from the raw returns
  merge_canopy.py               adds the canopy columns to the catalogue
  scaling_analysis.py           every analysis in the paper -> analysis/
  recalibrate_confidence.py     re-measures the circular-score calibration (not needed to
                                reproduce the catalogue, which is rectangular only)
requirements.txt                package versions used
SHA256SUMS                      checksums of every file
```

## The catalogue

**`detections_tau016.csv`** has one row per candidate structure.
`(filename, building_id)` is the unique key. Coordinates are given both as
WGS84 `latitude`/`longitude` and as UTM `easting`/`northing` in the tile's own
zone: 15N (EPSG:32615) or 16N (EPSG:32616), read from each tile's GeoTIFF
header. The columns are:
- footprint geometry (`width_m`, `height_m`, `area_m2`, `orientation_deg`, …)
- the terms of the shape score and the score itself (`building_confidence`)
- depth below the local ground surface (`peak_height_m`, `mean_height_m`)
- the agglomeration the detection belongs to (`agglomeration_id`, where 1 is
  the largest group and −1 means isolated; `agglomeration_n`;
  `agglomeration_class`)
- the distance to the nearest other detection (`nn_dist_m`)
- canopy cover over the footprint (`canopy_cover`, `canopy_p95_m`,
  `canopy_class`)

**`agglomerations_tau016.csv`** has one row per agglomeration: number of
structures, built area, convex-hull area and extent, density, nearest-neighbour
spacing, orientation coherence, and the tiles and datasets it spans.

**`data_dictionary.csv`** defines every column of both tables in English and
Spanish.

**Agglomerations** are DBSCAN clusters (ε = 50 m, minimum 3 points) over the
detection centroids. The centroids are projected to local metres with
per-point cosine scaling of latitude and longitude, so a group can cross a
tile or dataset boundary (41 do).

## Reproducing the deposit

The pipeline runs in Python 3.12 with the packages in `requirements.txt`.
Run every command from this folder.

```bash
pip install -r requirements.txt

python tools/download_tiles.py            # 1. 528 tiles, ~183 GB, into data/<dataset>/
python tools/data_inventory.py            # 2. -> intermediate/tile_inventory.csv            (seconds)
python detect_ruins_and_map.py            # 3. -> outputs/detector/ruins.csv, ungated        (~1.5 h)
python tools/filter_catalogue.py          # 4. depth gates -> intermediate/gated_candidates.csv
python tools/agglomerations.py            # 5. tau = 0.16 + DBSCAN -> catalogue/ (35 columns)
python tools/canopy_cover.py --all        # 6. -> catalogue/canopy_cover.csv                (~2.5 h)
python tools/merge_canopy.py              # 7. canopy columns -> catalogue/ (38 columns)
python tools/scaling_analysis.py          # 8. -> analysis/                                  (~4 min)
```

Steps 5–8 need only the deposited `intermediate/` and `catalogue/` files, not
the tiles. You can therefore check the catalogue and every number in the paper
in a few minutes without downloading anything.

Notes on the pipeline:
- **Depth gates (step 4).** Candidates are kept when their peak depth below
  the local ground is between 0.25 and 1.80 m and their mean depth is at least
  0.15 m. The gates are applied by `filter_catalogue.py` to the
  `peak_height_m` and `mean_height_m` columns of the detector output, not
  inside the detector. The reason is given in the comment on the configuration
  in `detect_ruins_and_map.py`.
- **Score threshold (step 5).** The published catalogue keeps rectangular
  candidates with `building_confidence ≥ 0.16`. Pass `--tau` to
  `agglomerations.py` to build the catalogue at another threshold.
- **Side outputs of step 3.** `detect_ruins_and_map.py` also writes
  interactive HTML maps and a sample of cross-section plots to
  `outputs/detector/`. They are for inspection only and are not used
  downstream.

### What has been verified

Each step was rerun from this folder and compared with the deposited files:

- **Steps 5 and 7:** reproduce `detections_tau016.csv`,
  `agglomerations_tau016.csv`, `data_dictionary.csv` and `stats.json` byte for
  byte.
- **Step 8:** reproduces `analysis/results.json` and every `analysis/*.csv`
  byte for byte. The bootstraps use a fixed seed.
- **Step 2:** reproduces `tile_inventory.csv` byte for byte.
- **Steps 3–4:** rerun on four tiles from four datasets, they reproduce the
  corresponding 155 rows of `gated_candidates.csv` exactly, every column.

A full rerun of steps 3 and 6 over all 528 tiles has not been repeated from
this package. The deposited files come from the original full run, made with
the same code.

## Data source and licence

The input data are G-LiHT (Goddard's LiDAR, Hyperspectral and Thermal Imager;
Cook et al. 2013, *Remote Sensing* 5:4045–4066) scans from the AMIGACarb
(April–May 2013) and Kiuic (April 2013) campaigns. They are distributed by NASA
Goddard at https://glihtdata.gsfc.nasa.gov/ and are not redistributed here,
only listed.

Code: MIT licence (`LICENSE`). Catalogue, intermediate tables and analysis
outputs: Creative Commons Attribution 4.0 (CC BY 4.0).

## Citation

Please cite the paper and this deposit (see `CITATION.cff`).
