# Tropical cyclone precipitation evaluation with collocation

We provide the computational workflow accompanying our comparison of CONUS404, CONUS404BA, MRMS Radar-Only QPE, and PERSIANN-CDR during ten tropical cyclones from 2016 to 2022. We evaluated daily precipitation on the common 0.25-degree PERSIANN-CDR grid across 23 watershed domains.

Our repository covers data acquisition, track-based event windows, conservative grid alignment, matched observations, separate TC/ETC triplets, four-product extended collocation (EC), storm-block bootstrap, leave-one-storm-out sensitivity, and direct-reference agreement.
## Start with the reproducible example

We recommend first creating the supplied environment and running the synthetic example. It does not download the precipitation archives or require our original file locations.

```bash
conda env create -f environment.yml
conda activate tc-precip-collocation
python examples/run_synthetic.py
python -m pytest tests
```

The example checks the statistical workflow using synthetic observations with known error structure. It is a numerical validation example, not a substitute for the ten-storm observational sample.

## Reproduce the study workflow

We configure paths with `configs/study.example.json` and the `TC_PRECIP_PROJECT`, `TC_PRECIP_RESULTS`, `TC_PRECIP_WATERSHEDS`, and `TC_PRECIP_STATES` environment variables. We keep native precipitation inputs outside the source tree. The pipeline runner provides stage selection and a dry-run preview; its command-line help describes the available stages.

```bash
python scripts/run_pipeline.py --help
python scripts/run_pipeline.py --config configs/study.example.json --dry-run
```

Our processing order is:

1. Retrieve a recorded HURDAT2 track snapshot and construct watershed-specific windows.
2. Retrieve daily precipitation for the expanded windows from all four products and assemble the PERSIANN daily files into native-grid event files.
3. Align CONUS404, CONUS404BA, and MRMS to native PERSIANN-CDR cells using EPSG:5070 overlap areas.
4. Pool matched observations across storm-calendar windows at each selected grid cell.
5. Run four-product EC, the 5,000-replicate storm-block bootstrap, and storm-omission sensitivity.
6. Calculate the two TC/ETC triplets separately, retaining separate reference estimates.
7. Calculate daily model-reference bias and RMSE without collocation or bootstrap screening.
8. Generate figures, tables, supplemental diagnostics, and optional GeoTIFF exports.

For example, the primary statistical calculation can be run directly after preparation:

```bash
python scripts/run_gridded_extended_collocation.py --input path/to/hurricane_daily_all4_persiann_grid.nc --output-dir path/to/ec_results --sample-mode all_windows --bootstrap 5000 --seed 404
```

We describe the scientific choices and equations in [Computational methods](docs/METHODS.md), reusable functions in [Statistical API](docs/STATISTICAL_API.md), data sources and version limits in [Data access](docs/DATA_ACCESS.md), and verification scope in [Reproducibility](docs/REPRODUCIBILITY.md). The code inventory and provenance records identify the study scripts and their portable release copies.

## Quantities and spatial subsets

We distinguish TC/ETC random-error variance and SD from direct RMSE against another product. ETC latent R2 describes estimated squared correlation with an unknown precipitation signal. Four-product EC additionally estimates the CONUS404-CONUS404BA error covariance. It does not independently identify all six error correlations among the four products.

Our final `all_windows` sample contains 78 unique core days and 127 expanded days.


## Supplemental material

We provide the supplement's figure-generation routine, figure-ready data, captions, and provenance with the release. 


## Scope of verification

We tested the portable numerical routines with synthetic cases and checked available study results against stored benchmarks. We did not repeat the full native-data acquisition and 5,000-replicate ten-storm calculation in this repository. New downloads can differ when upstream archives or catalogs have been revised. The verification report records which commands were actually run.

## Reuse and citation

We release our study-specific code under the MIT license (This is a preliminary under revision version). Source datasets and third-party boundary products retain their own terms; see [Notice and attribution](NOTICE.md) and [Data attribution](DATA_LICENSES.md). We exclude credentials, personal storage paths, large raw archives, vendor runtimes, and proprietary basemap imagery.

We credit Miguel Diaz as the repository author in `CITATION.cff`. We will add the public repository URL and archived release metadata when available. No software DOI has been assigned by this local package. We recommend archiving the eventual tagged GitHub release with a repository-preservation service and citing both that version and the accompanying paper.
