# Supplemental figures

We prepared six supporting figures using geographic panels, including our north symbol and thin WGS84 ruler scale bar, calibrated at 27.35°N. We used saved estimates and did not rerun collocation or bootstrap sampling when making these figures.

| Figure | Purpose |
| --- | --- |
| S1 | Correlation between CONUS404 and CONUS404BA random errors from four-product EC |
| S2 | Membership in the paired EC and common-admissible-triplet spatial subsets |
| S3 | Admissible bootstrap-draw fractions and conditional interval widths |
| S4 | Raw versus log1p latent R² sensitivity on identical admissible cells |
| S5 | Storm calendar-window sampling and the duplicate-date policy |
| S6 | Nonnegative, negative, and unavailable variance estimates in the separate triplets |

Each figure directory contains a 600 dpi PNG and LZW-compressed TIFF, a PDF and editable SVG, a lower-resolution preview, a caption, metadata, and a CSV table of the plotted values or counts. The TIFFs are publication figures, not georeferenced data rasters. The combined `manifest.json` records the source checksums, figure paths, and numerical summaries; `captions.md` collects the captions in manuscript order.

## Reproduce the figures

We provide the standalone code in `source/create_supplementary_figures.py` with its `brown_green_cartography.py` helper. The repository also includes both in `scripts/`. The plotting routine requires Python 3.11 or newer, NumPy, Matplotlib, netCDF4, Pillow, Shapely 2, pyproj, and SciPy. Times New Roman must be installed to reproduce the exact publication typography; we do not redistribute font files.

We included compact, derived figure inputs in the repository so readers can reproduce all six figures without obtaining the original NetCDF files or rerunning inference:

```bash
python scripts/create_supplementary_figures.py --input-bundle data/figure_inputs --output results/supplementary
```

The input bundle contains only the saved fields needed for these figures, plus the geographic boundary layers. It does not contain the full precipitation archive. We store numeric and text arrays in compressed NPZ files with `allow_pickle=False`; dimension and date-unit metadata are retained. Original NetCDF checksums link the bundle to the study estimates. The plotting routine checks bundle checksums before reading.

Times New Roman is the default. On systems without that font, select an installed alternative explicitly, for example `--font "Liberation Serif"`, or set the `TC_PRECIP_FONT` environment variable. The resulting metadata record the requested font family. Exact publication reproduction uses Times New Roman.

To regenerate from the original saved analysis NetCDF files instead, run the following command from the repository root. Replace the example paths with the corresponding files on your computer. Each triplet has its own directory even though the raw/log NetCDF basenames are shared.

```bash
python scripts/create_supplementary_figures.py \
  --ec results/gridded_extended_collocation_all_windows.nc \
  --agreement results/gridded_dataset_agreement.nc \
  --daily data/processed/hurricane_daily_all4_persiann_grid.nc \
  --triplet-a-raw results/triplet_a/triplet_error_estimates_raw.nc \
  --triplet-b-raw results/triplet_b/triplet_error_estimates_raw.nc \
  --triplet-a-log results/triplet_a/triplet_error_estimates_log1p.nc \
  --triplet-b-log results/triplet_b/triplet_error_estimates_log1p.nc \
  --boundaries data/boundaries/map_boundaries.gpkg \
  --output results/supplementary
```

The GeoPackage must provide `conus_states`, `conus_boundary`, and `watersheds_hurricane_analysis` layers in EPSG:4326. This figure routine reproduces our 23-watershed, 683-center-selected-cell study; the statistical routines elsewhere in the repository provide the reusable analysis workflow.

## Interpretation

We estimate only the CONUS404–CONUS404BA error correlation in the specified four-product EC model. The other error-pair covariances are assumptions. The common-triplet subset and bootstrap-screened paired EC subset are different selection criteria; neither is assumed to be nested within the other. We compare dimensionless raw/log1p latent R² as a transformation sensitivity diagnostic, while recognizing that the transformations represent different latent signals. The stacked-bar percentages in S6 are cell-count percentages, not watershed-area percentages.

## Verification

We verified identical input grids, the saved paired-comparison masks, the EC correlation identity, raster dimensions and 600 dpi metadata, and figure text placement. We inspected every preview. The source-data SHA256 checksums were unchanged after export. The captions and CSV tables retain the distinctions among the paired EC subset, common-admissible triplets, and product-specific algebraic estimates.
