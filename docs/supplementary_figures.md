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


## Interpretation

We estimate only the CONUS404–CONUS404BA error correlation in the specified four-product EC model. The other error-pair covariances are assumptions. The common-triplet subset and bootstrap-screened paired EC subset are different selection criteria; neither is assumed to be nested within the other. We compare dimensionless raw/log1p latent R² as a transformation sensitivity diagnostic, while recognizing that the transformations represent different latent signals. The stacked-bar percentages in S6 are cell-count percentages, not watershed-area percentages.

## Verification

We verified identical input grids, the saved paired-comparison masks, the EC correlation identity, raster dimensions and 600 dpi metadata, and figure text placement. We inspected every preview. The source-data SHA256 checksums were unchanged after export. The captions and CSV tables retain the distinctions among the paired EC subset, common-admissible triplets, and product-specific algebraic estimates.
