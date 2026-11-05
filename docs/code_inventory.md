# Code inventory

We provide the routines used to acquire, align, analyze, summarize, and plot our precipitation data. We retained the numerical bodies of the original routines and replaced computer-specific paths with configurable locations. `provenance/source_inventory.json` records the SHA-256 hashes of each original source and its released counterpart. We added the orchestration, synthetic examples, and tests when preparing this public release.

## Final gridded workflow

| Component | Routines | Role |
|---|---|---|
| Event tracks and windows | `download_hurdat.py`; `build_tc_event_windows_spherical.py` | We read Atlantic HURDAT2, interpolate hourly on great-circle paths, measure spherical distance to watershed polygons, and derive basin windows. The original window generator uses all available track statuses; it does not filter to TS/HU records. |
| Precipitation acquisition | `download_conus404_events.py`; `download_conus404ba_events.py`; `download_mrms_events.py`; `download_persiann_cdr_events.py` | We acquire event daily files with matched UTC day definitions and record source metadata. We use PERSIANN-CDR v01r01 and MRMS RadarOnly_QPE_24H. The HyTEST catalog URL is mutable; the source metadata do not establish a fully pinned historical store version. |
| Native PERSIANN event assembly | `assemble_persiann_events.py` | We extracted the variable-selection and event-concatenation functions from cells 8–9 of our original PERSIANN notebook. We concatenate native daily fields with exact coordinate matching and filename-based UTC dates, retaining float32 compression and performing no spatial interpolation or temporal aggregation. We added rejection of ambiguous duplicate source versions. |
| Spatial alignment | `prepare_gridded_persiann_etc.py` and shared functions in `prepare_watershed_day_etc.py` | We conservatively average source–target overlap areas in EPSG:5070 onto the PERSIANN-CDR grid in EPSG:4326. PERSIANN-CDR remains on its native grid. The standalone watershed-day analysis is a separate exploratory workflow. |
| Four-product EC | `run_gridded_extended_collocation.py` | We estimate the CONUS404–CONUS404BA correlated error block; other cross-error covariances are assumed zero. We retain raw and log1p analyses, storm-block bootstrap, and leave-one-storm-out outputs. MRMS/PERSIANN EC diagnostic variances average two supported triplet formulations; the separate triplet results below remain unaveraged. |
| Separate TC/ETC triplets | `run_separate_triplet_error_maps.py` | We estimate error variance, error SD and latent squared correlation separately for CONUS404–MRMS–PERSIANN-CDR and CONUS404BA–MRMS–PERSIANN-CDR. We preserve negative algebraic estimates and full-triplet validity flags. |
| Paired support and optional agreement diagnostics | `run_gridded_dataset_agreement.py` | We derive the paired model support from EC and bootstrap validity, and retain optional standardized-difference/tail diagnostics. These tail scores are not p-values or probabilities that a product is accurate. |
| Direct daily bias and RMSE | `extract_direct_reference_agreement.py`; `create_daily_agreement_violins_600dpi.py` | We calculate temporal metrics at each pixel on matched, eligible data without EC/bootstrap screening, then area-average the pixel metrics. |
| Numerical summaries | `extract_word_results_metrics.py` | We extract the paper-specific paired summaries and verify known spatial-support counts. This routine intentionally checks the original study and is not a generic new-event summary function. |
| GeoTIFF export | `export_gridded_results_to_geotiff.py` | We export stored grids and metadata without altering estimation or masks. |

## Publication and supplemental figures

We retain the final brown–cream–green triplet figures in `create_triplet_brown_green_600dpi.py`, together with their historical metadata-producing dependencies (`create_all_available_triplet_variance_figures.py`, `create_all_available_triplet_sd_r2_figures.py`, `create_large_triplet_error_variance_figure.py`, and `build_word_triplet_metric_figures.py`). We retain paired uncertainty and leave-one-storm-out figures in `create_paired_bootstrap_categories_600dpi.py` and `create_paired_loso_600dpi.py`. `build_word_primary_figures.py` supplies shared definitions for those plots. These historical reproduction scripts include explicit checks for the original study's counts and means.

We provide additional diagnostic panels in `create_supplementary_figures.py`; its command-line arguments specify every data input. The supplemental plots preserve the study's 683-cell grid definition and are intended to reproduce this study, while the covariance API and gridded estimators support other datasets. `brown_green_cartography.py` supplies the shared palette, north symbols, and thin ruler scale bars. Exact typography requires Times New Roman; several historical plotting routines permit a serif fallback if it is unavailable.

We retain Harvey accumulation and event-window illustrations in the `build_word_harvey_figure.py`, `create_harvey_contrasted_600dpi.py`, `plot_harvey_precipitation_comparison.py`, `create_hurdat2_window_figure.py`, `create_sanjacinto_harvey_window_figure.py`, and `create_harvey_all_watersheds_window_figure.py` scripts. The schematic window illustration is explicitly distinguished from the real track map. We retain earlier plot/audit routines for provenance; the supplemental index identifies the recommended final panels.

## Shared modules and alternatives

`map_overlays.py` provides boundary layers; `prepare_boundary_files.py` extracts single-layer GeoPackages so a reader cannot accidentally load a states layer as watershed polygons. `study_paths.py` resolves local data paths. `run_pipeline.py` runs selected stages, supports `--dry-run`, and records commands.

`prepare_study_track_subset.py` retains complete HURDAT2 records for our ten selected storms, and `derive_event_table.py` derives event-wide bounds from the watershed-specific contact inventory. We bundle the small selected-track snapshot and preserve the full-archive and subset hashes. The `windows` stage writes a new inventory under the results directory and does not overwrite the frozen study metadata.

`run_hurricane_extended_collocation.py` analyzes watershed-day averages and belongs to the earlier exploratory workflow. We did not use that spatial aggregation for the final pixelwise results. We include it to preserve the analytical history and because `prepare_watershed_day_etc.py` also supplies reusable geometry and file-reading helpers to the final gridded preparation.

The reusable `tc_precip_collocation` Python package exposes the same covariance estimators through `four_product_ec` and `triple_collocation`. The synthetic example and tests check known error moments, admissibility, matched-day sampling, date de-duplication, volume-conserving averaging, and missing-area masking. We have not rerun the full remote-data acquisition or 5,000-replicate analysis as part of packaging this release.
