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
| Direct daily bias and RMSE | `extract_direct_reference_agreement.py` | We calculate temporal metrics at each pixel on matched, eligible data without EC/bootstrap screening, then area-average the pixel metrics. |
| GeoTIFF export | `export_gridded_results_to_geotiff.py` | We export stored grids and metadata without altering estimation or masks. |



