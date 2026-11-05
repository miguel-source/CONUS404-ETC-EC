# Data access and provenance

We provide the acquisition routines and the event inventory used in our study. We keep full precipitation archives outside version control. Our code license does not change the terms of the source datasets.

## Products we evaluated

| Product | Acquisition route used by our routines | Daily variable or convention |
|---|---|---|
| CONUS404 | HyTEST intake catalog entry `conus404-daily-osn` | `PREC_ACC_NC`; daily accumulated precipitation, mm |
| CONUS404BA | HyTEST intake catalog entry `conus404-daily-ba-osn` | `RAIN`; daily accumulated precipitation, mm |
| MRMS Radar-Only QPE | Iowa State University MRMS MTArchive | `RadarOnly_QPE_24H`; file valid at D+1 00:00 UTC represents day D |
| PERSIANN-CDR | NOAA NCEI daily archive, product `v01r01` | Native daily 0.25-degree precipitation |
| Storm tracks | NOAA National Hurricane Center Atlantic HURDAT2 | Track positions, dates, and status records |
| Watersheds | The selected watershed polygons supplied with the study | HUC codes identify 20 HUC6 and three HUC8 domains |

We downloaded the precipitation subsets for the expanded event windows, retaining core-day flags for the narrower analysis. Our standard daily interval is [D 00:00 UTC, D+1 00:00 UTC). The MRMS routine preserves version-era information because the 2016-2022 sample crosses the October 2020 MRMS operational transition. We extracted the PERSIANN daily-to-event assembly from the study notebook into `scripts/assemble_persiann_events.py`. This step concatenates the required daily fields without spatial interpolation or temporal reaggregation, validates exact dates and matching grids, and writes the event files consumed by preparation. The acquisition pipeline calls it after downloading the four products.

## Authoritative source documentation

- CONUS404 data release: https://doi.org/10.5066/P9PHPK4F
- CONUS404BA data release: https://doi.org/10.5066/P9JE61P7
- HyTEST catalog: https://github.com/hytest-org/hytest/tree/main/dataset_catalog
- HyTEST catalog file used by the acquisition routines: https://raw.githubusercontent.com/hytest-org/hytest/main/dataset_catalog/hytest_intake_catalog.yml
- PERSIANN-CDR product and access documentation: https://www.ncei.noaa.gov/products/climate-data-records/precipitation-persiann
- PERSIANN-CDR dataset citation: https://doi.org/10.7289/V51V5BWQ
- PERSIANN-CDR archive: https://www.ncei.noaa.gov/data/precipitation-persiann/access/
- MRMS archive: https://mtarchive.geol.iastate.edu/
- NOAA MRMS documentation: https://www.nssl.noaa.gov/projects/mrms/
- Atlantic HURDAT2 information: https://www.nhc.noaa.gov/data/#hurdat
- HURDAT2 files: https://www.nhc.noaa.gov/data/hurdat/
- USGS Watershed Boundary Dataset information: https://www.usgs.gov/national-hydrography/watershed-boundary-dataset

## Version information and reproducibility

We identify the actual inputs through the product IDs, variables, date windows, grids, saved metadata, and processing manifests. The recorded HyTEST catalog URL points to a mutable `main` branch. Our acquisition metadata does not establish an exact numbered USGS release for those OSN stores. We therefore do not retrospectively label the evaluated data as the newest CONUS404 or CONUS404BA release. A current dataset landing page may describe a later revision or a longer period of record than the data we analyzed.

For a new application, we recommend saving the exact catalog file, its commit identifier where available, downloaded file hashes, track snapshot, and all run manifests. For a paper reproduction, we compare the resulting inputs and summaries with the checked-in study inventory and benchmark values. Regenerating event windows from revised HURDAT2 records is a new processing run and can change the selected days.

The original preparation cache fingerprints used source file size and modification time together with grid hashes. They were not full-file SHA-256 checksums of every precipitation input. New release provenance distinguishes code content hashes from those original source-file fingerprints. Some original storage locations were cloud placeholders during packaging; we did not perform a complete new download and rerun of all native archives.

## Boundary inputs

We use geographic watershed polygons with HUC identifiers and storm-assignment attributes. The event-window routine expects longitude-latitude geometry; a different input CRS must be transformed before spherical distance calculations. Watershed geometry and identifiers affect event windows, center selection, fractional area weights, and all area summaries.

The plotting routines can read the bundled boundary GeoPackage containing `watersheds_hurricane_analysis`, `conus_states`, and `conus_boundary`. The configuration also supplies single-layer watershed and state files to avoid ambiguity when readers choose the first layer by default. We attribute the Natural Earth state outlines and USGS WBD watershed derivatives in `DATA_LICENSES.md`. We do not redistribute commercial basemap tiles, Esri map imagery, or screenshots as study software.

## Preparing another region or storm sample

We update the storm inventory, HURDAT2 selection, watershed polygons, download dates, and source coverage together. We retain the distinction between a storm's assigned watershed windows and the `all_windows` gridded sample. We then reevaluate the minimum number of independent storms, matched days, and collocation admissibility rather than lowering thresholds solely to increase map coverage.

We require no embedded account keys in the repository. If an upstream service requires authentication in a future workflow, credentials belong in the user's environment or provider-supported credential store, not in scripts, configuration examples, notebooks, or manifests committed to Git.
