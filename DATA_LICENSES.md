# Data attribution and rights

We apply the MIT license to our study-specific code and documentation. We retain the source attribution and terms of the data products below. This file does not grant new rights over third-party datasets.

## Plotting boundaries

We derived the `conus_states` and `conus_boundary` plotting layers from Natural Earth's 1:50m `admin_1_states_provinces_lakes` data through the study's `map_overlays.py` routine. Natural Earth states that its vector and raster map data are in the public domain: https://www.naturalearthdata.com/about/terms-of-use/ . We credit these layers as "Made with Natural Earth." The source file does not record an exact Natural Earth version.

Our watershed source package identifies the USGS Watershed Boundary Dataset (WBD) and the ArcGIS Living Atlas WBD HUC6 access service. The source collection included 27 watershed features; the final hurricane analysis uses 23 selected watershed domains. We preserve the HUC identifiers and analysis selection rather than treating all 27 source features as study domains.

- USGS WBD: https://www.usgs.gov/national-hydrography/watershed-boundary-dataset
- USGS WBD service: https://hydro.nationalmap.gov/arcgis/rest/services/wbd/FeatureServer
- Source HUC6 access service: https://services.arcgis.com/P3ePLMYs2RVChkJx/arcgis/rest/services/Watershed_Boundary_Dataset_HUC_6s/FeatureServer/0
- USGS catalog record: https://data.usgs.gov/datacatalog/data/USGS%3A0101bc32-916e-481d-8654-db7f8509fd0c

We use the public-domain USGS WBD provenance for these derivatives and retain the access-service lineage. The supplied source documentation records a HUC6 service edit date of 2025-04-09, not an exact complete WBD release version for every polygon. The GeoPackage itself contains no detailed source-license metadata table. We therefore provide this attribution separately. We do not include ArcGIS basemap tiles, commercial imagery, or screenshots in the code release.

## Tracks and precipitation

We retain NOAA/NHC attribution for any bundled HURDAT2 snapshot and identify the track filename and content hash. HURDAT2 documentation is at https://www.nhc.noaa.gov/data/#hurdat .

We identify CONUS404, CONUS404BA, MRMS, and PERSIANN-CDR source documentation and citations in `docs/DATA_ACCESS.md`. We distribute code, selected derived diagnostics, and figure-ready arrays rather than the complete source precipitation archives. Users should cite the original products and consult their current dataset terms when obtaining or redistributing data.

## Derived figures and tables

We generated the supplemental figures and tables from the stated study inputs. The associated metadata records the source files, content hashes, spatial masks, units, and transformations. We make these study-generated materials available with the accompanying code while preserving the underlying data attribution above. Reusing a derived table or figure does not establish independence among its source products.
