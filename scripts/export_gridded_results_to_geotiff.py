"""Export all hurricane EC/agreement NetCDF result layers for ArcGIS Pro.

Every variable containing latitude and longitude is exploded into one
single-band GeoTIFF per combination of its other dimensions.  The folder and
filename retain window, transform, product, pair, and omitted-storm labels.
All rasters use the native PERSIANN-CDR 0.25-degree grid in EPSG:4326.
"""

from __future__ import annotations

from study_paths import PROJECT_ROOT, RESULTS_ROOT, METADATA_ROOT, REPO_ROOT, STATES_FILE, WATERSHEDS_FILE, DAILY_INPUT, EC_DIRECTORY, EC_INPUT, LOSO_INPUT, AGREEMENT_DIRECTORY, AGREEMENT_INPUT


import argparse
import itertools
import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
import xarray as xr

from map_overlays import write_boundary_geopackage


PAPER_ROOT = (PROJECT_ROOT)
EVENT_ROOT = PAPER_ROOT / "ETC_StormEvents"
DEFAULT_EXPORT_ROOT = RESULTS_ROOT / "ArcGIS_Exports"
DAILY_COMBINED = DAILY_INPUT
RESULT_SOURCES = {
    "01_EC_Point": EC_INPUT,
    "02_EC_LOSO": LOSO_INPUT,
    "03_Agreement_Point": AGREEMENT_INPUT,
    "04_Agreement_LOSO": AGREEMENT_DIRECTORY / "gridded_dataset_agreement_leave_one_storm_out.nc",
}

DIMENSION_ABBREVIATIONS = {
    "window": "win",
    "transform": "xform",
    "product": "prod",
    "dataset_pair": "pair",
    "omitted_storm": "omit",
    "time": "date",
    "observation": "obs",
    "storm": "storm",
}


def _safe_name(value: Any, maximum: int = 90) -> str:
    if isinstance(value, np.datetime64):
        text = np.datetime_as_string(value, unit="D")
    elif isinstance(value, pd.Timestamp):
        text = value.strftime("%Y-%m-%d")
    elif isinstance(value, bytes):
        text = value.decode("utf-8")
    else:
        text = str(value)
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-_.")
    return (text or "value")[:maximum]


def _grid_spec(lat: np.ndarray, lon: np.ndarray) -> dict[str, Any]:
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    if lat.ndim != 1 or lon.ndim != 1 or len(lat) < 2 or len(lon) < 2:
        raise ValueError("GeoTIFF export requires one-dimensional lat/lon grids.")
    dx = float(np.median(np.abs(np.diff(lon))))
    dy = float(np.median(np.abs(np.diff(lat))))
    if not np.allclose(np.abs(np.diff(lon)), dx, rtol=0, atol=1.0e-8):
        raise ValueError("Longitude spacing is not regular.")
    if not np.allclose(np.abs(np.diff(lat)), dy, rtol=0, atol=1.0e-8):
        raise ValueError("Latitude spacing is not regular.")
    west = float(np.min(lon) - dx / 2)
    north = float(np.max(lat) + dy / 2)
    return {
        "transform": from_origin(west, north, dx, dy),
        "flip_y": bool(lat[0] < lat[-1]),
        "flip_x": bool(lon[0] > lon[-1]),
        "width": len(lon),
        "height": len(lat),
        "resolution_x": dx,
        "resolution_y": dy,
        "bounds": (
            west,
            float(np.min(lat) - dy / 2),
            float(np.max(lon) + dx / 2),
            north,
        ),
    }


def _storage_type(dtype: np.dtype) -> tuple[str, int | float]:
    dtype = np.dtype(dtype)
    if dtype.kind == "u" and dtype.itemsize <= 1:
        return "uint8", 255
    if dtype.kind in "bi" and dtype.itemsize <= 2:
        return "int16", -32768
    if dtype.kind in "biu" and dtype.itemsize <= 4:
        return "int32", -2147483648
    return "float32", -9999.0


def _coordinate_value(array: xr.DataArray, dim: str, index: int) -> Any:
    if dim in array.coords:
        value = array.coords[dim].values[index]
        return value.item() if hasattr(value, "item") and not isinstance(value, np.datetime64) else value
    return index


def _iter_spatial_slices(
    array: xr.DataArray,
) -> Iterable[tuple[dict[str, int], dict[str, Any], xr.DataArray]]:
    extra_dims = [dim for dim in array.dims if dim not in {"lat", "lon"}]
    if not extra_dims:
        yield {}, {}, array.transpose("lat", "lon")
        return
    ranges = [range(array.sizes[dim]) for dim in extra_dims]
    for indices in itertools.product(*ranges):
        isel = dict(zip(extra_dims, indices))
        labels = {
            dim: _coordinate_value(array, dim, index)
            for dim, index in isel.items()
        }
        yield isel, labels, array.isel(isel).transpose("lat", "lon")


def _slice_filename(labels: dict[str, Any]) -> str:
    if not labels:
        return "layer.tif"
    pieces = []
    for dim, value in labels.items():
        abbreviation = DIMENSION_ABBREVIATIONS.get(dim, _safe_name(dim, 12))
        pieces.append(f"{abbreviation}-{_safe_name(value, 55)}")
    return "__".join(pieces) + ".tif"


def _write_tiff(
    path: Path,
    data: np.ndarray,
    source_dtype: np.dtype,
    grid: dict[str, Any],
    *,
    domain_mask: np.ndarray | None,
    preserve_outside_domain: bool,
    tags: dict[str, Any],
    overwrite: bool,
) -> dict[str, Any]:
    if path.exists() and not overwrite:
        with rasterio.open(path) as existing:
            expected_shape = (grid["height"], grid["width"])
            if (
                existing.count != 1
                or existing.shape != expected_shape
                or str(existing.crs) != "EPSG:4326"
                or existing.transform != grid["transform"]
            ):
                raise RuntimeError(
                    f"Existing GeoTIFF has an incompatible grid: {path}. "
                    "Rerun with overwrite=True to replace it."
                )
            stored = existing.read(1, masked=True)
            valid = np.asarray(stored.compressed())
            minimum = float(valid.min()) if valid.size else np.nan
            maximum = float(valid.max()) if valid.size else np.nan
            return {
                "status": "existing",
                "dtype": existing.dtypes[0],
                "nodata": existing.nodata,
                "valid_cells": int(valid.size),
                "minimum": minimum,
                "maximum": maximum,
            }
    if grid["flip_y"]:
        data = np.flipud(data)
        if domain_mask is not None:
            domain_mask = np.flipud(domain_mask)
    if grid["flip_x"]:
        data = np.fliplr(data)
        if domain_mask is not None:
            domain_mask = np.fliplr(domain_mask)
    dtype, nodata = _storage_type(source_dtype)
    invalid = ~np.isfinite(data.astype(np.float64, copy=False))
    if domain_mask is not None and not preserve_outside_domain:
        invalid |= ~domain_mask
    output = np.where(invalid, nodata, data).astype(dtype)
    finite = output[output != nodata]
    minimum = float(finite.min()) if finite.size else np.nan
    maximum = float(finite.max()) if finite.size else np.nan
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": grid["height"],
        "width": grid["width"],
        "count": 1,
        "dtype": dtype,
        "crs": "EPSG:4326",
        "transform": grid["transform"],
        "nodata": nodata,
        "compress": "DEFLATE",
        "predictor": 3 if dtype.startswith("float") else 2,
        "zlevel": 6,
        "tiled": True,
        "blockxsize": 128,
        "blockysize": 128,
        "BIGTIFF": "IF_SAFER",
    }
    with rasterio.open(path, "w", **profile) as destination:
        destination.write(output, 1)
        clean_tags = {
            key: str(value)[:32000]
            for key, value in tags.items()
            if value is not None
        }
        clean_tags.update(
            AREA_OR_POINT="Area",
            STATISTICS_MINIMUM=minimum,
            STATISTICS_MAXIMUM=maximum,
            STATISTICS_VALID_PERCENT=(100.0 * finite.size / output.size),
        )
        destination.update_tags(**clean_tags)
    return {
        "status": "written",
        "dtype": dtype,
        "nodata": nodata,
        "valid_cells": int(finite.size),
        "minimum": minimum,
        "maximum": maximum,
    }


def export_netcdf(
    source_path: Path,
    group_name: str,
    export_root: Path,
    external_domain_mask: np.ndarray | None,
    *,
    overwrite: bool,
    apply_domain_mask: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with xr.open_dataset(source_path, decode_times=True) as dataset:
        if "lat" not in dataset.coords or "lon" not in dataset.coords:
            raise ValueError(f"No lat/lon coordinates in {source_path}")
        grid = _grid_spec(dataset.lat.values, dataset.lon.values)
        local_domain = (
            np.asarray(dataset["study_domain_mask"].values, dtype=bool)
            if "study_domain_mask" in dataset
            else external_domain_mask
        )
        if local_domain is not None and local_domain.shape != (
            grid["height"],
            grid["width"],
        ):
            raise ValueError(f"Domain mask grid differs for {source_path}")
        for variable_name, array in dataset.data_vars.items():
            if "lat" not in array.dims or "lon" not in array.dims:
                continue
            variable_dir = export_root / group_name / _safe_name(variable_name)
            preserve_outside = variable_name in {
                "study_domain_mask",
                "study_watershed_fraction",
            }
            for _, labels, spatial in _iter_spatial_slices(array):
                output_path = variable_dir / _slice_filename(labels)
                tags = {
                    "source_netcdf": str(source_path.resolve()),
                    "analysis_group": group_name,
                    "variable": variable_name,
                    "selectors_json": json.dumps(
                        {key: str(value) for key, value in labels.items()},
                        sort_keys=True,
                    ),
                    "long_name": array.attrs.get("long_name", ""),
                    "description": array.attrs.get("description", ""),
                    "units": array.attrs.get("units", ""),
                    "grid": "PERSIANN-CDR 0.25 degree",
                }
                stats = _write_tiff(
                    output_path,
                    np.asarray(spatial.values),
                    array.dtype,
                    grid,
                    domain_mask=local_domain if apply_domain_mask else None,
                    preserve_outside_domain=preserve_outside,
                    tags=tags,
                    overwrite=overwrite,
                )
                rows.append(
                    {
                        "analysis_group": group_name,
                        "source_netcdf": str(source_path.resolve()),
                        "variable": variable_name,
                        "selectors": json.dumps(
                            {key: str(value) for key, value in labels.items()},
                            sort_keys=True,
                        ),
                        "geotiff": str(output_path.resolve()),
                        "units": array.attrs.get("units", ""),
                        **stats,
                    }
                )
    return rows


def _write_arcgis_guides(export_root: Path) -> tuple[Path, Path]:
    styles = pd.DataFrame(
        [
            ["*difference*; *covariance*; *correlation*", "Diverging", "Center at 0; use blue-white-red. Correlation range -1 to 1."],
            ["agreement_probability*", "Sequential", "Fixed range 0 to 0.5; darker means greater agreement."],
            ["agreement_score*; estimated_r2*", "Sequential", "Fixed range 0 to 1."],
            ["valid_*; sample_eligible; within_*", "Unique values", "0 = no, 1 = yes, NoData outside study domain."],
            ["invalid_reason_code", "Unique values", "Categorical bitmask; consult NetCDF flag_meanings."],
            ["n_observations; n_contributing_storms; *_count", "Graduated colors", "Integer counts."],
            ["error_sd*; error_variance*; signal_variance*; snr*", "Sequential", "Use a nonnegative sequential color ramp."],
        ],
        columns=["variable_pattern", "arcgis_renderer", "recommendation"],
    )
    style_path = export_root / "ArcGIS_symbology_guide.csv"
    styles.to_csv(style_path, index=False)
    readme_path = export_root / "README_ArcGIS.txt"
    readme_path.write_text(
        """ArcGIS Pro use
==============
1. Add 00_Boundaries/map_boundaries.gpkg. Use conus_boundary and
   watersheds_hurricane_analysis as unfilled outline layers above the rasters.
2. Add GeoTIFFs from 01_EC_Point, 02_EC_LOSO, 03_Agreement_Point, or
   04_Agreement_LOSO. Every raster is EPSG:4326 on the exact 0.25-degree
   PERSIANN-CDR cells.
3. Use the validity raster belonging to the analysis before interpreting a
   continuous estimate. EC primary results use valid_primary_ec. Diagnostic
   reference-product maps also require reference_diagnostics_valid. Agreement
   outputs use valid_agreement.
4. Use ArcGIS_symbology_guide.csv for recommended renderer types and ranges.
5. For a publication layout like the supplied example, use a full-CONUS map
   frame plus regional inset map frames. Do not stretch the hurricane raster
   to the Pacific Northwest: its actual extent is -104 to -72.75 longitude
   and 26.75 to 44.25 latitude.

GeoTIFF filenames encode every nonspatial selector. The complete mapping to
source NetCDF, variable, selectors, units, range, and NoData is in
GeoTIFF_manifest.csv.
""",
        encoding="utf-8",
    )
    return style_path, readme_path


def run_export(
    export_root: Path | str = DEFAULT_EXPORT_ROOT,
    *,
    groups: Iterable[str] | None = None,
    include_event_inputs: bool = False,
    include_combined_daily: bool = False,
    overwrite: bool = False,
    apply_domain_mask: bool = True,
) -> dict[str, Any]:
    export_root = Path(export_root)
    export_root.mkdir(parents=True, exist_ok=True)
    selected_groups = set(groups) if groups else set(RESULT_SOURCES)
    unknown = selected_groups.difference(RESULT_SOURCES)
    if unknown:
        raise ValueError(f"Unknown result groups: {sorted(unknown)}")
    for name in selected_groups:
        if not RESULT_SOURCES[name].exists():
            raise FileNotFoundError(RESULT_SOURCES[name])
    if not DAILY_COMBINED.exists():
        raise FileNotFoundError(DAILY_COMBINED)
    with xr.open_dataset(DAILY_COMBINED, engine="netcdf4") as daily:
        domain_mask = np.asarray(daily.study_domain_mask.values, dtype=bool)

    boundary_path = write_boundary_geopackage(
        export_root / "00_Boundaries" / "map_boundaries.gpkg",
        overwrite=overwrite,
    )
    manifest_rows: list[dict[str, Any]] = []
    for group_name in RESULT_SOURCES:
        if group_name not in selected_groups:
            continue
        print(f"Exporting {group_name}...")
        manifest_rows.extend(
            export_netcdf(
                RESULT_SOURCES[group_name],
                group_name,
                export_root,
                domain_mask,
                overwrite=overwrite,
                apply_domain_mask=apply_domain_mask,
            )
        )

    if include_event_inputs:
        event_files = sorted(
            (EVENT_ROOT / "Gridded_PERSIANN" / "events").glob("*.nc")
        )
        for event_file in event_files:
            group_name = f"05_Regridded_Events/{_safe_name(event_file.stem)}"
            print(f"Exporting {event_file.name}...")
            manifest_rows.extend(
                export_netcdf(
                    event_file,
                    group_name,
                    export_root,
                    domain_mask,
                    overwrite=overwrite,
                    apply_domain_mask=apply_domain_mask,
                )
            )
    if include_combined_daily:
        manifest_rows.extend(
            export_netcdf(
                DAILY_COMBINED,
                "06_Combined_Daily",
                export_root,
                domain_mask,
                overwrite=overwrite,
                apply_domain_mask=apply_domain_mask,
            )
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = export_root / "GeoTIFF_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    style_path, readme_path = _write_arcgis_guides(export_root)
    summary = {
        "export_root": str(export_root.resolve()),
        "geotiff_count": int(len(manifest)),
        "written_count": int((manifest.status == "written").sum()),
        "existing_count": int((manifest.status == "existing").sum()),
        "boundary_geopackage": str(boundary_path.resolve()),
        "manifest": str(manifest_path.resolve()),
        "symbology_guide": str(style_path.resolve()),
        "arcgis_readme": str(readme_path.resolve()),
    }
    summary_path = export_root / "export_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT_ROOT)
    parser.add_argument(
        "--groups",
        nargs="*",
        choices=list(RESULT_SOURCES),
        default=None,
        help="Omit to export all four calculation-result groups.",
    )
    parser.add_argument("--include-event-inputs", action="store_true")
    parser.add_argument("--include-combined-daily", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-domain-mask", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    run_export(
        export_root=arguments.export_root,
        groups=arguments.groups,
        include_event_inputs=arguments.include_event_inputs,
        include_combined_daily=arguments.include_combined_daily,
        overwrite=arguments.overwrite,
        apply_domain_mask=not arguments.no_domain_mask,
    )
