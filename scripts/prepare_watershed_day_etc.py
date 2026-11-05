"""
Prepare matched watershed-day precipitation observations for paired ETC analyses.

VS Code / Jupyter use
---------------------
This file uses ``# %%`` cell markers. Open it in VS Code and use "Run Cell", or
run the complete file from the CONUS404 Python environment.

Primary analysis design
-----------------------
The four products remain on their native grids. For each product, daily grid
cells are intersected with the same watershed polygon in an equal-area CRS and
are averaged using exact cell-polygon overlap areas. The resulting spatial
support is therefore the watershed, so a common PERSIANN raster grid is not
needed for the watershed-level ETC analysis.

Outputs include a long table, a matched four-product wide table, a NetCDF
observation table, QC summaries, and paired triplet CSV files for both the
expanded and core event windows.
"""

# %% Imports
from __future__ import annotations

from study_paths import PROJECT_ROOT, RESULTS_ROOT, METADATA_ROOT, REPO_ROOT, STATES_FILE, WATERSHEDS_FILE, DAILY_INPUT, EC_DIRECTORY, EC_INPUT, LOSO_INPUT, AGREEMENT_DIRECTORY, AGREEMENT_INPUT


import argparse
import hashlib
import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import xarray as xr

# Importing rioxarray activates the .rio accessor used below.
import rioxarray  # noqa: F401


# %% Configuration
PAPER_ROOT = (PROJECT_ROOT)

DATA_ROOT = PAPER_ROOT / "Hurricane_Events_P_Data"
WATERSHED_FILE = WATERSHEDS_FILE
DEFAULT_OUTPUT_DIR = PAPER_ROOT / "ETC_nc" / "Hurricane_watershed_day_ETC"

# CONUS Albers: used only for computing intersection areas in square metres.
AREA_CRS = "EPSG:5070"

# A watershed-day is ETC-ready only if every product covers at least this
# fraction of its watershed with valid precipitation values.
MIN_VALID_AREA_FRACTION = 0.95

# This is a QC ceiling, not a clipping threshold. Values above it stop the run
# so that units or fill-value handling can be inspected.
MAX_PLAUSIBLE_DAILY_MM = 2_000.0

PRODUCTS = {
    "conus404": {
        "directory": DATA_ROOT / "CONUS404",
        "variable": "PREC_ACC_NC",
    },
    "conus404ba": {
        "directory": DATA_ROOT / "CONUS404BA",
        "variable": "RAIN",
    },
    "mrms": {
        "directory": DATA_ROOT / "MRMS",
        "variable": "RadarOnly_QPE_24H",
    },
    "persiann_cdr": {
        "directory": DATA_ROOT / "PERSIANN-CDR" / "Event_Files",
        "variable": "precipitation",
    },
}

EVENTS = pd.DataFrame(
    [
        ("Matthew", 2016, "AL142016", 4, "2016-10-05", "2016-10-12", "2016-10-01", "2016-10-13"),
        ("Harvey", 2017, "AL092017", 8, "2017-08-23", "2017-09-03", "2017-08-19", "2017-09-04"),
        ("Irma", 2017, "AL112017", 3, "2017-09-08", "2017-09-14", "2017-09-04", "2017-09-15"),
        ("Florence", 2018, "AL062018", 3, "2018-09-12", "2018-09-20", "2018-09-08", "2018-09-21"),
        ("Imelda", 2019, "AL112019", 2, "2019-09-16", "2019-09-21", "2019-09-12", "2019-09-22"),
        ("Laura", 2020, "AL132020", 1, "2020-08-25", "2020-08-30", "2020-08-21", "2020-08-31"),
        ("Zeta", 2020, "AL282020", 1, "2020-10-27", "2020-10-31", "2020-10-23", "2020-11-01"),
        ("Eta", 2020, "AL292020", 1, "2020-11-07", "2020-11-15", "2020-11-03", "2020-11-16"),
        ("Ida", 2021, "AL092021", 9, "2021-08-27", "2021-09-04", "2021-08-23", "2021-09-05"),
        ("Ian", 2022, "AL092022", 1, "2022-09-26", "2022-10-02", "2022-09-22", "2022-10-03"),
    ],
    columns=[
        "event_name",
        "storm_year",
        "storm_id",
        "expected_watersheds",
        "core_start",
        "core_end",
        "expanded_start",
        "expanded_end",
    ],
)

for _column in ["core_start", "core_end", "expanded_start", "expanded_end"]:
    EVENTS[_column] = pd.to_datetime(EVENTS[_column])


# %% Input discovery and watershed-event membership
def _normalise_huc(value: object, huc_level: object) -> str:
    """Preserve leading zeroes if a DBF reader interpreted a HUC as numeric."""
    text = re.sub(r"\.0$", "", str(value).strip())
    match = re.search(r"(\d+)", str(huc_level))
    return text.zfill(int(match.group(1))) if match else text


def load_watersheds(path: Path = WATERSHED_FILE) -> gpd.GeoDataFrame:
    watersheds = gpd.read_file(path)
    required = {
        "BASIN_ID",
        "BASIN_NM",
        "STORMS",
        "HUC_CODES",
        "HUC_LEVEL",
        "HUC_NAME",
        "AREA_SQKM",
        "geometry",
    }
    missing = required.difference(watersheds.columns)
    if missing:
        raise ValueError(f"Watershed file is missing fields: {sorted(missing)}")
    if watersheds.crs is None:
        raise ValueError("The watershed shapefile has no CRS.")

    watersheds = watersheds.copy()
    watersheds["geometry"] = watersheds.geometry.map(shapely.make_valid)
    watersheds["huc_code"] = [
        _normalise_huc(value, level)
        for value, level in zip(watersheds["HUC_CODES"], watersheds["HUC_LEVEL"])
    ]
    watersheds["watershed_id"] = (
        watersheds["BASIN_ID"].astype(str).str.strip()
        + "__"
        + watersheds["huc_code"]
    )
    if watersheds["watershed_id"].duplicated().any():
        duplicate_ids = watersheds.loc[
            watersheds["watershed_id"].duplicated(False), "watershed_id"
        ].tolist()
        raise ValueError(f"Non-unique watershed IDs: {duplicate_ids}")
    if watersheds.geometry.is_empty.any():
        raise ValueError("At least one watershed has an empty geometry.")
    return watersheds


def watersheds_for_event(
    watersheds: gpd.GeoDataFrame, event: pd.Series
) -> gpd.GeoDataFrame:
    token = rf"\b{re.escape(event.event_name)}\s*\({event.storm_year}\)"
    selected = watersheds[
        watersheds["STORMS"].fillna("").str.contains(
            token, case=False, regex=True, na=False
        )
    ].copy()
    if len(selected) != int(event.expected_watersheds):
        raise ValueError(
            f"{event.event_name}: expected {event.expected_watersheds} watersheds "
            f"but the STORMS field selected {len(selected)}."
        )
    return selected


def find_event_file(product: str, event: pd.Series) -> Path:
    config = PRODUCTS[product]
    pattern = f"*_{event.event_name}_{event.storm_id}_expanded_*.nc"
    files = sorted(config["directory"].glob(pattern))
    if len(files) != 1:
        raise FileNotFoundError(
            f"{product}/{event.event_name}: expected exactly one file matching "
            f"{pattern!r}; found {len(files)}: {files}"
        )
    return files[0]


# %% Standardise only names and metadata; do not reproject or interpolate
def _normalised_dates(values: np.ndarray) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(pd.to_datetime(values))
    if dates.tz is not None:
        dates = dates.tz_convert("UTC").tz_localize(None)
    return dates.normalize()


def open_precipitation(
    product: str, path: Path
) -> tuple[xr.Dataset, xr.DataArray]:
    """Open one event and expose a common (time, y, x) interface."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Variable\(s\) referenced in bounds not in variables"
        )
        ds = xr.open_dataset(
            path,
            engine="netcdf4",
            decode_coords="all",
            chunks={"time": 1},
        )

    variable = PRODUCTS[product]["variable"]
    if variable not in ds:
        ds.close()
        raise KeyError(f"{path.name}: expected variable {variable!r}.")
    da = ds[variable]

    if product == "mrms":
        da = da.rename({"latitude": "y", "longitude": "x"})
        native_crs = ds.rio.crs
        if native_crs is None:
            ds.close()
            raise ValueError(f"{path.name}: MRMS CRS was not decoded.")
        da = da.rio.write_crs(native_crs)
    elif product == "persiann_cdr":
        da = da.rename({"lat": "y", "lon": "x"})
        # Convert 0..360 longitude to -180..180 without changing values.
        shifted_x = ((da["x"] + 180.0) % 360.0) - 180.0
        da = da.assign_coords(x=shifted_x).sortby("x")
        native_crs = "EPSG:4326"
        da = da.rio.write_crs(native_crs)
    else:
        native_crs = ds.rio.crs
        if native_crs is None:
            ds.close()
            raise ValueError(f"{path.name}: native projected CRS was not decoded.")
        da = da.rio.write_crs(native_crs)

    da = da.rio.set_spatial_dims(x_dim="x", y_dim="y")
    da = da.transpose("time", "y", "x")
    da = da.assign_coords(time=_normalised_dates(da.time.values))

    units = str(da.attrs.get("units", "")).lower().strip()
    if units not in {"mm", "millimeter", "millimeters"}:
        ds.close()
        raise ValueError(f"{path.name}: expected millimetres, found units={units!r}.")

    # NetCDF fill values have already decoded to NaN. Negative MRMS flags and
    # any other negative precipitation values must remain missing, never zero.
    da = da.where(np.isfinite(da) & (da >= 0.0))
    # xarray operations can drop the grid-mapping association even when the
    # scalar CRS coordinate remains. Restore it explicitly after cleaning.
    da = da.rio.write_crs(native_crs).rio.set_spatial_dims(x_dim="x", y_dim="y")
    da.name = product
    return ds, da


def validate_event_dates(da: xr.DataArray, event: pd.Series, path: Path) -> None:
    expected = pd.date_range(event.expanded_start, event.expanded_end, freq="D")
    actual = _normalised_dates(da.time.values)
    if not actual.is_unique:
        raise ValueError(f"{path.name}: duplicate daily timestamps.")
    if not actual.equals(expected):
        missing = expected.difference(actual).strftime("%Y-%m-%d").tolist()
        extra = actual.difference(expected).strftime("%Y-%m-%d").tolist()
        raise ValueError(
            f"{path.name}: dates do not match the expanded event window. "
            f"Missing={missing}; extra={extra}."
        )


# %% Exact fractional-cell intersection weights
@dataclass
class WeightSpec:
    row_start: int
    row_stop: int
    col_start: int
    col_stop: int
    weights_m2: np.ndarray
    basin_area_m2: float
    grid_overlap_area_m2: float
    geometric_coverage_fraction: float
    grid_hash: str


def coordinate_edges(centres: np.ndarray) -> np.ndarray:
    centres = np.asarray(centres, dtype=np.float64)
    if centres.ndim != 1 or centres.size < 2:
        raise ValueError("Spatial coordinates must be one-dimensional with >=2 cells.")
    differences = np.diff(centres)
    if not (np.all(differences > 0) or np.all(differences < 0)):
        raise ValueError("Spatial coordinates must be strictly monotonic.")
    edges = np.empty(centres.size + 1, dtype=np.float64)
    edges[1:-1] = (centres[:-1] + centres[1:]) / 2.0
    edges[0] = centres[0] - differences[0] / 2.0
    edges[-1] = centres[-1] + differences[-1] / 2.0
    return edges


def grid_hash(da: xr.DataArray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(da.x.values, dtype="<f8").tobytes())
    digest.update(np.asarray(da.y.values, dtype="<f8").tobytes())
    digest.update(da.rio.crs.to_wkt().encode("utf-8"))
    return digest.hexdigest()[:20]


def _weight_cache_path(
    cache_dir: Path, product: str, watershed_id: str, signature: str
) -> Path:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", watershed_id)
    return cache_dir / f"{product}__{safe_id}__{signature}.npz"


def _load_weight_spec(path: Path) -> WeightSpec:
    with np.load(path, allow_pickle=False) as saved:
        return WeightSpec(
            row_start=int(saved["row_start"]),
            row_stop=int(saved["row_stop"]),
            col_start=int(saved["col_start"]),
            col_stop=int(saved["col_stop"]),
            weights_m2=saved["weights_m2"],
            basin_area_m2=float(saved["basin_area_m2"]),
            grid_overlap_area_m2=float(saved["grid_overlap_area_m2"]),
            geometric_coverage_fraction=float(saved["geometric_coverage_fraction"]),
            grid_hash=str(saved["grid_hash"].item()),
        )


def _save_weight_spec(path: Path, spec: WeightSpec) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        row_start=spec.row_start,
        row_stop=spec.row_stop,
        col_start=spec.col_start,
        col_stop=spec.col_stop,
        weights_m2=spec.weights_m2,
        basin_area_m2=spec.basin_area_m2,
        grid_overlap_area_m2=spec.grid_overlap_area_m2,
        geometric_coverage_fraction=spec.geometric_coverage_fraction,
        grid_hash=np.asarray(spec.grid_hash),
    )


def build_weight_spec(
    da: xr.DataArray,
    watershed_geometry,
    watershed_crs,
    *,
    coverage_tolerance: float = 0.02,
) -> WeightSpec:
    """
    Intersect native raster cells with one watershed in an equal-area CRS.

    Each transformed cell polygon is intersected with the watershed polygon.
    Its exact overlap area in EPSG:5070 becomes the averaging weight.
    """
    native_crs = da.rio.crs
    if native_crs is None:
        raise ValueError("DataArray has no native CRS.")

    geometry = shapely.make_valid(watershed_geometry)
    basin_native = gpd.GeoSeries([geometry], crs=watershed_crs).to_crs(native_crs).iloc[0]
    basin_equal_area = (
        gpd.GeoSeries([geometry], crs=watershed_crs).to_crs(AREA_CRS).iloc[0]
    )
    basin_area_m2 = float(basin_equal_area.area)

    x_edges = coordinate_edges(da.x.values)
    y_edges = coordinate_edges(da.y.values)
    x_low = np.minimum(x_edges[:-1], x_edges[1:])
    x_high = np.maximum(x_edges[:-1], x_edges[1:])
    y_low = np.minimum(y_edges[:-1], y_edges[1:])
    y_high = np.maximum(y_edges[:-1], y_edges[1:])

    min_x, min_y, max_x, max_y = basin_native.bounds
    columns = np.flatnonzero((x_high > min_x) & (x_low < max_x))
    rows = np.flatnonzero((y_high > min_y) & (y_low < max_y))
    if rows.size == 0 or columns.size == 0:
        raise ValueError("Watershed does not intersect this product grid.")

    row_start, row_stop = int(rows.min()), int(rows.max()) + 1
    col_start, col_stop = int(columns.min()), int(columns.max()) + 1

    cell_x_low = x_low[col_start:col_stop]
    cell_x_high = x_high[col_start:col_stop]
    cell_y_low = y_low[row_start:row_stop]
    cell_y_high = y_high[row_start:row_stop]
    left, bottom = np.meshgrid(cell_x_low, cell_y_low)
    right, top = np.meshgrid(cell_x_high, cell_y_high)
    native_cells = shapely.box(left, bottom, right, top)

    # Reprojecting each cell polygon before intersection gives square-metre
    # weights and correctly handles geographic grids whose cell areas vary.
    equal_area_cells = gpd.GeoSeries(
        native_cells.ravel(), crs=native_crs
    ).to_crs(AREA_CRS)
    intersections = shapely.intersection(
        equal_area_cells.array, basin_equal_area
    )
    weights = np.asarray(shapely.area(intersections), dtype=np.float64).reshape(
        native_cells.shape
    )
    weights[weights < 1.0e-6] = 0.0

    overlap_area_m2 = float(weights.sum())
    coverage = overlap_area_m2 / basin_area_m2
    if not np.isfinite(coverage) or coverage <= 0:
        raise ValueError("Computed watershed-grid overlap is empty.")
    if coverage > 1.0 + coverage_tolerance:
        raise ValueError(
            f"Watershed-grid overlap is {coverage:.4f} of basin area (>1). "
            "Inspect the CRS and grid coordinates."
        )

    return WeightSpec(
        row_start=row_start,
        row_stop=row_stop,
        col_start=col_start,
        col_stop=col_stop,
        weights_m2=weights,
        basin_area_m2=basin_area_m2,
        grid_overlap_area_m2=overlap_area_m2,
        geometric_coverage_fraction=coverage,
        grid_hash=grid_hash(da),
    )


def get_weight_spec(
    da: xr.DataArray,
    product: str,
    watershed: pd.Series,
    watershed_crs,
    cache_dir: Path,
) -> WeightSpec:
    signature = grid_hash(da)
    cache_path = _weight_cache_path(
        cache_dir, product, watershed.watershed_id, signature
    )
    if cache_path.exists():
        return _load_weight_spec(cache_path)
    spec = build_weight_spec(da, watershed.geometry, watershed_crs)
    _save_weight_spec(cache_path, spec)
    return spec


# %% Native-grid watershed averages
def area_weighted_daily_mean(
    da: xr.DataArray, spec: WeightSpec
) -> xr.Dataset:
    subset = da.isel(
        y=slice(spec.row_start, spec.row_stop),
        x=slice(spec.col_start, spec.col_stop),
    )
    weights = xr.DataArray(
        spec.weights_m2,
        dims=("y", "x"),
        coords={"y": subset.y, "x": subset.x},
        name="intersection_area_m2",
    )
    positive_weights = weights > 0
    valid = subset.notnull() & positive_weights
    valid_area = weights.where(valid, 0.0).sum(("y", "x"))
    numerator = (subset.where(valid, 0.0) * weights).sum(("y", "x"))
    mean = (numerator / valid_area).where(valid_area > 0)
    # Divide by the full basin area, not only by the part reached by the grid.
    # This combines geometric domain coverage and day-specific valid-data
    # coverage in the single fraction used for complete-case filtering.
    valid_fraction = valid_area / spec.basin_area_m2
    result = xr.Dataset(
        {
            "precip_mm": mean,
            "valid_area_fraction": valid_fraction,
        }
    ).compute()

    finite = result["precip_mm"].values
    finite = finite[np.isfinite(finite)]
    if finite.size and (finite.min() < 0 or finite.max() > MAX_PLAUSIBLE_DAILY_MM):
        raise ValueError(
            "Computed basin precipitation lies outside the QC range "
            f"0..{MAX_PLAUSIBLE_DAILY_MM} mm/day: "
            f"min={finite.min()}, max={finite.max()}."
        )
    return result


def aggregate_event_product(
    product: str,
    event: pd.Series,
    event_watersheds: gpd.GeoDataFrame,
    cache_dir: Path,
) -> list[dict]:
    source_file = find_event_file(product, event)
    ds, da = open_precipitation(product, source_file)
    try:
        validate_event_dates(da, event, source_file)
        mrms_era_by_date: dict[pd.Timestamp, int] = {}
        if product == "mrms" and "mrms_era_code" in ds:
            mrms_dates = _normalised_dates(ds.time.values)
            mrms_era_by_date = {
                date: int(code)
                for date, code in zip(mrms_dates, ds["mrms_era_code"].values)
            }
        records: list[dict] = []
        for watershed in event_watersheds.itertuples(index=False):
            watershed_series = pd.Series(watershed._asdict())
            spec = get_weight_spec(
                da,
                product,
                watershed_series,
                event_watersheds.crs,
                cache_dir,
            )
            aggregate = area_weighted_daily_mean(da, spec)
            touched_cells = int(np.count_nonzero(spec.weights_m2 > 0))
            dates = _normalised_dates(aggregate.time.values)
            for index, date in enumerate(dates):
                records.append(
                    {
                        "event_name": event.event_name,
                        "storm_id": event.storm_id,
                        "storm_year": int(event.storm_year),
                        "date": date,
                        "expanded_day_index": int(
                            (date - event.expanded_start).days + 1
                        ),
                        "is_core_day": bool(
                            event.core_start <= date <= event.core_end
                        ),
                        "watershed_id": watershed.watershed_id,
                        "basin_id": watershed.BASIN_ID,
                        "basin_name": watershed.BASIN_NM,
                        "huc_code": watershed.huc_code,
                        "huc_level": watershed.HUC_LEVEL,
                        "huc_name": watershed.HUC_NAME,
                        "reported_area_sqkm": float(watershed.AREA_SQKM),
                        "product": product,
                        "precip_mm": float(aggregate.precip_mm.values[index]),
                        "mrms_era_code": mrms_era_by_date.get(date, np.nan),
                        "valid_area_fraction": float(
                            aggregate.valid_area_fraction.values[index]
                        ),
                        "geometric_coverage_fraction": float(
                            spec.geometric_coverage_fraction
                        ),
                        "intersecting_native_cells": touched_cells,
                        "grid_hash": spec.grid_hash,
                        "source_file": str(source_file),
                    }
                )
        return records
    finally:
        ds.close()


# %% Assemble exactly matched four-product rows
IDENTITY_COLUMNS = [
    "event_name",
    "storm_id",
    "storm_year",
    "date",
    "expanded_day_index",
    "is_core_day",
    "watershed_id",
    "basin_id",
    "basin_name",
    "huc_code",
    "huc_level",
    "huc_name",
    "reported_area_sqkm",
]


def make_wide_table(long_table: pd.DataFrame) -> pd.DataFrame:
    precipitation = long_table.pivot(
        index=IDENTITY_COLUMNS, columns="product", values="precip_mm"
    ).reset_index()
    coverage = long_table.pivot(
        index=IDENTITY_COLUMNS, columns="product", values="valid_area_fraction"
    ).reset_index()
    coverage = coverage.rename(
        columns={product: f"{product}_valid_area_fraction" for product in PRODUCTS}
    )
    wide = precipitation.merge(coverage, on=IDENTITY_COLUMNS, validate="one_to_one")
    mrms_metadata = long_table.loc[
        long_table["product"].eq("mrms"),
        IDENTITY_COLUMNS + ["mrms_era_code"],
    ]
    wide = wide.merge(mrms_metadata, on=IDENTITY_COLUMNS, validate="one_to_one")
    wide["mrms_era_code"] = wide["mrms_era_code"].astype(np.int16)

    expected_columns = list(PRODUCTS)
    for product in expected_columns:
        if product not in wide:
            wide[product] = np.nan
        coverage_column = f"{product}_valid_area_fraction"
        if coverage_column not in wide:
            wide[coverage_column] = np.nan

    value_ok = np.logical_and.reduce(
        [np.isfinite(wide[product]) for product in expected_columns]
    )
    coverage_ok = np.logical_and.reduce(
        [
            wide[f"{product}_valid_area_fraction"]
            >= MIN_VALID_AREA_FRACTION
            for product in expected_columns
        ]
    )
    wide["complete_case_all4"] = value_ok & coverage_ok
    wide["row_id"] = (
        wide["storm_id"].astype(str)
        + "|"
        + wide["watershed_id"].astype(str)
        + "|"
        + pd.to_datetime(wide["date"]).dt.strftime("%Y-%m-%d")
    )
    if wide["row_id"].duplicated().any():
        raise ValueError("Duplicate event-watershed-day row IDs were produced.")
    return wide.sort_values(
        ["storm_year", "event_name", "watershed_id", "date"]
    ).reset_index(drop=True)


def expected_row_count(events: pd.DataFrame, *, core_only: bool) -> int:
    start = events["core_start"] if core_only else events["expanded_start"]
    end = events["core_end"] if core_only else events["expanded_end"]
    days = (end - start).dt.days + 1
    return int((events["expected_watersheds"] * days).sum())


def validate_row_counts(wide: pd.DataFrame, events: pd.DataFrame) -> None:
    expected_expanded = expected_row_count(events, core_only=False)
    expected_core = expected_row_count(events, core_only=True)
    actual_expanded = len(wide)
    actual_core = int(wide["is_core_day"].sum())
    if actual_expanded != expected_expanded or actual_core != expected_core:
        raise ValueError(
            "Unexpected watershed-day row counts: "
            f"expanded {actual_expanded}/{expected_expanded}; "
            f"core {actual_core}/{expected_core}."
        )


def triplet_table(
    wide: pd.DataFrame, model_product: str, *, core_only: bool
) -> pd.DataFrame:
    selected = wide[wide["complete_case_all4"]].copy()
    if core_only:
        selected = selected[selected["is_core_day"]].copy()
    columns = [
        "row_id",
        "event_name",
        "storm_id",
        "storm_year",
        "date",
        "expanded_day_index",
        "is_core_day",
        "mrms_era_code",
        "watershed_id",
        "basin_id",
        "basin_name",
        "huc_code",
        model_product,
        "mrms",
        "persiann_cdr",
    ]
    result = selected[columns].rename(
        columns={
            model_product: f"{model_product}_mm",
            "mrms": "mrms_mm",
            "persiann_cdr": "persiann_cdr_mm",
        }
    )
    return result.sort_values("row_id").reset_index(drop=True)


def write_observation_netcdf(wide: pd.DataFrame, path: Path) -> None:
    table = wide.reset_index(drop=True)
    observation = np.arange(len(table), dtype=np.int32)
    data_vars = {
        product: (
            "observation",
            table[product].to_numpy(dtype=np.float32),
            {"long_name": f"Watershed-average daily precipitation: {product}", "units": "mm"},
        )
        for product in PRODUCTS
    }
    for product in PRODUCTS:
        data_vars[f"{product}_valid_area_fraction"] = (
            "observation",
            table[f"{product}_valid_area_fraction"].to_numpy(dtype=np.float32),
            {"units": "1"},
        )
    data_vars["is_core_day"] = (
        "observation",
        table["is_core_day"].astype(np.int8).to_numpy(),
        {"flag_values": np.array([0, 1], dtype=np.int8), "flag_meanings": "no yes"},
    )
    data_vars["complete_case_all4"] = (
        "observation",
        table["complete_case_all4"].astype(np.int8).to_numpy(),
        {"flag_values": np.array([0, 1], dtype=np.int8), "flag_meanings": "no yes"},
    )
    data_vars["mrms_era_code"] = (
        "observation",
        table["mrms_era_code"].to_numpy(dtype=np.int16),
        {
            "flag_values": np.array([0, 1, 2], dtype=np.int16),
            "flag_meanings": "mixed_v12_transition_day legacy_pre_v12 v12_or_later",
        },
    )
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "observation": observation,
            "row_id": ("observation", table["row_id"].astype(str).to_numpy()),
            "time": ("observation", pd.to_datetime(table["date"]).to_numpy()),
            "event_name": ("observation", table["event_name"].astype(str).to_numpy()),
            "storm_id": ("observation", table["storm_id"].astype(str).to_numpy()),
            "watershed_id": ("observation", table["watershed_id"].astype(str).to_numpy()),
            "huc_code": ("observation", table["huc_code"].astype(str).to_numpy()),
        },
        attrs={
            "title": "Matched watershed-day precipitation for paired ETC analyses",
            "spatial_aggregation": (
                "Exact fractional-cell area-weighted watershed means calculated "
                "on each product's native grid"
            ),
            "daily_interval": "[D 00:00 UTC, D+1 00:00 UTC)",
            "minimum_valid_area_fraction": MIN_VALID_AREA_FRACTION,
            "triplet_1": "CONUS404, MRMS RadarOnly QPE 24H, PERSIANN-CDR",
            "triplet_2": "CONUS404BA, MRMS RadarOnly QPE 24H, PERSIANN-CDR",
            "important_note": (
                "CONUS404BA is Daymet-adjusted; Daymet is not an independent "
                "triplet member for CONUS404BA."
            ),
        },
    )
    encoding = {
        variable: {"zlib": True, "complevel": 4}
        for variable in ds.data_vars
        if ds[variable].dtype.kind in "fiu"
    }
    ds.to_netcdf(path, engine="netcdf4", encoding=encoding)
    ds.close()


# %% Run the complete preparation workflow
def run_pipeline(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    event_names: Iterable[str] | None = None,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "weights_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    watersheds = load_watersheds()
    selected_events = EVENTS.copy()
    if event_names is not None:
        requested = {name.casefold() for name in event_names}
        selected_events = selected_events[
            selected_events["event_name"].str.casefold().isin(requested)
        ].copy()
        found = set(selected_events["event_name"].str.casefold())
        missing = requested.difference(found)
        if missing:
            raise ValueError(f"Unknown requested events: {sorted(missing)}")

    all_records: list[dict] = []
    for event in selected_events.itertuples(index=False):
        event_series = pd.Series(event._asdict())
        event_watersheds = watersheds_for_event(watersheds, event_series)
        print(
            f"\n{event.event_name} ({event.storm_id}): "
            f"{len(event_watersheds)} watersheds"
        )
        for product in PRODUCTS:
            print(f"  {product} ...", end="", flush=True)
            records = aggregate_event_product(
                product,
                event_series,
                event_watersheds,
                cache_dir,
            )
            all_records.extend(records)
            print(f" {len(records)} watershed-days")

    long_table = pd.DataFrame(all_records)
    wide_table = make_wide_table(long_table)
    validate_row_counts(wide_table, selected_events)

    long_path = output_dir / "watershed_day_precipitation_long.csv"
    wide_path = output_dir / "watershed_day_precipitation_matched_all4.csv"
    netcdf_path = output_dir / "watershed_day_precipitation_matched_all4.nc"
    long_table.to_csv(long_path, index=False, date_format="%Y-%m-%d")
    wide_table.to_csv(wide_path, index=False, date_format="%Y-%m-%d")
    write_observation_netcdf(wide_table, netcdf_path)

    triplet_paths: dict[str, Path] = {}
    for window, core_only in [("expanded", False), ("core", True)]:
        triplet_a = triplet_table(wide_table, "conus404", core_only=core_only)
        triplet_b = triplet_table(wide_table, "conus404ba", core_only=core_only)
        if not triplet_a["row_id"].equals(triplet_b["row_id"]):
            raise AssertionError("The paired ETC triplets do not use identical rows.")
        for model, table in [
            ("conus404", triplet_a),
            ("conus404ba", triplet_b),
        ]:
            path = output_dir / f"etc_triplet_{model}_{window}.csv"
            table.to_csv(path, index=False, date_format="%Y-%m-%d")
            triplet_paths[f"{model}_{window}"] = path

    summary = (
        wide_table.groupby(["event_name", "storm_id"], as_index=False)
        .agg(
            expanded_watershed_days=("row_id", "size"),
            core_watershed_days=("is_core_day", "sum"),
            complete_expanded_rows=("complete_case_all4", "sum"),
        )
    )
    core_complete = (
        wide_table[wide_table["is_core_day"]]
        .groupby(["event_name", "storm_id"])["complete_case_all4"]
        .sum()
        .rename("complete_core_rows")
        .reset_index()
    )
    summary = summary.merge(core_complete, on=["event_name", "storm_id"])
    summary_path = output_dir / "qc_summary_by_event.csv"
    summary.to_csv(summary_path, index=False)

    weight_qc_columns = [
        "product",
        "watershed_id",
        "basin_name",
        "huc_code",
        "reported_area_sqkm",
        "geometric_coverage_fraction",
        "intersecting_native_cells",
        "grid_hash",
    ]
    weight_qc = (
        long_table[weight_qc_columns]
        .drop_duplicates()
        .sort_values(["product", "watershed_id"])
    )
    weight_qc_path = output_dir / "qc_native_grid_watershed_weights.csv"
    weight_qc.to_csv(weight_qc_path, index=False)

    manifest = {
        "method": "native-grid exact fractional-cell area-weighted watershed means",
        "area_crs": AREA_CRS,
        "minimum_valid_area_fraction": MIN_VALID_AREA_FRACTION,
        "events": selected_events["event_name"].tolist(),
        "expanded_rows_expected": expected_row_count(selected_events, core_only=False),
        "expanded_rows_written": int(len(wide_table)),
        "core_rows_expected": expected_row_count(selected_events, core_only=True),
        "core_rows_written": int(wide_table["is_core_day"].sum()),
        "complete_expanded_rows": int(wide_table["complete_case_all4"].sum()),
        "complete_core_rows": int(
            wide_table.loc[wide_table["is_core_day"], "complete_case_all4"].sum()
        ),
        "inputs": {
            "watersheds": str(WATERSHED_FILE),
            **{
                product: str(config["directory"])
                for product, config in PRODUCTS.items()
            },
        },
    }
    manifest_path = output_dir / "processing_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nFinished")
    print(summary.to_string(index=False))
    print(f"\nMatched expanded rows: {len(wide_table)}")
    print(f"Matched core rows: {int(wide_table['is_core_day'].sum())}")
    print(
        "Complete four-product expanded rows: "
        f"{int(wide_table['complete_case_all4'].sum())}"
    )
    print(f"Outputs: {output_dir}")

    return {
        "long": long_path,
        "wide": wide_path,
        "netcdf": netcdf_path,
        "summary": summary_path,
        "weight_qc": weight_qc_path,
        "manifest": manifest_path,
        **triplet_paths,
    }


# %% Command-line entry point
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for tables, NetCDF, QC files, and cached weights.",
    )
    parser.add_argument(
        "--events",
        nargs="+",
        help="Optional event names. Omit to process all ten events.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Process Ian only (one event and one watershed).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    requested_events = ["Ian"] if arguments.smoke_test else arguments.events
    run_pipeline(arguments.output_dir, requested_events)
