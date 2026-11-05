#!/usr/bin/env python3
"""Export daily CONUS404-BA precipitation for the hurricane event windows.

The module is intended for import from a VS Code Jupyter notebook, although it
can also be run as a command-line script.  It writes one multi-day NetCDF file
per event, using the same event table and file organization as the companion
CONUS404 exporter.

Important dataset details
-------------------------
* HyTEST dataset: ``conus404-daily-ba-osn``
* Precipitation variable: ``RAIN`` (millimetres)
* Native grid: 1-km Lambert conformal grid, 3840 rows x 4608 columns
* A daily value labelled D is a 24-hour accumulation for
  ``[D 00:00 UTC, D+1 00:00 UTC)``.  The daily store is used directly; the
  hourly precipitation rate is not downloaded and resummed.
* CONUS404-BA precipitation was downscaled and bias adjusted using Daymet as
  the observational reference.  It therefore must not be treated as
  independent of Daymet in a triple-collocation triplet.

The current HyTEST CONUS404-BA Zarr metadata includes a ``checksum`` argument
for the Zstandard codec.  Some legacy Python environments
has numcodecs 0.12.1, whose Zstd constructor predates that metadata argument.
This script registers a narrow read-compatibility wrapper when needed; it does
not modify the source data or the active environment.
"""

from __future__ import annotations

from study_paths import PROJECT_ROOT, RESULTS_ROOT, METADATA_ROOT, REPO_ROOT, STATES_FILE, WATERSHEDS_FILE, DAILY_INPUT, EC_DIRECTORY, EC_INPUT, LOSO_INPUT, AGREEMENT_DIRECTORY, AGREEMENT_INPUT


import argparse
import csv
import hashlib
import inspect
import os
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

# Recommended by HyTEST for recent boto3 releases. These variables must be set
# before intake/s3fs/boto3 are imported.
os.environ.setdefault("AWS_REQUEST_CHECKSUM_CALCULATION", "when_required")
os.environ.setdefault("AWS_RESPONSE_CHECKSUM_VALIDATION", "when_required")

try:
    import intake
    import numpy as np
    import pandas as pd
    import xarray as xr
    from dask.diagnostics import ProgressBar
    from numcodecs.abc import Codec
    from numcodecs.registry import register_codec
    from numcodecs.zstd import Zstd as _InstalledZstd
except ImportError as exc:
    raise ImportError(
        "This script requires intake, intake-xarray, xarray, dask, s3fs, "
        "zarr<3, numcodecs, numpy, pandas, and netCDF4. Install the acquisition dependencies listed in environment.yml."
    ) from exc


CATALOG_URL = (
    "https://raw.githubusercontent.com/hytest-org/hytest/main/"
    "dataset_catalog/hytest_intake_catalog.yml"
)
DATASET_ID = "conus404-daily-ba-osn"
VARIABLE = "RAIN"
DATA_RELEASE_DOI = "https://doi.org/10.5066/P9JE61P7"
DAILY_INTERVAL = "[D 00:00 UTC, D+1 00:00 UTC)"
DEFAULT_OUTPUT_DIR = (PROJECT_ROOT / 'Hurricane_Events_P_Data/CONUS404BA')


class _ZstdChecksumMetadataCompat(Codec):
    """Allow old numcodecs to read Zarr metadata containing `checksum`.

    HyTEST currently writes ``checksum: false``.  The installed legacy codec
    can decode those frames; it only lacks the newer constructor argument.
    """

    codec_id = "zstd"

    def __init__(self, level: int = 0, checksum: bool = False):
        if checksum:
            raise RuntimeError(
                "This legacy numcodecs compatibility wrapper is safe only for "
                "Zarr arrays whose compressor metadata has checksum=false."
            )
        self.level = level
        self.checksum = checksum
        self._codec = _InstalledZstd(level=level)

    def encode(self, buffer):
        return self._codec.encode(buffer)

    def decode(self, buffer, out=None):
        return self._codec.decode(buffer, out=out)

    def get_config(self) -> dict[str, object]:
        return {
            "id": self.codec_id,
            "level": self.level,
            "checksum": self.checksum,
        }


def register_zstd_metadata_compatibility() -> bool:
    """Register the compatibility codec only when the installed one needs it."""
    try:
        parameters = inspect.signature(_InstalledZstd).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "checksum" in parameters:
        return False
    register_codec(_ZstdChecksumMetadataCompat)
    return True


ZSTD_COMPATIBILITY_ACTIVE = register_zstd_metadata_compatibility()


@dataclass(frozen=True)
class Event:
    name: str
    year: int
    storm_id: str
    watershed_count: int
    core_start: date
    core_end: date
    expected_core_days: int
    expanded_start: date
    expanded_end: date
    expected_expanded_days: int

    def bounds(self, window: str) -> tuple[date, date]:
        if window == "core":
            return self.core_start, self.core_end
        if window == "expanded":
            return self.expanded_start, self.expanded_end
        raise ValueError(f"Unknown window: {window}")


EVENTS = (
    Event(
        "Matthew", 2016, "AL142016", 4,
        date(2016, 10, 5), date(2016, 10, 12), 8,
        date(2016, 10, 1), date(2016, 10, 13), 13,
    ),
    Event(
        "Harvey", 2017, "AL092017", 8,
        date(2017, 8, 23), date(2017, 9, 3), 12,
        date(2017, 8, 19), date(2017, 9, 4), 17,
    ),
    Event(
        "Irma", 2017, "AL112017", 3,
        date(2017, 9, 8), date(2017, 9, 14), 7,
        date(2017, 9, 4), date(2017, 9, 15), 12,
    ),
    Event(
        "Florence", 2018, "AL062018", 3,
        date(2018, 9, 12), date(2018, 9, 20), 9,
        date(2018, 9, 8), date(2018, 9, 21), 14,
    ),
    Event(
        "Imelda", 2019, "AL112019", 2,
        date(2019, 9, 16), date(2019, 9, 21), 6,
        date(2019, 9, 12), date(2019, 9, 22), 11,
    ),
    Event(
        "Laura", 2020, "AL132020", 1,
        date(2020, 8, 25), date(2020, 8, 30), 6,
        date(2020, 8, 21), date(2020, 8, 31), 11,
    ),
    Event(
        "Zeta", 2020, "AL282020", 1,
        date(2020, 10, 27), date(2020, 10, 31), 5,
        date(2020, 10, 23), date(2020, 11, 1), 10,
    ),
    Event(
        "Eta", 2020, "AL292020", 1,
        date(2020, 11, 7), date(2020, 11, 15), 9,
        date(2020, 11, 3), date(2020, 11, 16), 14,
    ),
    Event(
        "Ida", 2021, "AL092021", 9,
        date(2021, 8, 27), date(2021, 9, 4), 9,
        date(2021, 8, 23), date(2021, 9, 5), 14,
    ),
    Event(
        "Ian", 2022, "AL092022", 1,
        date(2022, 9, 26), date(2022, 10, 2), 7,
        date(2022, 9, 22), date(2022, 10, 3), 12,
    ),
)


def validate_event_table() -> None:
    """Verify that the transcribed inclusive date counts are correct."""
    for event in EVENTS:
        core_days = (event.core_end - event.core_start).days + 1
        expanded_days = (event.expanded_end - event.expanded_start).days + 1
        if core_days != event.expected_core_days:
            raise RuntimeError(
                f"{event.name}: core window is {core_days} days, not "
                f"{event.expected_core_days}."
            )
        if expanded_days != event.expected_expanded_days:
            raise RuntimeError(
                f"{event.name}: expanded window is {expanded_days} days, not "
                f"{event.expected_expanded_days}."
            )


def choose_events(event_names: Sequence[str] | None) -> list[Event]:
    """Resolve event names case-insensitively while preserving requested order."""
    if not event_names:
        return list(EVENTS)

    by_name = {event.name.casefold(): event for event in EVENTS}
    selected: list[Event] = []
    unknown: list[str] = []
    seen: set[str] = set()
    for name in event_names:
        key = name.casefold()
        if key not in by_name:
            unknown.append(name)
        elif key not in seen:
            selected.append(by_name[key])
            seen.add(key)

    if unknown:
        choices = ", ".join(event.name for event in EVENTS)
        raise ValueError(f"Unknown event(s): {', '.join(unknown)}. Choices: {choices}")
    return selected


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def open_conus404ba_daily() -> xr.Dataset:
    """Open and validate the live HyTEST daily CONUS404-BA dataset lazily."""
    print("Opening the HyTEST catalog and daily CONUS404-BA OSN store ...")
    if ZSTD_COMPATIBILITY_ACTIVE:
        print(
            "Using the numcodecs Zstd metadata compatibility wrapper required "
            "by this Python environment."
        )

    hytest_catalog = intake.open_catalog(CATALOG_URL)
    conus404_catalog = hytest_catalog["conus404-catalog"]
    dataset = conus404_catalog[DATASET_ID].to_dask()

    if VARIABLE not in dataset:
        raise KeyError(f"{VARIABLE} is not present in {DATASET_ID}")
    for coordinate in ("time", "x", "y"):
        if coordinate not in dataset.coords:
            raise KeyError(f"Required coordinate {coordinate!r} is missing")
    if "crs" not in dataset:
        raise KeyError("Required native grid-mapping variable 'crs' is missing")
    if dataset[VARIABLE].dims != ("time", "y", "x"):
        raise RuntimeError(
            f"Unexpected {VARIABLE} dimensions: {dataset[VARIABLE].dims}"
        )

    integration = str(dataset[VARIABLE].attrs.get("integration_length", ""))
    units = str(dataset[VARIABLE].attrs.get("units", ""))
    if "24-hour" not in integration:
        raise RuntimeError(
            f"Unexpected {VARIABLE} integration_length: {integration!r}. "
            "Refusing to treat it as daily precipitation."
        )
    if units.lower() not in {"mm", "millimeter", "millimeters"}:
        raise RuntimeError(f"Unexpected {VARIABLE} units: {units!r}")

    times = pd.DatetimeIndex(dataset.time.values)
    if not times.is_monotonic_increasing or not times.is_unique:
        raise RuntimeError("CONUS404-BA daily time coordinates are not sorted and unique")
    if len(times) > 1 and not (
        times[1:] - times[:-1] == pd.Timedelta(days=1)
    ).all():
        raise RuntimeError("CONUS404-BA time coordinate is not exactly daily")

    x_values = np.asarray(dataset.x.values, dtype=np.float64)
    y_values = np.asarray(dataset.y.values, dtype=np.float64)
    if not np.isfinite(x_values).all() or not np.isfinite(y_values).all():
        raise RuntimeError("CONUS404-BA x/y coordinates contain non-finite values")
    if len(x_values) < 2 or len(y_values) < 2:
        raise RuntimeError("CONUS404-BA spatial coordinates are unexpectedly short")
    # The published coordinates have sub-metre float quantization at their
    # million-metre magnitudes (observed increments span 999.875--1000.125 m).
    if not np.allclose(np.diff(x_values), 1000.0, rtol=0.0, atol=0.2):
        raise RuntimeError("CONUS404-BA x spacing is not nominally 1,000 metres")
    if not np.allclose(np.diff(y_values), 1000.0, rtol=0.0, atol=0.2):
        raise RuntimeError("CONUS404-BA y spacing is not nominally 1,000 metres")

    print(
        f"Opened {DATASET_ID}: {times[0]:%Y-%m-%d} through "
        f"{times[-1]:%Y-%m-%d}; grid={dataset.sizes['y']} x "
        f"{dataset.sizes['x']} at 1-km spacing."
    )
    return dataset


def spatial_subset_suffix(spatial_subset: Mapping[str, slice] | None) -> str:
    """Keep full-domain and test/subset outputs from sharing a filename."""
    if not spatial_subset:
        return ""
    normalized = []
    for dimension in sorted(spatial_subset):
        selector = spatial_subset[dimension]
        normalized.append(
            (
                dimension,
                selector.start,
                selector.stop,
                selector.step,
            )
        )
    digest = hashlib.sha1(repr(normalized).encode("utf-8")).hexdigest()[:10]
    return f"_subset-{digest}"


def event_output_path(
    output_dir: Path,
    event: Event,
    window: str,
    spatial_subset: Mapping[str, slice] | None = None,
) -> Path:
    start, end = event.bounds(window)
    return output_dir / (
        f"CONUS404BA_{VARIABLE}_{event.name}_{event.storm_id}_{window}_"
        f"{start:%Y%m%d}_{end:%Y%m%d}_UTC00"
        f"{spatial_subset_suffix(spatial_subset)}.nc"
    )


def expected_event_times(event: Event, window: str) -> pd.DatetimeIndex:
    start, end = event.bounds(window)
    return pd.date_range(start=start, end=end, freq="D")


def validate_netcdf(path: Path, expected_times: pd.DatetimeIndex) -> None:
    """Verify the output structure, time labels, bounds, grid, and a data read."""
    with xr.open_dataset(path, engine="netcdf4") as check:
        for required in (VARIABLE, "crs", "time_bnds", "x", "y"):
            if required not in check:
                raise RuntimeError(f"{path.name}: missing {required}")

        actual = pd.DatetimeIndex(check.time.values)
        if not actual.equals(expected_times):
            raise RuntimeError(f"{path.name}: output dates do not match the event window")
        expected_bounds = np.column_stack(
            (
                expected_times.values.astype("datetime64[ns]"),
                expected_times.values.astype("datetime64[ns]") + np.timedelta64(1, "D"),
            )
        )
        actual_bounds = check["time_bnds"].values.astype("datetime64[ns]")
        if not np.array_equal(actual_bounds, expected_bounds):
            raise RuntimeError(f"{path.name}: output time bounds are incorrect")
        if check[VARIABLE].dims != ("time", "y", "x"):
            raise RuntimeError(
                f"{path.name}: unexpected {VARIABLE} dimensions "
                f"{check[VARIABLE].dims}"
            )
        if check[VARIABLE].attrs.get("units") != "mm":
            raise RuntimeError(f"{path.name}: {VARIABLE} units are not millimetres")
        if check[VARIABLE].attrs.get("daily_interval") != DAILY_INTERVAL:
            raise RuntimeError(f"{path.name}: daily interval metadata is incorrect")
        if check.sizes["x"] < 1 or check.sizes["y"] < 1:
            raise RuntimeError(f"{path.name}: output spatial grid is empty")
        if check.sizes["x"] > 1 and not np.allclose(
            np.diff(check.x.values), 1000.0, rtol=0.0, atol=0.2
        ):
            raise RuntimeError(f"{path.name}: x spacing is incorrect")
        if check.sizes["y"] > 1 and not np.allclose(
            np.diff(check.y.values), 1000.0, rtol=0.0, atol=0.2
        ):
            raise RuntimeError(f"{path.name}: y spacing is incorrect")

        # Force a small data read so a truncated/corrupt file is detected.
        check[VARIABLE].isel(time=0, y=0, x=0).load()


def prepare_event_dataset(
    source: xr.Dataset,
    event: Event,
    window: str,
    spatial_subset: Mapping[str, slice] | None = None,
) -> tuple[xr.Dataset, pd.DatetimeIndex]:
    """Create a lazy event dataset with explicit UTC time bounds."""
    start, end = event.bounds(window)
    expected = expected_event_times(event, window)
    available = pd.DatetimeIndex(source.time.values)
    missing = expected.difference(available)
    if len(missing):
        missing_text = ", ".join(item.strftime("%Y-%m-%d") for item in missing)
        raise RuntimeError(f"Source coverage is missing: {missing_text}")

    event_data = source[[VARIABLE, "crs"]].sel(
        time=slice(start.isoformat(), end.isoformat())
    )
    actual = pd.DatetimeIndex(event_data.time.values)
    if not actual.equals(expected):
        raise RuntimeError(
            f"{event.name}: selected source dates differ from the requested dates"
        )

    if spatial_subset:
        invalid_dimensions = set(spatial_subset).difference({"x", "y"})
        if invalid_dimensions:
            raise ValueError(
                "spatial_subset may contain only x/y index slices; invalid: "
                + ", ".join(sorted(invalid_dimensions))
            )
        event_data = event_data.isel(**dict(spatial_subset))
    if event_data.sizes["x"] < 1 or event_data.sizes["y"] < 1:
        raise ValueError("spatial_subset produced an empty grid")

    time_values = event_data.time.values.astype("datetime64[ns]")
    time_bounds = np.column_stack(
        (time_values, time_values + np.timedelta64(1, "D"))
    )
    event_data = event_data.assign_coords(bounds=("bounds", np.array([0, 1])))
    event_data["time_bnds"] = xr.DataArray(
        time_bounds,
        dims=("time", "bounds"),
        coords={"time": event_data.time, "bounds": event_data.bounds},
    )

    # Remove Zarr-specific encoding before defining NetCDF encoding below.
    for variable_name in event_data.variables:
        event_data[variable_name].encoding = {}

    event_data.time.attrs["bounds"] = "time_bnds"
    event_data.time.attrs["long_name"] = "UTC calendar-day label"
    event_data["time_bnds"].attrs.update(
        {
            "long_name": "UTC precipitation accumulation interval bounds",
            "description": "Each interval is [start, end), with end exclusive.",
        }
    )

    source_cell_methods = str(event_data[VARIABLE].attrs.get("cell_methods", ""))
    event_data[VARIABLE].attrs.update(
        {
            "cell_methods": "time: sum (interval: 1 day)",
            "source_zarr_cell_methods": source_cell_methods,
            "daily_interval": DAILY_INTERVAL,
            "bias_adjustment_reference": "Daymet",
        }
    )

    event_data.attrs = {
        "title": "Daily CONUS404-BA precipitation for a hurricane event",
        "Conventions": "CF-1.8",
        "event_name": event.name,
        "noaa_storm_id": event.storm_id,
        "storm_year": event.year,
        "watershed_count_from_event_table": event.watershed_count,
        "event_window_type": window,
        "event_window_start_utc": f"{start.isoformat()}T00:00:00Z",
        "event_window_end_utc_exclusive": (
            f"{(pd.Timestamp(end) + pd.Timedelta(days=1)):%Y-%m-%dT00:00:00Z}"
        ),
        "source_catalog": CATALOG_URL,
        "source_dataset_id": DATASET_ID,
        "source_variable": VARIABLE,
        "source_variable_units": "mm",
        "source_data_release": DATA_RELEASE_DOI,
        "source_model_configuration": str(
            source.attrs.get("model_configuration", "CONUS404_tpBiasCorr")
        ),
        "temporal_aggregation": (
            "HyTEST 24-hour accumulated precipitation; date D represents "
            f"{DAILY_INTERVAL}"
        ),
        "spatial_grid": "native CONUS404-BA 1-km Lambert conformal grid",
        "spatial_resampling_during_export": "none",
        "source_remap_attribute": str(
            source[VARIABLE].attrs.get("remap", "")
        ),
        "spatial_subset": "full native domain" if not spatial_subset else str(spatial_subset),
        "bias_adjustment_reference": "Daymet",
        "triple_collocation_note": (
            "CONUS404-BA precipitation is Daymet-bias-adjusted and is not "
            "independent of Daymet."
        ),
        "numcodecs_zstd_metadata_compatibility_applied": str(
            ZSTD_COMPATIBILITY_ACTIVE
        ).lower(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    return event_data, expected


def netcdf_encoding(dataset: xr.Dataset) -> dict[str, dict[str, object]]:
    """Create compressed, one-day NetCDF chunks for efficient later reading."""
    y_chunk = min(350, dataset.sizes["y"])
    x_chunk = min(350, dataset.sizes["x"])
    time_encoding: dict[str, object] = {
        "units": "days since 1970-01-01 00:00:00",
        "calendar": "proleptic_gregorian",
        "dtype": "int64",
        "_FillValue": None,
    }
    return {
        VARIABLE: {
            "dtype": "float32",
            "zlib": True,
            "complevel": 4,
            "shuffle": True,
            "chunksizes": (1, y_chunk, x_chunk),
            "_FillValue": np.float32(-9999.0),
        },
        "time": dict(time_encoding),
        "time_bnds": dict(time_encoding),
        "x": {"dtype": "float64", "_FillValue": None},
        "y": {"dtype": "float64", "_FillValue": None},
        "bounds": {"dtype": "int8", "_FillValue": None},
        "crs": {"dtype": "int64", "_FillValue": None},
    }


def write_event_file(
    event_data: xr.Dataset,
    expected_times: pd.DatetimeIndex,
    output_path: Path,
    overwrite: bool,
    retries: int,
) -> tuple[str, int, str]:
    """Write, validate, hash, and atomically promote one event NetCDF."""
    if output_path.exists() and not overwrite:
        validate_netcdf(output_path, expected_times)
        return (
            "skipped_valid_existing",
            output_path.stat().st_size,
            sha256_file(output_path),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(output_path.stem + ".part.nc")
    last_error: Exception | None = None

    for attempt in range(retries + 1):
        try:
            if partial.exists():
                partial.unlink()
            print(f"  Writing {output_path.name} ...")
            with ProgressBar():
                event_data.to_netcdf(
                    partial,
                    engine="netcdf4",
                    mode="w",
                    encoding=netcdf_encoding(event_data),
                )
            validate_netcdf(partial, expected_times)
            os.replace(partial, output_path)
            return "written", output_path.stat().st_size, sha256_file(output_path)
        except Exception as exc:
            last_error = exc
            if partial.exists():
                try:
                    partial.unlink()
                except OSError:
                    pass
            if attempt < retries:
                wait_seconds = min(2 ** attempt, 30)
                print(
                    f"  Attempt {attempt + 1} failed: {exc}. "
                    f"Retrying in {wait_seconds} second(s) ..."
                )
                time.sleep(wait_seconds)

    raise RuntimeError(f"Unable to write {output_path.name}: {last_error}")


def write_manifest(output_dir: Path, rows: list[dict[str, object]], window: str) -> Path:
    manifest_dir = output_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = manifest_dir / f"conus404ba_{window}_event_files.csv"
    temporary = manifest.with_name(manifest.name + ".tmp")
    fieldnames = [
        "event_name",
        "storm_year",
        "noaa_storm_id",
        "watershed_count",
        "window_type",
        "start_date_utc",
        "end_date_utc_inclusive",
        "day_count",
        "source_dataset",
        "source_variable",
        "source_data_release",
        "daily_interval",
        "native_grid",
        "bias_adjustment_reference",
        "output_path",
        "status",
        "size_bytes",
        "sha256",
        "completed_utc",
        "error",
    ]
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, manifest)
    return manifest


def download_conus404ba_events(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    event_names: Sequence[str] | None = None,
    window: str = "expanded",
    overwrite: bool = False,
    retries: int = 2,
    spatial_subset: Mapping[str, slice] | None = None,
) -> pd.DataFrame:
    """Export selected CONUS404-BA events and return the file manifest.

    Parameters
    ----------
    output_dir
        Directory where event NetCDF files and the CSV manifest are written.
    event_names
        Event names such as ``["Ian"]``.  ``None`` exports all ten events.
    window
        ``"expanded"`` (default) or ``"core"``.
    overwrite
        Replace a valid existing event file when True.
    retries
        Number of retries after the first write attempt.
    spatial_subset
        Optional xarray ``isel`` mapping for a rectangular native-grid subset,
        e.g. ``{"y": slice(1000, 1100), "x": slice(3000, 3100)}``.
        Leave as None for the full native 1-km domain.
    """
    if window not in {"expanded", "core"}:
        raise ValueError("window must be 'expanded' or 'core'")
    if retries < 0:
        raise ValueError("retries must be zero or greater")

    validate_event_table()
    selected_events = choose_events(event_names)
    destination = Path(output_dir)
    source = open_conus404ba_daily()
    rows: list[dict[str, object]] = []

    print(f"Output directory: {destination}")
    print(f"Window: {window}")
    print(f"Events: {', '.join(event.name for event in selected_events)}")
    print(
        "Spatial area: "
        + ("full native 1-km domain" if not spatial_subset else str(spatial_subset))
    )
    if not spatial_subset:
        print(
            "NOTE: Full-domain 1-km event files are large. Test one event first "
            "and confirm available disk space."
        )

    for event in selected_events:
        start, end = event.bounds(window)
        output_path = event_output_path(
            destination,
            event,
            window,
            spatial_subset=spatial_subset,
        )
        row: dict[str, object] = {
            "event_name": event.name,
            "storm_year": event.year,
            "noaa_storm_id": event.storm_id,
            "watershed_count": event.watershed_count,
            "window_type": window,
            "start_date_utc": start.isoformat(),
            "end_date_utc_inclusive": end.isoformat(),
            "day_count": (end - start).days + 1,
            "source_dataset": DATASET_ID,
            "source_variable": VARIABLE,
            "source_data_release": DATA_RELEASE_DOI,
            "daily_interval": DAILY_INTERVAL,
            "native_grid": "1-km Lambert conformal, 3840 x 4608",
            "bias_adjustment_reference": "Daymet",
            "output_path": str(output_path),
            "status": "",
            "size_bytes": "",
            "sha256": "",
            "completed_utc": "",
            "error": "",
        }
        print(f"{event.name}: {start} through {end}")
        try:
            event_data, expected_times = prepare_event_dataset(
                source,
                event,
                window,
                spatial_subset=spatial_subset,
            )
            status, size_bytes, checksum = write_event_file(
                event_data,
                expected_times,
                output_path,
                overwrite=overwrite,
                retries=retries,
            )
            row.update(
                {
                    "status": status,
                    "size_bytes": size_bytes,
                    "sha256": checksum,
                    "completed_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            print(f"  {status}: {size_bytes:,} bytes")
        except Exception as exc:
            row.update(
                {
                    "status": "failed",
                    "completed_utc": datetime.now(timezone.utc).isoformat(),
                    "error": str(exc),
                }
            )
            print(f"  FAILED: {exc}", file=sys.stderr)
        rows.append(row)
        # Update after every event so progress survives an interrupted run.
        write_manifest(destination, rows, window)

    source.close()
    manifest_path = write_manifest(destination, rows, window)
    manifest = pd.DataFrame(rows)
    failures = manifest.loc[manifest["status"] == "failed"]
    print(f"Manifest: {manifest_path}")
    print(
        f"Completed {len(manifest) - len(failures)} event file(s); "
        f"{len(failures)} failed."
    )
    if len(failures):
        failed_names = ", ".join(failures["event_name"].astype(str))
        raise RuntimeError(
            f"Some events failed ({failed_names}). Review the manifest and rerun; "
            "valid existing files will be skipped."
        )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export daily CONUS404-BA precipitation NetCDF files by event."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--event",
        action="append",
        metavar="NAME",
        help="Export one event; repeat for multiple events. Default: all.",
    )
    parser.add_argument("--window", choices=("expanded", "core"), default="expanded")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--retries", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        download_conus404ba_events(
            output_dir=args.output_dir,
            event_names=args.event,
            window=args.window,
            overwrite=args.overwrite,
            retries=args.retries,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
