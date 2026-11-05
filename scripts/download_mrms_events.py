#!/usr/bin/env python3
"""Download daily MRMS RadarOnly QPE for the study hurricane windows.

This file is designed to be imported from a VS Code Jupyter notebook, but it
can also be run from a terminal.  It writes one multi-day NetCDF file per
event, matching the event-file organization used for the other precipitation
datasets in this study.

Scientific choices made here
----------------------------
* Product: ``RadarOnly_QPE_24H`` rather than ``MultiSensor_QPE_24H``.  The
  radar-only product avoids intentionally bringing gauges into the MRMS leg of
  a triple-collocation triplet.
* Day definition: the value for observation date D is the MRMS 24-hour file
  valid at (D+1) 00:00 UTC and represents ``[D 00:00, D+1 00:00)``.
* Missing-data flags: all non-finite and negative MRMS values (including -1
  and -3) are stored as missing, never as zero precipitation.
* Archive: Iowa State University's MRMS MTArchive is used because it includes
  the pre-October-2020 record needed for the 2016--2019 storms.
* Version awareness: every day is tagged as legacy/pre-v12, v12-or-later, or
  the mixed transition day.  MRMS v12 became operational at 13:30 UTC on
  2020-10-14, so this is a relevant sensitivity check when storms are pooled.

The native MRMS grid is 0.01 degree and is large (3500 x 7000).  By default
the complete native domain is retained.  An optional longitude/latitude
bounding box can be supplied to reduce file sizes; it clips to whole native
grid cells and does not reproject or resample the data.

Minimal notebook use
--------------------
>>> from download_mrms_events import download_mrms_events
>>> manifest = download_mrms_events()  # all 10 expanded event windows

Test one event first
--------------------
>>> manifest = download_mrms_events(event_names=["Ian"])

Optional rectangular subset (west, south, east, north)
------------------------------------------------------
>>> manifest = download_mrms_events(
...     event_names=["Ian"],
...     bbox=(-85.0, 24.0, -75.0, 36.0),
... )
"""

from __future__ import annotations

from study_paths import PROJECT_ROOT, RESULTS_ROOT, METADATA_ROOT, REPO_ROOT, STATES_FILE, WATERSHEDS_FILE, DAILY_INPUT, EC_DIRECTORY, EC_INPUT, LOSO_INPUT, AGREEMENT_DIRECTORY, AGREEMENT_INPUT


import argparse
import csv
import gzip
import hashlib
import math
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, time as datetime_time, timedelta, timezone
from pathlib import Path
from typing import Iterator, Sequence

try:
    import netCDF4
    import numpy as np
    import pandas as pd
    import rasterio
    from rasterio.windows import Window, from_bounds
except ImportError as exc:
    raise ImportError(
        "This script requires numpy, pandas, rasterio, and netCDF4. "
        "In VS Code, select the same project Python environment used for the "
        "HyTEST workflow, or install those packages in the active kernel."
    ) from exc


PRODUCT = "RadarOnly_QPE_24H"
PRODUCT_VERSION = "00.00"
BASE_URL = "https://mtarchive.geol.iastate.edu"
DAILY_INTERVAL = "[D 00:00 UTC, D+1 00:00 UTC)"
FILL_VALUE = np.float32(-9999.0)
EPOCH_DATE = date(1970, 1, 1)
V12_OPERATIONAL_UTC = datetime(2020, 10, 14, 13, 30, tzinfo=timezone.utc)
DEFAULT_OUTPUT_DIR = (PROJECT_ROOT / 'Hurricane_Events_P_Data/MRMS')


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


@dataclass(frozen=True)
class DailySource:
    observation_date: date
    valid_end: datetime
    url: str
    gzip_path: Path
    era_code: int
    era_label: str


@dataclass(frozen=True)
class GridDefinition:
    width: int
    height: int
    transform: tuple[float, ...]
    crs_wkt: str


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
    """Verify the inclusive date counts transcribed from the event table."""
    for event in EVENTS:
        core_days = (event.core_end - event.core_start).days + 1
        expanded_days = (event.expanded_end - event.expanded_start).days + 1
        if core_days != event.expected_core_days:
            raise RuntimeError(
                f"{event.name}: core window has {core_days} days, not "
                f"{event.expected_core_days}."
            )
        if expanded_days != event.expected_expanded_days:
            raise RuntimeError(
                f"{event.name}: expanded window has {expanded_days} days, not "
                f"{event.expected_expanded_days}."
            )


def choose_events(event_names: Sequence[str] | None) -> list[Event]:
    """Resolve event names case-insensitively and reject misspellings."""
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


def inclusive_dates(start: date, end: date) -> list[date]:
    return [start + timedelta(days=offset) for offset in range((end - start).days + 1)]


def mrms_era(observation_date: date) -> tuple[int, str]:
    """Classify the daily interval relative to the MRMS v12 transition."""
    interval_start = datetime.combine(
        observation_date, datetime_time.min, tzinfo=timezone.utc
    )
    interval_end = interval_start + timedelta(days=1)
    if interval_end <= V12_OPERATIONAL_UTC:
        return 1, "legacy_pre_v12"
    if interval_start >= V12_OPERATIONAL_UTC:
        return 2, "v12_or_later"
    return 0, "mixed_v12_transition_day"


def daily_source(observation_date: date, output_dir: Path) -> DailySource:
    """Build the source URL for calendar day D using valid time D+1 00Z."""
    valid_end = datetime.combine(
        observation_date + timedelta(days=1),
        datetime_time.min,
        tzinfo=timezone.utc,
    )
    filename = (
        f"{PRODUCT}_{PRODUCT_VERSION}_{valid_end:%Y%m%d}-000000.grib2.gz"
    )
    relative = (
        f"{valid_end:%Y/%m/%d}/mrms/ncep/{PRODUCT}/{filename}"
    )
    era_code, era_label = mrms_era(observation_date)
    return DailySource(
        observation_date=observation_date,
        valid_end=valid_end,
        url=f"{BASE_URL}/{relative}",
        gzip_path=(
            output_dir
            / "raw"
            / PRODUCT
            / f"{valid_end:%Y}"
            / f"{valid_end:%m}"
            / f"{valid_end:%d}"
            / filename
        ),
        era_code=era_code,
        era_label=era_label,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_gzip_grib(path: Path) -> None:
    """Reject empty, truncated, non-gzip, or non-GRIB cached files."""
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Missing or empty source file: {path}")
    try:
        with gzip.open(path, "rb") as stream:
            first_chunk = stream.read(1024 * 1024)
            signature = first_chunk[:4]
            tail = first_chunk[-4:]
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                tail = (tail + chunk)[-4:]
    except (OSError, EOFError) as exc:
        raise RuntimeError(f"Invalid gzip stream: {path}") from exc
    if signature != b"GRIB":
        raise RuntimeError(f"Decompressed file does not start with GRIB: {path}")
    if tail != b"7777":
        raise RuntimeError(f"Decompressed file lacks the GRIB end marker: {path}")


def source_manifest_record(source: DailySource) -> dict[str, object]:
    start = datetime.combine(
        source.observation_date, datetime_time.min, tzinfo=timezone.utc
    )
    return {
        "observation_date_utc": source.observation_date.isoformat(),
        "interval_start_utc": start.isoformat().replace("+00:00", "Z"),
        "interval_end_utc_exclusive": source.valid_end.isoformat().replace(
            "+00:00", "Z"
        ),
        "mrms_valid_time_utc": source.valid_end.isoformat().replace("+00:00", "Z"),
        "mrms_era_code": source.era_code,
        "mrms_era_label": source.era_label,
        "source_url": source.url,
        "local_gzip_path": str(source.gzip_path),
        "download_status": "",
        "size_bytes": "",
        "sha256": "",
        "grib_element": "",
        "grib_unit": "",
        "valid_cell_count_stored_grid": "",
        "missing_cell_count_stored_grid": "",
        "valid_fraction_stored_grid": "",
        "minimum_mm_stored_grid": "",
        "maximum_mm_stored_grid": "",
        "completed_utc": "",
        "error": "",
    }


def download_daily_source(
    source: DailySource,
    overwrite_raw: bool,
    retries: int,
    timeout_seconds: int,
) -> dict[str, object]:
    """Download one compressed GRIB file with validation and atomic promotion."""
    record = source_manifest_record(source)
    target = source.gzip_path
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists() and not overwrite_raw:
        try:
            validate_gzip_grib(target)
            record.update(
                {
                    "download_status": "cached_valid",
                    "size_bytes": target.stat().st_size,
                    "sha256": sha256_file(target),
                    "completed_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            return record
        except Exception as exc:
            print(f"    Cached file is invalid and will be replaced: {exc}")

    temporary = target.with_name(target.name + ".part")
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            if temporary.exists():
                temporary.unlink()
            request = urllib.request.Request(
                source.url,
                headers={
                    "User-Agent": (
                        "USGS-CONUS404-extreme-precipitation-study/1.0 "
                        "(MRMS research download)"
                    )
                },
            )
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                expected_length_text = response.headers.get("Content-Length")
                expected_length = (
                    int(expected_length_text) if expected_length_text else None
                )
                with temporary.open("wb") as output:
                    shutil.copyfileobj(response, output, length=1024 * 1024)

            actual_length = temporary.stat().st_size
            if expected_length is not None and actual_length != expected_length:
                raise RuntimeError(
                    f"Incomplete response: received {actual_length:,} of "
                    f"{expected_length:,} bytes"
                )
            validate_gzip_grib(temporary)
            os.replace(temporary, target)
            record.update(
                {
                    "download_status": "downloaded",
                    "size_bytes": target.stat().st_size,
                    "sha256": sha256_file(target),
                    "completed_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            return record
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                last_error = RuntimeError(
                    f"MRMS source is absent (HTTP 404): {source.url}"
                )
                break
            last_error = exc
        except Exception as exc:
            last_error = exc

        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass
        if attempt < retries:
            delay = min(2 ** attempt, 30)
            print(
                f"    Download attempt {attempt + 1} failed: {last_error}. "
                f"Retrying in {delay} second(s) ..."
            )
            time.sleep(delay)

    if temporary.exists():
        try:
            temporary.unlink()
        except OSError:
            pass
    record.update(
        {
            "download_status": "failed",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "error": str(last_error),
        }
    )
    raise RuntimeError(str(last_error))


@contextmanager
def decompressed_grib(gzip_path: Path, scratch_dir: Path) -> Iterator[Path]:
    """Temporarily decompress one source file for GDAL/rasterio."""
    scratch_dir.mkdir(parents=True, exist_ok=True)
    temporary = scratch_dir / f"{gzip_path.stem}.{uuid.uuid4().hex}.grib2"
    try:
        with gzip.open(gzip_path, "rb") as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
        with temporary.open("rb") as stream:
            if stream.read(4) != b"GRIB":
                raise RuntimeError(f"Invalid decompressed GRIB: {gzip_path}")
        yield temporary
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass


def parse_valid_time(tags: dict[str, str], source_path: Path) -> datetime:
    raw = tags.get("GRIB_VALID_TIME")
    if raw is None:
        raise RuntimeError(f"{source_path.name}: GRIB_VALID_TIME is missing")
    try:
        return datetime.fromtimestamp(int(raw), tz=timezone.utc)
    except (TypeError, ValueError, OSError) as exc:
        raise RuntimeError(
            f"{source_path.name}: invalid GRIB_VALID_TIME {raw!r}"
        ) from exc


def inspect_grib(
    dataset: rasterio.io.DatasetReader,
    source: DailySource,
) -> tuple[GridDefinition, dict[str, str]]:
    """Validate product identity, units, valid time, and native grid."""
    if dataset.count != 1:
        raise RuntimeError(
            f"{source.gzip_path.name}: expected one GRIB band, found {dataset.count}"
        )
    tags = dataset.tags(1)
    element = tags.get("GRIB_ELEMENT", "")
    unit = tags.get("GRIB_UNIT", "")
    if element not in {PRODUCT, "RadarOnlyQPE24H"}:
        raise RuntimeError(
            f"{source.gzip_path.name}: expected {PRODUCT}, found {element!r}"
        )
    if "mm" not in unit.lower():
        raise RuntimeError(
            f"{source.gzip_path.name}: expected millimetres, found {unit!r}"
        )
    actual_valid = parse_valid_time(tags, source.gzip_path)
    if actual_valid != source.valid_end:
        raise RuntimeError(
            f"{source.gzip_path.name}: valid time is {actual_valid.isoformat()}, "
            f"expected {source.valid_end.isoformat()}"
        )
    if dataset.crs is None:
        raise RuntimeError(f"{source.gzip_path.name}: source CRS is missing")
    transform = dataset.transform
    if not math.isclose(transform.b, 0.0, abs_tol=1e-12) or not math.isclose(
        transform.d, 0.0, abs_tol=1e-12
    ):
        raise RuntimeError("Rotated MRMS grids are not supported by this exporter")
    grid = GridDefinition(
        width=dataset.width,
        height=dataset.height,
        transform=tuple(transform),
        crs_wkt=dataset.crs.to_wkt(),
    )
    return grid, tags


def assert_same_grid(actual: GridDefinition, expected: GridDefinition, name: str) -> None:
    if actual.width != expected.width or actual.height != expected.height:
        raise RuntimeError(f"{name}: source grid dimensions changed")
    if not np.allclose(actual.transform, expected.transform, rtol=0.0, atol=1e-10):
        raise RuntimeError(f"{name}: source affine transform changed")
    if actual.crs_wkt != expected.crs_wkt:
        raise RuntimeError(f"{name}: source CRS changed")


def stored_window(
    dataset: rasterio.io.DatasetReader,
    bbox: tuple[float, float, float, float] | None,
) -> Window:
    """Return the full grid or the whole-cell window intersecting a lon/lat bbox."""
    if bbox is None:
        return Window(0, 0, dataset.width, dataset.height)

    west, south, east, north = bbox
    if not all(np.isfinite([west, south, east, north])):
        raise ValueError("bbox values must all be finite")
    if west >= east or south >= north:
        raise ValueError("bbox must satisfy west < east and south < north")

    fractional = from_bounds(west, south, east, north, transform=dataset.transform)
    col_start = max(0, math.floor(fractional.col_off))
    row_start = max(0, math.floor(fractional.row_off))
    col_stop = min(dataset.width, math.ceil(fractional.col_off + fractional.width))
    row_stop = min(dataset.height, math.ceil(fractional.row_off + fractional.height))
    if col_start >= col_stop or row_start >= row_stop:
        raise ValueError(f"bbox {bbox} does not intersect the native MRMS grid")
    return Window(
        col_off=col_start,
        row_off=row_start,
        width=col_stop - col_start,
        height=row_stop - row_start,
    )


def bbox_suffix(bbox: tuple[float, float, float, float] | None) -> str:
    if bbox is None:
        return ""
    digest = hashlib.sha1(repr(tuple(float(value) for value in bbox)).encode()).hexdigest()
    return f"_bbox-{digest[:10]}"


def event_output_path(
    output_dir: Path,
    event: Event,
    window: str,
    bbox: tuple[float, float, float, float] | None,
) -> Path:
    start, end = event.bounds(window)
    return output_dir / (
        f"MRMS_{PRODUCT}_{event.name}_{event.storm_id}_{window}_"
        f"{start:%Y%m%d}_{end:%Y%m%d}_UTC00{bbox_suffix(bbox)}.nc"
    )


def create_netcdf_structure(
    dataset: netCDF4.Dataset,
    event: Event,
    window_type: str,
    dates: list[date],
    grid: GridDefinition,
    grid_window: Window,
    bbox: tuple[float, float, float, float] | None,
    block_rows: int,
) -> netCDF4.Variable:
    """Create CF-style coordinates, interval bounds, CRS, and precipitation."""
    row_start = int(grid_window.row_off)
    col_start = int(grid_window.col_off)
    height = int(grid_window.height)
    width = int(grid_window.width)
    transform = rasterio.Affine(*grid.transform[:6])

    longitude = transform.c + transform.a * (
        col_start + np.arange(width, dtype=np.float64) + 0.5
    )
    latitude = transform.f + transform.e * (
        row_start + np.arange(height, dtype=np.float64) + 0.5
    )
    time_values = np.asarray([(day - EPOCH_DATE).days for day in dates], dtype=np.int64)

    dataset.createDimension("time", len(dates))
    dataset.createDimension("bounds", 2)
    dataset.createDimension("latitude", height)
    dataset.createDimension("longitude", width)

    time_var = dataset.createVariable("time", "i8", ("time",))
    time_var.standard_name = "time"
    time_var.long_name = "UTC calendar-day label"
    time_var.units = "days since 1970-01-01 00:00:00"
    time_var.calendar = "proleptic_gregorian"
    time_var.bounds = "time_bnds"
    time_var.axis = "T"
    time_var[:] = time_values

    bounds_var = dataset.createVariable("time_bnds", "i8", ("time", "bounds"))
    bounds_var.units = time_var.units
    bounds_var.calendar = time_var.calendar
    bounds_var.long_name = "UTC precipitation accumulation interval bounds"
    bounds_var.comment = "Each interval is [start, end), with end exclusive."
    bounds_var[:, 0] = time_values
    bounds_var[:, 1] = time_values + 1

    lat_var = dataset.createVariable("latitude", "f4", ("latitude",))
    lat_var.standard_name = "latitude"
    lat_var.long_name = "latitude at grid-cell center"
    lat_var.units = "degrees_north"
    lat_var.axis = "Y"
    lat_var[:] = latitude.astype(np.float32)

    lon_var = dataset.createVariable("longitude", "f4", ("longitude",))
    lon_var.standard_name = "longitude"
    lon_var.long_name = "longitude at grid-cell center"
    lon_var.units = "degrees_east"
    lon_var.axis = "X"
    lon_var[:] = longitude.astype(np.float32)

    crs_var = dataset.createVariable("crs", "i1")
    crs_var.grid_mapping_name = "latitude_longitude"
    crs_var.semi_major_axis = 6378160.0
    crs_var.inverse_flattening = 298.253916296469
    crs_var.longitude_of_prime_meridian = 0.0
    crs_var.spatial_ref = grid.crs_wkt
    crs_var.crs_wkt = grid.crs_wkt
    crs_var.GeoTransform = " ".join(
        str(value)
        for value in (
            transform.c,
            transform.a,
            transform.b,
            transform.f,
            transform.d,
            transform.e,
        )
    )

    era_var = dataset.createVariable("mrms_era_code", "i2", ("time",))
    era_var.long_name = "MRMS operational-build era classification"
    era_var.flag_values = np.asarray([0, 1, 2], dtype=np.int16)
    era_var.flag_meanings = (
        "mixed_v12_transition_day legacy_pre_v12 v12_or_later"
    )
    era_var.comment = (
        "MRMS v12 became operational at 2020-10-14 13:30 UTC. "
        "This code marks the major archive-era break; it is not a complete "
        "minor-version history."
    )
    era_var[:] = np.asarray([mrms_era(day)[0] for day in dates], dtype=np.int16)

    chunks = (1, min(block_rows, height), min(700, width))
    precipitation = dataset.createVariable(
        PRODUCT,
        "f4",
        ("time", "latitude", "longitude"),
        zlib=True,
        complevel=4,
        shuffle=True,
        chunksizes=chunks,
        fill_value=FILL_VALUE,
    )
    precipitation.long_name = "MRMS radar-only 24-hour precipitation accumulation"
    precipitation.units = "mm"
    precipitation.coordinates = "latitude longitude"
    precipitation.grid_mapping = "crs"
    precipitation.cell_methods = "time: sum (interval: 1 day)"
    precipitation.daily_interval = DAILY_INTERVAL
    precipitation.source_product = PRODUCT
    precipitation.missing_value = FILL_VALUE
    precipitation.valid_min = np.float32(0.0)
    precipitation.comment = (
        "Original negative MRMS flags, including -1 (missing) and -3 "
        "(no coverage), are encoded as the NetCDF fill value."
    )

    start, end = event.bounds(window_type)
    realized_west = float(transform.c + transform.a * col_start)
    realized_east = float(transform.c + transform.a * (col_start + width))
    realized_north = float(transform.f + transform.e * row_start)
    realized_south = float(transform.f + transform.e * (row_start + height))
    eras = sorted({mrms_era(day)[1] for day in dates})
    dataset.setncatts(
        {
            "title": "Daily MRMS radar-only precipitation for a hurricane event",
            "Conventions": "CF-1.8",
            "event_name": event.name,
            "noaa_storm_id": event.storm_id,
            "storm_year": event.year,
            "watershed_count_from_event_table": event.watershed_count,
            "event_window_type": window_type,
            "event_window_start_utc": f"{start.isoformat()}T00:00:00Z",
            "event_window_end_utc_exclusive": (
                f"{(end + timedelta(days=1)).isoformat()}T00:00:00Z"
            ),
            "source_archive": BASE_URL,
            "source_product": PRODUCT,
            "source_product_version_token": PRODUCT_VERSION,
            "source_units": "mm",
            "temporal_aggregation": (
                "MRMS 24-hour accumulation valid at D+1 00:00 UTC; "
                f"date D represents {DAILY_INTERVAL}"
            ),
            "mrms_operational_eras_present": ", ".join(eras),
            "mrms_v12_transition_utc": "2020-10-14T13:30:00Z",
            "spatial_grid": "native MRMS 0.01-degree geographic grid",
            "spatial_resampling": "none",
            "requested_bbox_wsen": "full native domain" if bbox is None else str(bbox),
            "stored_grid_bounds_wsen": str(
                (realized_west, realized_south, realized_east, realized_north)
            ),
            "source_grid_shape_rows_columns": f"{grid.height}, {grid.width}",
            "stored_grid_shape_rows_columns": f"{height}, {width}",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "history": "Downloaded from IEM MTArchive and converted without resampling.",
        }
    )
    return precipitation


def write_one_day(
    precipitation: netCDF4.Variable,
    time_index: int,
    source: DailySource,
    grib_path: Path,
    expected_grid: GridDefinition,
    grid_window: Window,
    block_rows: int,
) -> dict[str, object]:
    """Stream one GRIB field into one NetCDF time slice and return QC values."""
    valid_count = 0
    missing_count = 0
    valid_sum = 0.0
    minimum = math.inf
    maximum = -math.inf

    with rasterio.open(grib_path) as grib:
        actual_grid, tags = inspect_grib(grib, source)
        assert_same_grid(actual_grid, expected_grid, source.gzip_path.name)

        source_col = int(grid_window.col_off)
        source_row = int(grid_window.row_off)
        width = int(grid_window.width)
        height = int(grid_window.height)
        for local_row in range(0, height, block_rows):
            row_count = min(block_rows, height - local_row)
            read_window = Window(
                col_off=source_col,
                row_off=source_row + local_row,
                width=width,
                height=row_count,
            )
            values = grib.read(1, window=read_window, out_dtype="float32")
            valid = np.isfinite(values) & (values >= 0.0)
            count = int(np.count_nonzero(valid))
            valid_count += count
            missing_count += int(values.size - count)
            if count:
                valid_values = values[valid]
                valid_sum += float(valid_values.sum(dtype=np.float64))
                minimum = min(minimum, float(valid_values.min()))
                maximum = max(maximum, float(valid_values.max()))
            values[~valid] = FILL_VALUE
            precipitation[
                time_index,
                local_row : local_row + row_count,
                :,
            ] = values

    total = valid_count + missing_count
    return {
        "grib_element": tags.get("GRIB_ELEMENT", ""),
        "grib_unit": tags.get("GRIB_UNIT", ""),
        "valid_cell_count_stored_grid": valid_count,
        "missing_cell_count_stored_grid": missing_count,
        "valid_fraction_stored_grid": (valid_count / total) if total else math.nan,
        "minimum_mm_stored_grid": minimum if valid_count else math.nan,
        "maximum_mm_stored_grid": maximum if valid_count else math.nan,
        "mean_mm_stored_grid": (valid_sum / valid_count) if valid_count else math.nan,
    }


def validate_event_netcdf(path: Path, event: Event, window: str) -> None:
    """Verify exact time labels/bounds, variable dimensions, and readable data."""
    expected_dates = inclusive_dates(*event.bounds(window))
    expected_time = np.asarray(
        [(day - EPOCH_DATE).days for day in expected_dates], dtype=np.int64
    )
    with netCDF4.Dataset(path, "r") as dataset:
        if PRODUCT not in dataset.variables:
            raise RuntimeError(f"{path.name}: missing {PRODUCT}")
        if "time_bnds" not in dataset.variables:
            raise RuntimeError(f"{path.name}: missing time_bnds")
        actual_time = np.asarray(dataset.variables["time"][:], dtype=np.int64)
        if not np.array_equal(actual_time, expected_time):
            raise RuntimeError(f"{path.name}: daily dates do not match the event window")
        actual_bounds = np.asarray(dataset.variables["time_bnds"][:], dtype=np.int64)
        expected_bounds = np.column_stack((expected_time, expected_time + 1))
        if not np.array_equal(actual_bounds, expected_bounds):
            raise RuntimeError(f"{path.name}: time bounds are incorrect")
        variable = dataset.variables[PRODUCT]
        if variable.dimensions != ("time", "latitude", "longitude"):
            raise RuntimeError(
                f"{path.name}: unexpected {PRODUCT} dimensions {variable.dimensions}"
            )
        if variable.shape[0] != len(expected_dates):
            raise RuntimeError(f"{path.name}: incorrect number of daily fields")
        if variable.shape[1] < 1 or variable.shape[2] < 1:
            raise RuntimeError(f"{path.name}: empty spatial grid")
        # Force a small read to detect a corrupt or truncated NetCDF file.
        variable[0, 0:1, 0:1]


def write_event_file(
    event: Event,
    window_type: str,
    output_path: Path,
    sources: list[DailySource],
    source_records: dict[date, dict[str, object]],
    scratch_dir: Path,
    bbox: tuple[float, float, float, float] | None,
    block_rows: int,
    overwrite: bool,
) -> tuple[str, int, str]:
    """Create, validate, hash, and atomically promote one event NetCDF."""
    if output_path.exists() and not overwrite:
        validate_event_netcdf(output_path, event, window_type)
        return (
            "skipped_valid_existing",
            output_path.stat().st_size,
            sha256_file(output_path),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(output_path.stem + ".part.nc")
    if partial.exists():
        partial.unlink()

    first = sources[0]
    try:
        with decompressed_grib(first.gzip_path, scratch_dir) as first_grib:
            with rasterio.open(first_grib) as first_dataset:
                expected_grid, _ = inspect_grib(first_dataset, first)
                grid_window = stored_window(first_dataset, bbox)

        dates = [source.observation_date for source in sources]
        with netCDF4.Dataset(partial, "w", format="NETCDF4") as output:
            precipitation = create_netcdf_structure(
                output,
                event,
                window_type,
                dates,
                expected_grid,
                grid_window,
                bbox,
                block_rows,
            )
            for time_index, source in enumerate(sources):
                print(
                    f"    {source.observation_date.isoformat()} "
                    f"<- valid {source.valid_end:%Y-%m-%d %H:%MZ}"
                )
                with decompressed_grib(source.gzip_path, scratch_dir) as grib_path:
                    qc = write_one_day(
                        precipitation,
                        time_index,
                        source,
                        grib_path,
                        expected_grid,
                        grid_window,
                        block_rows,
                    )
                source_records[source.observation_date].update(qc)
                output.sync()

        validate_event_netcdf(partial, event, window_type)
        os.replace(partial, output_path)
        return "written", output_path.stat().st_size, sha256_file(output_path)
    except Exception:
        if partial.exists():
            try:
                partial.unlink()
            except OSError:
                pass
        raise


SOURCE_MANIFEST_FIELDS = [
    "observation_date_utc",
    "interval_start_utc",
    "interval_end_utc_exclusive",
    "mrms_valid_time_utc",
    "mrms_era_code",
    "mrms_era_label",
    "source_url",
    "local_gzip_path",
    "download_status",
    "size_bytes",
    "sha256",
    "grib_element",
    "grib_unit",
    "valid_cell_count_stored_grid",
    "missing_cell_count_stored_grid",
    "valid_fraction_stored_grid",
    "minimum_mm_stored_grid",
    "maximum_mm_stored_grid",
    "mean_mm_stored_grid",
    "completed_utc",
    "error",
]

EVENT_MANIFEST_FIELDS = [
    "event_name",
    "storm_year",
    "noaa_storm_id",
    "watershed_count",
    "window_type",
    "start_date_utc",
    "end_date_utc_inclusive",
    "day_count",
    "source_archive",
    "source_product",
    "daily_interval",
    "mrms_eras_present",
    "bbox_wsen",
    "output_path",
    "status",
    "size_bytes",
    "sha256",
    "completed_utc",
    "error",
]


def write_csv_atomic(
    path: Path,
    rows: list[dict[str, object]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def download_mrms_events(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    event_names: Sequence[str] | None = None,
    window: str = "expanded",
    bbox: tuple[float, float, float, float] | None = None,
    overwrite: bool = False,
    overwrite_raw: bool = False,
    keep_source_gzip: bool = True,
    retries: int = 3,
    timeout_seconds: int = 120,
    block_rows: int = 350,
) -> pd.DataFrame:
    """Download MRMS and write one daily-precipitation NetCDF per event.

    Parameters
    ----------
    output_dir
        Destination for event NetCDFs, source cache, scratch space, and CSV
        manifests.  The default is the configured MRMS data directory.
    event_names
        Event names such as ``["Ian"]``.  ``None`` processes all 10 events.
    window
        ``"expanded"`` (default) or ``"core"``.
    bbox
        Optional ``(west, south, east, north)`` bounding box in longitude and
        latitude.  ``None`` retains the entire native MRMS domain.
    overwrite
        Replace valid existing event NetCDF files.
    overwrite_raw
        Redownload compressed source GRIB files even when a cached copy is valid.
    keep_source_gzip
        Keep the compressed source files for provenance and restartability.
        The default is True.  False removes them only if every event succeeds.
    retries
        Number of retries after the first download attempt.
    timeout_seconds
        Network timeout for each source-file request.
    block_rows
        Number of raster rows processed at a time.  This limits memory use.
    """
    if window not in {"expanded", "core"}:
        raise ValueError("window must be 'expanded' or 'core'")
    if retries < 0:
        raise ValueError("retries must be zero or greater")
    if timeout_seconds < 1:
        raise ValueError("timeout_seconds must be positive")
    if block_rows < 1:
        raise ValueError("block_rows must be positive")
    if bbox is not None:
        if len(bbox) != 4:
            raise ValueError("bbox must contain west, south, east, north")
        bbox = tuple(float(value) for value in bbox)

    validate_event_table()
    selected_events = choose_events(event_names)
    destination = Path(output_dir)
    manifest_dir = destination / "manifests"
    scratch_dir = destination / "_scratch"
    destination.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    event_manifest_path = manifest_dir / f"mrms_{window}_event_files.csv"
    source_manifest_path = manifest_dir / f"mrms_{window}_daily_sources.csv"
    source_records: dict[date, dict[str, object]] = {}
    source_objects: dict[date, DailySource] = {}
    event_rows: list[dict[str, object]] = []

    print(f"Output directory: {destination}")
    print(f"Product: {PRODUCT}")
    print(f"Window: {window}")
    print(f"Events: {', '.join(event.name for event in selected_events)}")
    print(f"Stored area: {'full native MRMS domain' if bbox is None else bbox}")
    if bbox is None:
        print(
            "NOTE: Full-domain event NetCDF files are large and the complete run "
            "can require many gigabytes."
        )

    for event in selected_events:
        start, end = event.bounds(window)
        dates = inclusive_dates(start, end)
        output_path = event_output_path(destination, event, window, bbox)
        era_labels = sorted({mrms_era(day)[1] for day in dates})
        event_row: dict[str, object] = {
            "event_name": event.name,
            "storm_year": event.year,
            "noaa_storm_id": event.storm_id,
            "watershed_count": event.watershed_count,
            "window_type": window,
            "start_date_utc": start.isoformat(),
            "end_date_utc_inclusive": end.isoformat(),
            "day_count": len(dates),
            "source_archive": BASE_URL,
            "source_product": PRODUCT,
            "daily_interval": DAILY_INTERVAL,
            "mrms_eras_present": ", ".join(era_labels),
            "bbox_wsen": "full native domain" if bbox is None else str(bbox),
            "output_path": str(output_path),
            "status": "",
            "size_bytes": "",
            "sha256": "",
            "completed_utc": "",
            "error": "",
        }
        print(f"{event.name}: {start} through {end} ({len(dates)} days)")
        try:
            if output_path.exists() and not overwrite:
                validate_event_netcdf(output_path, event, window)
                status = "skipped_valid_existing"
                size_bytes = output_path.stat().st_size
                checksum = sha256_file(output_path)
            else:
                event_sources: list[DailySource] = []
                for day in dates:
                    source = source_objects.get(day)
                    if source is None:
                        source = daily_source(day, destination)
                        print(
                            f"  Source {day.isoformat()} "
                            f"(valid {source.valid_end:%Y-%m-%d 00Z})"
                        )
                        try:
                            record = download_daily_source(
                                source,
                                overwrite_raw=overwrite_raw,
                                retries=retries,
                                timeout_seconds=timeout_seconds,
                            )
                        except Exception as exc:
                            failed_record = source_manifest_record(source)
                            failed_record.update(
                                {
                                    "download_status": "failed",
                                    "completed_utc": datetime.now(timezone.utc).isoformat(),
                                    "error": str(exc),
                                }
                            )
                            source_records[day] = failed_record
                            raise
                        source_objects[day] = source
                        source_records[day] = record
                    event_sources.append(source)

                status, size_bytes, checksum = write_event_file(
                    event,
                    window,
                    output_path,
                    event_sources,
                    source_records,
                    scratch_dir,
                    bbox,
                    block_rows,
                    overwrite,
                )

            event_row.update(
                {
                    "status": status,
                    "size_bytes": size_bytes,
                    "sha256": checksum,
                    "completed_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            print(f"  {status}: {output_path.name} ({size_bytes:,} bytes)")
        except Exception as exc:
            event_row.update(
                {
                    "status": "failed",
                    "completed_utc": datetime.now(timezone.utc).isoformat(),
                    "error": str(exc),
                }
            )
            print(f"  FAILED: {exc}", file=sys.stderr)

        event_rows.append(event_row)
        write_csv_atomic(event_manifest_path, event_rows, EVENT_MANIFEST_FIELDS)
        if source_records:
            ordered_sources = [source_records[key] for key in sorted(source_records)]
            write_csv_atomic(
                source_manifest_path,
                ordered_sources,
                SOURCE_MANIFEST_FIELDS,
            )

    failures = [row for row in event_rows if row["status"] == "failed"]
    if not keep_source_gzip and not failures:
        for day, source in source_objects.items():
            if source.gzip_path.exists():
                source.gzip_path.unlink()
                source_records[day]["download_status"] = "removed_after_conversion"
        if source_records:
            ordered_sources = [source_records[key] for key in sorted(source_records)]
            write_csv_atomic(
                source_manifest_path,
                ordered_sources,
                SOURCE_MANIFEST_FIELDS,
            )

    try:
        scratch_dir.rmdir()  # Removed only when empty.
    except OSError:
        pass

    manifest = pd.DataFrame(event_rows)
    print(f"Event manifest: {event_manifest_path}")
    if source_records:
        print(f"Daily-source manifest: {source_manifest_path}")
    print(
        f"Completed {len(event_rows) - len(failures)} event file(s); "
        f"{len(failures)} failed."
    )
    if failures:
        failed_names = ", ".join(str(row["event_name"]) for row in failures)
        raise RuntimeError(
            f"Some events failed ({failed_names}). Review the CSV manifests and "
            "rerun; valid cached downloads and completed NetCDFs will be reused."
        )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download daily MRMS RadarOnly QPE NetCDF files by event."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--event",
        action="append",
        metavar="NAME",
        help="Process one event; repeat for multiple events. Default: all.",
    )
    parser.add_argument("--window", choices=("expanded", "core"), default="expanded")
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite-raw", action="store_true")
    parser.add_argument(
        "--discard-source-gzip",
        action="store_true",
        help="Remove cached .grib2.gz files only after every event succeeds.",
    )
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--block-rows", type=int, default=350)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        download_mrms_events(
            output_dir=args.output_dir,
            event_names=args.event,
            window=args.window,
            bbox=tuple(args.bbox) if args.bbox else None,
            overwrite=args.overwrite,
            overwrite_raw=args.overwrite_raw,
            keep_source_gzip=not args.discard_source_gzip,
            retries=args.retries,
            timeout_seconds=args.timeout_seconds,
            block_rows=args.block_rows,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
