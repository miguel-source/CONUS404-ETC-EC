"""
Conservatively align the four hurricane precipitation products to PERSIANN-CDR.

The program keeps PERSIANN-CDR on its native 0.25-degree cells.  For
CONUS404, CONUS404-BA, and MRMS it constructs exact source-cell/target-cell
intersection areas in EPSG:5070 and uses those areas to calculate a target-cell
mean precipitation depth.  It therefore does not use bilinear interpolation
and does not sum millimetres from fine-grid cells.

Outputs
-------
1. One restartable four-product NetCDF file per storm.
2. One combined observation-by-grid-cell NetCDF used by
   ``run_gridded_extended_collocation.py``.
3. Cached sparse remapping weights and a JSON processing manifest.

VS Code / Jupyter
-----------------
The ``# %%`` markers make this file cell-friendly.  The public function is
``run_preparation``.  A small Ian/Tampa Bay two-day check is available with
``--smoke-test``.
"""

# %% Imports
from __future__ import annotations

from study_paths import PROJECT_ROOT, RESULTS_ROOT, METADATA_ROOT, REPO_ROOT, STATES_FILE, WATERSHEDS_FILE, DAILY_INPUT, EC_DIRECTORY, EC_INPUT, LOSO_INPUT, AGREEMENT_DIRECTORY, AGREEMENT_INPUT


import argparse
import hashlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import sparse
import shapely
import xarray as xr

from prepare_watershed_day_etc import (
    AREA_CRS,
    EVENTS,
    MAX_PLAUSIBLE_DAILY_MM,
    PAPER_ROOT,
    PRODUCTS,
    build_weight_spec,
    coordinate_edges,
    find_event_file,
    grid_hash,
    load_watersheds,
    open_precipitation,
    validate_event_dates,
    watersheds_for_event,
)


# %% Configuration
DEFAULT_OUTPUT_DIR = (
    PAPER_ROOT / "ETC_StormEvents" / "Gridded_PERSIANN"
)
SCRIPT_DIR = RESULTS_ROOT
DEFAULT_SMOKE_OUTPUT_DIR = SCRIPT_DIR / "gridded_persiann_smoke"

PRODUCT_ORDER = ["conus404", "conus404ba", "mrms", "persiann_cdr"]
REGRID_PRODUCTS = ["conus404", "conus404ba", "mrms"]
TARGET_CRS = "EPSG:4326"
TARGET_RESOLUTION_DEGREES = 0.25
MIN_VALID_SOURCE_AREA_FRACTION = 0.95
MIN_GEOMETRIC_COVERAGE_FRACTION = 0.98
PREPARATION_SCHEMA_VERSION = 2


@dataclass
class TargetGrid:
    """PERSIANN target coordinates and watershed-domain masks."""

    x: np.ndarray
    y: np.ndarray
    x_edges: np.ndarray
    y_edges: np.ndarray
    cell_polygons_wgs84: np.ndarray
    cell_polygons_equal_area: np.ndarray
    cell_area_m2: np.ndarray
    study_fraction: np.ndarray
    study_mask: np.ndarray
    assigned_fraction: np.ndarray
    assigned_mask: np.ndarray
    storm_ids: list[str]
    event_names: list[str]

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.y), len(self.x)

    @property
    def active_flat_indices(self) -> np.ndarray:
        return np.flatnonzero(self.study_mask.ravel())


@dataclass
class SparseRemap:
    """Sparse exact-overlap remapping weights for one native product grid."""

    product: str
    source_grid_hash: str
    target_grid_hash: str
    row_start: int
    row_stop: int
    col_start: int
    col_stop: int
    matrix: sparse.csr_matrix
    target_area_m2: np.ndarray
    geometric_coverage_fraction: np.ndarray
    active_flat_indices: np.ndarray


# %% Event selection and target-grid construction
def select_events(event_names: Iterable[str] | None = None) -> pd.DataFrame:
    selected = EVENTS.copy()
    if event_names:
        requested = {str(name).strip().lower() for name in event_names}
        selected = selected[
            selected["event_name"].str.lower().isin(requested)
        ].copy()
        missing = requested.difference(selected["event_name"].str.lower())
        if missing:
            raise ValueError(f"Unknown event names: {sorted(missing)}")
    if selected.empty:
        raise ValueError("No events were selected.")
    return selected.reset_index(drop=True)


def _unique_event_watersheds(
    watersheds: gpd.GeoDataFrame, events: pd.DataFrame
) -> gpd.GeoDataFrame:
    parts = [
        watersheds_for_event(watersheds, event)
        for _, event in events.iterrows()
    ]
    selected = pd.concat(parts, ignore_index=True)
    selected = gpd.GeoDataFrame(selected, geometry="geometry", crs=watersheds.crs)
    return selected.drop_duplicates("watershed_id").reset_index(drop=True)


def _geometry_union(frame: gpd.GeoDataFrame):
    geometry = shapely.union_all(frame.geometry.array)
    return shapely.make_valid(geometry)


def _target_hash(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(x, dtype="<f8").tobytes())
    digest.update(np.asarray(y, dtype="<f8").tobytes())
    digest.update(np.asarray(mask, dtype=np.uint8).tobytes())
    digest.update(TARGET_CRS.encode("utf-8"))
    return digest.hexdigest()[:20]


def _fractional_overlap(
    equal_area_cells: np.ndarray,
    cell_area_m2: np.ndarray,
    geometry,
    geometry_crs,
) -> np.ndarray:
    equal_geometry = (
        gpd.GeoSeries([shapely.make_valid(geometry)], crs=geometry_crs)
        .to_crs(AREA_CRS)
        .iloc[0]
    )
    intersections = shapely.intersection(equal_area_cells.ravel(), equal_geometry)
    overlap = np.asarray(shapely.area(intersections), dtype=np.float64)
    fraction = np.divide(
        overlap,
        cell_area_m2.ravel(),
        out=np.zeros_like(overlap),
        where=cell_area_m2.ravel() > 0,
    )
    return np.clip(fraction, 0.0, 1.0).reshape(cell_area_m2.shape)


def build_target_grid(
    watersheds: gpd.GeoDataFrame,
    events: pd.DataFrame,
) -> TargetGrid:
    """Subset the exact PERSIANN grid to the selected hurricane watersheds."""
    selected_watersheds = _unique_event_watersheds(watersheds, events)
    study_union = _geometry_union(selected_watersheds)
    study_wgs84 = (
        gpd.GeoSeries([study_union], crs=selected_watersheds.crs)
        .to_crs(TARGET_CRS)
        .iloc[0]
    )

    first_event = events.iloc[0]
    target_path = find_event_file("persiann_cdr", first_event)
    target_ds, target_da = open_precipitation("persiann_cdr", target_path)
    try:
        min_x, min_y, max_x, max_y = study_wgs84.bounds
        half = TARGET_RESOLUTION_DEGREES / 2.0
        x_all = np.asarray(target_da.x.values, dtype=np.float64)
        y_all = np.asarray(target_da.y.values, dtype=np.float64)
        x = x_all[(x_all + half > min_x) & (x_all - half < max_x)]
        y = y_all[(y_all + half > min_y) & (y_all - half < max_y)]
    finally:
        target_ds.close()

    if len(x) == 0 or len(y) == 0:
        raise ValueError("The selected watersheds do not intersect PERSIANN-CDR.")
    if not np.allclose(np.abs(np.diff(x)), TARGET_RESOLUTION_DEGREES):
        raise ValueError("Unexpected PERSIANN longitude spacing.")
    if not np.allclose(np.abs(np.diff(y)), TARGET_RESOLUTION_DEGREES):
        raise ValueError("Unexpected PERSIANN latitude spacing.")

    x_edges = coordinate_edges(x)
    y_edges = coordinate_edges(y)
    x_low = np.minimum(x_edges[:-1], x_edges[1:])
    x_high = np.maximum(x_edges[:-1], x_edges[1:])
    y_low = np.minimum(y_edges[:-1], y_edges[1:])
    y_high = np.maximum(y_edges[:-1], y_edges[1:])
    left, bottom = np.meshgrid(x_low, y_low)
    right, top = np.meshgrid(x_high, y_high)
    cells = shapely.box(left, bottom, right, top)
    equal_cells = np.asarray(
        gpd.GeoSeries(cells.ravel(), crs=TARGET_CRS)
        .to_crs(AREA_CRS)
        .array,
        dtype=object,
    ).reshape(cells.shape)
    cell_area = np.asarray(shapely.area(equal_cells), dtype=np.float64)

    study_fraction = _fractional_overlap(
        equal_cells, cell_area, study_union, selected_watersheds.crs
    )
    center_x, center_y = np.meshgrid(x, y)
    center_points = shapely.points(center_x, center_y)
    study_mask = np.asarray(
        shapely.intersects(study_wgs84, center_points), dtype=bool
    )
    if not study_mask.any():
        raise ValueError("No PERSIANN cell centres fall in the study watersheds.")

    assigned_fraction = []
    assigned_mask = []
    storm_ids: list[str] = []
    event_names: list[str] = []
    for _, event in events.iterrows():
        event_watersheds = watersheds_for_event(watersheds, event)
        event_union = _geometry_union(event_watersheds)
        event_wgs84 = (
            gpd.GeoSeries([event_union], crs=event_watersheds.crs)
            .to_crs(TARGET_CRS)
            .iloc[0]
        )
        fraction = _fractional_overlap(
            equal_cells, cell_area, event_union, event_watersheds.crs
        )
        mask = np.asarray(
            shapely.intersects(event_wgs84, center_points), dtype=bool
        )
        assigned_fraction.append(fraction)
        assigned_mask.append(mask)
        storm_ids.append(str(event.storm_id))
        event_names.append(str(event.event_name))

    return TargetGrid(
        x=x,
        y=y,
        x_edges=x_edges,
        y_edges=y_edges,
        cell_polygons_wgs84=cells,
        cell_polygons_equal_area=equal_cells,
        cell_area_m2=cell_area,
        study_fraction=study_fraction,
        study_mask=study_mask,
        assigned_fraction=np.stack(assigned_fraction),
        assigned_mask=np.stack(assigned_mask),
        storm_ids=storm_ids,
        event_names=event_names,
    )


# %% Sparse exact-overlap remapping weights
def _remap_cache_stem(product: str, source_hash: str, target_hash: str) -> str:
    return f"{product}__{source_hash}__to_persiann__{target_hash}"


def _save_sparse_remap(cache_dir: Path, remap: SparseRemap) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = _remap_cache_stem(
        remap.product, remap.source_grid_hash, remap.target_grid_hash
    )
    sparse.save_npz(cache_dir / f"{stem}.matrix.npz", remap.matrix)
    np.savez_compressed(
        cache_dir / f"{stem}.metadata.npz",
        product=np.asarray(remap.product),
        source_grid_hash=np.asarray(remap.source_grid_hash),
        target_grid_hash=np.asarray(remap.target_grid_hash),
        row_start=remap.row_start,
        row_stop=remap.row_stop,
        col_start=remap.col_start,
        col_stop=remap.col_stop,
        target_area_m2=remap.target_area_m2,
        geometric_coverage_fraction=remap.geometric_coverage_fraction,
        active_flat_indices=remap.active_flat_indices,
    )


def _load_sparse_remap(
    cache_dir: Path,
    product: str,
    source_hash: str,
    target_hash: str,
) -> SparseRemap | None:
    stem = _remap_cache_stem(product, source_hash, target_hash)
    matrix_path = cache_dir / f"{stem}.matrix.npz"
    metadata_path = cache_dir / f"{stem}.metadata.npz"
    if not matrix_path.exists() or not metadata_path.exists():
        return None
    with np.load(metadata_path, allow_pickle=False) as saved:
        remap = SparseRemap(
            product=str(saved["product"].item()),
            source_grid_hash=str(saved["source_grid_hash"].item()),
            target_grid_hash=str(saved["target_grid_hash"].item()),
            row_start=int(saved["row_start"]),
            row_stop=int(saved["row_stop"]),
            col_start=int(saved["col_start"]),
            col_stop=int(saved["col_stop"]),
            matrix=sparse.load_npz(matrix_path).tocsr(),
            target_area_m2=saved["target_area_m2"],
            geometric_coverage_fraction=saved["geometric_coverage_fraction"],
            active_flat_indices=saved["active_flat_indices"],
        )
    return remap


def build_sparse_remap(
    product: str,
    da: xr.DataArray,
    target: TargetGrid,
    cache_dir: Path,
) -> SparseRemap:
    """Build or load exact source/target intersection weights."""
    source_hash = grid_hash(da)
    target_hash = _target_hash(target.x, target.y, target.study_mask)
    cached = _load_sparse_remap(
        cache_dir, product, source_hash, target_hash
    )
    if cached is not None:
        return cached

    active = target.active_flat_indices
    specs = []
    total = len(active)
    print(f"Building exact {product} -> PERSIANN weights for {total} cells...")
    for position, flat_index in enumerate(active, start=1):
        polygon = target.cell_polygons_wgs84.ravel()[flat_index]
        spec = build_weight_spec(da, polygon, TARGET_CRS)
        specs.append(spec)
        if position == 1 or position % 100 == 0 or position == total:
            print(f"  {position}/{total} target cells")

    row_start = min(spec.row_start for spec in specs)
    row_stop = max(spec.row_stop for spec in specs)
    col_start = min(spec.col_start for spec in specs)
    col_stop = max(spec.col_stop for spec in specs)
    crop_width = col_stop - col_start
    crop_size = (row_stop - row_start) * crop_width

    sparse_rows: list[np.ndarray] = []
    sparse_columns: list[np.ndarray] = []
    sparse_values: list[np.ndarray] = []
    target_areas = np.empty(total, dtype=np.float64)
    coverage = np.empty(total, dtype=np.float64)
    for target_row, spec in enumerate(specs):
        local_r, local_c = np.nonzero(spec.weights_m2 > 0)
        global_r = local_r + spec.row_start - row_start
        global_c = local_c + spec.col_start - col_start
        source_flat = global_r * crop_width + global_c
        sparse_rows.append(np.full(len(source_flat), target_row, dtype=np.int64))
        sparse_columns.append(source_flat.astype(np.int64))
        sparse_values.append(spec.weights_m2[local_r, local_c])
        target_areas[target_row] = spec.basin_area_m2
        coverage[target_row] = spec.geometric_coverage_fraction

    matrix = sparse.csr_matrix(
        (
            np.concatenate(sparse_values),
            (np.concatenate(sparse_rows), np.concatenate(sparse_columns)),
        ),
        shape=(total, crop_size),
        dtype=np.float64,
    )
    if np.nanmin(coverage) < MIN_GEOMETRIC_COVERAGE_FRACTION:
        raise ValueError(
            f"{product}: minimum target-cell geometric grid coverage is "
            f"{np.nanmin(coverage):.4f}, below "
            f"{MIN_GEOMETRIC_COVERAGE_FRACTION:.2f}."
        )

    remap = SparseRemap(
        product=product,
        source_grid_hash=source_hash,
        target_grid_hash=target_hash,
        row_start=row_start,
        row_stop=row_stop,
        col_start=col_start,
        col_stop=col_stop,
        matrix=matrix,
        target_area_m2=target_areas,
        geometric_coverage_fraction=coverage,
        active_flat_indices=active,
    )
    _save_sparse_remap(cache_dir, remap)
    return remap


def apply_sparse_remap(
    da: xr.DataArray,
    remap: SparseRemap,
    target: TargetGrid,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply sparse weights one day at a time to bound peak memory."""
    if grid_hash(da) != remap.source_grid_hash:
        raise ValueError(f"{remap.product}: native grid differs from cached weights.")
    subset = da.isel(
        y=slice(remap.row_start, remap.row_stop),
        x=slice(remap.col_start, remap.col_stop),
    )
    n_time = subset.sizes["time"]
    n_y, n_x = target.shape
    output = np.full((n_time, n_y * n_x), np.nan, dtype=np.float32)
    coverage_output = np.full_like(output, np.nan)

    for time_index in range(n_time):
        values = np.asarray(
            subset.isel(time=time_index).compute().values, dtype=np.float64
        ).ravel()
        valid = np.isfinite(values) & (values >= 0.0)
        numerator = remap.matrix @ np.where(valid, values, 0.0)
        valid_area = remap.matrix @ valid.astype(np.float64)
        valid_fraction = np.divide(
            valid_area,
            remap.target_area_m2,
            out=np.zeros_like(valid_area),
            where=remap.target_area_m2 > 0,
        )
        mean = np.divide(
            numerator,
            valid_area,
            out=np.full_like(numerator, np.nan),
            where=valid_area > 0,
        )
        mean[valid_fraction < MIN_VALID_SOURCE_AREA_FRACTION] = np.nan
        output[time_index, remap.active_flat_indices] = mean.astype(np.float32)
        coverage_output[time_index, remap.active_flat_indices] = (
            valid_fraction.astype(np.float32)
        )
        print(f"    day {time_index + 1}/{n_time}", end="\r")
    print(" " * 40, end="\r")

    output = output.reshape(n_time, n_y, n_x)
    coverage_output = coverage_output.reshape(n_time, n_y, n_x)
    output[:, ~target.study_mask] = np.nan
    coverage_output[:, ~target.study_mask] = np.nan
    return output, coverage_output


# %% Event processing and NetCDF writing
def _direct_persiann_subset(
    da: xr.DataArray,
    target: TargetGrid,
) -> tuple[np.ndarray, np.ndarray]:
    subset = da.sel(x=target.x, y=target.y)
    if not np.allclose(subset.x.values, target.x):
        raise ValueError("PERSIANN target longitude coordinates do not match.")
    if not np.allclose(subset.y.values, target.y):
        raise ValueError("PERSIANN target latitude coordinates do not match.")
    values = np.asarray(subset.compute().values, dtype=np.float32)
    valid = np.isfinite(values) & (values >= 0.0)
    values = np.where(valid, values, np.nan).astype(np.float32)
    coverage = valid.astype(np.float32)
    values[:, ~target.study_mask] = np.nan
    coverage[:, ~target.study_mask] = np.nan
    return values, coverage


def _event_output_path(output_dir: Path, event: pd.Series) -> Path:
    return (
        output_dir
        / "events"
        / f"all4_persiann_grid_{event.event_name}_{event.storm_id}.nc"
    )


def _netcdf_encoding(dataset: xr.Dataset) -> dict:
    encoding: dict[str, dict] = {}
    for name, variable in dataset.data_vars.items():
        if variable.dtype.kind in "f":
            chunks = tuple(
                1 if dim in {"time", "observation", "storm"}
                else min(variable.sizes[dim], 64)
                for dim in variable.dims
            )
            encoding[name] = {
                "zlib": True,
                "complevel": 4,
                "shuffle": True,
                "dtype": "float32",
                "_FillValue": np.float32(-9999.0),
                "chunksizes": chunks,
            }
        elif variable.dtype.kind in "biu":
            encoding[name] = {"zlib": True, "complevel": 4, "shuffle": True}
    return encoding


def _event_assignment_hash(target: TargetGrid, event_index: int) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(target.x, dtype="<f8").tobytes())
    digest.update(np.asarray(target.y, dtype="<f8").tobytes())
    digest.update(
        np.asarray(target.assigned_mask[event_index], dtype=np.uint8).tobytes()
    )
    digest.update(
        np.asarray(target.assigned_fraction[event_index], dtype="<f4").tobytes()
    )
    return digest.hexdigest()[:20]


def _source_file_fingerprints(source_paths: dict[str, Path]) -> str:
    fingerprints = {}
    for product, path in source_paths.items():
        resolved = path.resolve()
        status = resolved.stat()
        fingerprints[product] = {
            "path": str(resolved),
            "size_bytes": int(status.st_size),
            "modified_time_ns": int(status.st_mtime_ns),
        }
    return json.dumps(fingerprints, sort_keys=True)


def _expected_event_dates(
    event: pd.Series, max_days: int | None
) -> pd.DatetimeIndex:
    dates = pd.date_range(event.expanded_start, event.expanded_end, freq="D")
    return dates if max_days is None else dates[:max_days]


def _existing_event_is_reusable(
    event_path: Path,
    event: pd.Series,
    event_index: int,
    target: TargetGrid,
    remaps: dict[str, SparseRemap],
    source_paths: dict[str, Path],
    max_days: int | None,
) -> tuple[bool, str]:
    """Validate a restart file before allowing it into a new combined archive."""
    required = set(PRODUCT_ORDER)
    required.update(f"{name}_valid_area_fraction" for name in PRODUCT_ORDER)
    required.update({"event_watershed_fraction", "event_watershed_mask"})
    expected_dates = _expected_event_dates(event, max_days)
    expected_source_files = json.dumps(
        {name: str(path.resolve()) for name, path in source_paths.items()},
        sort_keys=True,
    )
    expected_grid_hashes = json.dumps(
        {name: remaps[name].source_grid_hash for name in REGRID_PRODUCTS},
        sort_keys=True,
    )
    checks = {
        "preparation_schema_version": PREPARATION_SCHEMA_VERSION,
        "event_name": str(event.event_name),
        "storm_id": str(event.storm_id),
        "target_grid_hash": _target_hash(target.x, target.y, target.study_mask),
        "event_assignment_hash": _event_assignment_hash(target, event_index),
        "max_days_debug_limit": -1 if max_days is None else int(max_days),
        "source_files_json": expected_source_files,
        "source_file_fingerprints_json": _source_file_fingerprints(source_paths),
        "source_grid_hashes_json": expected_grid_hashes,
    }
    try:
        with xr.open_dataset(event_path, engine="netcdf4") as existing:
            for name, expected in checks.items():
                if existing.attrs.get(name) != expected:
                    return False, f"attribute {name!r} differs"
            missing = sorted(required.difference(existing.data_vars))
            if missing:
                return False, f"missing variables {missing}"
            if existing.sizes.get("time") != len(expected_dates):
                return False, "time length differs"
            if not np.array_equal(
                np.asarray(existing.time.values, dtype="datetime64[ns]"),
                expected_dates.to_numpy(dtype="datetime64[ns]"),
            ):
                return False, "dates differ"
            if not np.array_equal(existing.lon.values, target.x):
                return False, "longitude grid differs"
            if not np.array_equal(existing.lat.values, target.y):
                return False, "latitude grid differs"
            if not np.array_equal(
                existing.event_watershed_mask.values,
                target.assigned_mask[event_index].astype(np.int8),
            ):
                return False, "event watershed mask differs"
            if not np.allclose(
                existing.event_watershed_fraction.values,
                target.assigned_fraction[event_index],
                rtol=0.0,
                atol=1.0e-7,
                equal_nan=True,
            ):
                return False, "event watershed fractions differ"
    except Exception as exc:
        return False, f"could not validate existing file: {exc}"
    return True, "validated"


def process_event(
    event: pd.Series,
    event_index: int,
    target: TargetGrid,
    remaps: dict[str, SparseRemap],
    output_dir: Path,
    *,
    max_days: int | None = None,
    overwrite: bool = False,
) -> Path:
    event_path = _event_output_path(output_dir, event)
    event_path.parent.mkdir(parents=True, exist_ok=True)
    source_paths = {
        product: find_event_file(product, event) for product in PRODUCT_ORDER
    }
    if event_path.exists() and not overwrite:
        reusable, reason = _existing_event_is_reusable(
            event_path,
            event,
            event_index,
            target,
            remaps,
            source_paths,
            max_days,
        )
        if reusable:
            print(f"Using validated existing {event_path.name}")
            return event_path
        print(f"Rebuilding {event_path.name}: {reason}")

    arrays: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
    dates: pd.DatetimeIndex | None = None
    source_files: dict[str, str] = {}
    mrms_era_code: np.ndarray | None = None

    for product in PRODUCT_ORDER:
        source_path = source_paths[product]
        source_files[product] = str(source_path.resolve())
        source_ds, da = open_precipitation(product, source_path)
        try:
            validate_event_dates(da, event, source_path)
            if max_days is not None:
                da = da.isel(time=slice(0, max_days))
            product_dates = pd.DatetimeIndex(pd.to_datetime(da.time.values)).normalize()
            if dates is None:
                dates = product_dates
            elif not product_dates.equals(dates):
                raise ValueError(
                    f"{event.event_name}: product dates are not identical."
                )

            print(f"  {event.event_name}: {product}")
            if product == "persiann_cdr":
                values, valid_fraction = _direct_persiann_subset(da, target)
            else:
                values, valid_fraction = apply_sparse_remap(
                    da, remaps[product], target
                )
            finite = values[np.isfinite(values)]
            if finite.size and (
                finite.min() < 0 or finite.max() > MAX_PLAUSIBLE_DAILY_MM
            ):
                raise ValueError(
                    f"{event.event_name}/{product}: remapped precipitation outside "
                    f"0..{MAX_PLAUSIBLE_DAILY_MM} mm/day."
                )
            arrays[product] = (("time", "lat", "lon"), values)
            arrays[f"{product}_valid_area_fraction"] = (
                ("time", "lat", "lon"),
                valid_fraction,
            )
            if product == "mrms" and "mrms_era_code" in source_ds:
                mrms_era_code = np.asarray(
                    source_ds["mrms_era_code"].isel(time=slice(0, len(product_dates))).values,
                    dtype=np.int16,
                )
        finally:
            source_ds.close()

    assert dates is not None
    is_core = (
        (dates >= pd.Timestamp(event.core_start))
        & (dates <= pd.Timestamp(event.core_end))
    )
    arrays["event_watershed_fraction"] = (
        ("lat", "lon"),
        target.assigned_fraction[event_index].astype(np.float32),
    )
    arrays["event_watershed_mask"] = (
        ("lat", "lon"),
        target.assigned_mask[event_index].astype(np.int8),
    )
    if mrms_era_code is not None:
        arrays["mrms_era_code"] = (("time",), mrms_era_code)

    result = xr.Dataset(
        data_vars=arrays,
        coords={
            "time": dates.to_numpy(dtype="datetime64[ns]"),
            "lat": target.y.astype(np.float64),
            "lon": target.x.astype(np.float64),
            "is_core_day": ("time", is_core.astype(np.int8)),
        },
        attrs={
            "title": "Daily hurricane precipitation aligned to PERSIANN-CDR cells",
            "event_name": str(event.event_name),
            "storm_id": str(event.storm_id),
            "storm_year": int(event.storm_year),
            "target_grid": "Native PERSIANN-CDR 0.25-degree cell centres",
            "target_crs": TARGET_CRS,
            "remapping_method": (
                "Exact source-cell/target-cell overlap-area mean in EPSG:5070; "
                "PERSIANN-CDR retained unchanged"
            ),
            "minimum_valid_source_area_fraction": MIN_VALID_SOURCE_AREA_FRACTION,
            "daily_interval": "[D 00:00 UTC, D+1 00:00 UTC)",
            "preparation_schema_version": PREPARATION_SCHEMA_VERSION,
            "target_grid_hash": _target_hash(
                target.x, target.y, target.study_mask
            ),
            "event_assignment_hash": _event_assignment_hash(
                target, event_index
            ),
            "max_days_debug_limit": -1 if max_days is None else int(max_days),
            "source_files_json": json.dumps(source_files, sort_keys=True),
            "source_file_fingerprints_json": _source_file_fingerprints(
                source_paths
            ),
            "source_grid_hashes_json": json.dumps(
                {
                    product: remaps[product].source_grid_hash
                    for product in REGRID_PRODUCTS
                },
                sort_keys=True,
            ),
        },
    )
    result["crs"] = xr.DataArray(
        np.int8(0),
        attrs={
            "grid_mapping_name": "latitude_longitude",
            "epsg_code": TARGET_CRS,
            "semi_major_axis": 6378137.0,
            "inverse_flattening": 298.257223563,
        },
    )
    for product in PRODUCT_ORDER:
        result[product].attrs.update(
            units="mm",
            long_name=f"Daily precipitation from {product} on PERSIANN-CDR grid",
            cell_methods="time: sum (interval: 1 day)",
            grid_mapping="crs",
        )
        result[f"{product}_valid_area_fraction"].attrs.update(
            units="1",
            valid_min=0.0,
            valid_max=1.0,
        )
    result["lat"].attrs.update(
        standard_name="latitude", units="degrees_north", axis="Y"
    )
    result["lon"].attrs.update(
        standard_name="longitude", units="degrees_east", axis="X"
    )
    result.to_netcdf(
        event_path,
        engine="netcdf4",
        encoding=_netcdf_encoding(result),
    )
    result.close()
    print(f"  saved {event_path}")
    return event_path


def combine_event_files(
    event_paths: list[Path],
    events: pd.DataFrame,
    target: TargetGrid,
    output_dir: Path,
) -> Path:
    """Stack event files on an observation axis without losing storm identity."""
    product_variables = PRODUCT_ORDER + [
        f"{product}_valid_area_fraction" for product in PRODUCT_ORDER
    ]
    parts = []
    dates: list[np.ndarray] = []
    storm_indices: list[np.ndarray] = []
    core_flags: list[np.ndarray] = []
    era_codes: list[np.ndarray] = []

    for storm_index, path in enumerate(event_paths):
        with xr.open_dataset(path, engine="netcdf4") as event_ds:
            loaded = event_ds[product_variables].load()
            n_time = event_ds.sizes["time"]
            loaded = loaded.rename({"time": "observation"})
            loaded = loaded.assign_coords(observation=np.arange(n_time))
            parts.append(loaded)
            dates.append(np.asarray(event_ds.time.values, dtype="datetime64[ns]"))
            storm_indices.append(np.full(n_time, storm_index, dtype=np.int16))
            core_flags.append(
                np.asarray(event_ds.is_core_day.values, dtype=np.int8)
            )
            if "mrms_era_code" in event_ds:
                era_codes.append(
                    np.asarray(event_ds.mrms_era_code.values, dtype=np.int16)
                )
            else:
                era_codes.append(np.full(n_time, -1, dtype=np.int16))

    combined = xr.concat(
        parts,
        dim="observation",
        data_vars="minimal",
        coords="minimal",
        compat="override",
        join="exact",
    )
    n_obs = combined.sizes["observation"]
    combined = combined.assign_coords(observation=np.arange(n_obs, dtype=np.int32))
    combined = combined.assign_coords(
        date=("observation", np.concatenate(dates)),
        storm_index=("observation", np.concatenate(storm_indices)),
        is_core_day=("observation", np.concatenate(core_flags)),
        storm=np.arange(len(events), dtype=np.int16),
        storm_id=("storm", events["storm_id"].astype(str).to_numpy()),
        event_name=("storm", events["event_name"].astype(str).to_numpy()),
    )
    combined["mrms_era_code"] = (
        ("observation",), np.concatenate(era_codes)
    )
    combined["study_domain_mask"] = (
        ("lat", "lon"), target.study_mask.astype(np.int8)
    )
    combined["study_watershed_fraction"] = (
        ("lat", "lon"), target.study_fraction.astype(np.float32)
    )
    combined["assigned_storm_mask"] = (
        ("storm", "lat", "lon"), target.assigned_mask.astype(np.int8)
    )
    combined["assigned_storm_watershed_fraction"] = (
        ("storm", "lat", "lon"), target.assigned_fraction.astype(np.float32)
    )
    combined["crs"] = xr.DataArray(
        np.int8(0),
        attrs={
            "grid_mapping_name": "latitude_longitude",
            "epsg_code": TARGET_CRS,
            "semi_major_axis": 6378137.0,
            "inverse_flattening": 298.257223563,
        },
    )
    for product in PRODUCT_ORDER:
        combined[product].attrs["grid_mapping"] = "crs"
    duplicate_dates = pd.DatetimeIndex(np.concatenate(dates))
    duplicate_values = sorted(
        set(duplicate_dates[duplicate_dates.duplicated(keep=False)].strftime("%Y-%m-%d"))
    )
    # xr.concat preserves the first event's global attributes.  They describe
    # one source file and would be misleading on the multi-event product.
    combined.attrs = {}
    duplicate_note = (
        "Duplicate calendar dates are retained as separate event records in "
        "this archive. The all-windows EC program keeps the first EVENTS-order "
        "copy only; assigned-storm mode retains records under their storm masks."
        if duplicate_values
        else "No duplicate calendar dates occur in this selected event subset."
    )
    combined.attrs.update(
        title="Matched daily hurricane precipitation on the PERSIANN-CDR grid",
        products_in_order=",".join(PRODUCT_ORDER),
        target_grid="Native PERSIANN-CDR 0.25-degree cells",
        target_crs=TARGET_CRS,
        remapping_method=(
            "Exact source-cell/target-cell overlap-area mean in EPSG:5070; "
            "PERSIANN-CDR unchanged"
        ),
        minimum_valid_source_area_fraction=MIN_VALID_SOURCE_AREA_FRACTION,
        daily_interval="[D 00:00 UTC, D+1 00:00 UTC)",
        observation_definition="storm-event record x UTC calendar day",
        duplicate_calendar_dates_json=json.dumps(duplicate_values),
        duplicate_date_note=duplicate_note,
    )
    output_path = output_dir / "hurricane_daily_all4_persiann_grid.nc"
    combined.to_netcdf(
        output_path,
        engine="netcdf4",
        encoding=_netcdf_encoding(combined),
    )
    combined.close()
    print(f"Saved combined file: {output_path}")
    return output_path


# %% Complete workflow
def run_preparation(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    event_names: Iterable[str] | None = None,
    max_days: int | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    if max_days is not None and max_days < 1:
        raise ValueError("max_days must be a positive integer when supplied.")
    output_dir.mkdir(parents=True, exist_ok=True)
    events = select_events(event_names)
    watersheds = load_watersheds()
    target = build_target_grid(watersheds, events)
    print(
        f"Target grid: {len(target.y)} x {len(target.x)}; "
        f"{target.study_mask.sum()} watershed-centre cells"
    )

    remaps: dict[str, SparseRemap] = {}
    first_event = events.iloc[0]
    for product in REGRID_PRODUCTS:
        source_path = find_event_file(product, first_event)
        source_ds, source_da = open_precipitation(product, source_path)
        try:
            remaps[product] = build_sparse_remap(
                product,
                source_da,
                target,
                output_dir / "weights",
            )
        finally:
            source_ds.close()

    event_paths = []
    for event_index, (_, event) in enumerate(events.iterrows()):
        print(f"\nProcessing {event.event_name} ({event.storm_id})")
        event_paths.append(
            process_event(
                event,
                event_index,
                target,
                remaps,
                output_dir,
                max_days=max_days,
                overwrite=overwrite,
            )
        )

    combined_path = combine_event_files(event_paths, events, target, output_dir)
    manifest = {
        "output_directory": str(output_dir),
        "combined_file": str(combined_path),
        "event_files": [str(path) for path in event_paths],
        "events": events[["event_name", "storm_id"]].to_dict("records"),
        "products": PRODUCT_ORDER,
        "target_grid_shape": list(target.shape),
        "study_domain_cell_count": int(target.study_mask.sum()),
        "target_resolution_degrees": TARGET_RESOLUTION_DEGREES,
        "target_crs": TARGET_CRS,
        "method": (
            "exact source-target cell intersection area in EPSG:5070, "
            "normalized to valid source area"
        ),
        "minimum_valid_source_area_fraction": MIN_VALID_SOURCE_AREA_FRACTION,
        "source_grid_hashes": {
            product: remaps[product].source_grid_hash
            for product in REGRID_PRODUCTS
        },
        "target_grid_hash": _target_hash(
            target.x, target.y, target.study_mask
        ),
        "max_days_debug_limit": max_days,
    }
    manifest_path = output_dir / "gridded_preparation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "combined": combined_path,
        "events": event_paths,
        "manifest": manifest_path,
    }


# %% Command-line entry point
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--events", nargs="*", default=None)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Process only the first two Ian days over Tampa Bay.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    if arguments.smoke_test:
        smoke_output = (
            arguments.output_dir
            if arguments.output_dir != DEFAULT_OUTPUT_DIR
            else DEFAULT_SMOKE_OUTPUT_DIR
        )
        run_preparation(
            smoke_output,
            event_names=["Ian"],
            max_days=2,
            overwrite=arguments.overwrite,
        )
    else:
        run_preparation(
            arguments.output_dir,
            event_names=arguments.events,
            max_days=arguments.max_days,
            overwrite=arguments.overwrite,
        )
