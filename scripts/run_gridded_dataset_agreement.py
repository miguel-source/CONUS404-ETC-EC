"""
Pixelwise dataset-agreement analysis for the gridded hurricane EC results.

This is the PERSIANN-grid adaptation of ``4_dataset_agreement.ipynb``.  It
compares the temporal median daily difference between each pair of products
with the EC random-error standard deviation of that difference:

    sigma_delta = sqrt(var(e_a) + var(e_b) - 2 cov(e_a, e_b))

The original notebook's "agreement probability" is retained for
compatibility as ``Phi(-abs(median_difference) / sigma_delta)``.  It is a
one-sided lower-tail overlap measure, not the probability that two continuous
variables are exactly equal.  It ranges from 0 to 0.5.  The script also saves
``agreement_score = 2 * agreement_probability`` (0 to 1) and the standardized
discrepancy itself.

CONUS404 versus CONUS404BA is the primary pair because its error covariance is
estimated by the four-system EC.  Other pairs are diagnostic and use the EC
assumption of zero cross-error covariance; MRMS/PERSIANN error variances are
also diagnostic estimates.

The file contains ``# %%`` markers and can be imported directly in a VS Code
Jupyter notebook.  The public entry point is ``run_agreement_analysis``.
"""

# %% Imports
from __future__ import annotations

from study_paths import PROJECT_ROOT, RESULTS_ROOT, METADATA_ROOT, REPO_ROOT, STATES_FILE, WATERSHEDS_FILE, DAILY_INPUT, EC_DIRECTORY, EC_INPUT, LOSO_INPUT, AGREEMENT_DIRECTORY, AGREEMENT_INPUT


import argparse
import itertools
import json
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import ndtr
import xarray as xr

from map_overlays import add_map_boundaries


# %% Configuration
PAPER_ROOT = (PROJECT_ROOT)
DEFAULT_DAILY_INPUT = DAILY_INPUT
DEFAULT_EC_INPUT = EC_INPUT
DEFAULT_LOSO_INPUT = LOSO_INPUT
DEFAULT_OUTPUT_DIR = AGREEMENT_DIRECTORY

PRODUCTS = ("conus404", "conus404ba", "mrms", "persiann_cdr")
PRODUCT_LABELS = {
    "conus404": "CONUS404",
    "conus404ba": "CONUS404BA",
    "mrms": "MRMS",
    "persiann_cdr": "PERSIANN-CDR",
}
WINDOWS = ("core", "expanded")
TRANSFORMS = ("raw", "log1p")
PRODUCT_PAIRS = tuple(itertools.combinations(range(len(PRODUCTS)), 2))
PAIR_IDS = tuple(f"{PRODUCTS[i]}__{PRODUCTS[j]}" for i, j in PRODUCT_PAIRS)
PRIMARY_PAIR_ID = "conus404__conus404ba"


# %% Input and sample helpers
def _decode_strings(values: np.ndarray) -> list[str]:
    decoded: list[str] = []
    for value in np.asarray(values).tolist():
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        decoded.append(str(value))
    return decoded


def _require_inputs(daily: xr.Dataset, ec: xr.Dataset) -> None:
    daily_required = set(PRODUCTS) | {
        "date",
        "storm_index",
        "storm_id",
        "is_core_day",
        "study_domain_mask",
        "study_watershed_fraction",
    }
    ec_required = {
        "error_variance",
        "error_covariance_model_pair",
        "valid_primary_ec",
        "reference_diagnostics_valid",
        "sample_eligible",
        "n_observations",
        "n_contributing_storms",
    }
    missing_daily = sorted(name for name in daily_required if name not in daily)
    missing_ec = sorted(name for name in ec_required if name not in ec)
    if missing_daily:
        raise ValueError(f"Daily aligned input is missing: {missing_daily}")
    if missing_ec:
        raise ValueError(f"EC input is missing: {missing_ec}")
    for product in PRODUCTS:
        if daily[product].dims != ("observation", "lat", "lon"):
            raise ValueError(
                f"{product} must have dimensions (observation, lat, lon); "
                f"found {daily[product].dims}."
            )
    if tuple(_decode_strings(ec["product"].values)) != PRODUCTS:
        raise ValueError("EC product order does not match the agreement script.")
    if not np.array_equal(daily.lat.values, ec.lat.values) or not np.array_equal(
        daily.lon.values, ec.lon.values
    ):
        raise ValueError("Daily and EC grids do not match exactly.")


def _selected_observations(
    daily: xr.Dataset, window: str
) -> tuple[np.ndarray, int]:
    dates = pd.DatetimeIndex(pd.to_datetime(daily["date"].values))
    keep = np.ones(len(dates), dtype=bool)
    if window == "core":
        keep &= np.asarray(daily["is_core_day"].values, dtype=bool)
    selected = np.flatnonzero(keep)
    selected_dates = dates[selected]
    duplicated = selected_dates.duplicated(keep="first")
    if duplicated.any():
        duplicate_dates = selected_dates[duplicated].unique()
        for duplicate_date in duplicate_dates:
            positions = selected[selected_dates == duplicate_date]
            reference_position = int(positions[0])
            for other_position in positions[1:]:
                for product in PRODUCTS:
                    reference = np.asarray(
                        daily[product].isel(observation=reference_position).values
                    )
                    candidate = np.asarray(
                        daily[product].isel(observation=int(other_position)).values
                    )
                    if not np.allclose(
                        reference,
                        candidate,
                        rtol=0.0,
                        atol=0.0,
                        equal_nan=True,
                    ):
                        raise ValueError(
                            "Conflicting duplicate daily fields on "
                            f"{pd.Timestamp(duplicate_date):%Y-%m-%d}: {product}."
                        )
        keep[selected[duplicated]] = False
    return np.flatnonzero(keep), int(duplicated.sum())


def _matched_sample(
    values: np.ndarray,
    storm_index: np.ndarray,
    number_of_storms: int,
    study_domain: np.ndarray,
    min_days_per_storm: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return four-product complete-case mask and its pixel sample counts."""
    common = np.isfinite(values).all(axis=1) & (values >= 0).all(axis=1)
    common &= study_domain[None, :, :]
    preliminary = np.zeros(
        (number_of_storms, values.shape[2], values.shape[3]), dtype=np.int16
    )
    for storm in range(number_of_storms):
        preliminary[storm] = common[storm_index == storm].sum(axis=0)
    qualifying = preliminary >= min_days_per_storm
    final = common & qualifying[storm_index]
    counts = final.sum(axis=0).astype(np.int16)
    contributing_storms = qualifying.sum(axis=0).astype(np.int8)
    return final, counts, contributing_storms


def _nan_summaries(
    difference: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    masked = np.where(valid, difference, np.nan)
    absolute = np.abs(masked)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(masked, axis=0)
        median = np.nanmedian(masked, axis=0)
        temporal_sd = np.nanstd(masked, axis=0, ddof=1)
        mean_absolute = np.nanmean(absolute, axis=0)
        median_absolute = np.nanmedian(absolute, axis=0)
    return mean, median, temporal_sd, mean_absolute, median_absolute


def _safe_pair_error_sd(
    variance_a: np.ndarray,
    variance_b: np.ndarray,
    covariance_ab: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    variance = variance_a + variance_b - 2.0 * covariance_ab
    scale = np.maximum.reduce(
        (
            np.abs(variance_a),
            np.abs(variance_b),
            2.0 * np.abs(covariance_ab),
            np.ones_like(variance_a),
        )
    )
    # Permit only negligible floating-point negatives.
    variance = np.where(
        (variance < 0) & (variance >= -1.0e-10 * scale), 0.0, variance
    )
    sd = np.sqrt(np.where(variance >= 0, variance, np.nan))
    return variance, sd


def _agreement_metrics(
    median_difference: np.ndarray, difference_error_sd: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    standardized = np.divide(
        np.abs(median_difference),
        difference_error_sd,
        out=np.full_like(median_difference, np.nan, dtype=np.float64),
        where=np.isfinite(difference_error_sd) & (difference_error_sd > 0),
    )
    probability = ndtr(-standardized)
    return standardized, probability, 2.0 * probability


# %% Main agreement calculation
def _build_point_dataset(
    daily: xr.Dataset,
    ec: xr.Dataset,
    minimum_bootstrap_valid_fraction: float,
) -> tuple[xr.Dataset, dict[str, dict[str, Any]]]:
    ny, nx = daily.sizes["lat"], daily.sizes["lon"]
    n_pairs = len(PRODUCT_PAIRS)
    shape = (len(WINDOWS), len(TRANSFORMS), n_pairs, ny, nx)
    float_outputs = {
        name: np.full(shape, np.nan, dtype=np.float32)
        for name in (
            "mean_difference",
            "median_difference",
            "temporal_sd_difference",
            "mean_absolute_daily_difference",
            "median_absolute_daily_difference",
            "absolute_median_difference",
            "ec_difference_error_variance",
            "ec_difference_error_sd",
            "ec_error_covariance_used",
            "standardized_median_difference",
            "agreement_probability",
            "agreement_score",
            "within_1sigma",
            "within_1_96sigma",
            "standardized_mean_difference",
            "agreement_probability_mean_difference",
            "agreement_score_mean_difference",
            "within_1sigma_mean_difference",
            "within_1_96sigma_mean_difference",
            "bootstrap_valid_fraction_used",
        )
    }
    valid_output = np.zeros(shape, dtype=np.uint8)
    window_cache: dict[str, dict[str, Any]] = {}

    all_values = np.stack(
        [np.asarray(daily[name].values, dtype=np.float64) for name in PRODUCTS],
        axis=1,
    )
    all_storm_index = np.asarray(daily["storm_index"].values, dtype=np.int64)
    number_of_storms = daily.sizes["storm"]
    study_domain = np.asarray(daily["study_domain_mask"].values, dtype=bool)
    min_days = int(ec.attrs.get("minimum_valid_days_per_storm", 3))

    for wi, window in enumerate(WINDOWS):
        selected, duplicates = _selected_observations(daily, window)
        selected_values = all_values[selected]
        selected_storm_index = all_storm_index[selected]
        matched, counts, contributing_storms = _matched_sample(
            selected_values,
            selected_storm_index,
            number_of_storms,
            study_domain,
            min_days,
        )
        expected_counts = np.asarray(
            ec["n_observations"].sel(window=window).values, dtype=np.int16
        )
        expected_storms = np.asarray(
            ec["n_contributing_storms"].sel(window=window).values,
            dtype=np.int8,
        )
        if not np.array_equal(counts, expected_counts):
            raise ValueError(
                f"{window}: agreement sample counts do not reproduce the EC counts."
            )
        if not np.array_equal(contributing_storms, expected_storms):
            raise ValueError(
                f"{window}: contributing-storm counts do not reproduce EC."
            )
        window_cache[window] = {
            "values": selected_values,
            "storm_index": selected_storm_index,
            "matched": matched,
            "duplicates_discarded": duplicates,
        }

        for ti, transform in enumerate(TRANSFORMS):
            transformed = (
                np.log1p(selected_values)
                if transform == "log1p"
                else selected_values
            )
            error_variance = np.asarray(
                ec["error_variance"]
                .sel(window=window, transform=transform)
                .values,
                dtype=np.float64,
            )
            model_covariance = np.asarray(
                ec["error_covariance_model_pair"]
                .sel(window=window, transform=transform)
                .values,
                dtype=np.float64,
            )
            primary_valid = np.asarray(
                ec["valid_primary_ec"]
                .sel(window=window, transform=transform)
                .values,
                dtype=bool,
            )
            reference_valid = np.asarray(
                ec["reference_diagnostics_valid"]
                .sel(window=window, transform=transform)
                .values,
                dtype=bool,
            )
            eligible = np.asarray(
                ec["sample_eligible"].sel(window=window).values, dtype=bool
            )
            primary_bootstrap = (
                np.asarray(
                    ec["bootstrap_valid_fraction"]
                    .sel(window=window, transform=transform)
                    .values,
                    dtype=np.float64,
                )
                if "bootstrap_valid_fraction" in ec
                else np.ones((ny, nx), dtype=np.float64)
            )
            reference_bootstrap = (
                np.asarray(
                    ec["bootstrap_reference_valid_fraction"]
                    .sel(window=window, transform=transform)
                    .values,
                    dtype=np.float64,
                )
                if "bootstrap_reference_valid_fraction" in ec
                else primary_bootstrap
            )

            for pi, (product_a, product_b) in enumerate(PRODUCT_PAIRS):
                difference = transformed[:, product_b] - transformed[:, product_a]
                (
                    mean,
                    median,
                    temporal_sd,
                    mean_absolute,
                    median_absolute,
                ) = _nan_summaries(difference, matched)
                primary_pair = (product_a, product_b) == (0, 1)
                if primary_pair:
                    covariance = model_covariance
                    point_valid = primary_valid
                    bootstrap_fraction = primary_bootstrap
                else:
                    # This zero is an EC model assumption, not an empirical
                    # estimate of the cross-error covariance.
                    covariance = np.zeros((ny, nx), dtype=np.float64)
                    point_valid = primary_valid & reference_valid
                    bootstrap_fraction = reference_bootstrap
                pair_variance, pair_sd = _safe_pair_error_sd(
                    error_variance[product_a],
                    error_variance[product_b],
                    covariance,
                )
                standardized, probability, score = _agreement_metrics(
                    median, pair_sd
                )
                standardized_mean, probability_mean, score_mean = (
                    _agreement_metrics(mean, pair_sd)
                )
                valid = (
                    eligible
                    & point_valid
                    & np.isfinite(median)
                    & np.isfinite(pair_sd)
                    & (pair_sd > 0)
                    & np.isfinite(bootstrap_fraction)
                    & (
                        bootstrap_fraction
                        >= minimum_bootstrap_valid_fraction
                    )
                )

                values_by_name = {
                    "mean_difference": mean,
                    "median_difference": median,
                    "temporal_sd_difference": temporal_sd,
                    "mean_absolute_daily_difference": mean_absolute,
                    "median_absolute_daily_difference": median_absolute,
                    "absolute_median_difference": np.abs(median),
                    "ec_difference_error_variance": pair_variance,
                    "ec_difference_error_sd": pair_sd,
                    "ec_error_covariance_used": covariance,
                    "standardized_median_difference": standardized,
                    "agreement_probability": probability,
                    "agreement_score": score,
                    "within_1sigma": (standardized <= 1.0).astype(float),
                    "within_1_96sigma": (standardized <= 1.96).astype(float),
                    "standardized_mean_difference": standardized_mean,
                    "agreement_probability_mean_difference": probability_mean,
                    "agreement_score_mean_difference": score_mean,
                    "within_1sigma_mean_difference": (
                        standardized_mean <= 1.0
                    ).astype(float),
                    "within_1_96sigma_mean_difference": (
                        standardized_mean <= 1.96
                    ).astype(float),
                    "bootstrap_valid_fraction_used": bootstrap_fraction,
                }
                for name, values in values_by_name.items():
                    float_outputs[name][wi, ti, pi] = np.where(
                        valid, values, np.nan
                    ).astype(np.float32)
                valid_output[wi, ti, pi] = valid.astype(np.uint8)

    coords = {
        "window": list(WINDOWS),
        "transform": list(TRANSFORMS),
        "dataset_pair": list(PAIR_IDS),
        "lat": daily.lat.values,
        "lon": daily.lon.values,
        "product_a": (
            "dataset_pair",
            [PRODUCTS[i] for i, _ in PRODUCT_PAIRS],
        ),
        "product_b": (
            "dataset_pair",
            [PRODUCTS[j] for _, j in PRODUCT_PAIRS],
        ),
        "pair_label": (
            "dataset_pair",
            [
                f"{PRODUCT_LABELS[PRODUCTS[i]]} vs {PRODUCT_LABELS[PRODUCTS[j]]}"
                for i, j in PRODUCT_PAIRS
            ],
        ),
        "pair_role": (
            "dataset_pair",
            [
                "primary" if pair_id == PRIMARY_PAIR_ID else "diagnostic"
                for pair_id in PAIR_IDS
            ],
        ),
    }
    dims = ("window", "transform", "dataset_pair", "lat", "lon")
    result = xr.Dataset(
        {name: (dims, values) for name, values in float_outputs.items()},
        coords=coords,
    )
    result["valid_agreement"] = (dims, valid_output)
    result["n_observations"] = ec["n_observations"].copy(deep=True)
    result["n_contributing_storms"] = ec["n_contributing_storms"].copy(
        deep=True
    )
    result["study_domain_mask"] = daily["study_domain_mask"].copy(deep=True)
    result["study_watershed_fraction"] = daily[
        "study_watershed_fraction"
    ].copy(deep=True)
    result["crs"] = xr.DataArray(
        np.int8(0),
        attrs={
            "grid_mapping_name": "latitude_longitude",
            "epsg_code": "EPSG:4326",
            "semi_major_axis": 6378137.0,
            "inverse_flattening": 298.257223563,
        },
    )
    for variable in result.data_vars.values():
        if "lat" in variable.dims and "lon" in variable.dims:
            variable.attrs["grid_mapping"] = "crs"

    result.attrs.update(
        title="Pixelwise dataset agreement during ten hurricane windows",
        difference_definition="product_b minus product_a for matched daily values",
        primary_pair=PRIMARY_PAIR_ID,
        primary_pair_reason=(
            "CONUS404/CONUS404BA is the only pair whose cross-error covariance "
            "is estimated explicitly by the four-system EC."
        ),
        diagnostic_pair_note=(
            "All other pairs assume zero cross-error covariance according to the "
            "EC error groups; MRMS/PERSIANN error variances are diagnostics."
        ),
        agreement_probability_definition=(
            "Phi(-abs(temporal median paired difference) / EC error SD of the "
            "difference); one-sided lower-tail overlap on 0..0.5, not a p-value "
            "or the probability of exact equality."
        ),
        agreement_score_definition=(
            "Two times agreement_probability; normalized to 0..1 with 1 at a "
            "zero median difference relative to the EC difference-error SD."
        ),
        inference_note=(
            "This compares practical discrepancy with daily random-error spread; "
            "it is not a hypothesis test of the temporal median and pixels are "
            "not spatially independent."
        ),
        recommended_extreme_precipitation_metric=(
            "Use agreement_probability_mean_difference as the primary bias/error "
            "comparison because the temporal median is frequently zero or near "
            "zero in daily precipitation. Retain agreement_probability as the "
            "original-notebook-compatible median-bias sensitivity."
        ),
        minimum_bootstrap_valid_fraction=float(
            minimum_bootstrap_valid_fraction
        ),
        minimum_valid_days_per_storm=min_days,
        duplicate_records_discarded=json.dumps(
            {
                window: int(window_cache[window]["duplicates_discarded"])
                for window in WINDOWS
            }
        ),
    )
    result["median_difference"].attrs.update(
        long_name="Temporal median paired daily difference (product_b - product_a)",
        units="mm day-1 for raw; log1p units for log1p",
    )
    result["ec_difference_error_sd"].attrs.update(
        long_name="EC random-error SD of the paired difference",
        units="mm day-1 for raw; log1p units for log1p",
    )
    result["agreement_probability"].attrs.update(
        long_name="Original-notebook one-sided agreement tail probability",
        valid_min=0.0,
        valid_max=0.5,
    )
    result["agreement_score"].attrs.update(
        long_name="Normalized EC agreement score", valid_min=0.0, valid_max=1.0
    )
    result["agreement_probability_mean_difference"].attrs.update(
        long_name="One-sided agreement tail probability based on temporal mean bias",
        valid_min=0.0,
        valid_max=0.5,
    )
    result["agreement_score_mean_difference"].attrs.update(
        long_name="Normalized EC agreement score based on temporal mean bias",
        valid_min=0.0,
        valid_max=1.0,
    )
    return result, window_cache


# %% Leave-one-storm-out agreement for the primary pair
def _build_loso_dataset(
    daily: xr.Dataset,
    ec: xr.Dataset,
    loso_ec: xr.Dataset,
    window_cache: dict[str, dict[str, Any]],
) -> xr.Dataset:
    storm_ids = _decode_strings(daily["storm_id"].values)
    omitted_ids = _decode_strings(loso_ec["omitted_storm"].values)
    if set(storm_ids) != set(omitted_ids):
        raise ValueError("LOSO storm IDs do not match the daily aligned input.")
    shape = (
        len(WINDOWS),
        len(TRANSFORMS),
        len(omitted_ids),
        daily.sizes["lat"],
        daily.sizes["lon"],
    )
    float_outputs = {
        name: np.full(shape, np.nan, dtype=np.float32)
        for name in (
            "mean_difference_ba_minus_conus404",
            "median_difference_ba_minus_conus404",
            "absolute_median_difference",
            "ec_difference_error_sd",
            "standardized_median_difference",
            "agreement_probability",
            "agreement_score",
            "standardized_mean_difference",
            "agreement_probability_mean_difference",
            "agreement_score_mean_difference",
        )
    }
    counts_output = np.zeros(shape, dtype=np.int16)
    valid_output = np.zeros(shape, dtype=np.uint8)

    min_observations = {
        "core": int(ec.attrs.get("minimum_core_observations", 60)),
        "expanded": int(ec.attrs.get("minimum_expanded_observations", 100)),
    }
    for wi, window in enumerate(WINDOWS):
        cache = window_cache[window]
        raw_values = cache["values"]
        storm_index = cache["storm_index"]
        matched = cache["matched"]
        for ti, transform in enumerate(TRANSFORMS):
            values = np.log1p(raw_values) if transform == "log1p" else raw_values
            difference = values[:, 1] - values[:, 0]
            for oi, omitted_id in enumerate(omitted_ids):
                omitted_index = storm_ids.index(omitted_id)
                retained = matched & (storm_index[:, None, None] != omitted_index)
                counts = retained.sum(axis=0).astype(np.int16)
                mean, median, _, _, _ = _nan_summaries(difference, retained)
                sd_a = np.asarray(
                    loso_ec["error_sd_conus404"]
                    .sel(
                        window=window,
                        transform=transform,
                        omitted_storm=omitted_id,
                    )
                    .values,
                    dtype=np.float64,
                )
                sd_b = np.asarray(
                    loso_ec["error_sd_conus404ba"]
                    .sel(
                        window=window,
                        transform=transform,
                        omitted_storm=omitted_id,
                    )
                    .values,
                    dtype=np.float64,
                )
                rho = np.asarray(
                    loso_ec["error_correlation_model_pair"]
                    .sel(
                        window=window,
                        transform=transform,
                        omitted_storm=omitted_id,
                    )
                    .values,
                    dtype=np.float64,
                )
                _, pair_sd = _safe_pair_error_sd(
                    sd_a**2, sd_b**2, rho * sd_a * sd_b
                )
                standardized, probability, score = _agreement_metrics(
                    median, pair_sd
                )
                standardized_mean, probability_mean, score_mean = (
                    _agreement_metrics(mean, pair_sd)
                )
                valid = (
                    np.asarray(
                        loso_ec["valid_primary_ec"]
                        .sel(
                            window=window,
                            transform=transform,
                            omitted_storm=omitted_id,
                        )
                        .values,
                        dtype=bool,
                    )
                    & (counts >= min_observations[window])
                    & np.isfinite(median)
                    & np.isfinite(pair_sd)
                    & (pair_sd > 0)
                )
                values_by_name = {
                    "mean_difference_ba_minus_conus404": mean,
                    "median_difference_ba_minus_conus404": median,
                    "absolute_median_difference": np.abs(median),
                    "ec_difference_error_sd": pair_sd,
                    "standardized_median_difference": standardized,
                    "agreement_probability": probability,
                    "agreement_score": score,
                    "standardized_mean_difference": standardized_mean,
                    "agreement_probability_mean_difference": probability_mean,
                    "agreement_score_mean_difference": score_mean,
                }
                for name, output_values in values_by_name.items():
                    float_outputs[name][wi, ti, oi] = np.where(
                        valid, output_values, np.nan
                    ).astype(np.float32)
                counts_output[wi, ti, oi] = counts
                valid_output[wi, ti, oi] = valid.astype(np.uint8)

    dims = ("window", "transform", "omitted_storm", "lat", "lon")
    result = xr.Dataset(
        {name: (dims, values) for name, values in float_outputs.items()},
        coords={
            "window": list(WINDOWS),
            "transform": list(TRANSFORMS),
            "omitted_storm": omitted_ids,
            "lat": daily.lat.values,
            "lon": daily.lon.values,
        },
        attrs={
            "title": "Leave-one-storm-out agreement: CONUS404 versus CONUS404BA",
            "difference_definition": "CONUS404BA minus CONUS404",
            "interpretation": (
                "Use the spread across omitted storms as a stability diagnostic; "
                "storm omissions are not independent spatial realizations."
            ),
        },
    )
    result["n_observations"] = (dims, counts_output)
    result["valid_agreement"] = (dims, valid_output)
    result["crs"] = xr.DataArray(
        np.int8(0),
        attrs={"grid_mapping_name": "latitude_longitude", "epsg_code": "EPSG:4326"},
    )
    for variable in result.data_vars.values():
        if "lat" in variable.dims and "lon" in variable.dims:
            variable.attrs["grid_mapping"] = "crs"
    return result


# %% Summaries, NetCDF writing, and figures
def _area_summary(result: xr.Dataset) -> pd.DataFrame:
    latitude_weight = np.cos(np.deg2rad(result.lat.values))[:, None]
    domain_weight = (
        np.asarray(result["study_watershed_fraction"].values, dtype=np.float64)
        * latitude_weight
    )
    rows: list[dict[str, Any]] = []
    for window in WINDOWS:
        for transform in TRANSFORMS:
            for pair_id in PAIR_IDS:
                subset = result.sel(
                    window=window, transform=transform, dataset_pair=pair_id
                )
                valid = np.asarray(subset["valid_agreement"].values, dtype=bool)
                weights = np.where(valid, domain_weight, 0.0)
                weight_sum = weights.sum()

                def weighted_mean(variable: str) -> float:
                    values = np.asarray(subset[variable].values, dtype=np.float64)
                    return (
                        float(np.nansum(values * weights) / weight_sum)
                        if weight_sum > 0
                        else np.nan
                    )

                rows.append(
                    {
                        "window": window,
                        "transform": transform,
                        "dataset_pair": pair_id,
                        "pair_label": str(subset.pair_label.values),
                        "pair_role": str(subset.pair_role.values),
                        "valid_pixels": int(valid.sum()),
                        "valid_domain_area_fraction": (
                            float(weight_sum / domain_weight.sum())
                            if domain_weight.sum() > 0
                            else np.nan
                        ),
                        "area_weighted_agreement_probability": weighted_mean(
                            "agreement_probability"
                        ),
                        "area_weighted_agreement_score": weighted_mean(
                            "agreement_score"
                        ),
                        "area_weighted_mean_bias_agreement_probability": weighted_mean(
                            "agreement_probability_mean_difference"
                        ),
                        "area_weighted_mean_bias_agreement_score": weighted_mean(
                            "agreement_score_mean_difference"
                        ),
                        "area_fraction_within_1sigma": weighted_mean(
                            "within_1sigma"
                        ),
                        "area_fraction_within_1_96sigma": weighted_mean(
                            "within_1_96sigma"
                        ),
                        "area_fraction_mean_bias_within_1sigma": weighted_mean(
                            "within_1sigma_mean_difference"
                        ),
                        "area_fraction_mean_bias_within_1_96sigma": weighted_mean(
                            "within_1_96sigma_mean_difference"
                        ),
                        "area_weighted_signed_mean_difference": weighted_mean(
                            "mean_difference"
                        ),
                        "area_weighted_signed_median_difference": weighted_mean(
                            "median_difference"
                        ),
                        "area_weighted_standardized_difference": weighted_mean(
                            "standardized_median_difference"
                        ),
                        "area_weighted_standardized_mean_difference": weighted_mean(
                            "standardized_mean_difference"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _loso_area_summary(
    loso_result: xr.Dataset, daily: xr.Dataset
) -> pd.DataFrame:
    latitude_weight = np.cos(np.deg2rad(daily.lat.values))[:, None]
    domain_weight = (
        np.asarray(daily["study_watershed_fraction"].values, dtype=np.float64)
        * latitude_weight
    )
    rows: list[dict[str, Any]] = []
    for window in WINDOWS:
        for transform in TRANSFORMS:
            for storm_id in _decode_strings(loso_result.omitted_storm.values):
                subset = loso_result.sel(
                    window=window,
                    transform=transform,
                    omitted_storm=storm_id,
                )
                valid = np.asarray(subset.valid_agreement.values, dtype=bool)
                weights = np.where(valid, domain_weight, 0.0)
                weight_sum = weights.sum()

                def weighted_mean(variable: str) -> float:
                    values = np.asarray(subset[variable].values, dtype=np.float64)
                    return (
                        float(np.nansum(values * weights) / weight_sum)
                        if weight_sum > 0
                        else np.nan
                    )

                rows.append(
                    {
                        "window": window,
                        "transform": transform,
                        "omitted_storm": storm_id,
                        "valid_pixels": int(valid.sum()),
                        "valid_domain_area_fraction": (
                            float(weight_sum / domain_weight.sum())
                            if domain_weight.sum() > 0
                            else np.nan
                        ),
                        "area_weighted_mean_bias_agreement_probability": weighted_mean(
                            "agreement_probability_mean_difference"
                        ),
                        "area_weighted_mean_bias_agreement_score": weighted_mean(
                            "agreement_score_mean_difference"
                        ),
                        "area_weighted_signed_mean_difference": weighted_mean(
                            "mean_difference_ba_minus_conus404"
                        ),
                        "area_weighted_standardized_mean_difference": weighted_mean(
                            "standardized_mean_difference"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _write_netcdf(dataset: xr.Dataset, path: Path) -> None:
    encoding = {
        name: (
            {"zlib": True, "complevel": 4, "shuffle": True}
            if variable.ndim > 0
            else {}
        )
        for name, variable in dataset.data_vars.items()
    }
    dataset.to_netcdf(path, engine="netcdf4", encoding=encoding)


def _primary_quicklook(
    result: xr.Dataset,
    variable: str,
    path: Path,
    *,
    cmap: str,
    label_raw: str,
    label_log: str | None = None,
    fixed_limits: tuple[float, float] | None = None,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    artists: list[Any] = [None, None]
    subset_pair = result.sel(dataset_pair=PRIMARY_PAIR_ID)
    limits_by_transform: dict[str, tuple[float, float]] = {}
    if fixed_limits is None:
        for transform in TRANSFORMS:
            pooled = subset_pair[variable].sel(transform=transform).where(
                subset_pair["valid_agreement"].sel(transform=transform) > 0
            )
            finite = np.asarray(pooled.values, dtype=float)
            finite = finite[np.isfinite(finite)]
            if variable in {"median_difference", "mean_difference"}:
                bound = (
                    float(np.nanmax(np.abs(finite))) if finite.size else 1.0
                )
                bound = max(bound, 1.0e-8)
                limits_by_transform[transform] = (-bound, bound)
            else:
                upper = float(np.nanmax(finite)) if finite.size else 1.0
                limits_by_transform[transform] = (0.0, max(upper, 1.0e-8))
    for wi, window in enumerate(WINDOWS):
        for ti, transform in enumerate(TRANSFORMS):
            ax = axes[wi, ti]
            subset = subset_pair.sel(window=window, transform=transform)
            values = subset[variable].where(subset["valid_agreement"] > 0)
            if fixed_limits is not None:
                vmin, vmax = fixed_limits
            else:
                vmin, vmax = limits_by_transform[transform]
            artist = ax.pcolormesh(
                result.lon,
                result.lat,
                values,
                shading="auto",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            add_map_boundaries(ax, watershed_scope="analysis")
            ax.set_title(f"{window.capitalize()} | {transform}")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            artists[ti] = artist
    if fixed_limits is not None and artists[0] is not None:
        figure.colorbar(artists[0], ax=axes, shrink=0.87, label=label_raw)
    else:
        labels = (label_raw, label_log or label_raw)
        for ti, artist in enumerate(artists):
            if artist is not None:
                figure.colorbar(
                    artist, ax=axes[:, ti], shrink=0.87, label=labels[ti]
                )
    figure.suptitle(
        "Dataset agreement: CONUS404 versus CONUS404BA", fontsize=15
    )
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _all_pair_quicklook(
    result: xr.Dataset,
    window: str,
    transform: str,
    path: Path,
    *,
    variable: str,
    subtitle: str,
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    artist = None
    for ax, pair_id in zip(axes.ravel(), PAIR_IDS):
        subset = result.sel(
            window=window, transform=transform, dataset_pair=pair_id
        )
        values = subset[variable].where(
            subset["valid_agreement"] > 0
        )
        artist = ax.pcolormesh(
            result.lon,
            result.lat,
            values,
            shading="auto",
            cmap="GnBu",
            vmin=0.0,
            vmax=0.5,
        )
        add_map_boundaries(ax, watershed_scope="analysis")
        role = str(subset.pair_role.values)
        ax.set_title(f"{str(subset.pair_label.values)}\n({role})")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    if artist is not None:
        figure.colorbar(
            artist,
            ax=axes,
            shrink=0.88,
            label="One-sided agreement tail probability",
        )
    figure.suptitle(
        f"{subtitle} | {window.capitalize()} | {transform}", fontsize=15
    )
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


# %% Public workflow
def run_agreement_analysis(
    daily_input: Path | str = DEFAULT_DAILY_INPUT,
    ec_input: Path | str = DEFAULT_EC_INPUT,
    loso_input: Path | str | None = DEFAULT_LOSO_INPUT,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    *,
    minimum_bootstrap_valid_fraction: float = 0.90,
    do_loso: bool = True,
) -> dict[str, Path]:
    """Calculate gridded agreement metrics and return generated artifact paths."""
    daily_input, ec_input = Path(daily_input), Path(ec_input)
    output_dir = Path(output_dir)
    if not 0.0 <= minimum_bootstrap_valid_fraction <= 1.0:
        raise ValueError("minimum_bootstrap_valid_fraction must be in [0, 1].")
    for path in (daily_input, ec_input):
        if not path.exists():
            raise FileNotFoundError(path)
    output_dir.mkdir(parents=True, exist_ok=True)

    daily = xr.open_dataset(daily_input, decode_times=True)
    ec = xr.open_dataset(ec_input, decode_times=True)
    _require_inputs(daily, ec)
    result, window_cache = _build_point_dataset(
        daily, ec, minimum_bootstrap_valid_fraction
    )

    outputs: dict[str, Path] = {}
    if do_loso:
        if loso_input is None:
            raise ValueError("do_loso=True requires a LOSO EC input file.")
        loso_path = Path(loso_input)
        if not loso_path.exists():
            raise FileNotFoundError(loso_path)
        loso_ec = xr.open_dataset(loso_path, decode_times=True)
        loso_result = _build_loso_dataset(
            daily, ec, loso_ec, window_cache
        )
        loso_output = output_dir / "gridded_dataset_agreement_leave_one_storm_out.nc"
        _write_netcdf(loso_result, loso_output)
        outputs["leave_one_storm_out"] = loso_output
        loso_summary_path = (
            output_dir
            / "gridded_dataset_agreement_leave_one_storm_out_area_summary.csv"
        )
        _loso_area_summary(loso_result, daily).to_csv(
            loso_summary_path, index=False
        )
        outputs["leave_one_storm_out_area_summary"] = loso_summary_path
        valid_loso = loso_result["agreement_probability"].where(
            loso_result["valid_agreement"] > 0
        )
        valid_loso_mean = loso_result[
            "agreement_probability_mean_difference"
        ].where(loso_result["valid_agreement"] > 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            result["primary_loso_agreement_probability_median"] = (
                valid_loso.median("omitted_storm", skipna=True)
            )
            result["primary_loso_agreement_probability_std"] = valid_loso.std(
                "omitted_storm", skipna=True, ddof=1
            )
            result[
                "primary_loso_mean_bias_agreement_probability_median"
            ] = valid_loso_mean.median("omitted_storm", skipna=True)
            result[
                "primary_loso_mean_bias_agreement_probability_std"
            ] = valid_loso_mean.std("omitted_storm", skipna=True, ddof=1)
        result["primary_loso_valid_count"] = loso_result[
            "valid_agreement"
        ].sum("omitted_storm")
        loso_ec.close()

    result_path = output_dir / "gridded_dataset_agreement.nc"
    _write_netcdf(result, result_path)
    outputs["results"] = result_path

    summary = _area_summary(result)
    summary_path = output_dir / "gridded_dataset_agreement_area_summary.csv"
    summary.to_csv(summary_path, index=False)
    outputs["area_summary"] = summary_path

    figure_specs = (
        (
            "agreement_probability_mean_difference",
            "primary_mean_bias_agreement_probability.png",
            "GnBu",
            "Mean-bias agreement tail probability",
            None,
            (0.0, 0.5),
        ),
        (
            "agreement_score_mean_difference",
            "primary_mean_bias_agreement_score.png",
            "viridis",
            "Mean-bias normalized agreement score",
            None,
            (0.0, 1.0),
        ),
        (
            "standardized_mean_difference",
            "primary_standardized_mean_difference.png",
            "magma",
            "|mean difference| / EC difference-error SD",
            None,
            (0.0, 1.25),
        ),
        (
            "mean_difference",
            "primary_mean_difference.png",
            "RdBu_r",
            "Mean BA - CONUS404 (mm day$^{-1}$)",
            "Mean BA - CONUS404 (log1p units)",
            None,
        ),
        (
            "agreement_probability",
            "primary_median_bias_agreement_probability.png",
            "GnBu",
            "One-sided agreement tail probability",
            None,
            (0.0, 0.5),
        ),
        (
            "agreement_score",
            "primary_median_bias_agreement_score.png",
            "viridis",
            "Normalized agreement score",
            None,
            (0.0, 1.0),
        ),
        (
            "standardized_median_difference",
            "primary_standardized_median_difference.png",
            "magma",
            "|median difference| / EC difference-error SD",
            None,
            (0.0, 0.75),
        ),
        (
            "median_difference",
            "primary_median_difference.png",
            "RdBu_r",
            "Median BA - CONUS404 (mm day$^{-1}$)",
            "Median BA - CONUS404 (log1p units)",
            None,
        ),
    )
    for variable, filename, cmap, label_raw, label_log, limits in figure_specs:
        path = output_dir / filename
        _primary_quicklook(
            result,
            variable,
            path,
            cmap=cmap,
            label_raw=label_raw,
            label_log=label_log,
            fixed_limits=limits,
        )
        outputs[variable + "_figure"] = path

    for window in WINDOWS:
        for transform in TRANSFORMS:
            for variable, tag, subtitle in (
                (
                    "agreement_probability_mean_difference",
                    "mean_bias_agreement",
                    "All product pairs: mean-bias agreement",
                ),
                (
                    "agreement_probability",
                    "median_bias_agreement",
                    "All product pairs: median-bias agreement sensitivity",
                ),
            ):
                path = output_dir / f"all_pairs_{window}_{transform}_{tag}.png"
                _all_pair_quicklook(
                    result,
                    window,
                    transform,
                    path,
                    variable=variable,
                    subtitle=subtitle,
                )
                outputs[f"all_pairs_{window}_{transform}_{tag}_figure"] = path

    manifest = {
        "configuration": {
            "daily_input": str(daily_input.resolve()),
            "ec_input": str(ec_input.resolve()),
            "loso_input": str(Path(loso_input).resolve()) if do_loso else None,
            "minimum_bootstrap_valid_fraction": minimum_bootstrap_valid_fraction,
            "primary_pair": PRIMARY_PAIR_ID,
            "diagnostic_pairs": [
                pair_id for pair_id in PAIR_IDS if pair_id != PRIMARY_PAIR_ID
            ],
        },
        "outputs": {name: str(path) for name, path in outputs.items()},
        "valid_pixel_counts": {
            f"{window}_{transform}_{pair_id}": int(
                result["valid_agreement"]
                .sel(
                    window=window,
                    transform=transform,
                    dataset_pair=pair_id,
                )
                .sum()
            )
            for window in WINDOWS
            for transform in TRANSFORMS
            for pair_id in PAIR_IDS
        },
    }
    manifest_path = output_dir / "gridded_dataset_agreement_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    outputs["manifest"] = manifest_path

    daily.close()
    ec.close()
    print("Finished gridded dataset-agreement analysis:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")
    return outputs


# %% Command-line entry point
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily-input", type=Path, default=DEFAULT_DAILY_INPUT)
    parser.add_argument("--ec-input", type=Path, default=DEFAULT_EC_INPUT)
    parser.add_argument("--loso-input", type=Path, default=DEFAULT_LOSO_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--minimum-bootstrap-valid-fraction", type=float, default=0.90
    )
    parser.add_argument("--no-loso", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    run_agreement_analysis(
        daily_input=arguments.daily_input,
        ec_input=arguments.ec_input,
        loso_input=arguments.loso_input,
        output_dir=arguments.output_dir,
        minimum_bootstrap_valid_fraction=(
            arguments.minimum_bootstrap_valid_fraction
        ),
        do_loso=not arguments.no_loso,
    )
