"""
Pixelwise four-system Extended Collocation (EC) on the PERSIANN-CDR grid.

This module is deliberately independent of the older notebook functions.  It
expects one *already aligned* NetCDF containing the four daily precipitation
fields ``conus404``, ``conus404ba``, ``mrms`` and ``persiann_cdr`` on
``(observation, lat, lon)``.  CONUS404 and CONUS404BA are treated as the one
allowed correlated-error pair (the EC equivalent of corr_sets=[1, 1, 2, 3]).

The file contains ``# %%`` cells so it can be opened directly in VS Code or
imported into a Jupyter notebook.  The public entry point is ``run_analysis``.
Run ``python run_gridded_extended_collocation.py --help`` for CLI options.
"""

# %% Imports
from __future__ import annotations

from study_paths import PROJECT_ROOT, RESULTS_ROOT, METADATA_ROOT, REPO_ROOT, STATES_FILE, WATERSHEDS_FILE, DAILY_INPUT, EC_DIRECTORY, EC_INPUT, LOSO_INPUT, AGREEMENT_DIRECTORY, AGREEMENT_INPUT


import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from map_overlays import add_map_boundaries


# %% Configuration
PAPER_ROOT = (PROJECT_ROOT)
DEFAULT_INPUT = DAILY_INPUT
DEFAULT_OUTPUT_DIR = EC_DIRECTORY

PRODUCTS = ("conus404", "conus404ba", "mrms", "persiann_cdr")
MODEL_PRODUCTS = ("conus404", "conus404ba")
WINDOWS = ("core", "expanded")
TRANSFORMS = ("raw", "log1p")
SAMPLE_MODES = ("all_windows", "assigned_storms")
DENOMINATOR_RTOL = 1.0e-8

# invalid_reason_code is a bit mask, so more than one warning can be retained.
INVALID_FLAGS = {
    1: "outside_study_domain",
    2: "insufficient_observations",
    4: "insufficient_storms",
    8: "nonfinite_covariance",
    16: "invalid_mrms_persiann_denominator",
    32: "nonpositive_model_error_variance",
    64: "non_psd_model_error_block",
    128: "negative_model_signal_variance",
    256: "invalid_reference_error_diagnostic",
}

PRODUCT_METRICS = (
    "error_variance",
    "error_sd",
    "signal_variance",
    "snr",
    "snr_db",
    "estimated_r2_to_truth",
    "estimated_r_to_truth",
)

MAP_METRICS = (
    "covariance_mrms_persiann",
    "correlation_mrms_persiann",
    "error_covariance_model_pair",
    "error_covariance_direction_01",
    "error_covariance_direction_10",
    "error_covariance_asymmetry",
    "error_correlation_model_pair",
    "model_error_min_eigenvalue",
    "model_error_determinant",
    "error_sd_difference_ba_minus_conus404",
    "error_sd_ratio_ba_over_conus404",
    "snr_db_difference_ba_minus_conus404",
    "estimated_r2_difference_ba_minus_conus404",
    "estimated_r_difference_ba_minus_conus404",
)


# %% Small numerical helpers
def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Elementwise division with NaN where the ratio is undefined."""
    numerator, denominator = np.broadcast_arrays(numerator, denominator)
    out = np.full(numerator.shape, np.nan, dtype=np.float64)
    np.divide(
        numerator,
        denominator,
        out=out,
        where=np.isfinite(numerator)
        & np.isfinite(denominator)
        & (denominator != 0),
    )
    return out


def _safe_sqrt(values: np.ndarray) -> np.ndarray:
    out = np.full(np.shape(values), np.nan, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    np.sqrt(values, out=out, where=np.isfinite(values) & (values >= 0))
    return out


def _valid_positive_cross(
    cross: np.ndarray,
    variance_a: np.ndarray,
    variance_b: np.ndarray,
    rtol: float = DENOMINATOR_RTOL,
) -> np.ndarray:
    scale = _safe_sqrt(np.maximum(variance_a, 0) * np.maximum(variance_b, 0))
    return (
        np.isfinite(cross)
        & np.isfinite(scale)
        & (scale > 0)
        & (cross > rtol * scale)
    )


def _metrics_from_covariance(
    covariance: np.ndarray,
    denominator_rtol: float = DENOMINATOR_RTOL,
) -> dict[str, np.ndarray]:
    """
    Vectorized EC metrics from covariance matrices with trailing shape (4, 4).

    The first two error variances and their error covariance are the primary
    four-system EC estimates.  MRMS and PERSIANN-CDR error variances are
    diagnostics: each is calculated once using CONUS404 and once using
    CONUS404BA, then the two finite formulations are averaged only when both
    are numerically supported.
    """
    c = np.asarray(covariance, dtype=np.float64)
    if c.shape[-2:] != (4, 4):
        raise ValueError("covariance must have trailing dimensions (4, 4).")

    finite_covariance = np.isfinite(c).all(axis=(-2, -1))
    variance = np.diagonal(c, axis1=-2, axis2=-1)
    denominator = c[..., 2, 3]
    denominator_valid = _valid_positive_cross(
        denominator, variance[..., 2], variance[..., 3], denominator_rtol
    )

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        e0 = c[..., 0, 0] - c[..., 0, 2] * c[..., 0, 3] / denominator
        e1 = c[..., 1, 1] - c[..., 1, 2] * c[..., 1, 3] / denominator
        e01a = c[..., 0, 1] - c[..., 0, 2] * c[..., 1, 3] / denominator
        e01b = c[..., 1, 0] - c[..., 1, 2] * c[..., 0, 3] / denominator

    # Diagnostic error variance of MRMS.  C03 and C13 are the alternative
    # denominators obtained from the C404/PERS and BA/PERS triplets.
    valid_e2a = _valid_positive_cross(
        c[..., 0, 3], variance[..., 0], variance[..., 3], denominator_rtol
    )
    valid_e2b = _valid_positive_cross(
        c[..., 1, 3], variance[..., 1], variance[..., 3], denominator_rtol
    )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        e2a = c[..., 2, 2] - c[..., 2, 0] * c[..., 2, 3] / c[..., 0, 3]
        e2b = c[..., 2, 2] - c[..., 2, 1] * c[..., 2, 3] / c[..., 1, 3]

    # Diagnostic error variance of PERSIANN-CDR.  C02 and C12 are the
    # alternative denominators from the C404/MRMS and BA/MRMS triplets.
    valid_e3a = _valid_positive_cross(
        c[..., 0, 2], variance[..., 0], variance[..., 2], denominator_rtol
    )
    valid_e3b = _valid_positive_cross(
        c[..., 1, 2], variance[..., 1], variance[..., 2], denominator_rtol
    )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        e3a = c[..., 3, 3] - c[..., 3, 0] * c[..., 3, 2] / c[..., 0, 2]
        e3b = c[..., 3, 3] - c[..., 3, 1] * c[..., 3, 2] / c[..., 1, 2]

    reference_formulations_supported = valid_e2a & valid_e2b & valid_e3a & valid_e3b
    e2 = np.where(reference_formulations_supported, (e2a + e2b) / 2.0, np.nan)
    e3 = np.where(reference_formulations_supported, (e3a + e3b) / 2.0, np.nan)

    denominator_mask = denominator_valid & finite_covariance
    e0 = np.where(denominator_mask, e0, np.nan)
    e1 = np.where(denominator_mask, e1, np.nan)
    e01a = np.where(denominator_mask, e01a, np.nan)
    e01b = np.where(denominator_mask, e01b, np.nan)
    e01 = (e01a + e01b) / 2.0
    error_variance = np.stack((e0, e1, e2, e3), axis=-1)

    # Closed-form smallest eigenvalue of the symmetric 2 x 2 model-error block.
    discriminant = _safe_sqrt((e0 - e1) ** 2 + 4.0 * e01**2)
    minimum_eigenvalue = (e0 + e1 - discriminant) / 2.0
    determinant = e0 * e1 - e01**2
    scale = np.maximum.reduce(
        (np.abs(e0), np.abs(e1), np.abs(e01), np.ones_like(e0))
    )
    block_psd = np.isfinite(minimum_eigenvalue) & (
        minimum_eigenvalue >= -1.0e-10 * scale
    )
    model_variances_positive = (e0 > 0) & (e1 > 0)

    signal_variance = variance - error_variance
    reference_diagnostics_valid = (
        reference_formulations_supported
        & np.isfinite(e2)
        & np.isfinite(e3)
        & (e2 > 0)
        & (e3 > 0)
        & (signal_variance[..., 2] >= 0)
        & (signal_variance[..., 3] >= 0)
    )
    model_signals_nonnegative = (signal_variance[..., 0] >= 0) & (
        signal_variance[..., 1] >= 0
    )
    primary_valid = (
        finite_covariance
        & denominator_valid
        & model_variances_positive
        & block_psd
        & model_signals_nonnegative
    )

    error_sd = _safe_sqrt(error_variance)
    snr = _safe_divide(signal_variance, error_variance)
    snr_db = np.full_like(snr, np.nan)
    np.log10(snr, out=snr_db, where=np.isfinite(snr) & (snr > 0))
    snr_db *= 10.0
    r2 = _safe_divide(signal_variance, variance)
    r = np.where((r2 >= 0) & (r2 <= 1), _safe_sqrt(r2), np.nan)
    error_correlation = _safe_divide(e01, _safe_sqrt(e0 * e1))
    denominator_scale = _safe_sqrt(
        np.maximum(variance[..., 2], 0) * np.maximum(variance[..., 3], 0)
    )

    return {
        "finite_covariance": finite_covariance,
        "denominator_valid": denominator_valid,
        "reference_diagnostics_valid": reference_diagnostics_valid,
        "model_variances_positive": model_variances_positive,
        "model_error_block_psd": block_psd,
        "model_signals_nonnegative": model_signals_nonnegative,
        "valid_primary_ec": primary_valid,
        "covariance_mrms_persiann": denominator,
        "correlation_mrms_persiann": _safe_divide(
            denominator, denominator_scale
        ),
        "error_variance": error_variance,
        "error_sd": error_sd,
        "signal_variance": signal_variance,
        "snr": snr,
        "snr_db": snr_db,
        "estimated_r2_to_truth": r2,
        "estimated_r_to_truth": r,
        "error_covariance_model_pair": e01,
        "error_covariance_direction_01": e01a,
        "error_covariance_direction_10": e01b,
        "error_covariance_asymmetry": np.abs(e01a - e01b),
        "error_correlation_model_pair": error_correlation,
        "model_error_min_eigenvalue": minimum_eigenvalue,
        "model_error_determinant": determinant,
        "error_sd_difference_ba_minus_conus404": (
            error_sd[..., 1] - error_sd[..., 0]
        ),
        "error_sd_ratio_ba_over_conus404": _safe_divide(
            error_sd[..., 1], error_sd[..., 0]
        ),
        "snr_db_difference_ba_minus_conus404": (
            snr_db[..., 1] - snr_db[..., 0]
        ),
        "estimated_r2_difference_ba_minus_conus404": (
            r2[..., 1] - r2[..., 0]
        ),
        "estimated_r_difference_ba_minus_conus404": r[..., 1] - r[..., 0],
    }


# %% Sufficient statistics
def _storm_sufficient_statistics(
    values: np.ndarray,
    base_mask: np.ndarray,
    storm_index: np.ndarray,
    number_of_storms: int,
    min_days_per_storm: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return per-storm n, sum(x), sum(xx') after the per-storm day rule."""
    observations, variables, pixels = values.shape
    if variables != 4 or base_mask.shape != (observations, pixels):
        raise ValueError("Unexpected value or mask dimensions.")

    preliminary_counts = np.zeros((number_of_storms, pixels), dtype=np.int32)
    for storm in range(number_of_storms):
        preliminary_counts[storm] = base_mask[storm_index == storm].sum(axis=0)
    qualifying_storm = preliminary_counts >= min_days_per_storm
    final_mask = base_mask & qualifying_storm[storm_index]
    safe_values = np.where(final_mask[:, None, :], values, 0.0)

    counts = np.zeros((number_of_storms, pixels), dtype=np.int32)
    sums = np.zeros((number_of_storms, 4, pixels), dtype=np.float64)
    sum_cross = np.zeros((number_of_storms, 4, 4, pixels), dtype=np.float64)
    for storm in range(number_of_storms):
        rows = storm_index == storm
        counts[storm] = final_mask[rows].sum(axis=0)
        block = safe_values[rows]
        sums[storm] = block.sum(axis=0)
        sum_cross[storm] = np.einsum(
            "oip,ojp->ijp", block, block, optimize=True
        )
    return counts, sums, sum_cross, qualifying_storm, final_mask


def _covariance_from_statistics(
    count: np.ndarray, sums: np.ndarray, sum_cross: np.ndarray
) -> np.ndarray:
    """Convert sufficient statistics to sample covariance (..., 4, 4)."""
    count = np.asarray(count, dtype=np.float64)
    # Inputs use (..., 4, pixel) and (..., 4, 4, pixel).  Move pixel before
    # variable axes so the numerical core always ends with (4, 4).
    sums_last = np.moveaxis(sums, -1, -2)
    cross_last = np.moveaxis(sum_cross, -1, -3)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        centered = cross_last - (
            sums_last[..., :, None] * sums_last[..., None, :]
        ) / count[..., None, None]
        covariance = centered / (count[..., None, None] - 1.0)
    return np.where((count > 1)[..., None, None], covariance, np.nan)


def _point_statistics(
    counts: np.ndarray, sums: np.ndarray, sum_cross: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    total_count = counts.sum(axis=0)
    total_sums = sums.sum(axis=0)
    total_cross = sum_cross.sum(axis=0)
    covariance = _covariance_from_statistics(total_count, total_sums, total_cross)
    contributing_storms = (counts > 0).sum(axis=0)
    return total_count, contributing_storms, covariance, total_sums


# %% Input validation and observation selection
def _require_input_schema(ds: xr.Dataset, sample_mode: str) -> None:
    required_variables = set(PRODUCTS) | {
        "storm_index",
        "storm_id",
        "date",
        "is_core_day",
        "study_domain_mask",
    }
    if sample_mode == "assigned_storms":
        required_variables.add("assigned_storm_mask")
    missing = sorted(name for name in required_variables if name not in ds)
    if missing:
        raise ValueError(f"Combined NetCDF is missing required variables: {missing}")
    for name in PRODUCTS:
        if set(ds[name].dims) != {"observation", "lat", "lon"}:
            raise ValueError(
                f"{name!r} must use exactly (observation, lat, lon); "
                f"found {ds[name].dims}."
            )
    if ds["study_domain_mask"].dims != ("lat", "lon"):
        raise ValueError("study_domain_mask must have dimensions (lat, lon).")
    if sample_mode == "assigned_storms" and set(
        ds["assigned_storm_mask"].dims
    ) != {"storm", "lat", "lon"}:
        raise ValueError(
            "assigned_storm_mask must have dimensions (storm, lat, lon)."
        )


def _decode_strings(values: np.ndarray) -> list[str]:
    decoded: list[str] = []
    for value in np.asarray(values).tolist():
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        decoded.append(str(value))
    return decoded


def _observation_indices(
    ds: xr.Dataset, sample_mode: str, window: str
) -> tuple[np.ndarray, int]:
    """Select observations, applying mandatory date de-duplication if needed."""
    dates = pd.DatetimeIndex(pd.to_datetime(ds["date"].values))
    keep = np.ones(len(dates), dtype=bool)
    if window == "core":
        keep &= np.asarray(ds["is_core_day"].values, dtype=bool)
    duplicates_discarded = 0
    if sample_mode == "all_windows":
        # Input observation order is the declared EVENTS order.  This avoids
        # double counting an overlapping calendar date while retaining its
        # storm block assignment from the first event in that order.
        selected_positions = np.flatnonzero(keep)
        selected_dates = dates[selected_positions]
        duplicated_within_window = selected_dates.duplicated(keep="first")
        if duplicated_within_window.any():
            # De-duplication is legitimate only for true copies.  Refuse to
            # choose silently if a future event archive contains conflicting
            # precipitation or common-validity fields on the same date.
            duplicate_dates = selected_dates[duplicated_within_window].unique()
            compare_variables = list(PRODUCTS)
            if "valid_all4" in ds:
                compare_variables.append("valid_all4")
            for duplicate_date in duplicate_dates:
                positions = selected_positions[selected_dates == duplicate_date]
                reference_position = int(positions[0])
                for other_position in positions[1:]:
                    for variable_name in compare_variables:
                        reference = np.asarray(
                            ds[variable_name]
                            .isel(observation=reference_position)
                            .values
                        )
                        candidate = np.asarray(
                            ds[variable_name]
                            .isel(observation=int(other_position))
                            .values
                        )
                        if not np.allclose(
                            reference,
                            candidate,
                            rtol=0.0,
                            atol=0.0,
                            equal_nan=True,
                        ):
                            raise ValueError(
                                "Duplicate calendar records are not identical: "
                                f"{pd.Timestamp(duplicate_date):%Y-%m-%d}, "
                                f"variable {variable_name!r}, observations "
                                f"{reference_position} and {int(other_position)}."
                            )
        keep[selected_positions[duplicated_within_window]] = False
        duplicates_discarded = int(duplicated_within_window.sum())
    return np.flatnonzero(keep), duplicates_discarded


def _stack_spatial(array: xr.DataArray, leading: tuple[str, ...]) -> xr.DataArray:
    return array.transpose(*leading, "lat", "lon").stack(pixel=("lat", "lon"))


# %% Output allocation
def _float_array(shape: tuple[int, ...]) -> np.ndarray:
    return np.full(shape, np.nan, dtype=np.float32)


def _allocate_outputs(
    ny: int, nx: int, number_of_storms: int, do_loso: bool, bootstrap: int
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    wt = (len(WINDOWS), len(TRANSFORMS), ny, nx)
    wtp = (len(WINDOWS), len(TRANSFORMS), len(PRODUCTS), ny, nx)
    main: dict[str, np.ndarray] = {
        metric: _float_array(wtp) for metric in PRODUCT_METRICS
    }
    main.update({metric: _float_array(wt) for metric in MAP_METRICS})
    main.update(
        {
            "valid_primary_ec": np.zeros(wt, dtype=np.uint8),
            "denominator_valid": np.zeros(wt, dtype=np.uint8),
            "reference_diagnostics_valid": np.zeros(wt, dtype=np.uint8),
            "invalid_reason_code": np.zeros(wt, dtype=np.int16),
            "n_observations": np.zeros((len(WINDOWS), ny, nx), dtype=np.int16),
            "n_contributing_storms": np.zeros(
                (len(WINDOWS), ny, nx), dtype=np.int8
            ),
            "sample_eligible": np.zeros((len(WINDOWS), ny, nx), dtype=np.uint8),
        }
    )

    loso: dict[str, np.ndarray] = {}
    if do_loso:
        loso_shape = (
            len(WINDOWS),
            len(TRANSFORMS),
            number_of_storms,
            ny,
            nx,
        )
        loso["error_sd_conus404"] = _float_array(loso_shape)
        loso["error_sd_conus404ba"] = _float_array(loso_shape)
        loso["error_sd_difference_ba_minus_conus404"] = _float_array(
            loso_shape
        )
        loso["error_correlation_model_pair"] = _float_array(loso_shape)
        loso["snr_db_difference_ba_minus_conus404"] = _float_array(loso_shape)
        loso["estimated_r2_difference_ba_minus_conus404"] = _float_array(
            loso_shape
        )
        loso["valid_primary_ec"] = np.zeros(loso_shape, dtype=np.uint8)

    boot: dict[str, np.ndarray] = {}
    if bootstrap > 0:
        ci_product_shape = wtp
        ci_map_shape = wt
        for suffix in ("ci_low", "ci_high"):
            boot[f"error_sd_{suffix}"] = _float_array(ci_product_shape)
            for metric in (
                "error_sd_difference_ba_minus_conus404",
                "error_correlation_model_pair",
                "snr_db_difference_ba_minus_conus404",
                "estimated_r2_difference_ba_minus_conus404",
            ):
                boot[f"{metric}_{suffix}"] = _float_array(ci_map_shape)
        boot["bootstrap_valid_fraction"] = _float_array(ci_map_shape)
        boot["bootstrap_reference_valid_fraction"] = _float_array(ci_map_shape)
    return main, loso, boot


def _assign_product_metric(
    target: np.ndarray,
    window_index: int,
    transform_index: int,
    pixel_indices: np.ndarray,
    values: np.ndarray,
) -> None:
    for product in range(len(PRODUCTS)):
        target[window_index, transform_index, product].reshape(-1)[pixel_indices] = (
            values[:, product].astype(np.float32)
        )


def _assign_map_metric(
    target: np.ndarray,
    window_index: int,
    transform_index: int,
    pixel_indices: np.ndarray,
    values: np.ndarray,
) -> None:
    target[window_index, transform_index].reshape(-1)[pixel_indices] = np.asarray(
        values, dtype=target.dtype
    )


# %% Bootstrap and LOSO calculations
def _bootstrap_summaries(
    weights: np.ndarray,
    counts: np.ndarray,
    sums: np.ndarray,
    sum_cross: np.ndarray,
    eligible: np.ndarray,
) -> dict[str, np.ndarray]:
    """Percentile summaries using identical complete-storm draws per pixel."""
    aggregate_count = weights @ counts.astype(np.float64)
    aggregate_sum = np.einsum("bs,sip->bip", weights, sums, optimize=True)
    aggregate_cross = np.einsum(
        "bs,sijp->bijp", weights, sum_cross, optimize=True
    )
    covariance = _covariance_from_statistics(
        aggregate_count, aggregate_sum, aggregate_cross
    )
    metrics = _metrics_from_covariance(covariance)
    valid = metrics["valid_primary_ec"] & eligible[None, :]
    reference_valid = valid & metrics["reference_diagnostics_valid"]

    def interval(
        values: np.ndarray, interval_mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        values = np.where(interval_mask, values, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            low, high = np.nanpercentile(values, (2.5, 97.5), axis=0)
        return low, high

    summary: dict[str, np.ndarray] = {}
    product_valid = np.broadcast_to(
        valid[..., None], metrics["error_sd"].shape
    ).copy()
    product_valid[..., 2:] = reference_valid[..., None]
    low, high = interval(metrics["error_sd"], product_valid)
    summary["error_sd_ci_low"] = low
    summary["error_sd_ci_high"] = high
    for name in (
        "error_sd_difference_ba_minus_conus404",
        "error_correlation_model_pair",
        "snr_db_difference_ba_minus_conus404",
        "estimated_r2_difference_ba_minus_conus404",
    ):
        low, high = interval(metrics[name], valid)
        summary[f"{name}_ci_low"] = low
        summary[f"{name}_ci_high"] = high
    summary["bootstrap_valid_fraction"] = valid.mean(axis=0)
    summary["bootstrap_reference_valid_fraction"] = reference_valid.mean(axis=0)
    return summary


def _loso_metrics(
    omitted_storm: int,
    counts: np.ndarray,
    sums: np.ndarray,
    sum_cross: np.ndarray,
    outside_domain: np.ndarray,
    min_observations: int,
    min_storms: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    remaining_count = counts.sum(axis=0) - counts[omitted_storm]
    remaining_sums = sums.sum(axis=0) - sums[omitted_storm]
    remaining_cross = sum_cross.sum(axis=0) - sum_cross[omitted_storm]
    remaining_storms = (counts > 0).sum(axis=0) - (
        counts[omitted_storm] > 0
    )
    eligible = (
        ~outside_domain
        & (remaining_count >= min_observations)
        & (remaining_storms >= min_storms)
    )
    covariance = _covariance_from_statistics(
        remaining_count, remaining_sums, remaining_cross
    )
    metrics = _metrics_from_covariance(covariance)
    valid = eligible & metrics["valid_primary_ec"]
    return metrics, valid


# %% NetCDF assembly and plotting
def _dataset_from_outputs(
    ds_input: xr.Dataset,
    main: dict[str, np.ndarray],
    boot: dict[str, np.ndarray],
    attrs: dict[str, Any],
) -> xr.Dataset:
    coords = {
        "window": list(WINDOWS),
        "transform": list(TRANSFORMS),
        "product": list(PRODUCTS),
        "lat": ds_input["lat"].values,
        "lon": ds_input["lon"].values,
    }
    data_vars: dict[str, Any] = {}
    for metric in PRODUCT_METRICS:
        data_vars[metric] = (
            ("window", "transform", "product", "lat", "lon"), main[metric]
        )
    for metric in MAP_METRICS:
        data_vars[metric] = (
            ("window", "transform", "lat", "lon"), main[metric]
        )
    for metric in (
        "valid_primary_ec",
        "denominator_valid",
        "reference_diagnostics_valid",
        "invalid_reason_code",
    ):
        data_vars[metric] = (
            ("window", "transform", "lat", "lon"), main[metric]
        )
    for metric in (
        "n_observations",
        "n_contributing_storms",
        "sample_eligible",
    ):
        data_vars[metric] = (("window", "lat", "lon"), main[metric])
    for name, values in boot.items():
        dims = (
            ("window", "transform", "product", "lat", "lon")
            if name.startswith("error_sd_ci")
            else ("window", "transform", "lat", "lon")
        )
        data_vars[name] = (dims, values)

    clean_attrs = {
        key: (json.dumps(value) if isinstance(value, (dict, list, tuple)) else value)
        for key, value in attrs.items()
    }
    result = xr.Dataset(data_vars=data_vars, coords=coords, attrs=clean_attrs)
    result["crs"] = xr.DataArray(
        np.int8(0),
        attrs={
            "grid_mapping_name": "latitude_longitude",
            "epsg_code": "EPSG:4326",
            "semi_major_axis": 6378137.0,
            "inverse_flattening": 298.257223563,
        },
    )
    for name, variable in result.data_vars.items():
        if "lat" in variable.dims and "lon" in variable.dims:
            variable.attrs["grid_mapping"] = "crs"
    result["error_variance"].attrs.update(
        long_name="EC random-error variance; MRMS/PERSIANN values are diagnostics"
    )
    result["error_sd"].attrs.update(
        long_name="EC random-error standard deviation",
        units="mm day-1 for raw; log1p transformed units for log1p",
    )
    result["error_correlation_model_pair"].attrs.update(
        long_name="Estimated random-error correlation, CONUS404 versus CONUS404BA"
    )
    result["estimated_r2_to_truth"].attrs.update(
        long_name="EC estimated squared correlation to latent truth; not observed R-squared"
    )
    result["invalid_reason_code"].attrs.update(
        long_name="Bit mask of EC sample/solution warnings",
        flag_masks=np.array(sorted(INVALID_FLAGS), dtype=np.int16),
        flag_meanings=" ".join(INVALID_FLAGS[key] for key in sorted(INVALID_FLAGS)),
    )
    result["valid_primary_ec"].attrs.update(
        long_name="One where correlated C404/C404BA EC block is admissible"
    )
    if "bootstrap_valid_fraction" in result:
        result["bootstrap_valid_fraction"].attrs.update(
            long_name=(
                "Fraction of storm-block bootstrap replicates with an admissible "
                "primary CONUS404/CONUS404BA EC solution"
            )
        )
    if "bootstrap_reference_valid_fraction" in result:
        result["bootstrap_reference_valid_fraction"].attrs.update(
            long_name=(
                "Fraction of storm-block bootstrap replicates with admissible "
                "primary EC and MRMS/PERSIANN diagnostic error variances"
            )
        )
    return result


def _loso_dataset(
    ds_input: xr.Dataset,
    loso: dict[str, np.ndarray],
    storm_ids: list[str],
    attrs: dict[str, Any],
) -> xr.Dataset:
    dims = ("window", "transform", "omitted_storm", "lat", "lon")
    result = xr.Dataset(
        {name: (dims, values) for name, values in loso.items()},
        coords={
            "window": list(WINDOWS),
            "transform": list(TRANSFORMS),
            "omitted_storm": storm_ids,
            "lat": ds_input["lat"].values,
            "lon": ds_input["lon"].values,
        },
        attrs={
            key: (json.dumps(value) if isinstance(value, (dict, list, tuple)) else value)
            for key, value in attrs.items()
        },
    )
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
    result.attrs["description"] = (
        "Leave-one-complete-storm-out spatial sensitivity maps. Values are "
        "NaN where the reduced sample or EC solution is inadmissible."
    )
    return result


def _write_netcdf(dataset: xr.Dataset, path: Path) -> None:
    try:
        import h5netcdf  # noqa: F401

        encoding = {
            name: (
                {"compression": "gzip", "compression_opts": 4, "shuffle": True}
                if variable.ndim > 0
                else {}
            )
            for name, variable in dataset.data_vars.items()
        }
        dataset.to_netcdf(path, engine="h5netcdf", encoding=encoding)
    except ImportError:
        dataset.to_netcdf(path)


def _quicklook(
    result: xr.Dataset, metric: str, path: Path, sample_mode: str
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    artists: list[Any] = [None, None]
    for wi, window in enumerate(WINDOWS):
        for ti, transform in enumerate(TRANSFORMS):
            ax = axes[wi, ti]
            values = result[metric].isel(window=wi, transform=ti).where(
                result["valid_primary_ec"].isel(window=wi, transform=ti) > 0
            )
            if metric == "error_sd_difference_ba_minus_conus404":
                transform_values = result[metric].isel(transform=ti).where(
                    result["valid_primary_ec"].isel(transform=ti) > 0
                ).values
                finite = np.asarray(transform_values, dtype=float)
                finite = finite[np.isfinite(finite)]
                bound = (
                    float(np.nanpercentile(np.abs(finite), 98))
                    if finite.size
                    else 1.0
                )
                bound = max(bound, 1.0e-6)
                vmin, vmax, cmap = -bound, bound, "RdBu_r"
            else:
                vmin, vmax, cmap = -1.0, 1.0, "coolwarm"
            artist = ax.pcolormesh(
                result["lon"],
                result["lat"],
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
    if metric == "error_sd_difference_ba_minus_conus404":
        labels = ("BA - CONUS404 error SD (mm day$^{-1}$)",
                  "BA - CONUS404 error SD (log1p units)")
        for ti, artist in enumerate(artists):
            if artist is not None:
                figure.colorbar(artist, ax=axes[:, ti], shrink=0.88, label=labels[ti])
    elif artists[0] is not None:
        figure.colorbar(
            artists[0], ax=axes, shrink=0.88,
            label="C404/C404BA random-error correlation"
        )
    figure.suptitle(f"Pixelwise extended collocation ({sample_mode})", fontsize=15)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


# %% Main analysis
def run_analysis(
    input_path: Path | str = DEFAULT_INPUT,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    *,
    sample_mode: str = "all_windows",
    min_storms: int = 8,
    min_core_observations: int = 60,
    min_expanded_observations: int = 100,
    min_days_per_storm: int = 3,
    bootstrap: int = 0,
    seed: int = 404,
    pixel_chunk: int = 64,
    do_loso: bool = True,
) -> dict[str, Path]:
    """Run pixelwise EC and return paths to the generated artifacts."""
    input_path, output_dir = Path(input_path), Path(output_dir)
    if sample_mode not in SAMPLE_MODES:
        raise ValueError(f"sample_mode must be one of {SAMPLE_MODES}.")
    if min_storms < 3 or min_days_per_storm < 2:
        raise ValueError("Use at least 3 storms and at least 2 days per storm.")
    if bootstrap < 0 or pixel_chunk < 1:
        raise ValueError("bootstrap must be nonnegative and pixel_chunk positive.")
    if not input_path.exists():
        raise FileNotFoundError(
            f"Combined PERSIANN-grid NetCDF was not found: {input_path}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = xr.open_dataset(input_path, decode_times=True)
    _require_input_schema(ds, sample_mode)
    storm_ids = _decode_strings(ds["storm_id"].values)
    number_of_storms = len(storm_ids)
    if min_storms > number_of_storms:
        raise ValueError(
            f"min_storms={min_storms} exceeds {number_of_storms} input storms."
        )
    storm_index_all = np.asarray(ds["storm_index"].values, dtype=np.int64)
    if storm_index_all.shape != (ds.sizes["observation"],):
        raise ValueError("storm_index must have dimension (observation,).")
    if (storm_index_all < 0).any() or (storm_index_all >= number_of_storms).any():
        raise ValueError("storm_index contains an out-of-range storm number.")

    ny, nx = ds.sizes["lat"], ds.sizes["lon"]
    pixels = ny * nx
    main, loso, boot = _allocate_outputs(
        ny, nx, number_of_storms, do_loso, bootstrap
    )
    study_domain = np.asarray(
        _stack_spatial(ds["study_domain_mask"], ()).values, dtype=bool
    )
    if study_domain.shape != (pixels,):
        raise ValueError("Unexpected flattened study_domain_mask shape.")
    active_pixels = np.flatnonzero(study_domain)
    if active_pixels.size == 0:
        raise ValueError("study_domain_mask does not contain any selected pixels.")
    # Outside-domain cells never enter a covariance or bootstrap calculation,
    # but retain an explicit reason code in every output map.
    for wi in range(len(WINDOWS)):
        for ti in range(len(TRANSFORMS)):
            main["invalid_reason_code"][wi, ti].reshape(-1)[~study_domain] = 1

    stacked_products = {
        name: _stack_spatial(ds[name], ("observation",)) for name in PRODUCTS
    }
    stacked_valid = (
        _stack_spatial(ds["valid_all4"], ("observation",))
        if "valid_all4" in ds
        else None
    )
    stacked_assigned = (
        _stack_spatial(ds["assigned_storm_mask"], ("storm",))
        if sample_mode == "assigned_storms"
        else None
    )

    rng = np.random.default_rng(seed)
    bootstrap_weights = None
    if bootstrap:
        draws = rng.integers(
            0, number_of_storms, size=(bootstrap, number_of_storms)
        )
        bootstrap_weights = np.zeros(
            (bootstrap, number_of_storms), dtype=np.int16
        )
        for replicate in range(bootstrap):
            bootstrap_weights[replicate] = np.bincount(
                draws[replicate], minlength=number_of_storms
            )

    duplicate_counts: dict[str, int] = {}
    for wi, window in enumerate(WINDOWS):
        observation_indices, duplicates_discarded = _observation_indices(
            ds, sample_mode, window
        )
        duplicate_counts[window] = duplicates_discarded
        selected_storm_index = storm_index_all[observation_indices]
        min_observations = (
            min_core_observations if window == "core" else min_expanded_observations
        )
        print(
            f"{window}: {len(observation_indices)} selected records; "
            f"{duplicates_discarded} duplicated calendar records discarded"
        )

        for ti, transform in enumerate(TRANSFORMS):
            print(
                f"  {transform}: processing {active_pixels.size:,} study-domain "
                f"pixels ({pixels:,} cells in the rectangular grid)"
            )
            for chunk_start in range(0, active_pixels.size, pixel_chunk):
                pixel_indices = active_pixels[
                    chunk_start : chunk_start + pixel_chunk
                ]
                raw_values = np.stack(
                    [
                        np.asarray(
                            stacked_products[name]
                            .isel(
                                observation=observation_indices,
                                pixel=pixel_indices,
                            )
                            .values,
                            dtype=np.float64,
                        )
                        for name in PRODUCTS
                    ],
                    axis=1,
                )
                nonnegative = (raw_values >= 0).all(axis=1)
                common_finite = np.isfinite(raw_values).all(axis=1) & nonnegative
                if stacked_valid is not None:
                    common_finite &= np.asarray(
                        stacked_valid.isel(
                            observation=observation_indices, pixel=pixel_indices
                        ).values,
                        dtype=bool,
                    )
                # Only selected study-domain cells are chunked.  This avoids
                # spending most bootstrap time on the empty bounding rectangle.
                domain_chunk = np.ones(pixel_indices.size, dtype=bool)
                if sample_mode == "all_windows":
                    analysis_mask = np.broadcast_to(
                        domain_chunk, common_finite.shape
                    )
                else:
                    assigned = np.asarray(
                        stacked_assigned.isel(pixel=pixel_indices).values,
                        dtype=bool,
                    )
                    analysis_mask = assigned[selected_storm_index] & domain_chunk
                base_mask = common_finite & analysis_mask
                values = np.log1p(raw_values) if transform == "log1p" else raw_values

                counts, sums, sum_cross, _, _ = _storm_sufficient_statistics(
                    values,
                    base_mask,
                    selected_storm_index,
                    number_of_storms,
                    min_days_per_storm,
                )
                total_count, contributing_storms, covariance, _ = _point_statistics(
                    counts, sums, sum_cross
                )
                eligible = (
                    domain_chunk
                    & (total_count >= min_observations)
                    & (contributing_storms >= min_storms)
                )
                metrics = _metrics_from_covariance(covariance)
                primary_valid = eligible & metrics["valid_primary_ec"]

                if ti == 0:
                    main["n_observations"][wi].reshape(-1)[pixel_indices] = (
                        total_count.astype(np.int16)
                    )
                    main["n_contributing_storms"][wi].reshape(-1)[pixel_indices] = (
                        contributing_storms.astype(np.int8)
                    )
                    main["sample_eligible"][wi].reshape(-1)[pixel_indices] = (
                        eligible.astype(np.uint8)
                    )

                # Preserve diagnostic negative variances but suppress values at
                # pixels whose sample itself does not meet the stated design.
                for metric in PRODUCT_METRICS:
                    metric_values = np.where(
                        eligible[:, None], metrics[metric], np.nan
                    )
                    _assign_product_metric(
                        main[metric], wi, ti, pixel_indices, metric_values
                    )
                for metric in MAP_METRICS:
                    metric_values = np.where(eligible, metrics[metric], np.nan)
                    _assign_map_metric(
                        main[metric], wi, ti, pixel_indices, metric_values
                    )
                _assign_map_metric(
                    main["valid_primary_ec"],
                    wi,
                    ti,
                    pixel_indices,
                    primary_valid.astype(np.uint8),
                )
                _assign_map_metric(
                    main["denominator_valid"],
                    wi,
                    ti,
                    pixel_indices,
                    (eligible & metrics["denominator_valid"]).astype(np.uint8),
                )
                _assign_map_metric(
                    main["reference_diagnostics_valid"],
                    wi,
                    ti,
                    pixel_indices,
                    (eligible & metrics["reference_diagnostics_valid"]).astype(
                        np.uint8
                    ),
                )

                reason = np.zeros(pixel_indices.size, dtype=np.int16)
                outside = ~domain_chunk
                reason[outside] |= 1
                inside = ~outside
                reason[inside & (total_count < min_observations)] |= 2
                reason[inside & (contributing_storms < min_storms)] |= 4
                reason[eligible & ~metrics["finite_covariance"]] |= 8
                reason[eligible & ~metrics["denominator_valid"]] |= 16
                reason[eligible & ~metrics["model_variances_positive"]] |= 32
                reason[eligible & ~metrics["model_error_block_psd"]] |= 64
                reason[eligible & ~metrics["model_signals_nonnegative"]] |= 128
                reason[eligible & ~metrics["reference_diagnostics_valid"]] |= 256
                _assign_map_metric(
                    main["invalid_reason_code"], wi, ti, pixel_indices, reason
                )

                if do_loso:
                    for omitted in range(number_of_storms):
                        lm, lv = _loso_metrics(
                            omitted,
                            counts,
                            sums,
                            sum_cross,
                            outside,
                            min_observations,
                            min_storms,
                        )
                        destination = (
                            wi,
                            ti,
                            omitted,
                        )
                        for product_index, name in enumerate(
                            ("error_sd_conus404", "error_sd_conus404ba")
                        ):
                            loso[name][destination].reshape(-1)[pixel_indices] = (
                                np.where(
                                    lv, lm["error_sd"][:, product_index], np.nan
                                ).astype(np.float32)
                            )
                        for name in (
                            "error_sd_difference_ba_minus_conus404",
                            "error_correlation_model_pair",
                            "snr_db_difference_ba_minus_conus404",
                            "estimated_r2_difference_ba_minus_conus404",
                        ):
                            loso[name][destination].reshape(-1)[pixel_indices] = (
                                np.where(lv, lm[name], np.nan).astype(np.float32)
                            )
                        loso["valid_primary_ec"][destination].reshape(-1)[
                            pixel_indices
                        ] = lv.astype(np.uint8)

                if bootstrap_weights is not None:
                    bs = _bootstrap_summaries(
                        bootstrap_weights,
                        counts,
                        sums,
                        sum_cross,
                        eligible,
                    )
                    for name, values_bs in bs.items():
                        if name.startswith("error_sd_ci"):
                            _assign_product_metric(
                                boot[name],
                                wi,
                                ti,
                                pixel_indices,
                                values_bs,
                            )
                        else:
                            _assign_map_metric(
                                boot[name], wi, ti, pixel_indices, values_bs
                            )

    attrs: dict[str, Any] = {
        "title": "Pixelwise four-system extended collocation on PERSIANN-CDR grid",
        "source_file": str(input_path.resolve()),
        "sample_mode": sample_mode,
        "product_order": PRODUCTS,
        "correlated_error_pair": MODEL_PRODUCTS,
        "ec_error_groups_equivalent": [1, 1, 2, 3],
        "windows": WINDOWS,
        "transforms": TRANSFORMS,
        "minimum_storms": min_storms,
        "minimum_core_observations": min_core_observations,
        "minimum_expanded_observations": min_expanded_observations,
        "minimum_valid_days_per_storm": min_days_per_storm,
        "bootstrap_replicates": bootstrap,
        "bootstrap_seed": seed,
        "all_windows_date_policy": (
            "Select the requested core/expanded window, verify duplicate records "
            "are numerically identical, then keep the first record in input "
            "EVENTS order and use that retained record's storm block."
        ),
        "duplicate_records_discarded": duplicate_counts,
        "invalid_reason_flags": INVALID_FLAGS,
        "primary_estimand": (
            "Hurricane-calendar-window-conditioned random-error covariance, not "
            "local storm-impact error at every pixel, mean bias, or total RMSE"
            if sample_mode == "all_windows"
            else "Assigned-storm/watershed-conditioned random-error covariance, "
            "not mean bias or total RMSE"
        ),
        "reference_assumption": (
            "MRMS and PERSIANN-CDR errors are assumed mutually independent and "
            "independent of the model-error block; violations can bias estimates."
        ),
        "spatial_note": (
            "All products were aligned before this analysis; maps inherit the "
            "PERSIANN-CDR support and are not native-grid estimates."
        ),
        "diagnostic_note": (
            "MRMS/PERSIANN error variances average the C404- and BA-based TC "
            "formulations and are diagnostic rather than the primary EC target."
        ),
    }
    result = _dataset_from_outputs(ds, main, boot, attrs)
    stem = f"gridded_extended_collocation_{sample_mode}"
    result_path = output_dir / f"{stem}.nc"
    _write_netcdf(result, result_path)

    paths: dict[str, Path] = {"results": result_path}
    if do_loso:
        loso_result = _loso_dataset(ds, loso, storm_ids, attrs)
        loso_path = output_dir / f"{stem}_leave_one_storm_out.nc"
        _write_netcdf(loso_result, loso_path)
        paths["leave_one_storm_out"] = loso_path

    difference_figure = output_dir / f"{stem}_error_sd_difference.png"
    correlation_figure = output_dir / f"{stem}_model_error_correlation.png"
    _quicklook(
        result,
        "error_sd_difference_ba_minus_conus404",
        difference_figure,
        sample_mode,
    )
    _quicklook(
        result,
        "error_correlation_model_pair",
        correlation_figure,
        sample_mode,
    )
    paths["error_sd_difference_figure"] = difference_figure
    paths["model_error_correlation_figure"] = correlation_figure

    manifest_path = output_dir / f"{stem}_manifest.json"
    valid_counts = {
        f"{window}_{transform}": int(
            result["valid_primary_ec"].sel(window=window, transform=transform).sum()
        )
        for window in WINDOWS
        for transform in TRANSFORMS
    }
    manifest = {
        "configuration": attrs,
        "outputs": {name: str(path) for name, path in paths.items()},
        "valid_pixel_counts": valid_counts,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    paths["manifest"] = manifest_path
    ds.close()
    print("Finished pixelwise EC:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    return paths


# %% Synthetic numerical self-test
def run_synthetic_self_test(seed: int = 404) -> dict[str, float]:
    """Check the vectorized estimator against known simulated error moments."""
    rng = np.random.default_rng(seed)
    n = 200_000
    truth = rng.normal(0, 4.0, n)
    shared = rng.normal(0, 0.5, n)
    errors = np.column_stack(
        (
            shared + rng.normal(0, 0.7, n),
            1.2 * shared + rng.normal(0, 0.8, n),
            rng.normal(0, 0.9, n),
            rng.normal(0, 1.1, n),
        )
    )
    slopes = np.array([1.0, 1.1, 0.9, 1.2])
    values = truth[:, None] * slopes + errors
    covariance = np.cov(values, rowvar=False, ddof=1)
    metrics = _metrics_from_covariance(covariance[None, ...])
    expected = np.array([0.5**2 + 0.7**2, (1.2 * 0.5) ** 2 + 0.8**2])
    estimated = metrics["error_variance"][0, :2]
    error_covariance_expected = 1.2 * 0.5**2
    error_covariance_estimated = float(metrics["error_covariance_model_pair"][0])
    if not np.allclose(estimated, expected, rtol=0.05, atol=0.03):
        raise AssertionError(
            f"Synthetic model error variances failed: {estimated} versus {expected}"
        )
    if not np.isclose(
        error_covariance_estimated,
        error_covariance_expected,
        rtol=0.10,
        atol=0.03,
    ):
        raise AssertionError("Synthetic model error covariance check failed.")
    result = {
        "estimated_error_variance_conus404": float(estimated[0]),
        "expected_error_variance_conus404": float(expected[0]),
        "estimated_error_variance_conus404ba": float(estimated[1]),
        "expected_error_variance_conus404ba": float(expected[1]),
        "estimated_model_error_covariance": error_covariance_estimated,
        "expected_model_error_covariance": error_covariance_expected,
    }
    print(json.dumps(result, indent=2))
    return result


# %% CLI
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-mode", choices=SAMPLE_MODES, default="all_windows")
    parser.add_argument("--min-storms", type=int, default=8)
    parser.add_argument("--min-core-observations", type=int, default=60)
    parser.add_argument("--min-expanded-observations", type=int, default=100)
    parser.add_argument("--min-days-per-storm", type=int, default=3)
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        help="Complete-storm bootstrap replicates (0 disables; recommended first run).",
    )
    parser.add_argument("--seed", type=int, default=404)
    parser.add_argument("--pixel-chunk", type=int, default=64)
    parser.add_argument("--no-loso", action="store_true")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run a fast synthetic estimator check and exit.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.self_test:
        run_synthetic_self_test(seed=args.seed)
        return
    run_analysis(
        input_path=args.input,
        output_dir=args.output_dir,
        sample_mode=args.sample_mode,
        min_storms=args.min_storms,
        min_core_observations=args.min_core_observations,
        min_expanded_observations=args.min_expanded_observations,
        min_days_per_storm=args.min_days_per_storm,
        bootstrap=args.bootstrap,
        seed=args.seed,
        pixel_chunk=args.pixel_chunk,
        do_loso=not args.no_loso,
    )


if __name__ == "__main__":
    main()
