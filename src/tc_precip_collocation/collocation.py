"""We estimate error moments using the same equations as our study scripts.

These functions are copied from the archived study routines without changes
to their numerical composites. Input arrays contain sample covariance matrices.
"""

from __future__ import annotations

import numpy as np

DENOMINATOR_RTOL = 1.0e-8

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

def triplet_metrics(covariance):
    """TC variance equations; no averaging or four-system validity mask.

    A finite zero error variance is retained as a boundary solution. Negative
    variances are never abs-transformed/clipped. SNR is undefined at zero error.
    """
    c = np.asarray(covariance, dtype=np.float64)
    if c.shape[-2:] != (3, 3):
        raise ValueError("Expected trailing covariance dimensions (3, 3).")
    variance = np.diagonal(c, axis1=-2, axis2=-1)
    error = np.full(variance.shape, np.nan)
    signal = np.full(variance.shape, np.nan)
    denominators = np.zeros(variance.shape, dtype=bool)
    for i in range(3):
        j, k = [n for n in range(3) if n != i]
        supported = _valid_positive_cross(c[..., j, k], variance[..., j], variance[..., k])
        denominators[..., i] = supported
        with np.errstate(divide="ignore", invalid="ignore"):
            estimate = c[..., i, j] * c[..., i, k] / c[..., j, k]
        signal[..., i] = np.where(supported, estimate, np.nan)
        error[..., i] = variance[..., i] - signal[..., i]
    finite_covariance = np.isfinite(c).all(axis=(-2, -1))
    valid_product = (
        finite_covariance[..., None]
        & denominators
        & np.isfinite(variance) & (variance > 0)
        & np.isfinite(error) & (error >= 0)
        & np.isfinite(signal) & (signal >= 0)
    )
    valid_triplet = valid_product.all(axis=-1)
    sd = np.full(error.shape, np.nan)
    np.sqrt(error, out=sd, where=np.isfinite(error) & (error >= 0))
    snr = np.full(error.shape, np.nan)
    np.divide(signal, error, out=snr, where=valid_product & (error > 0))
    r2 = np.full(error.shape, np.nan)
    np.divide(signal, variance, out=r2, where=valid_product)
    return {
        "error_variance_unmasked": error,
        "signal_variance_unmasked": signal,
        "observed_variance": variance,
        "error_sd_unmasked": sd,
        "snr_unmasked": snr,
        "estimated_r2_to_truth_unmasked": r2,
        "valid_product_estimate": valid_product,
        "valid_triplet": valid_triplet,
        "denominator_valid": denominators,
        "zero_error_boundary": valid_product & (error == 0),
    }

four_product_ec = _metrics_from_covariance
triple_collocation = triplet_metrics
