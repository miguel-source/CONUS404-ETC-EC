"""
Four-system Extended Collocation (EC) for hurricane watershed-day precipitation.

The correlated-error pair is CONUS404 / CONUS404BA. MRMS RadarOnly QPE 24H
and PERSIANN-CDR are the two nominally independent systems. The program also
reports the two ordinary TC triplets as a diagnostic, bootstraps complete
storms, and performs leave-one-storm-out analysis.

Run from a VS Code PowerShell terminal with the CONUS404 environment. This file
also contains # %% markers so its definitions can be used interactively.
"""

# %% Imports
from __future__ import annotations

from study_paths import PROJECT_ROOT, RESULTS_ROOT, METADATA_ROOT, REPO_ROOT, STATES_FILE, WATERSHEDS_FILE, DAILY_INPUT, EC_DIRECTORY, EC_INPUT, LOSO_INPUT, AGREEMENT_DIRECTORY, AGREEMENT_INPUT


import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# %% Configuration
PAPER_ROOT = (PROJECT_ROOT)
DEFAULT_INPUT = (
    PAPER_ROOT
    / "ETC_nc"
    / "Hurricane_watershed_day_ETC"
    / "watershed_day_precipitation_matched_all4.csv"
)
DEFAULT_OUTPUT_DIR = PAPER_ROOT / "ETC_nc" / "Hurricane_EC_results"

PRODUCT_COLUMNS = ["conus404", "conus404ba", "mrms", "persiann_cdr"]
MODEL_PRODUCTS = ["conus404", "conus404ba"]
WINDOWS = ["core", "expanded"]
TRANSFORMS = ["raw", "log1p"]
DENOMINATOR_RTOL = 1.0e-8

# Metrics for which percentile bootstrap intervals are written.
BOOTSTRAP_METRICS = [
    "error_variance_conus404",
    "error_variance_conus404ba",
    "error_sd_conus404",
    "error_sd_conus404ba",
    "error_sd_difference_ba_minus_conus404",
    "error_sd_ratio_ba_over_conus404",
    "error_covariance_model_pair",
    "error_correlation_model_pair",
    "snr_conus404",
    "snr_conus404ba",
    "snr_db_conus404",
    "snr_db_conus404ba",
    "snr_db_difference_ba_minus_conus404",
    "estimated_r2_to_truth_conus404",
    "estimated_r2_to_truth_conus404ba",
    "estimated_r2_difference_ba_minus_conus404",
    "estimated_r_to_truth_conus404",
    "estimated_r_to_truth_conus404ba",
    "estimated_r_difference_ba_minus_conus404",
]


# %% Input and transformations
def load_matched_table(path: Path = DEFAULT_INPUT) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Matched input was not found: {path}\n"
            "Run prepare_watershed_day_etc.py first."
        )
    table = pd.read_csv(path, parse_dates=["date"])
    required = {
        "row_id",
        "storm_id",
        "event_name",
        "watershed_id",
        "is_core_day",
        "complete_case_all4",
        *PRODUCT_COLUMNS,
    }
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Matched input is missing columns: {sorted(missing)}")

    table["is_core_day"] = table["is_core_day"].astype(bool)
    table["complete_case_all4"] = table["complete_case_all4"].astype(bool)
    table = table[table["complete_case_all4"]].copy()
    if table.empty:
        raise ValueError("No complete four-product rows remain.")
    if table["row_id"].duplicated().any():
        raise ValueError("row_id must be unique before EC analysis.")
    values = table[PRODUCT_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Complete-case table still contains non-finite values.")
    if (values < 0).any():
        raise ValueError("Precipitation cannot be negative.")
    if table["storm_id"].nunique() < 3:
        raise ValueError("At least three storm blocks are required.")
    return table.sort_values(["storm_id", "watershed_id", "date"]).reset_index(
        drop=True
    )


def select_analysis(
    table: pd.DataFrame, window: str, transform: str
) -> tuple[pd.DataFrame, np.ndarray]:
    if window not in WINDOWS:
        raise ValueError(f"window must be one of {WINDOWS}")
    if transform not in TRANSFORMS:
        raise ValueError(f"transform must be one of {TRANSFORMS}")
    selected = table if window == "expanded" else table[table["is_core_day"]]
    selected = selected.copy()
    values = selected[PRODUCT_COLUMNS].to_numpy(dtype=np.float64)
    if transform == "log1p":
        values = np.log1p(values)
    return selected, values


# %% Extended Collocation estimator
def _safe_sqrt(value: float) -> float:
    return float(np.sqrt(value)) if np.isfinite(value) and value >= 0 else np.nan


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0:
        return np.nan
    return float(numerator / denominator)


def estimate_extended_collocation(values: np.ndarray) -> dict[str, float | int | bool]:
    """
    Estimate the CONUS404/CONUS404BA error covariance submatrix.

    Column order must be:
      0 CONUS404, 1 CONUS404BA, 2 MRMS, 3 PERSIANN-CDR.

    The two off-diagonal EC expressions are averaged to produce a symmetric
    sample estimate. Their difference is retained as a finite-sample diagnostic.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("values must have shape (observations, 4).")
    if len(values) < 4:
        raise ValueError("At least four matched observations are required.")
    if not np.isfinite(values).all():
        raise ValueError("EC input must be finite complete cases.")

    covariance = np.cov(values, rowvar=False, ddof=1)
    denominator = float(covariance[2, 3])
    denominator_scale = np.sqrt(
        max(float(covariance[2, 2]), 0.0)
        * max(float(covariance[3, 3]), 0.0)
    )
    denominator_ok = bool(
        np.isfinite(denominator)
        and denominator_scale > 0
        and denominator > DENOMINATOR_RTOL * denominator_scale
    )

    result: dict[str, float | int | bool] = {
        "n_rows": int(len(values)),
        "covariance_mrms_persiann": denominator,
        "correlation_mrms_persiann": _safe_ratio(denominator, denominator_scale),
        "denominator_valid": denominator_ok,
    }
    if not denominator_ok:
        for metric in BOOTSTRAP_METRICS:
            result[metric] = np.nan
        result.update(
            {
                "error_covariance_direction_01": np.nan,
                "error_covariance_direction_10": np.nan,
                "error_covariance_asymmetry": np.nan,
                "signal_variance_conus404": np.nan,
                "signal_variance_conus404ba": np.nan,
                "model_error_min_eigenvalue": np.nan,
                "model_error_determinant": np.nan,
                "valid_ec_solution": False,
            }
        )
        return result

    # Error variances of the two model products, using MRMS/PERSIANN as the
    # independent systems. These equal the target-product variance estimates
    # from the corresponding ordinary TC triplets.
    error_variance_c404 = float(
        covariance[0, 0]
        - covariance[0, 2] * covariance[0, 3] / denominator
    )
    error_variance_ba = float(
        covariance[1, 1]
        - covariance[1, 2] * covariance[1, 3] / denominator
    )

    error_covariance_01 = float(
        covariance[0, 1]
        - covariance[0, 2] * covariance[1, 3] / denominator
    )
    error_covariance_10 = float(
        covariance[1, 0]
        - covariance[1, 2] * covariance[0, 3] / denominator
    )
    error_covariance = (error_covariance_01 + error_covariance_10) / 2.0

    model_error_matrix = np.array(
        [
            [error_variance_c404, error_covariance],
            [error_covariance, error_variance_ba],
        ],
        dtype=np.float64,
    )
    eigenvalues = np.linalg.eigvalsh(model_error_matrix)
    determinant = float(np.linalg.det(model_error_matrix))
    tolerance = 1.0e-10 * max(float(np.max(np.abs(model_error_matrix))), 1.0)
    positive_variances = error_variance_c404 > 0 and error_variance_ba > 0
    psd = bool(eigenvalues.min() >= -tolerance)

    signal_variance_c404 = float(covariance[0, 0] - error_variance_c404)
    signal_variance_ba = float(covariance[1, 1] - error_variance_ba)
    positive_signals = signal_variance_c404 >= 0 and signal_variance_ba >= 0
    valid = bool(positive_variances and psd and positive_signals)

    error_sd_c404 = _safe_sqrt(error_variance_c404)
    error_sd_ba = _safe_sqrt(error_variance_ba)
    error_correlation = (
        _safe_ratio(
            error_covariance,
            np.sqrt(error_variance_c404 * error_variance_ba),
        )
        if positive_variances and psd
        else np.nan
    )
    snr_c404 = _safe_ratio(signal_variance_c404, error_variance_c404)
    snr_ba = _safe_ratio(signal_variance_ba, error_variance_ba)
    snr_db_c404 = 10.0 * np.log10(snr_c404) if snr_c404 > 0 else np.nan
    snr_db_ba = 10.0 * np.log10(snr_ba) if snr_ba > 0 else np.nan
    r2_c404 = _safe_ratio(signal_variance_c404, covariance[0, 0])
    r2_ba = _safe_ratio(signal_variance_ba, covariance[1, 1])
    r_c404 = _safe_sqrt(r2_c404) if 0 <= r2_c404 <= 1 else np.nan
    r_ba = _safe_sqrt(r2_ba) if 0 <= r2_ba <= 1 else np.nan

    result.update(
        {
            "error_variance_conus404": error_variance_c404,
            "error_variance_conus404ba": error_variance_ba,
            "error_sd_conus404": error_sd_c404,
            "error_sd_conus404ba": error_sd_ba,
            "error_sd_difference_ba_minus_conus404": error_sd_ba - error_sd_c404,
            "error_sd_ratio_ba_over_conus404": _safe_ratio(error_sd_ba, error_sd_c404),
            "error_covariance_model_pair": error_covariance,
            "error_covariance_direction_01": error_covariance_01,
            "error_covariance_direction_10": error_covariance_10,
            "error_covariance_asymmetry": abs(error_covariance_01 - error_covariance_10),
            "error_correlation_model_pair": error_correlation,
            "signal_variance_conus404": signal_variance_c404,
            "signal_variance_conus404ba": signal_variance_ba,
            "snr_conus404": snr_c404,
            "snr_conus404ba": snr_ba,
            "snr_db_conus404": snr_db_c404,
            "snr_db_conus404ba": snr_db_ba,
            "snr_db_difference_ba_minus_conus404": snr_db_ba - snr_db_c404,
            "estimated_r2_to_truth_conus404": r2_c404,
            "estimated_r2_to_truth_conus404ba": r2_ba,
            "estimated_r2_difference_ba_minus_conus404": r2_ba - r2_c404,
            "estimated_r_to_truth_conus404": r_c404,
            "estimated_r_to_truth_conus404ba": r_ba,
            "estimated_r_difference_ba_minus_conus404": r_ba - r_c404,
            "model_error_min_eigenvalue": float(eigenvalues.min()),
            "model_error_determinant": determinant,
            "valid_ec_solution": valid,
        }
    )
    return result


# %% Ordinary TC diagnostic for the two paired triplets
def estimate_tc_triplet(
    values: np.ndarray, names: Iterable[str]
) -> pd.DataFrame:
    values = np.asarray(values, dtype=np.float64)
    names = list(names)
    if values.ndim != 2 or values.shape[1] != 3 or len(names) != 3:
        raise ValueError("TC input must contain exactly three named columns.")
    covariance = np.cov(values, rowvar=False, ddof=1)
    rows: list[dict] = []
    for i, (j, k) in enumerate([(1, 2), (0, 2), (0, 1)]):
        denominator = covariance[j, k]
        error_variance = (
            covariance[i, i]
            - covariance[i, j] * covariance[i, k] / denominator
            if denominator != 0
            else np.nan
        )
        signal_variance = covariance[i, i] - error_variance
        snr = _safe_ratio(signal_variance, error_variance)
        rows.append(
            {
                "product": names[i],
                "error_variance": float(error_variance),
                "error_sd": _safe_sqrt(float(error_variance)),
                "signal_variance": float(signal_variance),
                "snr": snr,
                "snr_db": 10.0 * np.log10(snr) if snr > 0 else np.nan,
                "estimated_r2_to_truth": _safe_ratio(
                    signal_variance, covariance[i, i]
                ),
            }
        )
    result = pd.DataFrame(rows)
    result["estimated_r_to_truth"] = np.where(
        result["estimated_r2_to_truth"].between(0, 1),
        np.sqrt(result["estimated_r2_to_truth"]),
        np.nan,
    )
    return result


def paired_tc_diagnostics(
    values: np.ndarray, ec_result: dict[str, float | int | bool]
) -> pd.DataFrame:
    frames = []
    for triplet_name, indices, names in [
        (
            "CONUS404-MRMS-PERSIANN",
            [0, 2, 3],
            ["conus404", "mrms", "persiann_cdr"],
        ),
        (
            "CONUS404BA-MRMS-PERSIANN",
            [1, 2, 3],
            ["conus404ba", "mrms", "persiann_cdr"],
        ),
    ]:
        frame = estimate_tc_triplet(values[:, indices], names)
        frame.insert(0, "triplet", triplet_name)
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)

    # The target-model TC estimates must exactly match the corresponding EC
    # diagonal estimates. This is an internal formula check.
    checks = {
        "conus404": float(ec_result["error_variance_conus404"]),
        "conus404ba": float(ec_result["error_variance_conus404ba"]),
    }
    for product, expected in checks.items():
        actual = float(result.loc[result["product"].eq(product), "error_variance"].iloc[0])
        if not np.isclose(actual, expected, rtol=1.0e-10, atol=1.0e-10):
            raise AssertionError(f"TC/EC formula mismatch for {product}.")
    return result


# %% Storm-block bootstrap and leave-one-storm-out
def storm_blocks(metadata: pd.DataFrame, values: np.ndarray) -> tuple[list[str], list[np.ndarray]]:
    storms = sorted(metadata["storm_id"].unique().tolist())
    blocks = [values[metadata["storm_id"].eq(storm).to_numpy()] for storm in storms]
    return storms, blocks


def bootstrap_ec(
    metadata: pd.DataFrame,
    values: np.ndarray,
    draws: np.ndarray,
) -> pd.DataFrame:
    storms, blocks = storm_blocks(metadata, values)
    if draws.ndim != 2 or draws.shape[1] != len(storms):
        raise ValueError("Bootstrap draw matrix does not match the storm count.")
    records: list[dict] = []
    for replicate, sampled_indices in enumerate(draws):
        sample = np.concatenate([blocks[index] for index in sampled_indices], axis=0)
        estimate = estimate_extended_collocation(sample)
        estimate["bootstrap_replicate"] = replicate
        estimate["sampled_storm_indices"] = " ".join(map(str, sampled_indices))
        records.append(estimate)
    return pd.DataFrame(records)


def summarise_bootstrap(
    point: dict[str, float | int | bool], bootstrap: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict] = []
    for metric in BOOTSTRAP_METRICS:
        values = pd.to_numeric(bootstrap[metric], errors="coerce").to_numpy(float)
        finite = values[np.isfinite(values)]
        rows.append(
            {
                "metric": metric,
                "point_estimate": point.get(metric, np.nan),
                "bootstrap_median": np.median(finite) if finite.size else np.nan,
                "ci_2_5": np.percentile(finite, 2.5) if finite.size else np.nan,
                "ci_97_5": np.percentile(finite, 97.5) if finite.size else np.nan,
                "finite_bootstrap_count": int(finite.size),
                "finite_bootstrap_fraction": float(finite.size / len(values)),
                "valid_ec_bootstrap_fraction": float(
                    bootstrap["valid_ec_solution"].astype(bool).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def leave_one_storm_out(
    metadata: pd.DataFrame, values: np.ndarray
) -> pd.DataFrame:
    records: list[dict] = []
    for omitted in sorted(metadata["storm_id"].unique()):
        keep = ~metadata["storm_id"].eq(omitted).to_numpy()
        estimate = estimate_extended_collocation(values[keep])
        estimate["omitted_storm_id"] = omitted
        records.append(estimate)
    return pd.DataFrame(records)


# %% Figures
def _metric_interval(
    summary: pd.DataFrame, window: str, transform: str, metric: str
) -> pd.Series:
    selected = summary[
        summary["window"].eq(window)
        & summary["transform"].eq(transform)
        & summary["metric"].eq(metric)
    ]
    if len(selected) != 1:
        raise ValueError(f"Expected one summary row for {window}/{transform}/{metric}.")
    return selected.iloc[0]


def plot_error_sd(summary: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    colors = {"conus404": "#2c7fb8", "conus404ba": "#d95f0e"}
    for axis, transform in zip(axes, TRANSFORMS):
        for product_index, product in enumerate(MODEL_PRODUCTS):
            metric = f"error_sd_{product}"
            points, low, high = [], [], []
            for window in WINDOWS:
                row = _metric_interval(summary, window, transform, metric)
                points.append(float(row.point_estimate))
                low.append(float(row.point_estimate - row.ci_2_5))
                high.append(float(row.ci_97_5 - row.point_estimate))
            x = np.arange(len(WINDOWS)) + (product_index - 0.5) * 0.12
            axis.errorbar(
                x,
                points,
                yerr=np.vstack([low, high]),
                fmt="o",
                capsize=4,
                color=colors[product],
                label=product.upper(),
            )
        axis.set_xticks(np.arange(len(WINDOWS)), [w.title() for w in WINDOWS])
        axis.set_title("Raw precipitation" if transform == "raw" else "log1p sensitivity")
        axis.set_ylabel(
            "EC random-error SD (mm day$^{-1}$)"
            if transform == "raw"
            else "EC random-error SD [log(1 + mm)]"
        )
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    figure.suptitle("CONUS404 versus CONUS404BA: storm-block bootstrap 95% intervals")
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_error_correlation(summary: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    labels, points, low, high = [], [], [], []
    for transform in TRANSFORMS:
        for window in WINDOWS:
            row = _metric_interval(
                summary, window, transform, "error_correlation_model_pair"
            )
            labels.append(f"{window.title()}\n{transform}")
            points.append(float(row.point_estimate))
            low.append(float(row.point_estimate - row.ci_2_5))
            high.append(float(row.ci_97_5 - row.point_estimate))
    x = np.arange(len(labels))
    axis.errorbar(x, points, yerr=np.vstack([low, high]), fmt="o", capsize=5)
    interval_lows = np.asarray(points) - np.asarray(low)
    interval_highs = np.asarray(points) + np.asarray(high)
    y_min = max(-1.05, float(interval_lows.min()) - 0.04)
    y_max = min(1.05, float(interval_highs.max()) + 0.02)
    axis.axhline(1, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    axis.set_xticks(x, labels)
    axis.set_ylabel("EC error correlation: CONUS404 vs CONUS404BA")
    axis.set_ylim(y_min, y_max)
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


# %% Complete analysis
def run_analysis(
    input_csv: Path = DEFAULT_INPUT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    n_bootstrap: int = 5_000,
    seed: int = 404,
) -> dict[str, Path]:
    if n_bootstrap < 100:
        raise ValueError("Use at least 100 storm-block bootstrap replicates.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = load_matched_table(input_csv)
    storms = sorted(table["storm_id"].unique())
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(storms), size=(n_bootstrap, len(storms)))

    point_records: list[dict] = []
    bootstrap_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    loso_frames: list[pd.DataFrame] = []
    tc_frames: list[pd.DataFrame] = []

    for window in WINDOWS:
        for transform in TRANSFORMS:
            metadata, values = select_analysis(table, window, transform)
            point = estimate_extended_collocation(values)
            point.update(
                {
                    "window": window,
                    "transform": transform,
                    "n_storms": int(metadata["storm_id"].nunique()),
                }
            )
            point_records.append(point)

            tc = paired_tc_diagnostics(values, point)
            tc.insert(0, "transform", transform)
            tc.insert(0, "window", window)
            tc_frames.append(tc)

            print(
                f"{window:8s} {transform:5s}: "
                f"n={len(metadata)}, storms={metadata['storm_id'].nunique()}, "
                f"SD C404={point['error_sd_conus404']:.3f}, "
                f"SD BA={point['error_sd_conus404ba']:.3f}, "
                f"error rho={point['error_correlation_model_pair']:.3f}"
            )

            boot = bootstrap_ec(metadata, values, draws)
            boot.insert(0, "transform", transform)
            boot.insert(0, "window", window)
            bootstrap_frames.append(boot)

            summary = summarise_bootstrap(point, boot)
            summary.insert(0, "transform", transform)
            summary.insert(0, "window", window)
            summary_frames.append(summary)

            loso = leave_one_storm_out(metadata, values)
            loso.insert(0, "transform", transform)
            loso.insert(0, "window", window)
            loso_frames.append(loso)

    point_table = pd.DataFrame(point_records)
    bootstrap_table = pd.concat(bootstrap_frames, ignore_index=True)
    bootstrap_summary = pd.concat(summary_frames, ignore_index=True)
    loso_table = pd.concat(loso_frames, ignore_index=True)
    tc_table = pd.concat(tc_frames, ignore_index=True)

    point_path = output_dir / "ec_point_estimates.csv"
    bootstrap_path = output_dir / "ec_storm_block_bootstrap_replicates.csv.gz"
    summary_path = output_dir / "ec_storm_block_bootstrap_summary.csv"
    loso_path = output_dir / "ec_leave_one_storm_out.csv"
    tc_path = output_dir / "paired_tc_diagnostics.csv"
    error_sd_figure = output_dir / "ec_error_sd_bootstrap.png"
    correlation_figure = output_dir / "ec_model_error_correlation.png"

    point_table.to_csv(point_path, index=False)
    bootstrap_table.to_csv(bootstrap_path, index=False, compression="gzip")
    bootstrap_summary.to_csv(summary_path, index=False)
    loso_table.to_csv(loso_path, index=False)
    tc_table.to_csv(tc_path, index=False)
    plot_error_sd(bootstrap_summary, error_sd_figure)
    plot_error_correlation(bootstrap_summary, correlation_figure)

    manifest = {
        "input": str(Path(input_csv)),
        "output_directory": str(output_dir),
        "products_in_order": PRODUCT_COLUMNS,
        "correlated_error_pair": ["conus404", "conus404ba"],
        "nominally_independent_systems": ["mrms", "persiann_cdr"],
        "windows": WINDOWS,
        "transforms": TRANSFORMS,
        "bootstrap_unit": "complete storm",
        "bootstrap_replicates": n_bootstrap,
        "random_seed": seed,
        "independent_storm_blocks": len(storms),
        "storm_ids": storms,
        "important_assumptions": [
            "MRMS RadarOnly and PERSIANN-CDR errors are mutually independent.",
            "Their errors are independent of both model products.",
            "Only CONUS404 and CONUS404BA are assigned correlated errors.",
            "Errors are orthogonal to the unknown true precipitation.",
            "The affine error model is stable across pooled storms and watersheds.",
        ],
        "interpretation_note": (
            "EC estimates random-error variance, not mean bias or total RMSE. "
            "log1p results are sensitivity estimates in transformed units."
        ),
    }
    manifest_path = output_dir / "ec_analysis_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nSaved EC results to {output_dir}")
    return {
        "point_estimates": point_path,
        "bootstrap_replicates": bootstrap_path,
        "bootstrap_summary": summary_path,
        "leave_one_storm_out": loso_path,
        "paired_tc": tc_path,
        "error_sd_figure": error_sd_figure,
        "error_correlation_figure": correlation_figure,
        "manifest": manifest_path,
    }


# %% Command-line entry point
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=404)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    run_analysis(
        arguments.input,
        arguments.output_dir,
        n_bootstrap=arguments.bootstrap,
        seed=arguments.seed,
    )
