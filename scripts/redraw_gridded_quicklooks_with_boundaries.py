"""Redraw existing gridded EC/agreement PNGs with map-boundary overlays.

This reads the completed result NetCDFs, so it does not repeat EC, LOSO,
bootstrap, or agreement calculations.
"""

from __future__ import annotations

from study_paths import PROJECT_ROOT, RESULTS_ROOT, METADATA_ROOT, REPO_ROOT, STATES_FILE, WATERSHEDS_FILE, DAILY_INPUT, EC_DIRECTORY, EC_INPUT, LOSO_INPUT, AGREEMENT_DIRECTORY, AGREEMENT_INPUT


from pathlib import Path

import xarray as xr

import run_gridded_dataset_agreement as agreement
import run_gridded_extended_collocation as ec


EVENT_ROOT = (PROJECT_ROOT / 'ETC_StormEvents')
EC_DIR = EVENT_ROOT / "Hurricane_gridded_EC_results"
AGREEMENT_DIR = EVENT_ROOT / "Hurricane_gridded_agreement_results"


def redraw() -> list[Path]:
    written: list[Path] = []
    ec_path = EC_DIR / "gridded_extended_collocation_all_windows.nc"
    with xr.open_dataset(ec_path, engine="netcdf4") as result:
        for metric, suffix in (
            (
                "error_sd_difference_ba_minus_conus404",
                "error_sd_difference.png",
            ),
            ("error_correlation_model_pair", "model_error_correlation.png"),
        ):
            output = EC_DIR / f"gridded_extended_collocation_all_windows_{suffix}"
            ec._quicklook(result, metric, output, "all_windows")
            written.append(output)

    agreement_path = AGREEMENT_DIR / "gridded_dataset_agreement.nc"
    primary_specs = (
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
    with xr.open_dataset(agreement_path, engine="netcdf4") as result:
        for variable, filename, cmap, raw_label, log_label, limits in primary_specs:
            output = AGREEMENT_DIR / filename
            agreement._primary_quicklook(
                result,
                variable,
                output,
                cmap=cmap,
                label_raw=raw_label,
                label_log=log_label,
                fixed_limits=limits,
            )
            written.append(output)
        for window in agreement.WINDOWS:
            for transform in agreement.TRANSFORMS:
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
                    output = AGREEMENT_DIR / (
                        f"all_pairs_{window}_{transform}_{tag}.png"
                    )
                    agreement._all_pair_quicklook(
                        result,
                        window,
                        transform,
                        output,
                        variable=variable,
                        subtitle=subtitle,
                    )
                    written.append(output)
    return written


if __name__ == "__main__":
    outputs = redraw()
    print(f"Redrew {len(outputs)} PNG files with boundaries:")
    for path in outputs:
        print(path)
