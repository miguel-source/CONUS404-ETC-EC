"""Read-only numerical audit of saved hurricane NC results for the Word report."""

from study_paths import PROJECT_ROOT, RESULTS_ROOT, METADATA_ROOT, REPO_ROOT, STATES_FILE, WATERSHEDS_FILE, DAILY_INPUT, EC_DIRECTORY, EC_INPUT, LOSO_INPUT, AGREEMENT_DIRECTORY, AGREEMENT_INPUT
from pathlib import Path
import json
import numpy as np
import xarray as xr

ROOT = (PROJECT_ROOT / 'ETC_StormEvents')
OUT = RESULTS_ROOT / 'hurricane_results_word'
SOURCES = {'ec': EC_INPUT, 'agreement': AGREEMENT_INPUT, 'loso': LOSO_INPUT, 'daily': DAILY_INPUT}

def main():
    datasets = {}
    for key, path in SOURCES.items():
        with xr.open_dataset(path, engine="netcdf4") as f:
            datasets[key] = f.load()
    ec, agreement, loso, daily = (datasets[k] for k in ("ec", "agreement", "loso", "daily"))
    weights = agreement.study_watershed_fraction.values.astype(float) * np.cos(np.deg2rad(agreement.lat.values))[:, None]
    total_weight = weights.sum()
    def mean(values, mask):
        values = np.asarray(values, dtype=float)
        keep = mask & np.isfinite(values)
        if not keep.any():
            return None
        return float(np.sum(values[keep] * weights[keep]) / weights[keep].sum())
    def support(mask):
        return {"cells": int(mask.sum()), "weighted_watershed_area_pct": float(100 * weights[mask].sum() / total_weight)}
    def rng(values, mask):
        v = np.asarray(values)[mask]
        return [float(np.nanmin(v)), float(np.nanmax(v))]
    storm_names = {str(s): str(n) for s, n in zip(daily.storm_id.values, daily.event_name.values)}
    output = {
        "sources": {k: str(p) for k, p in SOURCES.items()},
        "source_attributes": {k: dict(datasets[k].attrs) for k in ("ec", "agreement")},
        "area_weighting": "study_watershed_fraction * cos(latitude); denominator sums all stored study_watershed_fraction cells, including partial watershed cells outside the active mask",
        "primary_support_definition": "agreement.valid_agreement for dataset_pair=conus404__conus404ba, transform=raw; requires sample eligible, admissible primary EC point estimate, finite positive difference-error SD, and primary bootstrap_valid_fraction >=0.90",
        "reference_support_definition": "point: EC valid_primary_ec AND reference_diagnostics_valid (saved references average two triplet formulations); bootstrap-screened: additionally reference valid fraction >=0.90 and agreement pair valid mask",
        "sd_aggregation": "Area-weighted arithmetic mean of pixelwise sqrt(error variance); not sqrt(area-weighted mean error variance)",
        "confidence_interval": "Saved pixelwise 2.5th and 97.5th percentiles from admissible storm-block bootstrap draws; conditional on draw admissibility, not a confidence interval for a domain mean",
        "agreement_score": "2*Phi(-abs(temporal mean BA-minus-CONUS404 difference)/EC difference-error SD), bounded [0,1], higher indicates smaller mean discrepancy relative to modeled daily error spread; not a p-value or accuracy probability",
        "sample_mode": ec.attrs["sample_mode"],
        "spatial_interpretation": ec.attrs["primary_estimand"],
        "bootstrap_replicates": int(ec.attrs["bootstrap_replicates"]),
        "bootstrap_seed": int(ec.attrs["bootstrap_seed"]),
        "minimum_storms": int(ec.attrs["minimum_storms"]),
        "minimum_observations": {"core": int(ec.attrs["minimum_core_observations"]), "expanded": int(ec.attrs["minimum_expanded_observations"])},
        "minimum_valid_days_per_storm": int(ec.attrs["minimum_valid_days_per_storm"]),
        "storm_id_name_map": storm_names,
        "windows": {},
    }
    for window in ("core", "expanded"):
        e = ec.sel(window=window, transform="raw")
        a = agreement.sel(window=window, transform="raw", dataset_pair="conus404__conus404ba")
        mask = a.valid_agreement.values.astype(bool)
        primary = e.valid_primary_ec.values.astype(bool)
        reference = primary & e.reference_diagnostics_valid.values.astype(bool)
        reference_screen = agreement.valid_agreement.sel(window=window, transform="raw", dataset_pair="conus404__mrms").values.astype(bool)
        if window == "core":
            assert mask.sum() == 527 and reference.sum() == 339 and reference_screen.sum() == 18
        else:
            assert mask.sum() == 660 and reference.sum() == 401 and reference_screen.sum() == 56
        means = {p: {m: mean(e[m].sel(product=p), mask) for m in ("error_sd", "error_variance", "signal_variance", "estimated_r2_to_truth", "snr_db")} for p in ("conus404", "conus404ba")}
        lo = e.error_sd_difference_ba_minus_conus404_ci_low.values
        hi = e.error_sd_difference_ba_minus_conus404_ci_high.values
        assert np.isfinite(lo[mask]).all() and np.isfinite(hi[mask]).all()
        l = loso.sel(window=window, transform="raw")
        all_valid = l.valid_primary_ec.values.astype(bool).all(axis=0)
        fixed = mask & all_valid
        omissions = []
        for storm in l.omitted_storm.values:
            s = str(storm)
            q = l.sel(omitted_storm=storm)
            selected_dates = daily.date.where(daily.storm_id.isel(storm=daily.storm_index) == s, drop=True).values
            omissions.append({"storm_id": s, "storm": storm_names[s], "expanded_start": str(np.min(selected_dates))[:10], "expanded_end": str(np.max(selected_dates))[:10], "sd_conus404": mean(q.error_sd_conus404, fixed), "sd_conus404ba": mean(q.error_sd_conus404ba, fixed), "mean_delta_sd_ba_minus_conus404": mean(q.error_sd_difference_ba_minus_conus404, fixed), "all_primary_screen_cells_still_valid": bool(q.valid_primary_ec.values[mask].all())})
        dates = daily.date.values
        daykeep = np.ones(len(dates), dtype=bool)
        if window == "core":
            daykeep &= daily.is_core_day.values.astype(bool)
        entry = {
            "primary_bootstrap_screened_support": support(mask),
            "primary_full_sample_support": support(primary),
            "joint_reference_full_sample_support": support(reference),
            "joint_reference_bootstrap_screened_support": support(reference_screen),
            "unique_calendar_days": int(np.unique(dates[daykeep]).size),
            "products_on_primary_support": means,
            "mean_delta_sd_ba_minus_conus404": mean(e.error_sd_difference_ba_minus_conus404, mask),
            "relative_change_ratio_of_mean_sd_pct": 100 * (means["conus404ba"]["error_sd"] / means["conus404"]["error_sd"] - 1),
            "mean_model_error_correlation": mean(e.error_correlation_model_pair, mask),
            "area_pct_conditional_paired_ci": {"lower_BA_sd": 100 * mean(hi < 0, mask), "higher_BA_sd": 100 * mean(lo > 0, mask), "includes_zero": 100 * mean((lo <= 0) & (hi >= 0), mask)},
            "primary_bootstrap_valid_fraction_range": rng(e.bootstrap_valid_fraction, mask),
            "primary_full_sample_bootstrap_valid_fraction_range": rng(e.bootstrap_valid_fraction, primary),
            "reference_full_sample_bootstrap_valid_fraction_range": rng(e.bootstrap_reference_valid_fraction, reference),
            "n_observations_range_on_primary_support": rng(e.n_observations, mask),
            "n_contributing_storms_range_on_primary_support": rng(e.n_contributing_storms, mask),
            "agreement_on_primary_support": {name: mean(a[name], mask) for name in ("mean_difference", "median_difference", "mean_absolute_daily_difference", "ec_difference_error_sd", "agreement_score_mean_difference", "agreement_score", "standardized_mean_difference")},
            "loso": {"fixed_support": support(fixed), "fixed_support_same_as_primary": bool(np.array_equal(fixed, mask)), "full_sample_delta_sd_on_fixed_support": mean(e.error_sd_difference_ba_minus_conus404, fixed), "omissions": omissions},
        }
        output["windows"][window] = entry
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "verified_results_metrics.json"
    path.write_text(json.dumps(output, indent=2, default=lambda v: v.item() if isinstance(v, np.generic) else str(v)), encoding="utf-8")
    print(path)
    print(json.dumps(output["windows"], indent=2))

if __name__ == "__main__":
    main()
