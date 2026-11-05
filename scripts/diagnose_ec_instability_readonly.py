"""Read-only local-support, storm-influence and reference-variance diagnostics."""

from study_paths import PROJECT_ROOT, RESULTS_ROOT, METADATA_ROOT, REPO_ROOT, STATES_FILE, WATERSHEDS_FILE, DAILY_INPUT, EC_DIRECTORY, EC_INPUT, LOSO_INPUT, AGREEMENT_DIRECTORY, AGREEMENT_INPUT
from pathlib import Path
import json
import numpy as np
import xarray as xr
from run_gridded_extended_collocation import (
    _storm_sufficient_statistics, _point_statistics, _metrics_from_covariance,
)
from run_gridded_dataset_agreement import _selected_observations

root = (PROJECT_ROOT / 'ETC_StormEvents')
with xr.open_dataset(root / "Gridded_PERSIANN/hurricane_daily_all4_persiann_grid.nc", engine="netcdf4") as d:
    daily = d.load()
with xr.open_dataset(root / "Hurricane_gridded_EC_results/gridded_extended_collocation_all_windows.nc", engine="netcdf4") as d:
    saved = d.load()
domain = daily.study_domain_mask.values.astype(bool)
active = np.flatnonzero(domain.ravel())
weights = (daily.study_watershed_fraction.values.astype(float) * np.cos(np.deg2rad(daily.lat.values))[:, None]).ravel()[active]
weights /= weights.sum()
assigned = daily.assigned_storm_mask.values.reshape(daily.sizes["storm"], -1)[:, active].astype(bool)
storm_names = [str(x) for x in daily.event_name.values]
counts_assigned = assigned.sum(axis=0)
output = {"assigned_storm_counts_cells": {str(n): int((counts_assigned == n).sum()) for n in np.unique(counts_assigned)},
          "assigned_storm_counts_area_pct_within_active_domain": {str(n): float(weights[counts_assigned == n].sum() * 100) for n in np.unique(counts_assigned)}}

def area_pct(mask):
    return float(weights[np.asarray(mask)].sum() * 100)

for window in ["core", "expanded"]:
    positions, _ = _selected_observations(daily, window)
    sample = daily.isel(observation=positions)
    storms = sample.storm_index.values.astype(int)
    raw = np.stack([sample[p].values.reshape(len(positions), -1)[:, active].astype(float)
                    for p in ["conus404", "conus404ba", "mrms", "persiann_cdr"]], axis=1)
    base = np.isfinite(raw).all(axis=1) & (raw >= 0).all(axis=1)
    if "valid_all4" in sample:
        base &= sample.valid_all4.values.reshape(len(positions), -1)[:, active].astype(bool)
    result = {}
    for transform in ["raw", "log1p"]:
        values = raw if transform == "raw" else np.log1p(raw)
        n, sums, cross, _, matched = _storm_sufficient_statistics(values, base, storms, 10, 3)
        total_n, contributing, cov, total_sums = _point_statistics(n, sums, cross)
        m = _metrics_from_covariance(cov)
        original = saved.sel(window=window, transform=transform)
        expected = original.error_variance.values.reshape(4, -1)[:, active].T
        np.testing.assert_allclose(m["error_variance"], expected, rtol=2e-6, atol=1e-5, equal_nan=True)
        good = m["valid_primary_ec"] & m["reference_diagnostics_valid"]
        reference_error = m["error_variance"][:, 2]
        mrms_positive = np.isfinite(reference_error) & (reference_error > 0)
        info = {
            "reproduces_saved_error_variances": True,
            "nonpositive_mrms_error_variance_cells": int((np.isfinite(reference_error) & (reference_error <= 0)).sum()),
            "full_sample_joint_valid_cells": int(good.sum()),
        }
        # Positive contributions to the overall, globally centered MRMS sum of
        # squares, grouped by calendar storm block. This diagnoses leverage;
        # it is not the number of independent meteorological events.
        mrms = values[:, 2]
        mean = total_sums[2] / total_n
        deviation_sq = np.where(matched, (mrms - mean[None]) ** 2, 0)
        by_storm = np.stack([deviation_sq[storms == s].sum(axis=0) for s in range(10)])
        shares = by_storm / by_storm.sum(axis=0, keepdims=True)
        top_share = shares.max(axis=0)
        info["mrms_variability_one_block_dominance"] = {
            "weighted_mean_largest_block_fraction": float(np.sum(weights * top_share)),
            "active_domain_area_pct_top_block_gt_50pct": area_pct(top_share > 0.5),
            "active_domain_area_pct_top_block_gt_80pct": area_pct(top_share > 0.8),
            "largest_block_area_pct": {name: area_pct(shares.argmax(axis=0) == s) for s, name in enumerate(storm_names)},
        }
        changes = []
        any_ref_failure = np.zeros(len(active), dtype=bool)
        for s, name in enumerate(storm_names):
            keep_n = total_n - n[s]
            keep_sums = sums.sum(axis=0) - sums[s]
            keep_cross = cross.sum(axis=0) - cross[s]
            # Reuse the exactly same sufficient-statistics covariance function.
            _, _, covariance, _ = _point_statistics(keep_n[None], keep_sums[None], keep_cross[None])
            omitted = _metrics_from_covariance(covariance)
            now_invalid = ~(omitted["valid_primary_ec"] & omitted["reference_diagnostics_valid"])
            any_ref_failure |= good & now_invalid
            e2 = omitted["error_variance"][:, 2]
            flip = mrms_positive & ((~np.isfinite(e2)) | (e2 <= 0))
            changes.append({"omitted_storm": name,
                            "nonpositive_mrms_error_cells_after_omission": int((np.isfinite(e2) & (e2 <= 0)).sum()),
                            "previously_positive_mrms_cells_now_nonpositive_or_undefined": int(flip.sum()),
                            "original_joint_valid_cells_now_invalid": int((good & now_invalid).sum())})
        info["leave_one_storm_out_reference_checks"] = changes
        info["full_sample_joint_valid_cells_failing_any_omission"] = int(any_ref_failure.sum())
        info["fraction_of_joint_valid_cells_failing_any_omission"] = float(any_ref_failure.sum() / good.sum())
        model_sat_corr = cov[:, 0, 3] / np.sqrt(cov[:, 0, 0] * cov[:, 3, 3])
        info["conus404_persiann_observed_correlation_quantiles"] = np.quantile(model_sat_corr[np.isfinite(model_sat_corr)], [0, .25, .5, .75, 1]).tolist()
        result[transform] = info
    output[window] = result
print(json.dumps(output, indent=2))
