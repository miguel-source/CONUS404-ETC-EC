"""Read-only direct daily discrepancies on the broad, matched study support.

No collocation admissibility or bootstrap screen is imposed. Positive bias
means the CONUS404 product is wetter than the stated reference.
"""

from study_paths import PROJECT_ROOT, RESULTS_ROOT, METADATA_ROOT, REPO_ROOT, STATES_FILE, WATERSHEDS_FILE, DAILY_INPUT, EC_DIRECTORY, EC_INPUT, LOSO_INPUT, AGREEMENT_DIRECTORY, AGREEMENT_INPUT
from pathlib import Path
import json
import warnings
import numpy as np
import xarray as xr
from run_gridded_dataset_agreement import _selected_observations, _matched_sample

ROOT = (PROJECT_ROOT / 'ETC_StormEvents')
SOURCE = DAILY_INPUT
OUT = RESULTS_ROOT / 'hurricane_results_word/direct_reference_agreement.json'
PRODUCTS = ("conus404", "conus404ba", "mrms", "persiann_cdr")

def main():
    with xr.open_dataset(SOURCE, engine="netcdf4") as f:
        daily = f.load()
    domain = daily.study_domain_mask.values.astype(bool)
    weights = daily.study_watershed_fraction.values.astype(float) * np.cos(np.deg2rad(daily.lat.values))[:, None]
    def mean(values, mask):
        v = np.asarray(values, dtype=float)
        keep = mask & np.isfinite(v)
        assert keep.any()
        return float(np.sum(v[keep] * weights[keep]) / weights[keep].sum())
    result = {
        "source": str(SOURCE),
        "support_definition": "All-four-product complete, finite, nonnegative matched daily series within study_domain_mask; at least 3 matched days per contributing storm, at least 8 contributing storms, and at least 60 core or 100 expanded observations. No EC admissibility or bootstrap criterion. One common spatial support and one common matched temporal mask for all four pairs within each window.",
        "date_definition": "Select is_core_day for core or all observations for expanded; verify duplicate date fields are identical then retain first observation record for each unique calendar date in input EVENTS order.",
        "weighting": "study_watershed_fraction*cos(latitude); coverage denominator includes all stored study_watershed_fraction cells",
        "difference_direction": "model minus reference; positive bias means CONUS404/BA is wetter",
        "bias_definition": "At each pixel, temporal arithmetic mean of matched daily model-minus-reference precipitation, followed by weighted spatial arithmetic mean.",
        "mae_definition": "At each pixel, temporal arithmetic mean of absolute matched daily model-minus-reference precipitation, followed by weighted spatial arithmetic mean.",
        "rmse_definition": "At each pixel, sqrt(temporal arithmetic mean of squared matched daily model-minus-reference precipitation), followed by weighted spatial arithmetic mean; this is a mean of pixelwise RMSE, not pooled domain RMSE.",
        "reference_interpretation": "MRMS and PERSIANN-CDR are comparison references, not error-free precipitation truth. Bias, MAE and RMSE quantify disagreement, not demonstrated model error.",
        "units": "mm day^-1",
        "windows": {},
    }
    for window in ("core", "expanded"):
        selected, discarded = _selected_observations(daily, window)
        sample = daily.isel(observation=selected)
        values = np.stack([sample[p].values.astype(float) for p in PRODUCTS], axis=1)
        matched, counts, storms = _matched_sample(values, sample.storm_index.values, daily.sizes["storm"], domain, 3)
        mask = domain & (counts >= (60 if window == "core" else 100)) & (storms >= 8)
        v = np.where(matched[:, None], values, np.nan)
        entry = {
            "cells": int(mask.sum()),
            "weighted_watershed_area_pct": float(100 * weights[mask].sum() / weights.sum()),
            "unique_calendar_days": int(sample.date.size),
            "duplicate_calendar_records_discarded": int(discarded),
            "matched_days_range_on_support": [int(counts[mask].min()), int(counts[mask].max())],
            "contributing_storms_range_on_support": [int(storms[mask].min()), int(storms[mask].max())],
            "pairs": {},
        }
        for model in ("conus404", "conus404ba"):
            for reference in ("mrms", "persiann_cdr"):
                diff = v[:, PRODUCTS.index(model)] - v[:, PRODUCTS.index(reference)]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    bias = np.nanmean(diff, axis=0)
                    mae = np.nanmean(np.abs(diff), axis=0)
                    rmse = np.sqrt(np.nanmean(diff ** 2, axis=0))
                    model_mean = np.nanmean(v[:, PRODUCTS.index(model)], axis=0)
                    reference_mean = np.nanmean(v[:, PRODUCTS.index(reference)], axis=0)
                for a in (bias, mae, rmse):
                    assert np.isfinite(a[mask]).all()
                entry["pairs"][f"{model}_minus_{reference}"] = {
                    "model": model,
                    "reference": reference,
                    "area_weighted_mean_pixel_bias": mean(bias, mask),
                    "area_weighted_mean_pixel_mae": mean(mae, mask),
                    "area_weighted_mean_pixel_rmse": mean(rmse, mask),
                    "area_weighted_mean_pixel_model_daily_precipitation": mean(model_mean, mask),
                    "area_weighted_mean_pixel_reference_daily_precipitation": mean(reference_mean, mask),
                    "cells": int(mask.sum()),
                }
        result["windows"][window] = entry
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(OUT)
    print(json.dumps(result["windows"], indent=2))

if __name__ == "__main__":
    main()
