"""Core raw six-panel TC figures, with unaveraged triplet-specific references."""
from __future__ import annotations

from study_paths import PROJECT_ROOT, RESULTS_ROOT, METADATA_ROOT, REPO_ROOT, STATES_FILE, WATERSHEDS_FILE, DAILY_INPUT, EC_DIRECTORY, EC_INPUT, LOSO_INPUT, AGREEMENT_DIRECTORY, AGREEMENT_INPUT


import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import FixedLocator
import numpy as np
import xarray as xr

from map_overlays import add_map_boundaries

ROOT = RESULTS_ROOT
BASE = ROOT / "separate_triplet_results"
OUT = ROOT / "hurricane_results_word" / "figures"
SOURCES = (
    BASE / "CONUS404_MRMS_PERSIANN_CDR/triplet_error_estimates_raw.nc",
    BASE / "CONUS404BA_MRMS_PERSIANN_CDR/triplet_error_estimates_raw.nc",
)
EXTENT = (-105.5, -71.5, 26, 45)
LABELS = {"conus404": "CONUS404", "conus404ba": "CONUS404BA",
          "mrms": "MRMS", "persiann_cdr": "PERSIANN-CDR"}


def edges(a):
    a = np.asarray(a, dtype=float)
    d = np.diff(a)
    assert np.allclose(d, d[0])
    return np.r_[a[0] - d[0]/2, a + d[0]/2]


def weighted_mean(da, weights, mask):
    v = da.values
    m = mask.values & np.isfinite(v)
    return float(np.sum(v[m] * weights[m]) / np.sum(weights[m]))


def create_figure(datasets, variable, basename, cmap, *, limits=None, bins=None, label):
    fig, axes = plt.subplots(2, 3, figsize=(6.5, 4.1), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.085, right=0.865, bottom=0.14, top=0.85,
                        hspace=0.5, wspace=0.10)
    panel_means = []
    for row, ds in enumerate(datasets):
        mask = ds.valid_both_triplets.sel(window="core").astype(bool)
        weights = ds.study_watershed_fraction.values * np.cos(np.deg2rad(ds.lat.values))[:, None]
        for col, product in enumerate(ds.product.values):
            ax = axes[row, col]
            da = ds[variable].sel(window="core", product=product).where(mask)
            kwargs = {"cmap": cmap}
            if bins is not None:
                kwargs["norm"] = BoundaryNorm(bins, cmap.N)
            else:
                kwargs.update(vmin=limits[0], vmax=limits[1])
            mesh = ax.pcolormesh(edges(ds.lon), edges(ds.lat), da, shading="flat",
                                 rasterized=True, zorder=1, **kwargs)
            ax.set_xlim(EXTENT[:2])
            ax.set_ylim(EXTENT[2:])
            add_map_boundaries(ax)
            # Narrow panels need lighter cartographic lines to retain pixel colors.
            for collection in ax.collections[1:]:
                if hasattr(collection, "get_linewidths"):
                    collection.set_linewidth(collection.get_linewidths() * 0.75)
            ax.set_aspect("equal", adjustable="box")
            ax.xaxis.set_major_locator(FixedLocator([-100, -90, -80]))
            ax.yaxis.set_major_locator(FixedLocator([30, 40]))
            ax.tick_params(labelsize=8.5, length=2.5, width=0.5, pad=1,
                           labelleft=col == 0, labelbottom=row == 1)
            for spine in ax.spines.values():
                spine.set_linewidth(0.55)
                spine.set_color("#555555")
            mean = weighted_mean(da, weights, mask)
            fmt = f"{mean:.3f}" if variable == "estimated_r2_to_truth" else f"{mean:.2f}"
            ax.set_title(f"({'abcdef'[row*3+col]}) {LABELS[str(product)]}\nMean = {fmt}",
                         fontsize=10.2, linespacing=1.1, pad=3)
            panel_means.append({"triplet": "A" if row == 0 else "B", "product": str(product),
                                "core_area_weighted_mean": mean})
    fig.text(0.475, 0.965, "CONUS404 triplet", ha="center", va="top", fontsize=11)
    fig.text(0.475, 0.515, "BA triplet", ha="center", va="top", fontsize=11)
    fig.text(0.475, 0.035, "Longitude (°)", ha="center", fontsize=10)
    fig.text(0.012, 0.51, "Latitude (°)", va="center", rotation=90, fontsize=10)
    cax = fig.add_axes([0.89, 0.18, 0.022, 0.63])
    if bins is not None:
        cb = fig.colorbar(mesh, cax=cax, boundaries=bins, ticks=bins,
                          spacing="uniform")
    else:
        cb = fig.colorbar(mesh, cax=cax)
    cb.set_label(label, fontsize=10, labelpad=5)
    cb.ax.tick_params(labelsize=8.5, length=2.5, width=0.5)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"{basename}.{suffix}", dpi=400, facecolor="white")
    plt.close(fig)
    return {"file": basename, "variable": variable, "limits": limits,
            "boundaries": bins, "panel_means": panel_means}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    datasets = [xr.open_dataset(p, engine="netcdf4").load() for p in SOURCES]
    masks = [ds.valid_both_triplets.astype(bool) for ds in datasets]
    np.testing.assert_array_equal(masks[0], masks[1])
    counts = {w: int(masks[0].sel(window=w).sum()) for w in ("core", "expanded")}
    assert counts == {"core": 314, "expanded": 382}, counts
    rows = []
    for name, ds in zip(("A", "B"), datasets):
        weights = ds.study_watershed_fraction.values.astype(float) * np.cos(np.deg2rad(ds.lat.values))[:, None]
        for window in ("core", "expanded"):
            mask = ds.valid_both_triplets.sel(window=window).astype(bool)
            for product in ds.product.values:
                row = {"triplet": name, "model": "conus404" if name == "A" else "conus404ba",
                       "product": str(product), "window": window,
                       "cells": int(mask.sum()),
                       "weighted_area_coverage_percent": float(weights[mask.values].sum()/weights.sum()*100)}
                for variable in ("error_variance", "error_sd", "estimated_r2_to_truth"):
                    row[variable] = weighted_mean(ds[variable].sel(window=window, product=product), weights, mask)
                rows.append(row)
    core_var = [ds.error_variance.sel(window="core").where(ds.valid_both_triplets.sel(window="core")) for ds in datasets]
    maximum_var = max(float(x.max()) for x in core_var)
    variance_bins = [0, 5, 10, 25, 50, 100, 200, 400, 800, 1600, 3200, 6400]
    while variance_bins[-1] < maximum_var:
        variance_bins.append(variance_bins[-1]*2)
    sd_max = max(float(ds.error_sd.sel(window="core").where(ds.valid_both_triplets.sel(window="core")).max()) for ds in datasets)
    sd_limit = float(math.ceil(sd_max/10)*10)
    figs = [
        create_figure(datasets, "error_variance", "figure2_triplet_error_variance",
                      plt.get_cmap("YlGnBu", len(variance_bins)-1), bins=variance_bins,
                      label="Error variance [(mm day$^{-1}$)$^2$]"),
        create_figure(datasets, "error_sd", "figure3_triplet_tc_rmse",
                      plt.get_cmap("YlGnBu"), limits=[0, sd_limit],
                      label="TC error SD (mm day$^{-1}$)"),
        create_figure(datasets, "estimated_r2_to_truth", "figure4_triplet_latent_r2",
                      plt.get_cmap("viridis"), limits=[0, 1],
                      label="Estimated latent-signal $R^2$"),
    ]
    metadata = {"sources": [str(p) for p in SOURCES], "transform": "raw",
                "counts": counts, "scope": "Core raw maps; core and expanded metrics provided",
                "mask": "valid_both_triplets; common all-three admissibility across both triplets",
                "averaging": "No reference averaging; each product belongs to its own triplet",
                "bootstrap_filter": False, "extent_lonlat": list(EXTENT),
                "variance_bin_note": "Nonlinear color intervals of raw variance; data are not transformed",
                "rmse_note": "TC random-error SD, not observed-reference total RMSE",
                "latent_r2_note": "Estimated squared correlation with latent precipitation signal; not R2 against MRMS",
                "variance_maximum": maximum_var, "sd_maximum": sd_max,
                "figures": figs, "metric_rows": rows}
    (OUT / "triplet_metric_figures_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"counts": counts, "variance_maximum": maximum_var,
                      "sd_maximum": sd_max, "metric_rows": rows}, indent=2))
    for ds in datasets:
        ds.close()


if __name__ == "__main__":
    main()
