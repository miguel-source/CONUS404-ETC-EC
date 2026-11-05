"""Plot raw EC variance and SD on jointly admissible four-product support.

Uses saved full-sample flags, not the 90% reference-bootstrap screen.
Source NetCDFs are read-only. Output PNG/SVG/JSON files stay in this workspace.
Jupyter: %run followed by the absolute path of this script.
"""

from study_paths import PROJECT_ROOT, RESULTS_ROOT, METADATA_ROOT, REPO_ROOT, STATES_FILE, WATERSHEDS_FILE, DAILY_INPUT, EC_DIRECTORY, EC_INPUT, LOSO_INPUT, AGREEMENT_DIRECTORY, AGREEMENT_INPUT

from pathlib import Path
import json
import os

HERE = RESULTS_ROOT
OUTPUT = HERE / "raw_full_sample_error_variance_maps"
os.environ.setdefault("MPLCONFIGDIR", str(HERE / "_core_cells_mpl"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
import numpy as np
import xarray as xr

from map_overlays import add_map_boundaries

ROOT = (PROJECT_ROOT / 'ETC_StormEvents')
EC_SOURCE = ROOT / "Hurricane_gridded_EC_results" / "gridded_extended_collocation_all_windows.nc"
AGREEMENT_SOURCE = ROOT / "Hurricane_gridded_agreement_results" / "gridded_dataset_agreement.nc"
PRODUCTS = ("mrms", "conus404", "conus404ba", "persiann_cdr")
LABELS = {"mrms": "MRMS", "conus404": "CONUS404", "conus404ba": "CONUS404BA", "persiann_cdr": "PERSIANN-CDR"}
EXTENT = (-105.5, -71.5, 26.0, 45.0)


def coordinate_edges(values):
    values = np.asarray(values, dtype=float)
    step = np.diff(values)
    if len(values) < 2 or not np.allclose(step, step[0]):
        raise ValueError("Expected a regular grid.")
    return np.r_[values[0] - step[0] / 2, (values[:-1] + values[1:]) / 2, values[-1] + step[0] / 2]


def main():
    with xr.open_dataset(EC_SOURCE, engine="netcdf4") as f:
        ec = f.sel(transform="raw").load()
    with xr.open_dataset(AGREEMENT_SOURCE, engine="netcdf4") as f:
        agreement = f.sel(transform="raw").load()
    xr.testing.assert_equal(ec.lat, agreement.lat)
    xr.testing.assert_equal(ec.lon, agreement.lon)
    domain = agreement.study_domain_mask.values.astype(bool)
    weights = agreement.study_watershed_fraction.values.astype(float) * np.cos(np.deg2rad(ec.lat.values))[:, None]
    xe, ye = coordinate_edges(ec.lon.values), coordinate_edges(ec.lat.values)
    if not np.allclose(abs(np.diff(xe)), 0.25) or not np.allclose(abs(np.diff(ye)), 0.25):
        raise ValueError("Expected the shared 0.25-degree PERSIANN-CDR grid.")
    data = {}
    for window, expected_count in (("core", 339), ("expanded", 401)):
        q = ec.sel(window=window)
        # Use one common mask for all four products; do not cherry-pick
        # a separate set of positive estimates for each product.
        mask = q.valid_primary_ec.values.astype(bool) & q.reference_diagnostics_valid.values.astype(bool)
        if mask.sum() != expected_count or np.any(mask & ~domain):
            raise ValueError(f"Unexpected support for {window}: {mask.sum()} cells.")
        coverage = float(100 * weights[mask].sum() / weights.sum())
        fields, means = {}, {}
        for metric in ("error_variance", "error_sd"):
            fields[metric], means[metric] = {}, {}
            for product in PRODUCTS:
                values = q[metric].sel(product=product).transpose("lat", "lon").values.astype(float)
                if not np.all(np.isfinite(values[mask]) & (values[mask] > 0)):
                    raise ValueError(f"Invalid retained values: {window}/{metric}/{product}")
                fields[metric][product] = np.where(mask, values, np.nan)
                means[metric][product] = float(np.sum(weights[mask] * values[mask]) / weights[mask].sum())
        data[window] = {"mask": mask, "coverage": coverage, "fields": fields, "means": means}

    max_variance = max(np.nanmax(data[w]["fields"]["error_variance"][p]) for w in data for p in PRODUCTS)
    # Unequal, explicitly labeled absolute-value bins distinguish small values
    # while still including the largest extremes. No values are log-transformed,
    # winsorized, clipped, interpolated, or changed before plotting.
    candidate_bounds = [0, 5, 10, 25, 50, 100, 200, 400, 800, 1600, 3200, 6400]
    while candidate_bounds[-1] < max_variance:
        candidate_bounds.append(2 * candidate_bounds[-1])
    upper = next(i for i, value in enumerate(candidate_bounds) if value >= max_variance)
    variance_bounds = candidate_bounds[:upper + 1]
    variance_cmap = plt.get_cmap("viridis", len(variance_bounds) - 1)
    variance_norm = BoundaryNorm(variance_bounds, variance_cmap.N)
    max_sd = max(np.nanmax(data[w]["fields"]["error_sd"][p]) for w in data for p in PRODUCTS)
    sd_limit = float(np.ceil(max_sd / 10) * 10)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titleweight": "semibold", "svg.fonttype": "none", "savefig.facecolor": "white"})
    OUTPUT.mkdir(parents=True, exist_ok=True)

    metadata = {"ec_source": str(EC_SOURCE), "agreement_source_for_domain_and_weights": str(AGREEMENT_SOURCE),
                "sampling_scope": "all_windows", "transform": "raw",
                "mask": "valid_primary_ec AND reference_diagnostics_valid, same cells for all products",
                "bootstrap_screen_applied": False,
                "variance_units": "(mm/day)^2", "sd_units": "mm/day",
                "variance_color_boundaries": variance_bounds, "sd_color_limits": [0, sd_limit],
                "weighting": "study_watershed_fraction*cos(latitude); coverage denominator includes all stored watershed fractions",
                "note": "Full-sample admissibility is not proof of true errors or bootstrap stability; no full-domain ranking.", "windows": {}}
    for window, d in data.items():
        result = {"retained_cells": int(d["mask"].sum()), "weighted_watershed_area_pct": d["coverage"],
                  "area_weighted_means_of_pixelwise_metrics": d["means"], "figures": {}}
        for metric in ("error_variance", "error_sd"):
            variance = metric == "error_variance"
            metric_label = "random-error variance" if variance else "random-error standard deviation"
            units = "(mm/day)²" if variance else "mm/day"
            cmap = variance_cmap if variance else "viridis"
            norm = variance_norm if variance else Normalize(0, sd_limit)
            fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.5), sharex=True, sharey=True)
            fig.subplots_adjust(left=0.065, right=0.865, top=0.865, bottom=0.13, wspace=0.13, hspace=0.26)
            for ax, product, letter in zip(axes.ravel(), PRODUCTS, "abcd"):
                ax.set_xlim(EXTENT[:2])
                ax.set_ylim(EXTENT[2:])
                ax.pcolormesh(xe, ye, np.where(domain, 1.0, np.nan), cmap=ListedColormap(["#e1e7ec"]), vmin=0, vmax=1, shading="flat", zorder=1)
                artist = ax.pcolormesh(xe, ye, d["fields"][metric][product], cmap=cmap, norm=norm, shading="flat", zorder=3)
                add_map_boundaries(ax)
                mean_label = "mean error variance" if variance else "mean error SD"
                ax.set_title(f"({letter}) {LABELS[product]}\nArea-weighted {mean_label}: {d['means'][metric][product]:.2f} {units}", fontsize=12)
                ax.set_xlabel("Longitude (°)")
                ax.set_ylabel("Latitude (°)")
                ax.tick_params(labelbottom=True, labelleft=True, labelsize=9)
            color_ax = fig.add_axes([0.895, 0.195, 0.023, 0.575])
            if variance:
                bar = fig.colorbar(artist, cax=color_ax, boundaries=variance_bounds, ticks=variance_bounds, spacing="uniform")
            else:
                bar = fig.colorbar(artist, cax=color_ax)
            bar.set_label(f"EC-estimated {metric_label} ({units})")
            bar.ax.tick_params(labelsize=9)
            fig.suptitle(f"{window.capitalize()} windows • raw precipitation • EC {metric_label}\n{int(d['mask'].sum())} jointly admissible cells | {d['coverage']:.2f}% of weighted watershed area", fontsize=17, y=0.965)
            scale_note = "Shared absolute-value color bins; no transformation of variance values." if variance else "Shared linear color scale."
            fig.text(0.5, 0.060, scale_note + " Gray = study cells outside the joint-admissibility mask.", ha="center", fontsize=10)
            fig.text(0.5, 0.033, "Full-sample point estimates; no ≥90% bootstrap-validity screen. Watershed and CONUS/state boundaries shown.", ha="center", fontsize=10, color="#404040")
            stem = OUTPUT / f"{window}_raw_jointly_admissible_{metric}_four_products"
            paths = {}
            for extension in ("png", "svg"):
                path = stem.with_suffix("." + extension)
                fig.savefig(path, dpi=300)
                paths[extension] = str(path)
            plt.close(fig)
            result["figures"][metric] = paths
        rows, cols = np.where(d["mask"])
        result["cells"] = [{"latitude": float(ec.lat.values[row]), "longitude": float(ec.lon.values[col]),
                            "error_variance": {p: float(d["fields"]["error_variance"][p][row, col]) for p in PRODUCTS},
                            "error_sd": {p: float(d["fields"]["error_sd"][p][row, col]) for p in PRODUCTS}}
                           for row, col in zip(rows, cols)]
        metadata["windows"][window] = result
    (OUTPUT / "raw_jointly_admissible_maps_provenance.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"variance_color_boundaries": variance_bounds, "windows": {w: {k: v for k, v in result.items() if k != "cells"} for w, result in metadata["windows"].items()}}, indent=2))


if __name__ == "__main__":
    main()
