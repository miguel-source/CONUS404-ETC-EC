"""Make the two primary ETC figures for the short Word results report.

Source NetCDFs are read only.  The primary comparison mask is independent
of the stricter MRMS/PERSIANN reference-diagnostic bootstrap mask.
"""
from __future__ import annotations

from study_paths import PROJECT_ROOT, RESULTS_ROOT, METADATA_ROOT, REPO_ROOT, STATES_FILE, WATERSHEDS_FILE, DAILY_INPUT, EC_DIRECTORY, EC_INPUT, LOSO_INPUT, AGREEMENT_DIRECTORY, AGREEMENT_INPUT


import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FixedLocator
import numpy as np
import xarray as xr

from map_overlays import add_map_boundaries

HERE = RESULTS_ROOT
OUT = HERE / "hurricane_results_word" / "figures"
BASE = (PROJECT_ROOT / 'ETC_StormEvents')
EC = EC_INPUT
AGREEMENT = AGREEMENT_INPUT
EXTENT = (-105.5, -71.5, 26.0, 45.0)
WINDOWS = ("core", "expanded")


def edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    mid = (values[:-1] + values[1:]) / 2
    return np.concatenate(([values[0] - (mid[0] - values[0])], mid,
                           [values[-1] + (values[-1] - mid[-1])]))


def decorate(ax, *, left: bool, bottom: bool) -> None:
    ax.set_xlim(EXTENT[:2])
    ax.set_ylim(EXTENT[2:])
    add_map_boundaries(ax)
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(FixedLocator([-100, -90, -80]))
    ax.yaxis.set_major_locator(FixedLocator([30, 40]))
    ax.tick_params(axis="both", labelsize=10.0, length=3, width=0.6,
                   labelleft=left, labelbottom=bottom, pad=2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)
        spine.set_color("#555555")


def save(fig, basename: str) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"{basename}.{suffix}", dpi=400, facecolor="white")
    plt.close(fig)


def figure4() -> None:
    """Direct paired mean precipitation difference; it is not bias to truth."""
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "Times New Roman", "font.size": 11,
        "axes.titlesize": 12, "axes.labelsize": 11,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    with xr.open_dataset(AGREEMENT, engine="netcdf4") as ag:
        selected = ag.sel(dataset_pair="conus404__conus404ba", transform="raw")
        mask = selected.valid_agreement.astype(bool).load()
        difference = selected.mean_difference.where(mask).load()
        lon_edges, lat_edges = edges(ag.lon.values), edges(ag.lat.values)
        limit = float(math.ceil(float(np.abs(difference).max()) / 5) * 5)
        fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.55), sharex=True, sharey=True)
        fig.subplots_adjust(left=0.075, right=0.845, bottom=0.20, top=0.90,
                            wspace=0.10)
        for col, window in enumerate(WINDOWS):
            ax = axes[col]
            artist = ax.pcolormesh(
                lon_edges, lat_edges, difference.sel(window=window), cmap="RdBu_r",
                vmin=-limit, vmax=limit, shading="flat", rasterized=True, zorder=1,
            )
            decorate(ax, left=col == 0, bottom=True)
            ax.set_title(f"({'ab'[col]}) {window.capitalize()}", pad=4)
        cax = fig.add_axes([0.865, 0.26, 0.025, 0.60])
        cb = fig.colorbar(artist, cax=cax)
        cb.set_label("Mean ΔP (mm day$^{-1}$)", labelpad=6, fontsize=11)
        cb.ax.tick_params(labelsize=10, width=0.6, length=3)
        fig.text(0.46, 0.07, "Longitude (°)", ha="center", fontsize=11)
        fig.text(0.012, 0.56, "Latitude (°)", va="center", rotation=90, fontsize=11)
        save(fig, "figure4_mean_precipitation_difference")
        meta = {
            "source": str(AGREEMENT),
            "variable": "mean_difference",
            "dataset_pair": "conus404__conus404ba", "transform": "raw",
            "definition": "Temporal mean of CONUS404BA minus CONUS404 daily precipitation; not bias against truth",
            "support": "primary valid_agreement",
            "counts": {w: int(mask.sel(window=w).sum()) for w in WINDOWS},
            "actual_min": float(difference.min()), "actual_max": float(difference.max()),
            "color_limits": [-limit, limit], "extent_lonlat": list(EXTENT),
        }
        weights = ag.study_watershed_fraction.values.astype(float) * np.cos(
            np.deg2rad(ag.lat.values)
        )[:, None]
        meta["area_weighted_mean_difference_mm_day"] = {
            w: float(np.nansum(difference.sel(window=w).values * weights) /
                     weights[mask.sel(window=w).values].sum())
            for w in WINDOWS
        }
        (OUT / "figure4_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(json.dumps(meta, indent=2))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "Times New Roman", "font.size": 11,
        "axes.titlesize": 12, "axes.labelsize": 11,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "hatch.linewidth": 0.35,
    })
    with xr.open_dataset(EC, engine="netcdf4") as ds, xr.open_dataset(AGREEMENT, engine="netcdf4") as ag:
        mask = ag.valid_agreement.sel(
            dataset_pair="conus404__conus404ba", transform="raw"
        ).astype(bool).load()
        expected = ds.valid_primary_ec.sel(transform="raw").astype(bool) & (
            ds.bootstrap_valid_fraction.sel(transform="raw") >= 0.9
        )
        np.testing.assert_array_equal(mask.values, expected.values)
        counts = {w: int(mask.sel(window=w).sum()) for w in WINDOWS}
        assert counts == {"core": 527, "expanded": 660}, counts
        lon = ds.lon.values
        lat = ds.lat.values
        lon_edges, lat_edges = edges(lon), edges(lat)
        sd = ds.error_sd.sel(transform="raw", product=["conus404", "conus404ba"]).where(mask).load()
        sd_max = float(sd.max())
        sd_lim = float(math.ceil(sd_max / 10) * 10)
        fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.45), sharex=True, sharey=True)
        fig.subplots_adjust(left=0.075, right=0.845, bottom=0.105, top=0.925,
                            hspace=0.26, wspace=0.10)
        labels = ("CONUS404", "CONUS404BA")
        letters = iter("abcd")
        for row, window in enumerate(WINDOWS):
            for col, product in enumerate(("conus404", "conus404ba")):
                ax = axes[row, col]
                artist = ax.pcolormesh(
                    lon_edges, lat_edges, sd.sel(window=window, product=product),
                    cmap="YlGnBu", vmin=0, vmax=sd_lim, shading="flat",
                    rasterized=True, zorder=1,
                )
                decorate(ax, left=col == 0, bottom=row == 1)
                ax.set_title(f"({next(letters)}) {window.capitalize()}: {labels[col]}", pad=4)
        cax = fig.add_axes([0.865, 0.15, 0.025, 0.70])
        cb = fig.colorbar(artist, cax=cax)
        cb.set_label("Random-error SD (mm day$^{-1}$)", labelpad=6, fontsize=11)
        cb.ax.tick_params(labelsize=10, width=0.6, length=3)
        fig.text(0.46, 0.025, "Longitude (°)", ha="center", fontsize=11)
        fig.text(0.012, 0.52, "Latitude (°)", va="center", rotation=90, fontsize=11)
        save(fig, "figure2_primary_error_sd")

        delta = ds.error_sd_difference_ba_minus_conus404.sel(transform="raw").where(mask).load()
        low = ds.error_sd_difference_ba_minus_conus404_ci_low.sel(transform="raw").load()
        high = ds.error_sd_difference_ba_minus_conus404_ci_high.sel(transform="raw").load()
        finite_intervals = np.isfinite(low) & np.isfinite(high)
        spans_zero = mask & finite_intervals & (low <= 0) & (high >= 0)
        max_abs = float(np.abs(delta).max())
        delta_lim = float(math.ceil(max_abs / 5) * 5)
        fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.55), sharex=True, sharey=True)
        fig.subplots_adjust(left=0.075, right=0.845, bottom=0.25, top=0.90,
                            wspace=0.10)
        for col, window in enumerate(WINDOWS):
            ax = axes[col]
            artist = ax.pcolormesh(
                lon_edges, lat_edges, delta.sel(window=window), cmap="RdBu_r",
                vmin=-delta_lim, vmax=delta_lim, shading="flat",
                rasterized=True, zorder=1,
            )
            rr, cc = np.where(spans_zero.sel(window=window).values)
            patches = [Rectangle(
                (min(lon_edges[j], lon_edges[j+1]), min(lat_edges[i], lat_edges[i+1])),
                abs(lon_edges[j+1] - lon_edges[j]), abs(lat_edges[i+1] - lat_edges[i])
            ) for i, j in zip(rr, cc)]
            pc = PatchCollection(patches, facecolors="none", edgecolors="#424242",
                                 linewidths=0, hatch="////", zorder=4)
            ax.add_collection(pc)
            decorate(ax, left=col == 0, bottom=True)
            ax.set_title(f"({'ab'[col]}) {window.capitalize()}", pad=4)
        cax = fig.add_axes([0.865, 0.29, 0.025, 0.57])
        cb = fig.colorbar(artist, cax=cax, ticks=[-20, -10, 0, 10, 20])
        cb.set_label("BA − CONUS404 SD (mm day$^{-1}$)", labelpad=6, fontsize=10)
        cb.ax.tick_params(labelsize=10, width=0.6, length=3)
        fig.text(0.46, 0.16, "Longitude (°)", ha="center", fontsize=11)
        fig.text(0.012, 0.56, "Latitude (°)", va="center", rotation=90, fontsize=11)
        fig.legend(
            handles=[Patch(facecolor="white", edgecolor="#424242", hatch="////",
                           linewidth=0.4, label="Conditional bootstrap interval spans zero")],
            loc="lower center", bbox_to_anchor=(0.46, 0.005), frameon=False,
            fontsize=10, handlelength=1.8, handleheight=1.1,
        )
        save(fig, "figure3_paired_error_sd_difference")
        meta = {
            "source_ec": str(EC), "source_agreement": str(AGREEMENT),
            "transform": "raw", "extent_lonlat": list(EXTENT),
            "support": "valid_primary_ec AND primary bootstrap_valid_fraction >= 0.90; equal to primary valid_agreement",
            "counts": counts, "sd_max": sd_max, "sd_color_limits": [0, sd_lim],
            "delta_min": float(delta.min()), "delta_max": float(delta.max()),
            "delta_color_limits": [-delta_lim, delta_lim],
            "hatching": "Finite conditional paired-difference bootstrap interval includes zero; no unconditional inference implied",
            "intervals": "2.5th and 97.5th percentiles conditional on admissible bootstrap replicates",
            "hatched_counts": {w: int(spans_zero.sel(window=w).sum()) for w in WINDOWS},
            "reference_mask_used": False,
        }
        (OUT / "primary_figures_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    if "--figure4-only" in sys.argv:
        figure4()
    else:
        main()
        figure4()
