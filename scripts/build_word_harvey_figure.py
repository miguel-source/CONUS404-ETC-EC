"""Four-product raw Harvey storm totals for the Word results report.

Original archives are read only. Totals require all 17 expanded-window UTC
days, source-area QC, and identical spatial support in all four products.
"""
from __future__ import annotations

from study_paths import PROJECT_ROOT, RESULTS_ROOT, METADATA_ROOT, REPO_ROOT, STATES_FILE, WATERSHEDS_FILE, DAILY_INPUT, EC_DIRECTORY, EC_INPUT, LOSO_INPUT, AGREEMENT_DIRECTORY, AGREEMENT_INPUT


import json
import os
from pathlib import Path

ROOT = RESULTS_ROOT
OUT = ROOT / "hurricane_results_word"
os.environ.setdefault("MPLCONFIGDIR", str(OUT / "_mpl_harvey"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.font_manager import findfont, FontProperties
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from shapely.ops import unary_union
import xarray as xr

from map_overlays import load_conus_layers, load_watershed_layers, _line_segments

EVENT_DIR = (PROJECT_ROOT / 'ETC_StormEvents/Gridded_PERSIANN/events')
PRODUCTS = ["mrms", "conus404", "conus404ba", "persiann_cdr"]
LABELS = ["MRMS", "CONUS404", "CONUS404BA", "PERSIANN-CDR"]
STORM_ID = "AL092017"


def edges(values):
    values = np.asarray(values, dtype=float)
    delta = np.diff(values)
    assert np.allclose(delta, delta[0])
    return np.r_[values[0] - delta[0] / 2, (values[1:] + values[:-1]) / 2,
                 values[-1] + delta[0] / 2]


def add_boundaries(ax, geometries, color, width, order):
    segments = [part for geom in geometries for part in _line_segments(geom)]
    ax.add_collection(LineCollection(segments, colors=color, linewidths=width,
                                     zorder=order))


def main():
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    inventory = []
    selected = None
    for path in sorted(EVENT_DIR.glob("all4_persiann_grid_*.nc")):
        with xr.open_dataset(path, engine="netcdf4") as ds:
            times = pd.DatetimeIndex(ds.time.values)
            inventory.append({"event_name": ds.attrs["event_name"],
                              "storm_id": ds.attrs["storm_id"],
                              "expanded_days": len(times),
                              "core_days": int(ds.is_core_day.sum().values),
                              "first_day": str(times.min().date()),
                              "last_day": str(times.max().date())})
            if ds.attrs["storm_id"] == STORM_ID:
                if selected is not None:
                    raise ValueError("Ambiguous Harvey event files")
                selected = path
    if selected is None:
        raise ValueError("Harvey archive not found")
    assert len(inventory) == 10
    assert max(item["expanded_days"] for item in inventory) == 17
    with xr.open_dataset(selected, engine="netcdf4") as src:
        ds = src.sortby("lat", ascending=False).sortby("lon").load()
    expected = pd.date_range("2017-08-19", "2017-09-04", freq="D")
    assert pd.DatetimeIndex(ds.time.values).equals(expected)
    assert ds.attrs["storm_id"] == STORM_ID
    assert int(ds.is_core_day.sum()) == 12
    assert ds.attrs["daily_interval"] == "[D 00:00 UTC, D+1 00:00 UTC)"
    lat_edges, lon_edges = edges(ds.lat), edges(ds.lon)
    assert np.allclose(np.diff(lon_edges), 0.25)
    assert np.allclose(np.diff(lat_edges), -0.25)
    domain = ds.event_watershed_mask.values.astype(bool)
    common = domain.copy()
    totals, finite_counts = [], {}
    threshold = float(ds.attrs["minimum_valid_source_area_fraction"])
    for product in PRODUCTS:
        daily = ds[product].transpose("time", "lat", "lon")
        assert daily.attrs["units"] == "mm", (product, daily.attrs)
        valid = np.isfinite(daily.values) & (daily.values >= 0)
        fraction = ds[f"{product}_valid_area_fraction"].transpose("time", "lat", "lon").values
        valid &= np.isfinite(fraction) & (fraction >= threshold - 1e-6)
        complete = valid.all(axis=0)
        finite_counts[product] = int((complete & domain).sum())
        common &= complete
        # Explicit no-skip sum: incomplete series cannot become a low total.
        totals.append(daily.astype(np.float64).sum("time", skipna=False).values)
    assert common.any()
    totals = np.stack(totals)
    totals[:, ~common] = np.nan

    # Cell-area weights times watershed overlap fraction, consistently with
    # the aligned archive and domain-weighted summaries elsewhere in the report.
    west, north = np.meshgrid(lon_edges[:-1], lat_edges[:-1])
    east, south = np.meshgrid(lon_edges[1:], lat_edges[1:])
    cells = shapely.box(west.ravel(), south.ravel(), east.ravel(), north.ravel())
    areas = gpd.GeoSeries(cells, crs="EPSG:4326").to_crs("EPSG:5070").area.to_numpy().reshape(common.shape)
    weights = areas * ds.event_watershed_fraction.values
    weights[~common] = 0
    product_metrics = []
    for index, product in enumerate(PRODUCTS):
        rr, cc = np.unravel_index(np.nanargmax(totals[index]), common.shape)
        product_metrics.append({
            "product": product, "label": LABELS[index],
            "area_weighted_mean_storm_total_mm": float(np.nansum(totals[index] * weights) / weights.sum()),
            "peak_grid_cell_storm_total_mm": float(totals[index, rr, cc]),
            "peak_latitude": float(ds.lat[rr]), "peak_longitude": float(ds.lon[cc]),
        })

    all_watersheds, analysis_watersheds = load_watershed_layers()
    harvey = analysis_watersheds.loc[analysis_watersheds.STORMS.fillna("").str.contains("Harvey", case=False)].copy()
    assert len(analysis_watersheds) == 23
    assert len(harvey) == 8
    states, conus = load_conus_layers()
    bounds = harvey.total_bounds
    extent = (np.floor((bounds[0] - 0.45) * 2) / 2,
              np.ceil((bounds[2] + 0.45) * 2) / 2,
              np.floor((bounds[1] - 0.35) * 2) / 2,
              np.ceil((bounds[3] + 0.35) * 2) / 2)

    font_path = findfont(FontProperties(family="Times New Roman"), fallback_to_default=True)
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 10.5,
                         "axes.titlesize": 12, "axes.titleweight": "normal",
                         "axes.labelsize": 10.5, "xtick.labelsize": 10,
                         "ytick.labelsize": 10, "axes.linewidth": 0.6,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.35), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.075, right=0.865, bottom=0.14, top=0.94,
                        hspace=0.24, wspace=0.13)
    color_max = float(np.ceil(np.nanmax(totals) / 200) * 200)
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("white")
    for index, ax in enumerate(axes.flat):
        mesh = ax.pcolormesh(lon_edges, lat_edges, totals[index], cmap=cmap,
                             vmin=0, vmax=color_max, shading="flat", rasterized=True)
        add_boundaries(ax, states.geometry, "#b7b7b7", 0.35, 3)
        add_boundaries(ax, conus.geometry, "#777777", 0.45, 4)
        add_boundaries(ax, harvey.geometry, "#252525", 0.55, 5)
        ax.set_xlim(extent[:2]); ax.set_ylim(extent[2:])
        ax.set_aspect(1 / np.cos(np.deg2rad(np.mean(extent[2:]))))
        ax.set_title(f"({chr(97 + index)}) {LABELS[index]}", loc="left", pad=4)
        ax.xaxis.set_major_locator(MultipleLocator(4))
        ax.yaxis.set_major_locator(MultipleLocator(2))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{abs(x):g}°W"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{abs(x):g}°N"))
        ax.tick_params(length=2.5, pad=2)
        peak = product_metrics[index]
        ax.plot(peak["peak_longitude"], peak["peak_latitude"], marker="*", markersize=6,
                markerfacecolor="white", markeredgecolor="black", markeredgewidth=0.45, zorder=6)
        ax.text(0.97, 0.96, f"Peak: {peak['peak_grid_cell_storm_total_mm']:,.0f} mm",
                transform=ax.transAxes, ha="right", va="top", fontsize=10,
                bbox={"facecolor":"white", "edgecolor":"none", "alpha":0.9, "pad":1.5}, zorder=7)
    cax = fig.add_axes([0.89, 0.19, 0.019, 0.69])
    bar = fig.colorbar(mesh, cax=cax, ticks=np.arange(0, color_max + 1, 400))
    bar.set_label("Storm-total precipitation (mm)", labelpad=5, fontsize=11)
    bar.ax.tick_params(labelsize=10, length=2)
    fig.text(0.46, 0.055, "Harvey | 19 August–4 September 2017 | 17 UTC days", ha="center", fontsize=11)
    fig.text(0.46, 0.012, f"Common 0.25° grid; {int(common.sum())} complete cells; black outlines: eight assigned watersheds.",
             ha="center", fontsize=9.6)
    png = OUT / "figures" / "figure1_harvey_total.png"
    pdf = png.with_suffix(".pdf")
    fig.savefig(png, dpi=400)
    fig.savefig(pdf)
    plt.close(fig)
    metadata = {
        "source_archive": str(selected), "storm_id": STORM_ID,
        "event_selection": "Harvey has the longest archived core (12-day) and expanded (17-day) windows; eight assigned watersheds; Ida has more assigned watersheds (nine).",
        "event_inventory": inventory,
        "first_daily_label": "2017-08-19", "last_daily_label_inclusive": "2017-09-04",
        "utc_interval_start": "2017-08-19T00:00:00Z", "utc_interval_end_exclusive": "2017-09-05T00:00:00Z",
        "n_days": 17, "statistic": "Sum of untransformed daily precipitation (skipna=False)",
        "minimum_source_valid_area_fraction": threshold,
        "domain_cell_count": int(domain.sum()), "product_complete_domain_cell_counts": finite_counts,
        "spatial_selection": "PERSIANN grid-cell centers inside the union of the eight Harvey-assigned watershed polygons; fractional watershed overlap used for area-weighted means",
        "complete_series_coverage_of_selected_cells_percent": float(common.sum() / domain.sum() * 100),
        "common_complete_cell_count": int(common.sum()),
        "common_weighted_area_km2": float(weights.sum() / 1e6),
        "domain_weighted_area_km2": float(np.sum(areas * ds.event_watershed_fraction.values) / 1e6),
        "common_area_coverage_percent": float(weights.sum() / np.sum(areas * ds.event_watershed_fraction.values) * 100),
        "grid": {"resolution_degrees": 0.25, "crs": "EPSG:4326", "regional_extent_wsen": [extent[0],extent[2],extent[1],extent[3]]},
        "watershed_count_in_source": len(all_watersheds), "watershed_count_in_analysis": len(analysis_watersheds),
        "harvey_watershed_count": len(harvey), "harvey_watersheds": harvey.HUC_NAME.tolist(),
        "product_metrics": product_metrics, "color_limits_mm": [0, color_max],
        "font": {"name": "Times New Roman", "actual_font_path": font_path},
        "figure_inches": [6.5,4.35], "output_png": str(png), "output_pdf": str(pdf),
        "caption": "Untransformed storm-total precipitation during the expanded Harvey window (19 August–4 September 2017, inclusive) for (a) MRMS, (b) CONUS404, (c) CONUS404BA, and (d) PERSIANN-CDR. All 17 daily UTC accumulations were summed on the common 0.25° PERSIANN-CDR grid, retaining 269 identical cells with centers inside the eight assigned watershed polygons and complete, quality-controlled records in all four products. Black outlines identify those watersheds; gray lines show state and CONUS boundaries. Stars mark the largest total on the plotted common support. All panels use the same color scale. Peaks are coarse-grid cell means, not native-resolution or gauge maxima; the expanded window can include rainfall not attributable to Harvey.",
    }
    (OUT / "figure1_harvey_metrics.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
