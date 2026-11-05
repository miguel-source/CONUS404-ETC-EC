"""Replot the original expanded-window Harvey comparison with blue-teal colors.

Data, quality control, common cells, area weights, extent, and linear scale
match the existing Word-report figure; only cartographic styling is changed.
"""

from __future__ import annotations

from study_paths import PROJECT_ROOT, RESULTS_ROOT, METADATA_ROOT, REPO_ROOT, STATES_FILE, WATERSHEDS_FILE, DAILY_INPUT, EC_DIRECTORY, EC_INPUT, LOSO_INPUT, AGREEMENT_DIRECTORY, AGREEMENT_INPUT


import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import xarray as xr

from build_word_harvey_figure import PRODUCTS, LABELS, edges, add_boundaries
from map_overlays import load_watershed_layers, load_conus_layers


ROOT = RESULTS_ROOT
ORIGINAL_METADATA = ROOT / "hurricane_results_word" / "figure1_harvey_metrics.json"
OUT = ROOT / "Harvey_storm_total_contrasted" / "600dpi"
DPI = 600
SIZE = (14, 10.2)
COLOR_STOPS = [
    (0.00, "#F1F8FC"),
    (0.12, "#B1D3E4"),
    (0.24, "#7EB6D3"),
    (0.50, "#238B78"),
    (0.76, "#0E6255"),
    (1.00, "#043A32"),
]


def load_original_comparison(metadata):
    source_path = Path(metadata["source_archive"])
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    with xr.open_dataset(source_path, engine="netcdf4") as source:
        ds = source.sortby("lat", ascending=False).sortby("lon").load()
    assert ds.attrs["storm_id"] == "AL092017"
    expected = pd.date_range("2017-08-19", "2017-09-04", freq="D")
    assert pd.DatetimeIndex(ds.time.values).equals(expected)
    assert ds.attrs["daily_interval"] == "[D 00:00 UTC, D+1 00:00 UTC)"
    domain = ds.event_watershed_mask.values.astype(bool)
    common = domain.copy()
    totals = []
    threshold = float(ds.attrs["minimum_valid_source_area_fraction"])
    assert threshold == metadata["minimum_source_valid_area_fraction"]
    for product in PRODUCTS:
        daily = ds[product].transpose("time", "lat", "lon")
        assert daily.attrs["units"] == "mm"
        fraction = ds[f"{product}_valid_area_fraction"].transpose("time", "lat", "lon").values
        valid = (np.isfinite(daily.values) & (daily.values >= 0)
                 & np.isfinite(fraction) & (fraction >= threshold - 1e-6))
        common &= valid.all(axis=0)
        totals.append(daily.astype(np.float64).sum("time", skipna=False).values)
    totals = np.stack(totals)
    totals[:, ~common] = np.nan
    assert int(common.sum()) == metadata["common_complete_cell_count"] == 269
    lon_edges, lat_edges = edges(ds.lon), edges(ds.lat)
    west, north = np.meshgrid(lon_edges[:-1], lat_edges[:-1])
    east, south = np.meshgrid(lon_edges[1:], lat_edges[1:])
    cells = shapely.box(west.ravel(), south.ravel(), east.ravel(), north.ravel())
    areas = (gpd.GeoSeries(cells, crs="EPSG:4326").to_crs("EPSG:5070").area
             .to_numpy().reshape(common.shape))
    weights = areas * ds.event_watershed_fraction.values
    weights[~common] = 0
    metrics = []
    for index, product in enumerate(PRODUCTS):
        rr, cc = np.unravel_index(np.nanargmax(totals[index]), common.shape)
        result = {
            "product": product, "label": LABELS[index],
            "area_weighted_mean_storm_total_mm": float(np.nansum(totals[index] * weights) / weights.sum()),
            "peak_grid_cell_storm_total_mm": float(totals[index, rr, cc]),
            "peak_latitude": float(ds.lat[rr]), "peak_longitude": float(ds.lon[cc]),
        }
        original = metadata["product_metrics"][index]
        assert result["product"] == original["product"]
        for key in ("area_weighted_mean_storm_total_mm", "peak_grid_cell_storm_total_mm",
                    "peak_latitude", "peak_longitude"):
            np.testing.assert_allclose(result[key], original[key], rtol=1e-12, atol=1e-10)
        metrics.append(result)
    return ds, totals, metrics, source_path, source_hash


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    original = json.loads(ORIGINAL_METADATA.read_text(encoding="utf-8"))
    ds, totals, metrics, source_path, source_hash = load_original_comparison(original)
    _, watersheds = load_watershed_layers()
    harvey = watersheds.loc[watersheds.STORMS.fillna("").str.contains("Harvey", case=False)].copy()
    assert len(harvey) == 8
    assert set(harvey.HUC_NAME) == set(original["harvey_watersheds"])
    states, conus = load_conus_layers()
    west, south, east, north = original["grid"]["regional_extent_wsen"]
    limits = original["color_limits_mm"]
    assert limits == [0, 1800]
    assert np.nanmin(totals) >= limits[0] and np.nanmax(totals) <= limits[1]
    cmap = LinearSegmentedColormap.from_list("harvey_blue_teal", COLOR_STOPS, N=4096)
    cmap.set_bad("white")
    norm = Normalize(*limits, clip=False)
    plt.rcParams.update({
        "font.family": "Times New Roman", "font.size": 14,
        "text.color": "#202A33", "axes.labelcolor": "#202A33",
        "xtick.color": "#39454E", "ytick.color": "#39454E",
        "svg.fonttype": "none", "savefig.facecolor": "white",
    })
    fig = plt.figure(figsize=SIZE, facecolor="white")
    fig.text(.453, .976, "Hurricane Harvey: storm-window precipitation",
             ha="center", va="top", fontsize=24, fontweight="bold")
    fig.text(.453, .935, "19 August–4 September 2017  |  17 daily UTC accumulations",
             ha="center", fontsize=16, color="#53616A")
    width = .36
    aspect = 1 / np.cos(np.deg2rad((south + north) / 2))
    height = width * SIZE[0] * (north - south) / (east - west) * aspect / SIZE[1]
    for index in range(4):
        row, col = divmod(index, 2)
        ax = fig.add_axes([[.065, .48][col], [.545, .16][row], width, height])
        mesh = ax.pcolormesh(edges(ds.lon), edges(ds.lat), totals[index],
                             cmap=cmap, norm=norm, shading="flat", rasterized=True,
                             antialiased=False, edgecolors="none", zorder=1)
        add_boundaries(ax, states.geometry, "#B5BCC1", .55, 3)
        add_boundaries(ax, conus.geometry, "#7A858D", .75, 4)
        add_boundaries(ax, harvey.geometry, "#25343B", .9, 5)
        ax.set_xlim(west, east)
        ax.set_ylim(south, north)
        ax.set_aspect(aspect, adjustable="box")
        ax.set_title(f"({'abcd'[index]}) {LABELS[index]}", loc="left", pad=11,
                     fontsize=19, fontweight="bold")
        ax.xaxis.set_major_locator(MultipleLocator(4))
        ax.yaxis.set_major_locator(MultipleLocator(2))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{abs(v):g}°W"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{abs(v):g}°N"))
        ax.tick_params(length=4, width=.7, pad=4, labelsize=14,
                       labelleft=col == 0, labelbottom=row == 1)
        for spine in ax.spines.values():
            spine.set_linewidth(.8)
            spine.set_color("#8B959B")
        peak = metrics[index]
        ax.plot(peak["peak_longitude"], peak["peak_latitude"], marker="*", markersize=11,
                markerfacecolor="white", markeredgecolor="#172B32", markeredgewidth=.9,
                linestyle="none", zorder=6)
        ax.text(.975, .96, f"Peak: {peak['peak_grid_cell_storm_total_mm']:,.0f} mm",
                transform=ax.transAxes, ha="right", va="top", fontsize=15,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .94, "pad": 3},
                zorder=7)
    cax = fig.add_axes([.882, .197, .021, .633])
    bar = fig.colorbar(mesh, cax=cax, ticks=np.arange(0, 1801, 200))
    bar.set_label("Storm-total precipitation (mm)", labelpad=14, fontsize=17)
    bar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    bar.ax.tick_params(labelsize=14, length=4, width=.7, pad=6)
    bar.outline.set_linewidth(.7)
    fig.text(.453, .097, "Longitude", ha="center", fontsize=16)
    fig.text(.013, .518, "Latitude", rotation=90, va="center", ha="center", fontsize=16)
    handles = [
        Line2D([], [], color="#25343B", linewidth=1.1, label="Eight assigned watersheds"),
        Line2D([], [], linestyle="none", marker="*", markersize=11,
               markerfacecolor="white", markeredgecolor="#172B32", label="Maximum grid-cell total"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.453, .047),
               ncol=2, frameon=False, fontsize=13, columnspacing=2, handlelength=1.8)
    fig.text(.453, .020,
             "Common 0.25° PERSIANN-CDR grid • 269 complete cells in all four products • Identical linear 0–1,800 mm scale",
             ha="center", fontsize=12, color="#53616A")
    stem = OUT / "Harvey_storm_total_blue_teal_600dpi"
    for suffix in (".png", ".svg", ".tif"):
        kwargs = {"pil_kwargs": {"compression": "tiff_lzw"}} if suffix == ".tif" else {}
        fig.savefig(stem.with_suffix(suffix), dpi=DPI, **kwargs)
    fig.savefig(OUT / "preview.png", dpi=145)
    plt.close(fig)
    metadata = {
        "source_archive": str(source_path), "source_sha256": source_hash,
        "original_figure_metadata": str(ORIGINAL_METADATA),
        "data_and_statistics_match_original": True, "source_data_modified": False,
        "storm_id": "AL092017", "n_days": 17,
        "first_daily_label": "2017-08-19", "last_daily_label_inclusive": "2017-09-04",
        "utc_interval_end_exclusive": "2017-09-05T00:00:00Z",
        "common_complete_cell_count": 269, "harvey_watersheds": original["harvey_watersheds"],
        "grid": original["grid"], "color_limits_mm": limits,
        "color_mapping": "Linear normalization, custom continuous sequential blue-teal palette; no data transformation",
        "requested_anchor_colors": ["#7EB6D3", "#238B78"], "palette_stops": COLOR_STOPS,
        "dpi": DPI, "figure_inches": SIZE, "raster_dimensions": [round(v*DPI) for v in SIZE],
        "product_metrics": metrics,
        "outputs": {suffix: str(stem.with_suffix('.' + suffix)) for suffix in ("png", "tif", "svg")},
        "caption": original["caption"],
    }
    (OUT / "figure_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (OUT / "figure_caption.md").write_text(original["caption"] + "\n", encoding="utf-8")
    assert source_hash == hashlib.sha256(source_path.read_bytes()).hexdigest()
    ds.close()
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
