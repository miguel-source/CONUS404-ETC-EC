"""Four-panel, untransformed Harvey precipitation comparison.

Jupyter example (CONUS404 or CONUS404_ETC2 kernel):
    %run "scripts/plot_harvey_precipitation_comparison.py"

Default: sum the 12 daily accumulations dated 2017-08-23 through 2017-09-03.
This is an event-total map, NOT a map of maximum daily precipitation. The
star marks the largest event-total grid-cell value in each product.
All products use the already conservatively aligned PERSIANN-CDR grid.
Original native-resolution extremes are not preserved by spatial averaging.
"""

from __future__ import annotations

from study_paths import PROJECT_ROOT, RESULTS_ROOT, METADATA_ROOT, REPO_ROOT, STATES_FILE, WATERSHEDS_FILE, DAILY_INPUT, EC_DIRECTORY, EC_INPUT, LOSO_INPUT, AGREEMENT_DIRECTORY, AGREEMENT_INPUT


import argparse
import json
import os
from pathlib import Path

# Keep rendering/font-cache writes inside the script's own output location.
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / "_harvey_mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
import rasterio
from rasterio.transform import from_origin
import shapefile
from shapely.geometry import shape as geometry_from_shape
from shapely.ops import transform as transform_geometry, unary_union
import xarray as xr

PROJECT = (PROJECT_ROOT)
RESULT_ROOT = PROJECT / "ETC_StormEvents"
DEFAULT_INPUT = RESULT_ROOT / "Gridded_PERSIANN" / "events" / "all4_persiann_grid_Harvey_AL092017.nc"
DEFAULT_OUTPUT = RESULT_ROOT / "Hurricane_Harvey_figures"
DEFAULT_WATERSHEDS = WATERSHEDS_FILE
DEFAULT_STATES = (STATES_FILE)
PRODUCTS = ("mrms", "conus404", "conus404ba", "persiann_cdr")
LABELS = ("MRMS", "CONUS404", "CONUS404BA", "PERSIANN-CDR")
WINDOWS = {
    "core": ("2017-08-23", "2017-09-03"),
    "expanded": ("2017-08-19", "2017-09-04"),
}


def read_geometries(path: Path):
    """Read polygon geometry and attributes; explicitly transform to WGS84."""
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == '.gpkg':
        import geopandas as gpd
        frame = gpd.read_file(path).to_crs('EPSG:4326')
        return [(row.geometry, {key:value for key,value in row.items() if key != 'geometry'}) for _,row in frame.iterrows()]
    prj = path.with_suffix(".prj")
    if not prj.exists():
        raise FileNotFoundError(f"A .prj file is required to establish the boundary CRS: {prj}")
    source_crs = CRS.from_wkt(prj.read_text(encoding="utf-8-sig"))
    transform = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True).transform
    records = []
    with shapefile.Reader(str(path), encodingErrors="replace") as reader:
        for item in reader.iterShapeRecords():
            geom = geometry_from_shape(item.shape.__geo_interface__)
            if not geom.is_valid:
                geom = geom.buffer(0)  # Repair only the in-memory boundary copy.
            geom = transform_geometry(transform, geom)
            records.append((geom, item.record.as_dict()))
    return records


def segments(geom):
    """Convert polygon boundaries to Matplotlib segments, including holes."""
    boundary = geom.boundary
    if boundary.geom_type in {"LineString", "LinearRing"}:
        return [np.asarray(boundary.coords)]
    return [np.asarray(part.coords) for part in boundary.geoms
            if part.geom_type in {"LineString", "LinearRing"}]


def boundary_collection(geometries, color, width, order):
    return LineCollection(
        [part for geom in geometries for part in segments(geom)],
        colors=color, linewidths=width, zorder=order,
    )


def coordinate_edges(values):
    values = np.asarray(values, dtype=float)
    differences = np.diff(values)
    if len(values) < 2 or not np.allclose(differences, differences[0]):
        raise ValueError("Expected a regular coordinate axis with at least two cells.")
    return np.concatenate(([values[0] - differences[0] / 2],
                           (values[:-1] + values[1:]) / 2,
                           [values[-1] + differences[0] / 2]))


def calculate_maps(input_path, window, statistic):
    start, end = WINDOWS[window]
    expected_dates = pd.date_range(start, end, freq="D")
    with xr.open_dataset(input_path, engine="netcdf4") as source:
        if source.attrs.get("event_name") != "Harvey":
            raise ValueError("This script expects the Harvey event input.")
        missing = [name for name in PRODUCTS if name not in source]
        if missing:
            raise ValueError(f"Missing products: {missing}")
        data = source.sel(time=slice(start, end)).sortby("lat", ascending=False).sortby("lon").load()
    if not pd.DatetimeIndex(data.time.values).equals(expected_dates):
        raise ValueError("The requested window must contain every UTC day exactly once.")
    lon_edges = coordinate_edges(data.lon.values)
    lat_edges = coordinate_edges(data.lat.values)
    if not np.allclose(np.diff(lon_edges), 0.25) or not np.allclose(np.diff(lat_edges), -0.25):
        raise ValueError("Expected the already aligned 0.25-degree PERSIANN-CDR grid.")
    domain = data.event_watershed_mask.values.astype(bool)
    common = domain.copy()
    maps = []
    counts = []
    minimum_area = float(data.attrs.get("minimum_valid_source_area_fraction", 0.95))
    for name in PRODUCTS:
        da = data[name].transpose("time", "lat", "lon")
        if str(da.attrs.get("units", "")).lower() != "mm":
            raise ValueError(f"{name}: daily accumulated depth must have units mm.")
        values = da.values.astype(np.float64)
        finite = np.isfinite(values)
        if np.any((values < 0) & finite & domain[None, :, :]):
            raise ValueError(f"{name}: negative precipitation found in the Harvey domain.")
        fraction_name = name + "_valid_area_fraction"
        if fraction_name in data:
            fractions = data[fraction_name].transpose("time", "lat", "lon").values
            finite &= np.isfinite(fractions) & (fractions >= minimum_area - 1e-6)
        count = finite.sum(axis=0)
        common &= count == len(expected_dates)
        counts.append(count)
        # Do not use nansum: an incomplete day series must not become a low total.
        if statistic == "storm_total":
            maps.append(np.sum(values, axis=0))
        elif statistic == "max_daily":
            maps.append(np.max(values, axis=0))
        else:
            raise ValueError(statistic)
    if not common.any():
        raise ValueError("No Harvey cells have complete observations in all four products.")
    stacked = np.stack(maps)
    stacked[:, ~common] = np.nan
    maxima = []
    for i, name in enumerate(PRODUCTS):
        row, col = np.unravel_index(np.nanargmax(stacked[i]), common.shape)
        maxima.append({"product": name, "label": LABELS[i], "maximum_mm": float(stacked[i, row, col]),
                       "latitude": float(data.lat[row]), "longitude": float(data.lon[col])})
    return data, stacked, np.stack(counts), common, maxima, lon_edges, lat_edges


def draw_figure(maps, common, maxima, lon_edges, lat_edges, harvey, states, output_stem,
                window, statistic, view="regional", dpi=400):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.linewidth": 0.8, "axes.titleweight": "semibold",
        "svg.fonttype": "none", "savefig.facecolor": "white",
    })
    harvey_geometries = [geom for geom, _ in harvey]
    conus = unary_union(states)
    if view == "regional":
        west, south, east, north = unary_union(harvey_geometries).bounds
        extent = (np.floor((west - 0.6) * 2) / 2, np.ceil((east + 0.6) * 2) / 2,
                  np.floor((south - 0.5) * 2) / 2, np.ceil((north + 0.5) * 2) / 2)
    else:
        extent = (lon_edges.min(), lon_edges.max(), lat_edges.min(), lat_edges.max())
    # A single continuous, non-diverging scale is appropriate for positive depths.
    largest = float(np.nanmax(maps))
    tick_step = 300.0 if largest > 1000 else 100.0
    color_max = max(tick_step, np.ceil(largest / tick_step) * tick_step)
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 10.0), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.075, right=0.865, bottom=0.15, top=0.845,
                        hspace=0.22, wspace=0.14)
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("white")
    state_lines = [part for geom in states for part in segments(geom)]
    watershed_lines = [part for geom in harvey_geometries for part in segments(geom)]
    conus_lines = segments(conus)
    mesh = None
    for i, ax in enumerate(axes.flat):
        mesh = ax.pcolormesh(lon_edges, lat_edges, np.ma.masked_invalid(maps[i]),
                             cmap=cmap, vmin=0, vmax=color_max, shading="flat",
                             edgecolors="none", rasterized=True, zorder=2)
        ax.add_collection(LineCollection(state_lines, colors="#979797", linewidths=0.55, zorder=4))
        ax.add_collection(LineCollection(conus_lines, colors="#454545", linewidths=0.8, zorder=5))
        ax.add_collection(LineCollection(watershed_lines, colors="#252525", linewidths=0.75, zorder=6))
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect(1 / np.cos(np.deg2rad((extent[2] + extent[3]) / 2)))
        ax.set_title(f"({chr(97 + i)})  {LABELS[i]}", loc="left", fontsize=16, pad=9)
        ax.grid(color="#e4e4e4", linewidth=0.4, linestyle=":", zorder=0)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{abs(x):g}°W"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{abs(y):g}°N"))
        ax.tick_params(labelsize=10)
        maximum = maxima[i]
        ax.plot(maximum["longitude"], maximum["latitude"], marker="*", markersize=12,
                markerfacecolor="white", markeredgecolor="#151515", markeredgewidth=0.9,
                linestyle="none", zorder=8)
        ax.text(0.97, 0.94, f"Max: {maximum['maximum_mm']:,.0f} mm", transform=ax.transAxes,
                ha="right", va="top", fontsize=11.5, color="#252525", zorder=10,
                bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#bfbfbf", "alpha": 0.94})
    for ax in axes[:, 0]:
        ax.set_ylabel("Latitude", fontsize=11)
    for ax in axes[1]:
        ax.set_xlabel("Longitude", fontsize=11)
    bar_ax = fig.add_axes([0.892, 0.203, 0.018, 0.57])
    bar = fig.colorbar(mesh, cax=bar_ax, ticks=np.arange(0, color_max + tick_step / 2, tick_step))
    label = "Storm-total precipitation (mm)" if statistic == "storm_total" else "Maximum daily precipitation (mm)"
    bar.set_label(label, labelpad=13, fontsize=12)
    bar.ax.tick_params(labelsize=10)
    start, end = [pd.Timestamp(value) for value in WINDOWS[window]]
    title = "Storm-total precipitation" if statistic == "storm_total" else "Maximum daily precipitation"
    fig.suptitle(f"Hurricane Harvey | {title}", fontsize=22, fontweight="semibold", y=0.965)
    subtitle = (f"{start:%d %B}–{end:%d %B %Y}  ·  {(end - start).days + 1} UTC days  ·  "
                "Common 0.25° PERSIANN-CDR grid")
    fig.text(0.47, 0.912, subtitle, ha="center", fontsize=12, color="#4c4c4c")
    handles = [
        Line2D([0], [0], color="#252525", lw=1.0, label="Harvey-assigned watersheds"),
        Line2D([0], [0], color="#979797", lw=0.7, label="State / CONUS boundaries"),
        Line2D([0], [0], marker="*", markerfacecolor="white", markeredgecolor="#151515",
               color="none", markersize=11, label="Maximum grid-cell value"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.48, 0.063),
               ncol=3, frameon=False, fontsize=10.5, handlelength=2.2, columnspacing=2.0)
    computation = "sum of daily precipitation" if statistic == "storm_total" else "maximum of daily precipitation"
    fig.text(0.47, 0.044, f"Untransformed values; {computation}. Matched, complete cells intersecting the eight Harvey watersheds.",
             ha="center", fontsize=9.5, color="#555555")
    fig.text(0.47, 0.024, f"Shared support: {int(common.sum())} cells. Maxima are 0.25° cell means, not native-resolution or rain-gauge extremes.",
             ha="center", fontsize=9.5, color="#555555")
    png_path = output_stem.with_suffix(".png")
    svg_path = output_stem.with_suffix(".svg")
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(svg_path, dpi=300)
    plt.close(fig)
    return png_path, svg_path, color_max


def export_maps(data, maps, counts, common, maxima, output_stem, window, statistic, color_max,
                watershed_records):
    start, end = WINDOWS[window]
    crs_attrs = {"grid_mapping_name": "latitude_longitude", "epsg_code": "EPSG:4326",
                 "semi_major_axis": 6378137.0, "inverse_flattening": 298.257223563}
    result = xr.Dataset(
        {
            statistic: (("product", "lat", "lon"), maps),
            "valid_day_count": (("product", "lat", "lon"), counts.astype(np.int16)),
            "common_complete_mask": (("lat", "lon"), common.astype(np.int8)),
            "event_watershed_mask": (("lat", "lon"), data.event_watershed_mask.values.astype(np.int8)),
            "event_watershed_fraction": (("lat", "lon"), data.event_watershed_fraction.values),
            "crs": ((), np.int8(0), crs_attrs),
        },
        coords={"product": list(PRODUCTS), "lat": data.lat.values, "lon": data.lon.values},
        attrs={"title": "Harvey untransformed precipitation comparison", "event_name": "Harvey",
               "storm_id": "AL092017", "window": window, "first_day": start, "last_day_inclusive": end,
               "daily_interval": "[D 00:00 UTC, D+1 00:00 UTC)", "statistic": statistic,
               "number_of_days": int(data.sizes["time"]), "source_file": str(data.attrs["actual_input_file"]),
               "spatial_support": "Full 0.25-degree cells intersecting Harvey-assigned watersheds; common complete series in all four products",
               "remapping_method": str(data.attrs.get("remapping_method", "")),
               "temporal_missing_policy": "No incomplete totals; every date must be valid in every product"},
    )
    result[statistic].attrs = {"units": "mm", "long_name": "Storm-total precipitation" if statistic == "storm_total" else "Maximum daily precipitation",
                               "cell_methods": "time: sum" if statistic == "storm_total" else "time: maximum", "grid_mapping": "crs"}
    result.lat.attrs = {"standard_name": "latitude", "units": "degrees_north", "axis": "Y"}
    result.lon.attrs = {"standard_name": "longitude", "units": "degrees_east", "axis": "X"}
    nc_path = output_stem.with_suffix(".nc")
    encoding = {statistic: {"zlib": True, "complevel": 4, "_FillValue": np.nan}}
    result.to_netcdf(nc_path, engine="netcdf4", encoding=encoding)
    dx, dy = float(np.diff(data.lon)[0]), float(-np.diff(data.lat)[0])
    transform = from_origin(float(data.lon[0]) - dx / 2, float(data.lat[0]) + dy / 2, dx, dy)
    tif_folder = output_stem.parent / "GeoTIFFs"
    tif_folder.mkdir(exist_ok=True)
    tiffs = []
    for i, product in enumerate(PRODUCTS):
        tif_path = tif_folder / f"{output_stem.name}_{product}.tif"
        with rasterio.open(tif_path, "w", driver="GTiff", height=maps.shape[1], width=maps.shape[2],
                           count=1, dtype="float32", crs="EPSG:4326", transform=transform,
                           nodata=-9999.0, compress="DEFLATE", predictor=3, tiled=True) as dst:
            dst.write(np.where(np.isfinite(maps[i]), maps[i], -9999).astype(np.float32), 1)
            dst.set_band_description(1, f"{LABELS[i]}: {statistic}, mm")
            dst.update_tags(product=product, event="Harvey", statistic=statistic, units="mm",
                            start_day=start, end_day_inclusive=end, window=window,
                            n_days=str(data.sizes["time"]), grid="PERSIANN-CDR 0.25 degree",
                            support="Common complete cells intersecting Harvey-assigned watersheds")
        tiffs.append(str(tif_path))
    ending_midnight = pd.Timestamp(end) + pd.Timedelta(days=1)
    caption = (
        f"Figure. Spatial distribution of {'storm-total' if statistic == 'storm_total' else 'maximum daily'} precipitation during Hurricane Harvey "
        f"in (a) MRMS, (b) CONUS404, (c) CONUS404BA, and (d) PERSIANN-CDR. "
        f"Daily accumulation intervals dated {start} through {end}, inclusive, "
        f"cover {start} 00:00 UTC to {ending_midnight:%Y-%m-%d} 00:00 UTC. "
        f"Values are {'summed over the event window' if statistic == 'storm_total' else 'the largest daily accumulation in the window'} "
        "and are not log-transformed. The four products are compared on the native 0.25-degree PERSIANN-CDR grid, "
        "using the existing overlap-area conservative remapping for the other products. "
        f"The common support contains {int(common.sum())} cells with complete daily records in all four products and "
        "intersects eight Harvey-assigned watershed polygons. Black outlines indicate watershed boundaries; "
        "gray outlines indicate state/CONUS boundaries. Stars indicate each product's maximum on this common support. "
        "A shared color scale is used. These maxima are coarse-grid cell-mean accumulations, not "
        "native-grid or rain-gauge maxima; the calendar window alone does not establish that every accumulation was caused by Harvey."
    )
    output_stem.with_suffix(".txt").write_text(caption + "\n", encoding="utf-8")
    metadata = {
        "input_file": str(data.attrs["actual_input_file"]), "window": window, "statistic": statistic,
        "first_day": start, "last_day_inclusive": end, "utc_interval_end_exclusive": ending_midnight.isoformat(),
        "n_days": int(data.sizes["time"]), "n_harvey_domain_cells": int(data.event_watershed_mask.values.sum()),
        "n_common_complete_cells": int(common.sum()), "units": "mm", "grid_resolution_degrees": 0.25,
        "shared_color_limits": [0.0, float(color_max)], "maxima": maxima,
        "watersheds": [attrs.get("HUC_NAME", attrs.get("BASIN_NM", "")) for _, attrs in watershed_records],
        "files": {"png": str(output_stem.with_suffix(".png")), "svg": str(output_stem.with_suffix(".svg")),
                  "nc": str(nc_path), "geotiffs": tiffs}, "caption": caption,
    }
    output_stem.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def create_figure(input_path=DEFAULT_INPUT, output_dir=DEFAULT_OUTPUT, window="core",
                  statistic="storm_total", view="regional", dpi=400,
                  watershed_file=DEFAULT_WATERSHEDS, states_file=DEFAULT_STATES):
    input_path, output_dir = Path(input_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data, maps, counts, common, maxima, lon_edges, lat_edges = calculate_maps(input_path, window, statistic)
    data.attrs["actual_input_file"] = str(input_path)
    watershed_records = read_geometries(Path(watershed_file))
    harvey = [(geom, attrs) for geom, attrs in watershed_records
              if "harvey" in str(attrs.get("STORMS", "")).casefold()]
    if len(harvey) != 8:
        raise ValueError(f"Expected eight Harvey-assigned watershed polygons, found {len(harvey)}.")
    states = [geom for geom, attrs in read_geometries(Path(states_file))
              if attrs.get("admin") == "United States of America" and attrs.get("postal") not in {"AK", "HI"}]
    if not states:
        raise ValueError("Could not identify CONUS states in the boundary file.")
    start, end = WINDOWS[window]
    stem = output_dir / f"Harvey_{statistic}_{window}_{start.replace('-', '')}_{end.replace('-', '')}_{view}"
    png_path, svg_path, color_max = draw_figure(maps, common, maxima, lon_edges, lat_edges,
                                               harvey, states, stem, window, statistic, view, dpi)
    metadata = export_maps(data, maps, counts, common, maxima, stem, window, statistic, color_max, harvey)
    print(f"Figure: {png_path}")
    print(f"Editable SVG: {svg_path}")
    print(f"Common complete support: {int(common.sum())} cells; {data.sizes['time']} daily observations per cell")
    for item in maxima:
        print(f"{item['label']:14s} maximum = {item['maximum_mm']:9.3f} mm at "
              f"({item['longitude']:.3f}, {item['latitude']:.3f})")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--window", choices=tuple(WINDOWS), default="core")
    parser.add_argument("--statistic", choices=("storm_total", "max_daily"), default="storm_total")
    parser.add_argument("--view", choices=("regional", "study"), default="regional")
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--watersheds", type=Path, default=DEFAULT_WATERSHEDS)
    parser.add_argument("--states", type=Path, default=DEFAULT_STATES)
    args = parser.parse_args()
    create_figure(args.input, args.output, args.window, args.statistic, args.view, args.dpi,
                  args.watersheds, args.states)


if __name__ == "__main__":
    main()
