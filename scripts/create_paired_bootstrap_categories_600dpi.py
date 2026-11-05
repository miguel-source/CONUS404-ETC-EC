"""Publication-size maps of conditional paired EC bootstrap interval direction.

Source estimates are read only. These maps classify saved pixelwise intervals;
they do not rerun collocation, alter admissibility, or estimate domain-mean CIs.
"""

from __future__ import annotations

from study_paths import PROJECT_ROOT, RESULTS_ROOT, METADATA_ROOT, REPO_ROOT, STATES_FILE, WATERSHEDS_FILE, DAILY_INPUT, EC_DIRECTORY, EC_INPUT, LOSO_INPUT, AGREEMENT_DIRECTORY, AGREEMENT_INPUT


import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np
from PIL import Image
import xarray as xr

from build_word_primary_figures import AGREEMENT, EC, EXTENT, edges
from map_overlays import add_map_boundaries


HERE = RESULTS_ROOT
OUT = HERE / "paired_uncertainty_figures" / "bootstrap_categories" / "600dpi"
VERIFIED = HERE / "hurricane_results_word" / "verified_results_metrics.json"
DPI = 600
SIZE = (16.0, 7.8)
TEAL = "#238B78"
GRAY = "#CCD0D3"
BROWN = "#A66A27"
TEXT = "#202A33"
MUTED = "#53616A"
COLORS = (TEAL, GRAY, BROWN)
WINDOWS = ("core", "expanded")

CONFIGS = (
    {
        "key": "error_sd",
        "variable": "error_sd_difference_ba_minus_conus404",
        "title": "Paired storm-bootstrap comparison of random-error SD",
        "subtitle": "CONUS404BA minus CONUS404  |  5,000 storm-block resamples  |  Conditional pixelwise 95% intervals",
        "short_metric": "SD",
        "mean_label": "Mean delta SD",
        "mean_format": "+.2f",
        "units": "mm day$^{-1}$",
        "labels": ("Lower BA error SD", "Interval includes zero", "Higher BA error SD"),
    },
    {
        "key": "latent_r2",
        "variable": "estimated_r2_difference_ba_minus_conus404",
        "title": r"Paired storm-bootstrap comparison of latent-signal $R^2$",
        "subtitle": "CONUS404BA minus CONUS404  |  5,000 storm-block resamples  |  Conditional pixelwise 95% intervals",
        "short_metric": r"$R^2$",
        "mean_label": r"Mean $\Delta R^2$",
        "mean_format": "+.3f",
        "units": "",
        "labels": (r"Lower BA latent $R^2$", "Interval includes zero", r"Higher BA latent $R^2$"),
    },
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configure() -> None:
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 13,
        "mathtext.fontset": "stix",
        "axes.unicode_minus": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })


def decorate(ax, *, show_ylabels: bool) -> None:
    ax.set_xlim(EXTENT[:2])
    ax.set_ylim(EXTENT[2:])
    add_map_boundaries(ax)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    ax.xaxis.set_major_locator(FixedLocator([-100, -90, -80]))
    ax.yaxis.set_major_locator(FixedLocator([30, 35, 40]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{abs(v):.0f}°W"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}°N"))
    ax.tick_params(labelsize=14, length=4, width=.75, pad=5,
                   labelleft=show_ylabels)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#68727A")
        spine.set_linewidth(.8)


def draw(config: dict, ec: xr.Dataset, ag: xr.Dataset, verified: dict,
         source_hashes: dict) -> None:
    stem = f"paired_bootstrap_{config['key']}_categories"
    variable = config["variable"]
    weights = (ag.study_watershed_fraction.values.astype(float)
               * np.cos(np.deg2rad(ag.lat.values))[:, None])
    mask = ag.valid_agreement.transpose("window", "lat", "lon").values.astype(bool)
    expected = (ec.valid_primary_ec.values.astype(bool)
                & (ec.bootstrap_valid_fraction.values >= .90))
    np.testing.assert_array_equal(mask, expected)
    point = ec[variable].transpose("window", "lat", "lon").values
    low = ec[f"{variable}_ci_low"].transpose("window", "lat", "lon").values
    high = ec[f"{variable}_ci_high"].transpose("window", "lat", "lon").values
    lon_edges, lat_edges = edges(ec.lon.values), edges(ec.lat.values)
    cmap = ListedColormap(COLORS)
    cmap.set_bad("white")
    norm = BoundaryNorm([-1.5, -.5, .5, 1.5], cmap.N)

    fig = plt.figure(figsize=SIZE, facecolor="white")
    fig.text(.515, .974, config["title"], ha="center", va="top",
             fontsize=23, fontweight="bold", color=TEXT)
    fig.text(.515, .916, config["subtitle"], ha="center", fontsize=14, color=MUTED)
    width = .405
    map_height = width * SIZE[0] * (EXTENT[3]-EXTENT[2]) / (EXTENT[1]-EXTENT[0]) / SIZE[1]
    panels = []
    for index, window in enumerate(WINDOWS):
        mi = mask[index]
        finite = np.isfinite(low[index]) & np.isfinite(high[index])
        assert np.all(finite[mi]), f"Missing retained intervals: {window}"
        assert np.all(low[index][mi] <= high[index][mi])
        categories = np.full(mi.shape, np.nan)
        below = mi & finite & (high[index] < 0)
        above = mi & finite & (low[index] > 0)
        includes = mi & finite & (low[index] <= 0) & (high[index] >= 0)
        assert np.array_equal(below | above | includes, mi)
        categories[below], categories[includes], categories[above] = -1, 0, 1
        masks = (below, includes, above)
        total = float(weights[mi].sum())
        coverage = total / float(weights.sum()) * 100
        percentages = [float(weights[m].sum() / total * 100) for m in masks]
        counts = [int(m.sum()) for m in masks]
        assert int(mi.sum()) == {"core": 527, "expanded": 660}[window]
        np.testing.assert_allclose(sum(percentages), 100, atol=1e-9)
        if config["key"] == "error_sd":
            reference = verified["windows"][window]["area_pct_conditional_paired_ci"]
            np.testing.assert_allclose(percentages, [reference["lower_BA_sd"],
                                       reference["includes_zero"], reference["higher_BA_sd"]])
        mean = float(np.sum(point[index][mi] * weights[mi]) / total)
        left = (.065, .555)[index]
        ax = fig.add_axes([left, .365, width, map_height])
        ax.pcolormesh(lon_edges, lat_edges, categories, cmap=cmap, norm=norm,
                      shading="flat", rasterized=True, edgecolors="none",
                      antialiased=False, zorder=1)
        decorate(ax, show_ylabels=index == 0)
        ax.set_title(f"({'ab'[index]}) {window.capitalize()}", loc="left",
                     fontsize=19, fontweight="bold", pad=13)
        ax.set_title(f"{mi.sum()} cells  |  {coverage:.2f}% watershed area", loc="right",
                     fontsize=12.5, pad=14)
        mean_label = (r"Mean $\Delta$SD" if config["key"] == "error_sd"
                      else config["mean_label"])
        mean_text = f"{mean_label}: {format(mean, config['mean_format'])} {config['units']}"
        ax.text(.025, .967, mean_text.strip(), transform=ax.transAxes,
                 ha="left", va="top", fontsize=13,
                 bbox={"facecolor": "white", "edgecolor": "none", "alpha": .96, "pad": 3},
                 zorder=12)

        # Area summaries complement the map: widths are retained area, not cell counts.
        bar = fig.add_axes([left, .204, width, .062])
        offset = 0.0
        for value, color, code in zip(percentages, COLORS, (-1, 0, 1)):
            bar.barh([0], [value], left=offset, height=.74, color=color,
                      edgecolor="none", linewidth=0)
            inside = value >= 9
            label_y = 0 if inside else .70
            bar.text(offset+value/2, label_y, f"{value:.2f}%", ha="center", va="center",
                      fontsize=13, fontweight="bold",
                      color=("white" if code != 0 else TEXT) if inside else TEXT)
            offset += value
        bar.set_xlim(0, 100)
        bar.set_ylim(-.6, 1.15)
        bar.set_axis_off()
        bar.text(.5, -.25, "Share of retained weighted watershed area", transform=bar.transAxes,
                 ha="center", va="top", fontsize=12.5, color=MUTED)
        panels.append({
            "window": window, "retained_cells": int(mi.sum()),
            "weighted_watershed_area_coverage_percent": coverage,
            "area_weighted_mean_point_difference": mean,
            "category_order": ["interval_below_zero", "interval_includes_zero", "interval_above_zero"],
            "category_cell_counts": counts,
            "category_percent_retained_weighted_area": percentages,
        })
    fig.text(.515, .301, "Longitude", ha="center", fontsize=15)
    fig.text(.015, .595, "Latitude", ha="center", va="center", rotation=90, fontsize=15)
    legend = fig.legend(handles=[Patch(facecolor=color, edgecolor="none", label=label)
                                 for color, label in zip(COLORS, config["labels"])],
                        loc="lower center", bbox_to_anchor=(.515, .090), ncol=3,
                        frameon=False, fontsize=14, handlelength=1.8, handleheight=1.0,
                        columnspacing=2.5)
    fig.text(.515, .049, "White cells: outside the retained paired support. Black outlines: analyzed watersheds and CONUS boundary.",
             ha="center", fontsize=11.8, color=MUTED)
    fig.text(.515, .019, "Interval includes zero does not mean zero difference. Percentages use retained area within each panel.",
             ha="center", fontsize=11.8, color=MUTED)
    fig.canvas.draw()
    bbox = legend.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    assert bbox.x0 > 0 and bbox.x1 < 1 and bbox.y0 > 0, bbox

    paths = {}
    for suffix in ("png", "tif", "svg"):
        output = OUT / f"{stem}_600dpi.{suffix}"
        kwargs = {"pil_kwargs": {"compression": "tiff_lzw"}} if suffix == "tif" else {}
        fig.savefig(output, dpi=DPI, **kwargs)
        paths[suffix] = str(output)
    preview = OUT / f"{stem}_preview.png"
    fig.savefig(preview, dpi=145)
    plt.close(fig)
    for suffix in ("png", "tif"):
        with Image.open(paths[suffix]) as im:
            assert im.size == (9600, 4680), im.size
            assert all(abs(v-DPI) < .1 for v in im.info["dpi"])
    caption = (
        f"Paired storm-block bootstrap comparison of {'random-error standard deviation' if config['key'] == 'error_sd' else 'estimated squared correlation with the latent precipitation signal'} "
        "between CONUS404BA and CONUS404 during (a) core and (b) expanded event windows. "
        "All differences are CONUS404BA minus CONUS404. Teal identifies cells whose conditional pixelwise 95% interval lies entirely below zero, "
        "brown identifies intervals entirely above zero, and gray identifies intervals including zero. "
        "Intervals are saved 2.5th and 97.5th percentiles from admissible paired complete-storm draws among 5,000 resamples. "
        "The primary model-pair support requires an admissible full-sample four-product extended-collocation estimate and at least 90% admissible bootstrap draws, "
        "retaining 527 core-window and 660 expanded-window cells. It is not the common separate-triplet mask and does not require admissible reference-product variances. "
        "Summary bars give percentages of retained weighted watershed area, using watershed-overlap fraction times cosine latitude. "
        "Panel coverage percentages use all stored watershed-fraction cells as denominator. Mean differences are full-sample area-weighted means, not bootstrap means. "
        "Gray does not establish equality, and these pixelwise intervals are not confidence intervals for the domain-average difference or spatially multiplicity-adjusted tests. "
        "White indicates cells outside retained support. Black lines delineate the 23 analyzed watersheds and CONUS boundary; light gray lines delineate states. "
        "The maps inherit the common 0.25-degree geographic grid."
    )
    metadata = {
        "sources_sha256": source_hashes, "metric": config["key"], "variable": variable,
        "analysis": "paired four-product EC with CONUS404-CONUS404BA error covariance",
        "transform": "raw", "grid_crs": "EPSG:4326", "resolution_degrees": .25,
        "support": "valid_primary_ec AND bootstrap_valid_fraction >=0.90; verified equal to primary valid_agreement",
        "bootstrap_replicates": int(ec.attrs["bootstrap_replicates"]),
        "interval": "Conditional pixelwise 2.5th-97.5th percentile of admissible complete-storm paired draws",
        "category_codes": {"-1": "ci_high <0", "0": "ci_low <=0 AND ci_high >=0", "1": "ci_low >0"},
        "colors": {"below_zero": TEAL, "includes_zero": GRAY, "above_zero": BROWN},
        "weighting": "study_watershed_fraction*cos(latitude)",
        "panels": panels, "dpi": DPI, "figure_inches": list(SIZE),
        "raster_dimensions": [9600, 4680], "outputs": paths, "preview": str(preview),
        "source_data_modified": False, "estimation_rerun": False,
        "tiff_note": "TIFF is a publication figure, not a georeferenced data raster.",
    }
    (OUT / f"{stem}_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (OUT / f"{stem}_caption.md").write_text(caption+"\n", encoding="utf-8")
    print(json.dumps({"figure": stem, "outputs": paths, "preview": str(preview), "panels": panels}, indent=2), flush=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    configure()
    hashes = {str(path): sha(path) for path in (EC, AGREEMENT, VERIFIED)}
    verified = json.loads(VERIFIED.read_text(encoding="utf-8"))
    with xr.open_dataset(EC, engine="netcdf4") as source_ec, xr.open_dataset(AGREEMENT, engine="netcdf4") as source_ag:
        ec = source_ec.sel(transform="raw").load()
        ag = source_ag.sel(transform="raw", dataset_pair="conus404__conus404ba").load()
    assert int(ec.attrs["bootstrap_replicates"]) == 5000
    assert ec.window.values.tolist() == list(WINDOWS)
    for config in CONFIGS:
        draw(config, ec, ag, verified, hashes)
    assert hashes == {str(path): sha(path) for path in (EC, AGREEMENT, VERIFIED)}


if __name__ == "__main__":
    main()
