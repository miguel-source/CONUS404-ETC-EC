"""Publication-size brown/teal EC-change and direct-agreement figures.

The two figures deliberately preserve DIFFERENT support rules: the paired
four-system EC map retains its primary bootstrap screen; the direct daily
agreement chart uses the broad matched sample without collocation screening.
No source estimates or datasets are modified or re-estimated.
"""

from __future__ import annotations

from study_paths import PROJECT_ROOT, RESULTS_ROOT, METADATA_ROOT, REPO_ROOT, STATES_FILE, WATERSHEDS_FILE, DAILY_INPUT, EC_DIRECTORY, EC_INPUT, LOSO_INPUT, AGREEMENT_DIRECTORY, AGREEMENT_INPUT


import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter, MultipleLocator
import numpy as np
import xarray as xr

from build_word_primary_figures import EC, AGREEMENT, EXTENT, edges
from map_overlays import add_map_boundaries


ROOT = RESULTS_ROOT
OUT = ROOT / "agreement_narrative_figures" / "600dpi"
DIRECT = ROOT / "hurricane_results_word" / "direct_reference_agreement.json"
ORIGINAL_MAP = ROOT / "hurricane_results_word" / "figures" / "primary_figures_metadata.json"
DPI = 600
BROWN = "#A66A27"
TEAL = "#238B78"
TEXT = "#202A33"
MUTED = "#53616A"
MAP_STOPS = [
    (0.00, "#176C61"), (0.25, "#238B78"), (0.45, "#80C3BD"),
    (0.49, "#D8EFEB"), (0.50, "#FFFFFF"), (0.51, "#F0E8D3"),
    (0.55, "#D2B477"), (0.75, "#B7833B"), (1.00, "#88501D"),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_figure(fig, name, size, caption, metadata):
    stem = OUT / f"{name}_600dpi"
    fig.savefig(stem.with_suffix(".png"), dpi=DPI)
    fig.savefig(stem.with_suffix(".tif"), dpi=DPI,
                pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(stem.with_suffix(".svg"), dpi=DPI)
    fig.savefig(OUT / f"{name}_preview.png", dpi=145)
    plt.close(fig)
    metadata.update({
        "dpi": DPI, "figure_inches": list(size),
        "raster_dimensions": [round(v*DPI) for v in size],
        "source_data_modified": False, "estimation_rerun": False,
        "outputs": {s: str(stem.with_suffix('.'+s)) for s in ("png", "tif", "svg")},
    })
    (OUT / f"{name}_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (OUT / f"{name}_caption.md").write_text(caption + "\n", encoding="utf-8")
    print(json.dumps({"figure": name, "outputs": metadata["outputs"],
                      "panels": metadata.get("panels")}, indent=2), flush=True)


def map_figure():
    original = json.loads(ORIGINAL_MAP.read_text(encoding="utf-8"))
    hashes = {str(p): sha(p) for p in (EC, AGREEMENT)}
    with xr.open_dataset(EC, engine="netcdf4") as ec_file, xr.open_dataset(AGREEMENT, engine="netcdf4") as agreement_file:
        ec = ec_file.sel(transform="raw").load()
        agreement = agreement_file.sel(transform="raw", dataset_pair="conus404__conus404ba").load()
    mask = agreement.valid_agreement.values.astype(bool)
    expected = ec.valid_primary_ec.values.astype(bool) & (ec.bootstrap_valid_fraction.values >= .9)
    np.testing.assert_array_equal(mask, expected)
    delta = ec.error_sd_difference_ba_minus_conus404.values
    low = ec.error_sd_difference_ba_minus_conus404_ci_low.values
    high = ec.error_sd_difference_ba_minus_conus404_ci_high.values
    hatch = mask & np.isfinite(low) & np.isfinite(high) & (low <= 0) & (high >= 0)
    weights = (agreement.study_watershed_fraction.values.astype(float)
               * np.cos(np.deg2rad(agreement.lat.values))[:, None])
    assert np.isfinite(delta[mask]).all()
    assert delta[mask].min() >= -20 and delta[mask].max() <= 20
    cmap = LinearSegmentedColormap.from_list("teal_white_brown", MAP_STOPS, N=4096)
    cmap.set_bad("white")
    size = (16, 6.4)
    fig = plt.figure(figsize=size, facecolor="white")
    fig.text(.455, .972, "Bias-adjustment change in collocation random-error SD",
             ha="center", va="top", fontsize=23, fontweight="bold")
    fig.text(.455, .885, "CONUS404BA minus CONUS404  |  Raw daily precipitation  |  Paired four-product EC analysis",
             ha="center", fontsize=14, color=MUTED)
    width = .35
    height = width * size[0] * (EXTENT[3]-EXTENT[2]) / (EXTENT[1]-EXTENT[0]) / size[1]
    lon_edges, lat_edges = edges(ec.lon.values), edges(ec.lat.values)
    panels = []
    for index, window in enumerate(("core", "expanded")):
        selected_mask, selected_hatch = mask[index], hatch[index]
        assert int(selected_mask.sum()) == original["counts"][window]
        assert int(selected_hatch.sum()) == original["hatched_counts"][window]
        mean = float(np.sum(delta[index][selected_mask]*weights[selected_mask])/weights[selected_mask].sum())
        coverage = float(weights[selected_mask].sum()/weights.sum()*100)
        hatched_area = float(weights[selected_hatch].sum()/weights[selected_mask].sum()*100)
        ax = fig.add_axes([[.065, .48][index], .30, width, height])
        mesh = ax.pcolormesh(lon_edges, lat_edges, np.where(selected_mask, delta[index], np.nan),
                             cmap=cmap, norm=Normalize(-20, 20), shading="flat", rasterized=True,
                             edgecolors="none", antialiased=False, zorder=1)
        rr, cc = np.where(selected_hatch)
        rectangles = [Rectangle(
            (min(lon_edges[c], lon_edges[c+1]), min(lat_edges[r], lat_edges[r+1])),
            abs(lon_edges[c+1]-lon_edges[c]), abs(lat_edges[r+1]-lat_edges[r]),
        ) for r, c in zip(rr, cc)]
        ax.add_collection(PatchCollection(rectangles, facecolors="none", edgecolors="#626A6E",
                                          linewidths=0, hatch="///", zorder=4))
        ax.set_xlim(EXTENT[:2])
        ax.set_ylim(EXTENT[2:])
        add_map_boundaries(ax)
        ax.set_aspect("equal", adjustable="box")
        ax.xaxis.set_major_locator(FixedLocator([-100, -90, -80]))
        ax.yaxis.set_major_locator(FixedLocator([30, 35, 40]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{abs(v):.0f}°W"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}°N"))
        ax.tick_params(labelsize=13, length=4, width=.7, pad=4, labelleft=index == 0)
        for spine in ax.spines.values():
            spine.set_color("#8B959B")
            spine.set_linewidth(.75)
        ax.set_title(f"({'ab'[index]}) {window.capitalize()}", loc="left", fontsize=19,
                     fontweight="bold", pad=12)
        ax.set_title(f"{selected_mask.sum()} cells  |  {coverage:.2f}% area", loc="right",
                     fontsize=12.5, pad=13)
        ax.text(.025, .965, f"Mean ΔSD: {mean:+.2f} mm day$^{{-1}}$\nHatched area: {hatched_area:.1f}%",
                transform=ax.transAxes, ha="left", va="top", fontsize=12.5, linespacing=1.45,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .93, "pad": 3}, zorder=12)
        panels.append({
            "window": window, "retained_cells": int(selected_mask.sum()),
            "weighted_watershed_area_percent": coverage, "mean_delta_sd_mm_day": mean,
            "hatched_cells": int(selected_hatch.sum()), "hatched_percent_retained_area": hatched_area,
            "minimum": float(delta[index][selected_mask].min()),
            "maximum": float(delta[index][selected_mask].max()),
        })
    cax = fig.add_axes([.875, .335, .018, .415])
    cb = fig.colorbar(mesh, cax=cax, ticks=[-20, -10, 0, 10, 20])
    cb.set_label("BA − CONUS404 error SD (mm day$^{-1}$)", fontsize=14.5, labelpad=13)
    cb.ax.tick_params(labelsize=12.5, length=4, pad=5)
    cb.outline.set_linewidth(.7)
    fig.text(.45, .219, "Longitude", ha="center", fontsize=16)
    fig.text(.012, .54, "Latitude", ha="center", va="center", rotation=90, fontsize=16)
    fig.legend(handles=[Patch(facecolor="white", edgecolor="#626A6E", hatch="///",
                              linewidth=.5, label="Conditional storm-bootstrap 95% interval includes zero")],
               loc="lower center", bbox_to_anchor=(.45, .136), frameon=False, fontsize=13,
               handleheight=1.0, handlelength=2.1)
    fig.text(.252, .100, "Teal: lower estimated SD in BA", ha="center", fontsize=13, color="#176C61")
    fig.text(.655, .100, "Brown: higher estimated SD in BA", ha="center", fontsize=13, color="#88501D")
    fig.text(.45, .050, "Primary model-pair support with ≥90% admissible bootstrap replicates; not the separate-triplet diagnostic maps.",
             ha="center", fontsize=11.5, color=MUTED)
    fig.text(.45, .018, "Hatching does not establish equality. Conditional intervals do not validate the collocation error-independence assumptions.",
             ha="center", fontsize=11.5, color=MUTED)
    caption = (
        "Paired difference in raw-precipitation random-error standard deviation estimated by the four-product extended-collocation model, "
        "CONUS404BA minus CONUS404, during core and expanded storm-calendar windows. This is the original primary EC comparison, "
        "not a subtraction of the product-specific means or grids from the separate three-product diagnostic plots. "
        "Teal denotes lower and brown higher estimated error SD in BA. The shared linear normalization remains −20 to +20 mm day⁻¹; "
        "the custom sequential hues on either side of white improve contrast near zero without changing the values. "
        "Spatial support is valid_primary_ec and bootstrap_valid_fraction ≥0.90, identical to the original primary valid_agreement mask. "
        "The core and expanded panels retain 527 and 660 cells, respectively. Hatching indicates finite conditional 2.5th–97.5th percentile "
        "storm-bootstrap intervals that include zero; these intervals condition on admissible replicates and do not imply unconditional coverage "
        "or prove equality. Hatched-area percentages use retained panel area as denominator. Panel means use watershed-overlap fraction "
        "times cosine latitude; total coverage uses the stored watershed-fraction support as denominator. "
        "The model-pair mask does not require admissible MRMS/PERSIANN error variances and does not validate reference-error independence. "
        "Black lines delineate analysis watersheds and the CONUS boundary; gray lines delineate states."
    )
    metadata = {
        "sources": hashes, "transform": "raw", "source_variable": "error_sd_difference_ba_minus_conus404",
        "analysis": "four-product EC allowing CONUS404–CONUS404BA error covariance",
        "support": original["support"], "hatching": original["hatching"], "reference_mask_used": False,
        "color_limits": [-20, 20], "normalization": "linear", "palette_stops": MAP_STOPS,
        "extent_lonlat": list(EXTENT), "panels": panels,
    }
    save_figure(fig, "paired_ec_error_sd_change", size, caption, metadata)
    assert hashes == {str(p): sha(p) for p in (EC, AGREEMENT)}


def agreement_figure():
    source_hash = sha(DIRECT)
    data = json.loads(DIRECT.read_text(encoding="utf-8"))
    expected = {
        "core": {"conus404_minus_mrms": (.01, 14.71), "conus404ba_minus_mrms": (.19, 15.63),
                 "conus404_minus_persiann_cdr": (1.73, 16.77), "conus404ba_minus_persiann_cdr": (1.91, 17.53)},
        "expanded": {"conus404_minus_mrms": (-.12, 12.66), "conus404ba_minus_mrms": (.02, 13.37),
                     "conus404_minus_persiann_cdr": (.98, 14.19), "conus404ba_minus_persiann_cdr": (1.11, 14.79)},
    }
    for window in expected:
        assert data["windows"][window]["cells"] == 683
        for pair, values in expected[window].items():
            actual = data["windows"][window]["pairs"][pair]
            assert float(f"{actual['area_weighted_mean_pixel_bias']:.2f}") == values[0]
            assert float(f"{actual['area_weighted_mean_pixel_rmse']:.2f}") == values[1]
    size = (14.8, 8.8)
    fig, axes = plt.subplots(2, 2, figsize=size, facecolor="white")
    fig.subplots_adjust(left=.125, right=.96, bottom=.19, top=.80, wspace=.44, hspace=.72)
    fig.text(.535, .976, "Daily precipitation agreement: mean bias versus daily discrepancies",
             ha="center", va="top", fontsize=22, fontweight="bold")
    fig.text(.535, .932, "Area-weighted mean pixelwise metrics on identical matched samples  |  Model minus reference",
             ha="center", fontsize=14, color=MUTED)
    fig.legend(handles=[
        Line2D([], [], linestyle="none", marker="o", markersize=9, color=BROWN, label="CONUS404"),
        Line2D([], [], linestyle="none", marker="D", markersize=8, color=TEAL, label="CONUS404BA"),
    ], loc="lower center", bbox_to_anchor=(.535, .855), ncol=2, frameon=False,
       fontsize=14, columnspacing=2.2, handletextpad=.6)
    panels = []
    for row, window in enumerate(("core", "expanded")):
        pairs = data["windows"][window]["pairs"]
        days = data["windows"][window]["unique_calendar_days"]
        for col, metric in enumerate(("bias", "rmse")):
            ax = axes[row, col]
            variable = f"area_weighted_mean_pixel_{metric}"
            records = []
            for reference, label, y in (("mrms", "MRMS", 1), ("persiann_cdr", "PERSIANN-CDR", 0)):
                model_a = pairs[f"conus404_minus_{reference}"][variable]
                model_b = pairs[f"conus404ba_minus_{reference}"][variable]
                ax.plot([model_a, model_b], [y, y], color="#AEB8BE", linewidth=1.8, zorder=2)
                ax.scatter([model_a], [y], s=100, marker="o", color=BROWN, zorder=4)
                ax.scatter([model_b], [y], s=90, marker="D", color=TEAL, zorder=4)
                percent = (model_b/model_a - 1)*100 if metric == "rmse" else None
                text_a = f"{model_a:+.2f}" if metric == "bias" else f"{model_a:.2f}"
                text_b = f"{model_b:+.2f}" if metric == "bias" else f"{model_b:.2f} ({percent:+.1f}%)"
                for x, text, offset, va in ((model_a, text_a, 13, "bottom"), (model_b, text_b, -15, "top")):
                    ax.annotate(text, (x, y), xytext=(0, offset), textcoords="offset points",
                                ha="center", va=va, fontsize=13, color=TEXT, annotation_clip=False)
                records.append({"reference": reference, "conus404": model_a, "conus404ba": model_b,
                                "ba_minus_conus404": model_b-model_a,
                                "ba_rmse_percent_change_from_conus404": percent})
            ax.set_ylim(-.65, 1.65)
            ax.set_yticks([1, 0], ["MRMS", "PERSIANN-CDR"])
            ax.tick_params(axis="y", labelsize=13, length=0, pad=10)
            ax.tick_params(axis="x", labelsize=12.5, length=4, width=.7, pad=4)
            ax.set_axisbelow(True)
            ax.grid(axis="x", color="#E4E9EC", linewidth=.7)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color("#A4AFB6")
            ax.spines["bottom"].set_linewidth(.7)
            if metric == "bias":
                ax.set_xlim(-.35, 2.30)
                ax.xaxis.set_major_locator(FixedLocator([0, .5, 1, 1.5, 2]))
                ax.axvline(0, color="#89969F", linestyle=(0, (4, 3)), linewidth=.9, zorder=1)
                ax.set_xlabel("Mean bias (mm day$^{-1}$)", fontsize=14, labelpad=9)
                title = "mean bias"
            else:
                ax.set_xlim(0, 22)
                ax.xaxis.set_major_locator(MultipleLocator(5))
                ax.set_xlabel("Mean pixelwise RMSE (mm day$^{-1}$)", fontsize=14, labelpad=9)
                title = "daily RMSE"
            ax.set_title(f"({'abcd'[row*2+col]}) {window.capitalize()}: {title}", loc="left",
                         fontsize=17, fontweight="bold", pad=14)
            panels.append({"window": window, "metric": metric, "matched_cells": 683,
                           "unique_days": days, "records": records})
    fig.text(.535, .085,
             "BA moves expanded-window MRMS mean bias closer to zero, but raises mean pixelwise RMSE in all four comparisons.",
             ha="center", fontsize=13, color=TEXT)
    fig.text(.535, .044,
             "683 common cells • 92.45% weighted watershed area • 78 core / 127 expanded days • No EC or bootstrap screening",
             ha="center", fontsize=11.5, color=MUTED)
    fig.text(.535, .016,
             "Positive bias = model wetter than reference. RMSE percentages compare BA with CONUS404. References are not error-free truth.",
             ha="center", fontsize=11.5, color=MUTED)
    caption = (
        "Direct daily precipitation agreement of CONUS404 and CONUS404BA with MRMS Radar-Only QPE and PERSIANN-CDR. "
        "The upper row uses core storm windows (78 unique calendar days) and the lower row expanded windows (127 days). "
        "Left panels show signed model-minus-reference mean bias; right panels show mean pixelwise daily RMSE. "
        "Brown circles denote CONUS404 and teal diamonds denote CONUS404BA; gray segments connect the two model estimates for the same reference. "
        "Both columns use common axis limits across temporal windows. Text next to markers gives the point estimate; percentages accompanying "
        "BA RMSE are relative changes from CONUS404 calculated using full-precision values. Bias and RMSE were first calculated temporally "
        "at each pixel on the same four-product coincident sample, then spatially averaged using watershed-overlap fraction times cosine latitude. "
        "All four model–reference pairs use the same 683 cells (92.45% weighted watershed-area coverage) within each window. "
        "RMSE is an area-weighted mean of pixelwise RMSEs, not the square root of pooled domain mean squared difference. "
        "No collocation admissibility or bootstrap screening is used, and no uncertainty intervals or significance tests are inferred from the supplied table. "
        "BA reduces the magnitude of expanded-window mean bias against MRMS but has higher mean pixelwise RMSE than CONUS404 in all four comparisons. "
        "MRMS and PERSIANN-CDR are comparison references rather than error-free truth. Original duplicate trailing columns in the supplied table "
        "are omitted; all unique table values match the saved full-precision direct-agreement results."
    )
    metadata = {
        "source_summary": str(DIRECT), "source_summary_sha256": source_hash,
        "original_daily_source": data["source"], "support_definition": data["support_definition"],
        "weighting": data["weighting"], "bias_definition": data["bias_definition"],
        "rmse_definition": data["rmse_definition"], "reference_interpretation": data["reference_interpretation"],
        "colors": {"conus404": BROWN, "conus404ba": TEAL}, "units": "mm day-1",
        "inference": "Descriptive point estimates only; no statistical-significance claim or uncertainty intervals",
        "provided_table_unique_values_verified": True, "panels": panels,
    }
    save_figure(fig, "daily_model_reference_agreement", size, caption, metadata)
    assert sha(DIRECT) == source_hash


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "Times New Roman", "font.size": 13,
        "text.color": TEXT, "axes.labelcolor": TEXT,
        "xtick.color": "#39454E", "ytick.color": "#39454E",
        "svg.fonttype": "none", "savefig.facecolor": "white", "hatch.linewidth": .35,
    })
    map_figure()
    agreement_figure()


if __name__ == "__main__":
    main()
