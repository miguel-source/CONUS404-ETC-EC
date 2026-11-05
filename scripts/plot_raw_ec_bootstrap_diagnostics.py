"""Raw four-product EC tables/maps with bootstrap diagnostics, no point filtering.

Read saved estimates, percentiles and validity fractions (5,000 storm draws).
Keep all 339/401 jointly admissible full-sample cells. Never overwrite inputs.
Table output is Markdown and a publication-style image, not a workbook.
"""

from study_paths import PROJECT_ROOT, RESULTS_ROOT, METADATA_ROOT, REPO_ROOT, STATES_FILE, WATERSHEDS_FILE, DAILY_INPUT, EC_DIRECTORY, EC_INPUT, LOSO_INPUT, AGREEMENT_DIRECTORY, AGREEMENT_INPUT

from pathlib import Path
import json
import os

HERE = RESULTS_ROOT
OUTPUT = HERE / "raw_ec_bootstrap_diagnostics"
os.environ.setdefault("MPLCONFIGDIR", str(HERE / "_core_cells_mpl"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.patches import Patch, Rectangle
import numpy as np
import xarray as xr

from map_overlays import add_map_boundaries
from plot_raw_full_sample_error_variance_maps import coordinate_edges

ROOT = (PROJECT_ROOT / 'ETC_StormEvents')
EC_PATH = ROOT / "Hurricane_gridded_EC_results" / "gridded_extended_collocation_all_windows.nc"
AGREEMENT_PATH = ROOT / "Hurricane_gridded_agreement_results" / "gridded_dataset_agreement.nc"
PRODUCTS = ("mrms", "conus404", "conus404ba", "persiann_cdr")
TABLE_PRODUCTS = ("conus404", "conus404ba", "mrms", "persiann_cdr")
LABELS = {"mrms": "MRMS", "conus404": "CONUS404", "conus404ba": "CONUS404BA", "persiann_cdr": "PERSIANN-CDR"}
EXTENT = (-105.5, -71.5, 26.0, 45.0)


def discrete_scale(maximum, candidates):
    boundaries = list(candidates)
    while boundaries[-1] < maximum:
        boundaries.append(2 * boundaries[-1])
    last = next(i for i, value in enumerate(boundaries) if value >= maximum)
    boundaries = boundaries[:last + 1]
    cmap = plt.get_cmap("viridis", len(boundaries) - 1)
    return cmap, BoundaryNorm(boundaries, cmap.N), boundaries


def main():
    with xr.open_dataset(EC_PATH, engine="netcdf4") as f:
        ec = f.sel(transform="raw").load()
    with xr.open_dataset(AGREEMENT_PATH, engine="netcdf4") as f:
        agreement = f.sel(transform="raw").load()
    xr.testing.assert_equal(ec.lat, agreement.lat)
    xr.testing.assert_equal(ec.lon, agreement.lon)
    if ec.attrs.get("bootstrap_replicates") != 5000:
        raise ValueError("Expected the previously reported 5,000-draw bootstrap.")
    domain = agreement.study_domain_mask.values.astype(bool)
    weights = agreement.study_watershed_fraction.values.astype(float) * np.cos(np.deg2rad(ec.lat.values))[:, None]
    xe, ye = coordinate_edges(ec.lon.values), coordinate_edges(ec.lat.values)
    data, summaries = {}, {}
    for window, expected in (("core", 339), ("expanded", 401)):
        q = ec.sel(window=window)
        mask = q.valid_primary_ec.values.astype(bool) & q.reference_diagnostics_valid.values.astype(bool)
        if mask.sum() != expected or np.any(mask & ~domain):
            raise ValueError(f"Unexpected {window} full-sample support.")
        fields = {metric: {} for metric in ("error_variance", "error_sd", "bootstrap_valid_pct", "conditional_interval_width")}
        summary = {"retained_cells": int(mask.sum()), "weighted_watershed_area_pct": float(100 * weights[mask].sum() / weights.sum()), "products": {}}
        for product in PRODUCTS:
            variance = q.error_variance.sel(product=product).values.astype(float)
            sd = q.error_sd.sel(product=product).values.astype(float)
            fraction_name = "bootstrap_valid_fraction" if product in ("conus404", "conus404ba") else "bootstrap_reference_valid_fraction"
            fraction = q[fraction_name].values.astype(float)
            low = q.error_sd_ci_low.sel(product=product).values.astype(float)
            high = q.error_sd_ci_high.sel(product=product).values.astype(float)
            width = high - low
            for array in (variance, sd, fraction, low, high, width):
                if not np.isfinite(array[mask]).all():
                    raise ValueError(f"Missing retained {window}/{product} values.")
            if np.any(variance[mask] <= 0) or np.any(sd[mask] <= 0) or np.any(width[mask] < 0):
                raise ValueError("Inadmissible retained estimates or percentile ranges.")
            if np.any((fraction[mask] < 0) | (fraction[mask] > 1)):
                raise ValueError("Invalid bootstrap fractions.")
            for metric, array in (("error_variance", variance), ("error_sd", sd), ("bootstrap_valid_pct", 100 * fraction), ("conditional_interval_width", width)):
                fields[metric][product] = np.where(mask, array, np.nan)
            summary["products"][product] = {
                "area_weighted_mean_pixel_error_variance": float(np.sum(weights[mask] * variance[mask]) / weights[mask].sum()),
                "area_weighted_mean_pixel_error_sd": float(np.sum(weights[mask] * sd[mask]) / weights[mask].sum()),
                "median_pixel_conditional_interval_width": float(np.median(width[mask])),
                "median_pixel_bootstrap_valid_pct": float(100 * np.median(fraction[mask])),
                "cells_with_at_least_90pct_valid_draws": int(np.sum(mask & (fraction >= 0.9))),
                "bootstrap_validity_variable": fraction_name,
                "interval_definition": "saved 97.5th minus 2.5th percentile of error SD over admissible draws; conditional on draw admissibility",
            }
        data[window] = {"mask": mask, "fields": fields}
        summaries[window] = summary

    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titleweight": "semibold", "svg.fonttype": "none", "savefig.facecolor": "white"})
    varmax = max(np.nanmax(d["fields"]["error_variance"][p]) for d in data.values() for p in PRODUCTS)
    widthmax = max(np.nanmax(d["fields"]["conditional_interval_width"][p]) for d in data.values() for p in PRODUCTS)
    var_cmap, var_norm, var_ticks = discrete_scale(varmax, (0, 5, 10, 25, 50, 100, 200, 400, 800, 1600, 3200, 6400))
    width_cmap, width_norm, width_ticks = discrete_scale(widthmax, (0, 2, 5, 10, 20, 40, 80, 160))
    sdmax = max(np.nanmax(d["fields"]["error_sd"][p]) for d in data.values() for p in PRODUCTS)
    sd_limit = float(np.ceil(sdmax / 10) * 10)
    configurations = {
        "error_variance": ("Full-sample EC random-error variance", "(mm/day)²", var_cmap, var_norm, var_ticks),
        "error_sd": ("Full-sample EC random-error standard deviation", "mm/day", "viridis", Normalize(0, sd_limit), None),
        "bootstrap_valid_pct": ("Storm-block-bootstrap admissibility", "% of 5,000 draws", "RdYlGn", Normalize(0, 100), [0, 20, 40, 60, 80, 90, 100]),
        "conditional_interval_width": ("Conditional bootstrap percentile-range width of error SD", "mm/day", width_cmap, width_norm, width_ticks),
    }
    figure_paths = {}
    for window, d in data.items():
        summary = summaries[window]
        count = summary["retained_cells"]
        figure_paths[window] = {}
        for metric, (title, units, cmap, norm, ticks) in configurations.items():
            fig, axes = plt.subplots(2, 2, figsize=(14.5, 11.0), sharex=True, sharey=True)
            fig.subplots_adjust(left=0.065, right=0.865, top=0.865, bottom=0.165, wspace=0.13, hspace=0.26)
            for ax, product, letter in zip(axes.ravel(), PRODUCTS, "abcd"):
                ax.set_xlim(EXTENT[:2])
                ax.set_ylim(EXTENT[2:])
                ax.pcolormesh(xe, ye, np.where(domain, 1.0, np.nan), cmap=ListedColormap(["#e1e7ec"]), vmin=0, vmax=1, shading="flat", zorder=1)
                artist = ax.pcolormesh(xe, ye, d["fields"][metric][product], cmap=cmap, norm=norm, shading="flat", zorder=3)
                add_map_boundaries(ax)
                stats = summary["products"][product]
                if metric == "error_variance":
                    annotation = f"Mean variance: {stats['area_weighted_mean_pixel_error_variance']:.2f} {units}"
                elif metric == "error_sd":
                    annotation = f"Mean pixel error SD: {stats['area_weighted_mean_pixel_error_sd']:.2f} {units}"
                elif metric == "bootstrap_valid_pct":
                    annotation = f"Median valid draws: {stats['median_pixel_bootstrap_valid_pct']:.1f}%"
                else:
                    annotation = f"Median pixel range width: {stats['median_pixel_conditional_interval_width']:.2f} {units}"
                stable = stats["cells_with_at_least_90pct_valid_draws"]
                ax.set_title(f"({letter}) {LABELS[product]}\n{annotation} | ≥90% valid: {stable}/{count}", fontsize=10.5)
                # Keep point/interval values visible; hatching is a flag only.
                # True 0.25-degree rectangles avoid contour interpolation.
                if metric != "bootstrap_valid_pct":
                    flagged = d["mask"] & (d["fields"]["bootstrap_valid_pct"][product] < 90)
                    rr, cc = np.where(flagged)
                    patches = [Rectangle((min(xe[c:c+2]), min(ye[r:r+2])), abs(xe[c+1]-xe[c]), abs(ye[r+1]-ye[r])) for r, c in zip(rr, cc)]
                    ax.add_collection(PatchCollection(patches, facecolor="none", edgecolor="#343434", linewidth=0, hatch="///", alpha=0.65, zorder=8))
                ax.set_xlabel("Longitude (°)")
                ax.set_ylabel("Latitude (°)")
                ax.tick_params(labelbottom=True, labelleft=True, labelsize=9)
            color_ax = fig.add_axes([0.895, 0.225, 0.023, 0.555])
            discrete = metric in ("error_variance", "conditional_interval_width")
            kwargs = {"boundaries": ticks, "ticks": ticks, "spacing": "uniform"} if discrete else {"ticks": ticks} if ticks else {}
            bar = fig.colorbar(artist, cax=color_ax, **kwargs)
            bar.set_label(f"{title} ({units})", fontsize=10)
            bar.ax.tick_params(labelsize=9)
            fig.suptitle(f"{window.capitalize()} windows • raw precipitation • {title}\nAll {count} jointly admissible full-sample cells retained ({summary['weighted_watershed_area_pct']:.2f}% weighted watershed area)", fontsize=16, y=0.965)
            if metric != "bootstrap_valid_pct":
                fig.legend(handles=[Patch(facecolor="none", edgecolor="#343434", hatch="///", label="<90% admissible product-specific bootstrap draws; values retained"), Patch(facecolor="#e1e7ec", edgecolor="none", label="Outside common full-sample admissible support")], loc="lower center", bbox_to_anchor=(0.5, 0.105), ncol=2, frameon=False, fontsize=9)
            note = "CONUS404/BA validity: primary model-pair checks. MRMS/PERSIANN validity: primary plus joint reference checks."
            fig.text(0.5, 0.075, note, ha="center", fontsize=9)
            if metric == "conditional_interval_width":
                extra = "Width = 97.5th − 2.5th percentile of admissible draws only; conditional ranges, not confidence intervals for area-weighted means."
            elif metric == "bootstrap_valid_pct":
                extra = "Matching validity maps within product pairs follow their shared validation rule; they do not imply identical error estimates."
            else:
                extra = "Full-sample point estimates are unchanged. Bootstrap diagnoses sampling sensitivity; no point-estimate cells are removed."
            fig.text(0.5, 0.050, extra, ha="center", fontsize=9, color="#404040")
            fig.text(0.5, 0.026, "Same color scale across products and windows; different core/expanded spatial support. Watershed and CONUS/state boundaries shown.", ha="center", fontsize=9, color="#404040")
            stem = OUTPUT / f"{window}_raw_all_admissible_{metric}_bootstrap_diagnostics"
            paths = {}
            for extension in ("png", "svg"):
                path = stem.with_suffix("." + extension)
                fig.savefig(path, dpi=300)
                paths[extension] = str(path)
            plt.close(fig)
            figure_paths[window][metric] = paths

    # Table of point estimates plus separate pixelwise bootstrap diagnostics.
    # Never average percentile endpoints and label them a CI of the mean.
    table_rows = []
    markdown = ["# Raw EC point estimates and storm-block-bootstrap diagnostics", "", "All jointly admissible full-sample cells retained: core 339 (45.62% weighted watershed area); expanded 401 (54.10%).", "", "| Window | Product | Area-weighted mean pixel error SD (mm/day) | Median pixel conditional range width (mm/day) | Median pixel admissible bootstrap draws (%) | Pixels with ≥90% admissible draws |", "|---|---|---:|---:|---:|---:|"]
    for window, summary in summaries.items():
        for product in TABLE_PRODUCTS:
            s = summary["products"][product]
            row = [window.capitalize(), LABELS[product], f"{s['area_weighted_mean_pixel_error_sd']:.2f}", f"{s['median_pixel_conditional_interval_width']:.2f}", f"{s['median_pixel_bootstrap_valid_pct']:.1f}", f"{s['cells_with_at_least_90pct_valid_draws']}/{summary['retained_cells']}"]
            table_rows.append(row)
            markdown.append("| " + " | ".join(row) + " |")
    notes = [
        "Point-estimate SDs are area-weighted means; bootstrap diagnostic medians are unweighted medians across retained pixels.",
        "Range width is the saved 97.5th minus 2.5th percentile at each pixel, using admissible bootstrap draws only. It is conditional on draw admissibility, not an unconditional 95% confidence interval.",
        "CONUS404/BA intervals and valid fractions use primary model-pair checks. MRMS/PERSIANN require primary plus joint reference checks. Matching percentages within pairs reflect shared validation rules.",
        "The ≥90% column is a flag/count, not a filter. All 339 core and 401 expanded point-estimate cells remain included.",
        "Reference estimates have frequent inadmissible bootstrap draws; their conditional ranges must not be presented as definitive precision or domain-wide rankings.",
        "The bootstrap used 5,000 common storm-block draws, seed 404. Invalid draw estimates were not replaced with zero.",
    ]
    markdown += [""] + ["- " + note for note in notes]
    (OUTPUT / "raw_ec_point_estimates_and_bootstrap_table.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    fig, ax = plt.subplots(figsize=(16.5, 8.5))
    ax.axis("off")
    headers = ["Window", "Product", "Mean pixel error SD\n(mm/day)", "Median conditional\nrange width (mm/day)", "Median admissible\nbootstrap draws (%)", "Pixels with ≥90%\nadmissible draws"]
    table = ax.table(cellText=table_rows, colLabels=headers, cellLoc="center", colWidths=[0.085, 0.16, 0.17, 0.205, 0.195, 0.185], bbox=[0, 0.30, 1, 0.58])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#d1dbe5")
        cell.set_linewidth(0.7)
        if r == 0:
            cell.set_facecolor("#17334d")
            cell.set_text_props(color="white", weight="bold", fontsize=10.5)
            cell.set_height(cell.get_height() * 1.45)
        else:
            cell.set_facecolor("#eef4fa" if r <= 4 else "#eef7f4")
            if c == 1:
                cell.set_text_props(weight="bold")
    ax.set_title("Raw EC point estimates and storm-block-bootstrap diagnostics", fontsize=19, pad=28)
    ax.text(0.5, 0.94, "Core: all 339 jointly admissible cells (45.62% area) | Expanded: all 401 cells (54.10% area)", ha="center", fontsize=12)
    for y, note in zip([0.235, 0.188, 0.141, 0.094, 0.047], [
        "All point-estimate cells remain included. ≥90% is a diagnostic count, not a spatial filter.",
        "SD means are area-weighted; diagnostic medians summarize individual pixels without area weighting.",
        "Conditional range = pixelwise 2.5–97.5 percentiles of admissible draws only; not a CI for the reported mean SD.",
        "CONUS404/BA: model-pair validity. MRMS/PERSIANN: primary plus joint reference validity.",
        "5,000 storm-block draws; seed 404. Frequent inadmissible draws limit reference-estimate reliability.",
    ]):
        ax.text(0.5, y, note, ha="center", fontsize=10, color="#404040")
    fig.subplots_adjust(left=0.035, right=0.965, bottom=0.045, top=0.875)
    table_paths = {}
    for extension in ("png", "svg"):
        path = OUTPUT / f"raw_ec_point_estimates_and_bootstrap_table.{extension}"
        fig.savefig(path, dpi=300)
        table_paths[extension] = str(path)
    plt.close(fig)
    metadata = {"ec_source": str(EC_PATH), "agreement_source": str(AGREEMENT_PATH), "sampling_scope": "all_windows", "transform": "raw", "bootstrap_replicates": int(ec.attrs["bootstrap_replicates"]), "bootstrap_seed": int(ec.attrs["bootstrap_seed"]), "point_mask": "valid_primary_ec AND reference_diagnostics_valid", "point_mask_filtered_by_bootstrap": False, "product_summary": summaries, "notes": notes, "figures": figure_paths, "table": table_paths, "variance_color_bins": var_ticks, "interval_width_color_bins": width_ticks, "sd_color_limits": [0, sd_limit]}
    (OUTPUT / "raw_ec_bootstrap_diagnostics_provenance.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summaries, "table": table_paths, "figures": figure_paths}, indent=2))


if __name__ == "__main__":
    main()
