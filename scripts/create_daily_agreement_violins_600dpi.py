"""Area-weighted spatial violins of real pixelwise daily agreement metrics.

Reuses the direct-agreement complete-case daily sample, not a TC mask.
The distributions are calculated from individual pixels, never fabricated
from table averages. No source data or original results are overwritten.
"""

from __future__ import annotations

from study_paths import PROJECT_ROOT, RESULTS_ROOT, METADATA_ROOT, REPO_ROOT, STATES_FILE, WATERSHEDS_FILE, DAILY_INPUT, EC_DIRECTORY, EC_INPUT, LOSO_INPUT, AGREEMENT_DIRECTORY, AGREEMENT_INPUT


import hashlib
import json
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
import numpy as np
from scipy.stats import gaussian_kde
import xarray as xr

from run_gridded_dataset_agreement import _selected_observations, _matched_sample


ROOT = RESULTS_ROOT
SUMMARY = ROOT / "hurricane_results_word" / "direct_reference_agreement.json"
OUT = ROOT / "agreement_violin_figures" / "600dpi"
PRODUCTS = ("conus404", "conus404ba", "mrms", "persiann_cdr")
MODELS = PRODUCTS[:2]
REFERENCES = PRODUCTS[2:]
WINDOWS = ("core", "expanded")
COLORS = {
    "conus404_minus_mrms": "#FEEEF6",
    "conus404_minus_persiann_cdr": "#C3CFE8",
    "conus404ba_minus_mrms": "#68ACD4",
    "conus404ba_minus_persiann_cdr": "#016F5B",
}
PAIR_LABELS = {
    "conus404_minus_mrms": "CONUS404 − MRMS",
    "conus404_minus_persiann_cdr": "CONUS404 − PERSIANN-CDR",
    "conus404ba_minus_mrms": "CONUS404BA − MRMS",
    "conus404ba_minus_persiann_cdr": "CONUS404BA − PERSIANN-CDR",
}
LABELS = {"conus404": "CONUS404", "conus404ba": "CONUS404BA"}
DPI = 600
SIZE = (14.5, 9.3)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weighted_quantile(values, weights, quantiles):
    order = np.argsort(values, kind="stable")
    x, w = np.asarray(values)[order], np.asarray(weights)[order]
    cumulative = (np.cumsum(w) - .5*w) / w.sum()
    return np.interp(quantiles, cumulative, x, left=x[0], right=x[-1])


def load_metrics(summary):
    source_path = Path(summary["source"])
    source_hash = sha(source_path)
    with xr.open_dataset(source_path, engine="netcdf4") as source:
        daily = source.load()
    domain = daily.study_domain_mask.values.astype(bool)
    weights = (daily.study_watershed_fraction.values.astype(float)
               * np.cos(np.deg2rad(daily.lat.values))[:, None])
    assert int(domain.sum()) == 683
    shape = (2, 2, 2, daily.sizes["lat"], daily.sizes["lon"])
    metrics = {"mean_bias": np.full(shape, np.nan), "rmse": np.full(shape, np.nan)}
    masks, day_counts, storm_counts, records, window_records = [], [], [], [], []
    for wi, window in enumerate(WINDOWS):
        selected, discarded = _selected_observations(daily, window)
        sample = daily.isel(observation=selected)
        values = np.stack([sample[p].values.astype(float) for p in PRODUCTS], axis=1)
        matched, counts, storms = _matched_sample(
            values, sample.storm_index.values, daily.sizes["storm"], domain, 3,
        )
        mask = domain & (counts >= (60 if window == "core" else 100)) & (storms >= 8)
        assert int(mask.sum()) == summary["windows"][window]["cells"] == 683
        assert int(sample.date.size) == summary["windows"][window]["unique_calendar_days"]
        w = weights[mask]
        assert np.isfinite(w).all() and (w > 0).all()
        v = np.where(matched[:, None], values, np.nan)
        for mi, model in enumerate(MODELS):
            for ri, reference in enumerate(REFERENCES):
                difference = v[:, PRODUCTS.index(model)] - v[:, PRODUCTS.index(reference)]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    pixel_bias = np.nanmean(difference, axis=0)
                    pixel_rmse = np.sqrt(np.nanmean(difference**2, axis=0))
                pair = summary["windows"][window]["pairs"][f"{model}_minus_{reference}"]
                for metric, array, mean_key in (
                    ("mean_bias", pixel_bias, "area_weighted_mean_pixel_bias"),
                    ("rmse", pixel_rmse, "area_weighted_mean_pixel_rmse"),
                ):
                    x = array[mask]
                    assert np.isfinite(x).all()
                    if metric == "rmse":
                        assert (x >= 0).all()
                    metrics[metric][wi, mi, ri] = np.where(mask, array, np.nan)
                    mean = float(np.average(x, weights=w))
                    np.testing.assert_allclose(mean, pair[mean_key], rtol=1e-12, atol=1e-11)
                    q05, q25, median, q75, q95 = weighted_quantile(x, w, [.05, .25, .5, .75, .95])
                    records.append({
                        "window": window, "model": model, "reference": reference, "metric": metric,
                        "cells": int(x.size), "area_weighted_mean": mean,
                        "area_weighted_median": float(median),
                        "area_weighted_q05": float(q05), "area_weighted_q25": float(q25),
                        "area_weighted_q75": float(q75), "area_weighted_q95": float(q95),
                        "area_weighted_spatial_sd": float(np.sqrt(np.average((x-mean)**2, weights=w))),
                        "minimum": float(x.min()), "maximum": float(x.max()),
                    })
        masks.append(mask)
        day_counts.append(counts)
        storm_counts.append(storms)
        window_records.append({
            "window": window, "cells": int(mask.sum()), "unique_calendar_days": int(sample.date.size),
            "weighted_watershed_area_percent": float(100*weights[mask].sum()/weights.sum()),
            "matched_days_minimum": int(counts[mask].min()), "matched_days_maximum": int(counts[mask].max()),
            "contributing_storms_minimum": int(storms[mask].min()),
            "contributing_storms_maximum": int(storms[mask].max()),
            "duplicate_calendar_records_discarded": int(discarded),
        })
    np.testing.assert_array_equal(masks[0], masks[1])
    dims = ("window", "model", "reference", "lat", "lon")
    saved = xr.Dataset(
        {**{name: (dims, values) for name, values in metrics.items()},
         "common_support": (("window", "lat", "lon"), np.asarray(masks, dtype="int8")),
         "matched_days": (("window", "lat", "lon"), np.asarray(day_counts, dtype="int16")),
         "contributing_storms": (("window", "lat", "lon"), np.asarray(storm_counts, dtype="int8")),
         "spatial_weight": (("lat", "lon"), weights)},
        coords={"window": list(WINDOWS), "model": list(MODELS), "reference": list(REFERENCES),
                "lat": daily.lat.values, "lon": daily.lon.values},
        attrs={
            "title": "Pixelwise direct daily model-reference agreement for spatial violin plots",
            "source_daily": str(source_path), "source_daily_sha256": source_hash,
            "support": summary["support_definition"], "weighting": summary["weighting"],
            "date_definition": summary["date_definition"], "reference_interpretation": summary["reference_interpretation"],
            "TC_admissibility_filter_applied": "false", "bootstrap_filter_applied": "false",
        },
    )
    saved.mean_bias.attrs.update(units="mm day-1", definition=summary["bias_definition"])
    saved.rmse.attrs.update(units="mm day-1", definition=summary["rmse_definition"])
    saved.lat.attrs.update(units="degrees_north", standard_name="latitude")
    saved.lon.attrs.update(units="degrees_east", standard_name="longitude")
    saved.to_netcdf(OUT / "direct_agreement_pixel_metrics.nc", engine="netcdf4",
                    encoding={name: {"zlib": True, "complevel": 4} for name in saved.data_vars})
    daily.close()
    return saved, records, window_records, source_path, source_hash


def draw(saved, records, windows):
    plt.rcParams.update({
        "font.family": "Times New Roman", "font.size": 13,
        "text.color": "#202A33", "axes.labelcolor": "#202A33",
        "xtick.color": "#39454E", "ytick.color": "#39454E", "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(2, 2, figsize=SIZE, facecolor="white", sharey="row")
    fig.subplots_adjust(left=.088, right=.973, bottom=.18, top=.75, wspace=.16, hspace=.40)
    fig.text(.535, .975, "Spatial distributions of daily precipitation agreement",
             ha="center", va="top", fontsize=23, fontweight="bold")
    fig.text(.535, .931, "Pixelwise temporal metrics  |  683 common cells across 23 watersheds  |  Area-weighted distributions",
             ha="center", fontsize=14, color="#53616A")
    fig.legend(handles=[
        Patch(facecolor=color, edgecolor="#454545", linewidth=.7, label=PAIR_LABELS[pair])
        for pair, color in COLORS.items()
    ], loc="lower center", bbox_to_anchor=(.535, .865), frameon=False,
       ncol=4, fontsize=12, columnspacing=1.5, handlelength=1.5, handletextpad=.65)
    fig.legend(handles=[
        Line2D([], [], marker="D", markersize=7, markerfacecolor="none", markeredgecolor="#17252B",
               color="none", label="Weighted mean"),
        Line2D([], [], marker="o", markersize=5, color="#17252B", linestyle="none", label="Weighted median"),
        Line2D([], [], color="#17252B", linewidth=1.1, marker="_", markersize=8,
               label="5th–95th percentiles"),
    ], loc="lower center", bbox_to_anchor=(.535, .814), frameon=False,
       ncol=3, fontsize=12.5, columnspacing=2, handlelength=1.3, handletextpad=.65)
    density_records = []
    for row, metric in enumerate(("mean_bias", "rmse")):
        for col, window in enumerate(WINDOWS):
            ax = axes[row, col]
            mask = saved.common_support.sel(window=window).values.astype(bool)
            w = saved.spatial_weight.values[mask]
            for ri, reference in enumerate(REFERENCES):
                for mi, model in enumerate(MODELS):
                    pair_id = f"{model}_minus_{reference}"
                    x = saved[metric].sel(window=window, model=model, reference=reference).values[mask]
                    position = ri + (-.18 if mi == 0 else .18)
                    kde = gaussian_kde(x, weights=w, bw_method="scott")
                    support = np.linspace(x.min(), x.max(), 700)
                    density = kde(support)
                    widths = .148 * density / density.max()
                    ax.fill_betweenx(support, position-widths, position+widths,
                                     facecolor=COLORS[pair_id], alpha=1,
                                     edgecolor="#454545", linewidth=.85,
                                     linestyle="solid", zorder=2)
                    q05, median, q95 = weighted_quantile(x, w, [.05, .5, .95])
                    mean = float(np.average(x, weights=w))
                    ax.vlines(position, q05, q95, color="#17252B", linewidth=1.15, zorder=4)
                    ax.hlines([q05, q95], position-.028, position+.028,
                              color="#17252B", linewidth=1.15, zorder=4)
                    ax.scatter(position, median, s=24, marker="o", color="#17252B", zorder=6)
                    ax.scatter(position, mean, s=49, marker="D", facecolors="none",
                               edgecolors="#17252B", linewidths=1.0, zorder=7)
                    # Numeric labels repeat the table mean, not a new pooled
                    # RMSE. Fixed upper placement keeps labels off the KDE.
                    label = f"{mean:+.2f}" if metric == "mean_bias" else f"{mean:.2f}"
                    ax.text(position, .962, label, transform=ax.get_xaxis_transform(),
                            ha="center", va="top", fontsize=12.5, color="#202A33",
                            bbox={"facecolor": "white", "edgecolor": "none", "alpha": .88, "pad": 1.5})
                    density_records.append({
                        "window": window, "model": model, "reference": reference, "metric": metric,
                        "kernel": "weighted Gaussian", "bandwidth_rule": "Scott",
                        "bandwidth_factor": float(kde.factor),
                        "kde_weight_effective_n_for_bandwidth": float(kde.neff),
                        "support_minimum": float(support.min()), "support_maximum": float(support.max()),
                        "maximum_half_width": float(widths.max()),
                    })
            ax.set_xlim(-.60, 1.60)
            ax.set_xticks([0, 1], ["MRMS", "PERSIANN-CDR"])
            ax.tick_params(axis="x", labelsize=14, length=0, pad=8)
            ax.tick_params(axis="y", labelsize=12.5, length=4, width=.7, pad=5)
            ax.set_axisbelow(True)
            ax.grid(False, which="both", axis="both")
            for side in ("top", "right", "left", "bottom"):
                ax.spines[side].set_visible(True)
                ax.spines[side].set_color("#545454")
                ax.spines[side].set_linewidth(.85)
                ax.spines[side].set_linestyle("solid")
            if metric == "mean_bias":
                assert all(-14 <= r["minimum"] <= r["maximum"] <= 20
                           for r in records if r["metric"] == metric)
                # Reserve clear headroom for numeric means, so the labels
                # cannot cover the largest observed tail.
                ax.set_ylim(-14, 20)
                ax.yaxis.set_major_locator(MultipleLocator(5))
                ax.axhline(0, color="#747474", linestyle="solid", linewidth=.8, zorder=1)
                metric_label = "mean bias"
                if col == 0:
                    ax.set_ylabel("Pixelwise mean bias (mm day$^{-1}$)", fontsize=15, labelpad=10)
            else:
                assert all(0 <= r["minimum"] <= r["maximum"] <= 100
                           for r in records if r["metric"] == metric)
                ax.set_ylim(0, 100)
                ax.yaxis.set_major_locator(MultipleLocator(20))
                metric_label = "daily RMSE"
                if col == 0:
                    ax.set_ylabel("Pixelwise daily RMSE (mm day$^{-1}$)", fontsize=15, labelpad=10)
            days = windows[col]["unique_calendar_days"]
            ax.set_title(f"({'abcd'[row*2+col]}) {window.capitalize()}: {metric_label}",
                         loc="left", fontsize=17, fontweight="bold", pad=14)
            ax.set_title(f"{days} days", loc="right", fontsize=12.5, pad=16)
    fig.text(.535, .108, "Comparison reference", ha="center", fontsize=15)
    fig.text(.535, .066,
             "Numbers above violins are area-weighted means matching Table 3-7. Whiskers show spatial spread, not confidence intervals.",
             ha="center", fontsize=12, color="#53616A")
    fig.text(.535, .030,
             "92.45% weighted watershed area • No collocation or bootstrap screening • Bias is model minus reference • References are not truth",
             ha="center", fontsize=11.5, color="#53616A")
    stem = OUT / "daily_agreement_spatial_violins_600dpi"
    fig.savefig(stem.with_suffix(".png"), dpi=DPI)
    fig.savefig(stem.with_suffix(".tif"), dpi=DPI, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(stem.with_suffix(".svg"), dpi=DPI)
    fig.savefig(OUT / "preview.png", dpi=145)
    plt.close(fig)
    return density_records, {s: str(stem.with_suffix('.'+s)) for s in ("png", "tif", "svg")}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    summary_hash = sha(SUMMARY)
    pixel_cache = OUT / "direct_agreement_pixel_metrics.nc"
    previous_metadata = OUT / "figure_metadata.json"
    if pixel_cache.exists() and previous_metadata.exists():
        previous = json.loads(previous_metadata.read_text(encoding="utf-8"))
        source = Path(previous["source_daily"])
        source_hash = previous["source_daily_sha256"]
        assert sha(source) == source_hash and summary_hash == previous["source_summary_sha256"]
        pixel_cache_hash = sha(pixel_cache)
        with xr.open_dataset(pixel_cache, engine="netcdf4") as cache:
            saved = cache.load()
        records, windows = previous["distributions"], previous["windows"]
        recomputed = False
    else:
        saved, records, windows, source, source_hash = load_metrics(summary)
        pixel_cache_hash = sha(pixel_cache)
        recomputed = True
    densities, output_files = draw(saved, records, windows)
    caption = (
        "Area-weighted spatial distributions of pixelwise daily precipitation agreement metrics. The reference-example palette is adapted to "
        "the four comparisons: pale pink, CONUS404–MRMS; lavender, CONUS404–PERSIANN-CDR; blue, CONUS404BA–MRMS; and teal, CONUS404BA–PERSIANN-CDR. "
        "Panel frames and violin contours are continuous solid lines; internal gridlines are removed, retaining only the solid zero-bias reference. "
        "Core windows (78 unique calendar days) are shown in the left "
        "column and expanded windows (127 days) in the right column. The upper row shows pixelwise temporal mean model-minus-reference bias "
        "in mm day⁻¹, not percentage bias; the lower row shows pixelwise temporal RMSE in mm day⁻¹. Each violin is constructed from the same "
        "683 matched grid cells across the 23 study watersheds (92.45% of weighted watershed area), with one temporal metric per pixel. "
        "Daily complete cases, storm qualification, and date deduplication match Table 3-7, and no collocation-admissibility or bootstrap filter "
        "is imposed. Density estimates use Gaussian kernels, Scott's bandwidth rule, and watershed-overlap fraction times cosine latitude as "
        "weights; displayed densities are restricted to each distribution's actual minimum and maximum. Each violin is normalized to the same "
        "maximum width, so widths convey within-distribution relative density rather than counts or independent sample size. Hollow diamonds "
        "show area-weighted means, black circles weighted medians, and capped vertical lines weighted 5th–95th percentile ranges. Weighted "
        "quantiles interpolate through cumulative-weight midpoints. Numbers above the violins repeat their area-weighted means, which match "
        "the saved daily-agreement table. The figure retains all observed tails on common linear axes within each metric. Spatial percentiles "
        "are descriptive ranges, not confidence intervals, and spatially correlated pixels are not independent storm replicates. Mean pixelwise "
        "RMSE is not pooled domain RMSE or TC random-error SD. MRMS and PERSIANN-CDR are imperfect comparison references, not precipitation truth."
    )
    metadata = {
        "source_daily": str(source), "source_daily_sha256": source_hash,
        "source_summary": str(SUMMARY), "source_summary_sha256": summary_hash,
        "source_data_modified": False, "TC_estimation_rerun": False,
        "direct_pixel_metrics_recomputed": recomputed, "area_weighted_means_match_table": True,
        "figure_inches": list(SIZE), "dpi": DPI, "raster_dimensions": [round(v*DPI) for v in SIZE],
        "density_weighting": "study_watershed_fraction*cos(latitude)",
        "density_normalization": "Each violin has identical maximum width; density cut at observed min/max",
        "summary_markers": {"hollow_diamond": "area-weighted mean", "black_circle": "area-weighted median",
                            "capped_vertical_line": "weighted 5th–95th percentile range, not a confidence interval"},
        "weighted_quantile_definition": "Linear interpolation through (cumsum(sorted_weights)-0.5*sorted_weights)/sum(weights)",
        "bootstrap_screen_applied": False, "TC_mask_applied": False,
        "bias_axis_limits": [-14, 20], "rmse_axis_limits": [0, 100],
        "all_tails_retained": True, "colors": COLORS,
        "panel_frame": "Four continuous solid sides",
        "violin_contour": "Continuous solid dark outline",
        "internal_gridlines_shown": False, "zero_bias_reference": "Solid line",
        "windows": windows, "distributions": records, "density_details": densities,
        "outputs": output_files, "pixel_metrics_netcdf": str(OUT / "direct_agreement_pixel_metrics.nc"),
    }
    (OUT / "figure_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (OUT / "figure_caption.md").write_text(caption + "\n", encoding="utf-8")
    assert sha(source) == source_hash and sha(SUMMARY) == summary_hash
    assert sha(pixel_cache) == pixel_cache_hash
    saved.close()
    print(json.dumps({"outputs": output_files, "windows": windows,
                      "distributions": records}, indent=2), flush=True)


if __name__ == "__main__":
    main()
