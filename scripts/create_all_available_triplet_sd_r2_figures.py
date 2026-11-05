"""Large diagnostic SD and latent-R2 maps from existing raw triplet results.

Each panel has its own numerical-range support, not a joint-triplet mask.
Undefined SD from negative variance and out-of-range R2 are gray. No source
NetCDFs are modified, no TC estimation is rerun, and no X markers are drawn.
"""

from __future__ import annotations

from study_paths import PROJECT_ROOT, RESULTS_ROOT, METADATA_ROOT, REPO_ROOT, STATES_FILE, WATERSHEDS_FILE, DAILY_INPUT, EC_DIRECTORY, EC_INPUT, LOSO_INPUT, AGREEMENT_DIRECTORY, AGREEMENT_INPUT


import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np
import xarray as xr

from build_word_triplet_metric_figures import SOURCES, EXTENT, LABELS, edges
from map_overlays import add_map_boundaries


ROOT = RESULTS_ROOT
OUT = ROOT / "triplet_sd_r2_all_available"
GRAY = "#B9BEC3"
METRICS = {
    "error_sd": {
        "title": "Pixelwise precipitation random-error standard deviation",
        "label": "TC random-error SD (mm day$^{-1}$)",
        "units": "mm day-1",
        "cmap": "YlGnBu", "limits": [0, 70], "digits": 2,
        "gray_label": "Undefined SD (negative error variance)",
        "derivation": "sqrt(error_variance_unmasked), only where finite and >=0",
        "note": "TC random-error SD, not direct RMSE against MRMS or PERSIANN-CDR; additive and multiplicative bias are not included as total errors against truth.",
    },
    "latent_r2": {
        "title": "Pixelwise estimated squared correlation with latent precipitation",
        "label": "Estimated latent-signal $R^2$",
        "units": "1",
        "cmap": "viridis", "limits": [0, 1], "digits": 3,
        "gray_label": "$R^2$ outside [0, 1]",
        "derivation": "signal_variance_unmasked / observed_variance, only where observed_variance >0",
        "note": "Estimated squared correlation with the latent signal, not observed pairwise rainfall correlation or regression R-squared. Values outside [0,1] are diagnostic failures, not valid correlations.",
    },
}


def metric_fields(ds, window, product, metric):
    """Return the diagnostic value and disjoint colored/gray/missing masks."""
    domain = ds.study_domain_mask.values.astype(bool)
    eligible = domain & ds.sample_eligible.sel(window=window).values.astype(bool)
    variance = ds.error_variance_unmasked.sel(window=window, product=product).values
    if metric == "error_sd":
        values = np.full(variance.shape, np.nan)
        good_variance = np.isfinite(variance) & (variance >= 0)
        np.sqrt(variance, out=values, where=good_variance)
        np.testing.assert_allclose(
            values, ds.error_sd_unmasked.sel(window=window, product=product).values,
            equal_nan=True, rtol=1e-12, atol=1e-12,
        )
        colored = eligible & np.isfinite(values)
        gray = eligible & np.isfinite(variance) & (variance < 0)
    else:
        signal = ds.signal_variance_unmasked.sel(window=window, product=product).values
        observed = ds.observed_variance.sel(window=window, product=product).values
        values = np.full(signal.shape, np.nan)
        np.divide(signal, observed, out=values,
                  where=np.isfinite(signal) & np.isfinite(observed) & (observed > 0))
        # The source's R2_unmasked was already product-screened. Reconstruct
        # the same algebraic ratio to reveal, not hide, out-of-range failures.
        stored = ds.estimated_r2_to_truth_unmasked.sel(window=window, product=product).values
        np.testing.assert_allclose(values[np.isfinite(stored)], stored[np.isfinite(stored)],
                                   rtol=1e-12, atol=1e-12)
        colored = eligible & np.isfinite(values) & (values >= 0) & (values <= 1)
        gray = eligible & np.isfinite(values) & ((values < 0) | (values > 1))
    missing = domain & ~(colored | gray)
    np.testing.assert_array_equal(colored | gray | missing, domain)
    assert not np.any(colored & gray)
    return values, colored, gray, missing, eligible


def draw(datasets, window, metric, dpi, source_hashes):
    config = METRICS[metric]
    out = OUT / metric / window / f"{dpi}dpi"
    out.mkdir(parents=True, exist_ok=True)
    domain = datasets[0].study_domain_mask.values.astype(bool)
    np.testing.assert_array_equal(domain, datasets[1].study_domain_mask.values.astype(bool))
    assert int(domain.sum()) == 683
    weights = (datasets[0].study_watershed_fraction.values.astype(float)
               * np.cos(np.deg2rad(datasets[0].lat.values))[:, None])
    fig = plt.figure(figsize=(16, 9.3), facecolor="white")
    fig.text(.465, .977, config["title"], ha="center", va="top",
             fontsize=22, fontweight="bold")
    fig.text(.465, .935,
             f"{window.capitalize()} event windows  |  Raw precipitation  |  All available product-specific estimates",
             ha="center", fontsize=14, color="#53616A")
    row_titles = ["Triplet A: CONUS404 – MRMS – PERSIANN-CDR",
                  "Triplet B: CONUS404BA – MRMS – PERSIANN-CDR"]
    row_header_y, y_positions = [.897, .515], [.575, .192]
    x_positions, panel_width = [.055, .335, .615], .255
    panel_height = panel_width * 16 * ((EXTENT[3] - EXTENT[2]) / (EXTENT[1] - EXTENT[0])) / 9.3
    cmap = plt.get_cmap(config["cmap"])
    norm = Normalize(*config["limits"], clip=False)
    panels = []
    for row, ds in enumerate(datasets):
        fig.text(.4625, row_header_y[row], row_titles[row], ha="center",
                 fontsize=16, fontweight="bold")
        for col, product_value in enumerate(ds.product.values):
            product = str(product_value)
            values, colored, gray, missing, eligible = metric_fields(ds, window, product, metric)
            assert int(eligible.sum()) == 683
            assert colored.any()
            assert values[colored].min() >= config["limits"][0]
            assert values[colored].max() <= config["limits"][1]
            mean = float(np.sum(values[colored] * weights[colored]) / weights[colored].sum())
            coverage = float(weights[colored].sum() / weights.sum() * 100)
            individual = ds.valid_product_estimate.sel(window=window, product=product).values.astype(bool)
            other_invalid = colored & ~individual
            ax = fig.add_axes([x_positions[col], y_positions[row], panel_width, panel_height])
            ax.pcolormesh(edges(ds.lon), edges(ds.lat), np.where(gray, 1.0, np.nan),
                          cmap=ListedColormap([GRAY]), vmin=0, vmax=1,
                          shading="flat", rasterized=True, edgecolors="none", zorder=1)
            mesh = ax.pcolormesh(edges(ds.lon), edges(ds.lat), np.where(colored, values, np.nan),
                                 cmap=cmap, norm=norm, shading="flat", rasterized=True,
                                 edgecolors="none", antialiased=False, zorder=2)
            ax.set_xlim(EXTENT[:2])
            ax.set_ylim(EXTENT[2:])
            add_map_boundaries(ax)
            for collection in ax.collections[2:]:
                if hasattr(collection, "get_linewidths"):
                    collection.set_linewidth(collection.get_linewidths() * .85)
            ax.set_aspect("equal", adjustable="box")
            ax.xaxis.set_major_locator(FixedLocator([-100, -90, -80]))
            ax.yaxis.set_major_locator(FixedLocator([30, 35, 40]))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{abs(v):.0f}°W"))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}°N"))
            ax.tick_params(labelsize=12, length=3.5, width=.6, pad=3,
                           labelleft=col == 0, labelbottom=True)
            for spine in ax.spines.values():
                spine.set_color("#8B959B")
                spine.set_linewidth(.65)
            ax.set_title(f"({'abcdef'[row*3+col]}) {LABELS[product]}", loc="left",
                         fontsize=16, fontweight="bold", pad=10)
            ax.set_title(f"Mean = {mean:.{config['digits']}f}", loc="right", fontsize=12.5, pad=11)
            counts = f"Color: {colored.sum()}   Gray: {gray.sum()}   NA: {missing.sum()}"
            ax.text(.025, .045, counts, transform=ax.transAxes, fontsize=11,
                    ha="left", va="bottom", zorder=12,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": .90, "pad": 2})
            panels.append({
                "panel": "abcdef"[row*3+col], "triplet": "A" if row == 0 else "B",
                "product": product, "sample_eligible_cells": int(eligible.sum()),
                "colored_cells": int(colored.sum()), "gray_cells": int(gray.sum()),
                "unavailable_cells": int(missing.sum()),
                "colored_but_other_product_admissibility_failed": int(other_invalid.sum()),
                "area_weighted_mean_colored_values": mean,
                "colored_weighted_watershed_area_percent": coverage,
                "minimum_colored_value": float(values[colored].min()),
                "maximum_colored_value": float(values[colored].max()),
                "other_failure_coordinates": [
                    {"latitude": float(ds.lat.values[yy]), "longitude": float(ds.lon.values[xx])}
                    for yy, xx in zip(*np.where(other_invalid))
                ],
            })
    cax = fig.add_axes([.893, .219, .018, .590])
    ticks = np.arange(0, 71, 10) if metric == "error_sd" else np.linspace(0, 1, 6)
    cb = fig.colorbar(mesh, cax=cax, ticks=ticks)
    cb.ax.tick_params(labelsize=12, length=4, width=.6, pad=5)
    cb.outline.set_linewidth(.7)
    cb.set_label(config["label"], fontsize=15, labelpad=14)
    fig.text(.4625, .126, "Longitude", ha="center", fontsize=15)
    fig.text(.011, .500, "Latitude", ha="center", va="center", rotation=90, fontsize=15)
    fig.legend(handles=[
        Patch(facecolor=GRAY, edgecolor="#8D969C", label=config["gray_label"]),
        Patch(facecolor="white", edgecolor="#8D969C", label="Unavailable estimate (within study cells)"),
    ], loc="lower center", bbox_to_anchor=(.4625, .066), frameon=False,
       ncol=2, fontsize=11.5, handlelength=1.3, handleheight=.8, columnspacing=1.4)
    fig.text(.4625, .046,
             "Means use colored cells only; spatial support differs between panels. Colored cells do not establish full-triplet admissibility.",
             ha="center", fontsize=11.5, color="#53616A")
    fig.text(.4625, .020,
             "Diagnostic display • No joint-triplet or bootstrap screening • Same 0.25° grid and color scale • Reference estimates kept separate by triplet",
             ha="center", fontsize=11.5, color="#53616A")
    stem = out / f"triplet_{metric}_{window}_all_available_{dpi}dpi"
    fig.savefig(stem.with_suffix(".png"), dpi=dpi)
    fig.savefig(stem.with_suffix(".svg"), dpi=dpi)
    fig.savefig(stem.with_suffix(".tif"), dpi=dpi, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(out / "preview.png", dpi=145)
    plt.close(fig)
    metadata = {
        "source_netcdfs": [str(p) for p in SOURCES], "source_sha256": source_hashes,
        "metric": metric, "derivation": config["derivation"], "metric_note": config["note"],
        "window": window, "transform": "raw", "units": config["units"],
        "estimation_rerun": False, "source_data_modified": False,
        "support_rule": "Each product independently: sample_eligible AND study_domain_mask; finite in-range metric colored",
        "gray_rule": config["gray_label"], "missing_rule": "Unavailable estimates white within study cells",
        "joint_admissibility_mask_applied": False, "bootstrap_filter_applied": False,
        "other_admissibility_markers_shown": False,
        "means_rule": "Mean over each panel's colored cells only; not matched-support product rankings; SD mean is mean of pixelwise SDs, not square root of mean variance",
        "weight_rule": "study_watershed_fraction * cos(latitude)",
        "caution": "Diagnostic maps; numerical-range screening does not validate the triplet decomposition or error-independence assumptions.",
        "dpi": dpi, "figure_inches": [16, 9.3], "raster_dimensions": [16*dpi, round(9.3*dpi)],
        "color_limits": config["limits"], "extent_lonlat": list(EXTENT), "panels": panels,
    }
    (out / "figure_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metric_caption = (
        "TC random-error standard deviation is the square root of each finite nonnegative algebraic error-variance estimate. "
        "Gray cells have negative estimated error variance and therefore no real-valued error SD; they are not converted to absolute values or clipped to zero. "
        "This is collocation random-error SD, not direct RMSE against a reference product. "
        "Other individual-product admissibility conditions are not applied to remove nonnegative-variance SD values, and their diagnostic flags remain in the metadata. "
        if metric == "error_sd" else
        "Estimated latent-signal R² is reconstructed as signal_variance_unmasked / observed_variance for positive observed variance. "
        "The source's stored estimated_r2_to_truth_unmasked variable already excludes individual-product failures; reconstructing the ratio exposes those failures. "
        "Finite ratios within [0,1] are colored; ratios below zero or above one are gray, not clipped. These gray ratios are not physically admissible squared correlations. "
        "This metric is an estimated squared correlation with latent precipitation, not observed pairwise correlation or regression R-squared. "
    )
    caption = (
        f"Figure. Product-specific available {metric.replace('_', ' ')} diagnostics for raw daily precipitation during {window} event windows. "
        "The upper row uses CONUS404–MRMS–PERSIANN-CDR; the lower row uses CONUS404BA–MRMS–PERSIANN-CDR. "
        + metric_caption +
        "Unavailable estimates are white within watershed outlines. No X markers are shown. Each panel uses its own sample-eligible numerical-range support; "
        "no joint-triplet or bootstrap mask is applied. Mean is the area-weighted mean of colored values, with weights equal to fractional watershed overlap times cosine latitude. "
        "The SD figure averages pixelwise SDs rather than taking the square root of average error variance. "
        "Because support differs between panels, their means are diagnostic summaries, not matched-support or domain-wide product rankings. "
        "Colored values do not establish full-triplet admissibility. Common color scales are used across all products, triplets, and windows for each metric. "
        "Black outlines show the 23 analysis watersheds and the CONUS boundary; light-gray outlines show states. Source collocation estimates are not rerun or modified.\n"
    )
    (out / "figure_caption.md").write_text(caption, encoding="utf-8")
    print(json.dumps({"window": window, "metric": metric, "output": str(out),
                      "means": [p["area_weighted_mean_colored_values"] for p in panels],
                      "counts": [[p[k] for k in ("colored_cells", "gray_cells", "unavailable_cells")] for p in panels]}, indent=2), flush=True)
    return metadata


def main(dpi=600):
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 12, "text.color": "#202A33", "axes.labelcolor": "#202A33",
        "xtick.color": "#39454E", "ytick.color": "#39454E", "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCES}
    datasets = [xr.open_dataset(p, engine="netcdf4").load() for p in SOURCES]
    for window in ("core", "expanded"):
        for metric in METRICS:
            draw(datasets, window, metric, dpi, hashes)
    for ds in datasets:
        ds.close()
    assert hashes == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCES}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpi", type=int, default=600, choices=[300, 400, 600])
    main(dpi=parser.parse_args().dpi)
