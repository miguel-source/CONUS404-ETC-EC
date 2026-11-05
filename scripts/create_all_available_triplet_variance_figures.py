"""Display each product's available algebraic error variances without joint masks.

Negative estimates are gray and nonfinite/unavailable estimates are white.
Other individual-product admissibility failures remain in the metadata but
are not separately marked. No source results are edited or re-estimated.
"""

from __future__ import annotations

from study_paths import PROJECT_ROOT, RESULTS_ROOT, METADATA_ROOT, REPO_ROOT, STATES_FILE, WATERSHEDS_FILE, DAILY_INPUT, EC_DIRECTORY, EC_INPUT, LOSO_INPUT, AGREEMENT_DIRECTORY, AGREEMENT_INPUT


import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np
import xarray as xr

from build_word_triplet_metric_figures import SOURCES, EXTENT, LABELS, edges
from create_large_triplet_error_variance_figure import BOUNDARIES
from map_overlays import add_map_boundaries


ROOT = RESULTS_ROOT
OUT = ROOT / "triplet_error_variance_all_available"
NEGATIVE_COLOR = "#B9BEC3"


def draw(datasets, window, dpi):
    out = OUT / window / f"{dpi}dpi"
    out.mkdir(parents=True, exist_ok=True)
    domain = datasets[0].study_domain_mask.values.astype(bool)
    assert domain.sum() == 683
    np.testing.assert_array_equal(domain, datasets[1].study_domain_mask.values.astype(bool))
    weights = datasets[0].study_watershed_fraction.values.astype(float) * np.cos(np.deg2rad(datasets[0].lat.values))[:, None]
    fig = plt.figure(figsize=(16, 9.3), facecolor="white")
    fig.text(.465, .977, "Pixelwise precipitation random-error variance", ha="center", va="top",
             fontsize=22, fontweight="bold")
    fig.text(.465, .935, f"{window.capitalize()} event windows  |  Raw precipitation  |  All available product-specific estimates",
             ha="center", fontsize=14, color="#53616A")
    row_titles = ["Triplet A: CONUS404 – MRMS – PERSIANN-CDR",
                  "Triplet B: CONUS404BA – MRMS – PERSIANN-CDR"]
    row_header_y, y_positions = [.897, .515], [.575, .192]
    x_positions, panel_width = [.055, .335, .615], .255
    panel_height = panel_width * 16 * ((EXTENT[3] - EXTENT[2]) / (EXTENT[1] - EXTENT[0])) / 9.3
    cmap = plt.get_cmap("YlGnBu", len(BOUNDARIES) - 1)
    norm = BoundaryNorm(BOUNDARIES, cmap.N)
    panels = []
    for row, ds in enumerate(datasets):
        fig.text(.4625, row_header_y[row], row_titles[row], ha="center", fontsize=16, fontweight="bold")
        eligible = ds.sample_eligible.sel(window=window).values.astype(bool) & domain
        assert eligible.sum() == 683
        for col, product_value in enumerate(ds.product.values):
            product = str(product_value)
            raw = ds.error_variance_unmasked.sel(window=window, product=product).values
            positive = eligible & np.isfinite(raw) & (raw >= 0)
            negative = eligible & np.isfinite(raw) & (raw < 0)
            missing = domain & ~np.isfinite(raw)
            np.testing.assert_array_equal(positive | negative | missing, domain)
            assert not np.any(positive & negative)
            individual = ds.valid_product_estimate.sel(window=window, product=product).values.astype(bool)
            other_invalid = positive & ~individual
            assert positive.any()
            assert np.max(raw[positive]) <= BOUNDARIES[-1], "Shared scale would clip positive values"
            mean = float(np.sum(raw[positive] * weights[positive]) / weights[positive].sum())
            coverage = float(weights[positive].sum() / weights.sum() * 100)
            ax = fig.add_axes([x_positions[col], y_positions[row], panel_width, panel_height])
            ax.pcolormesh(edges(ds.lon), edges(ds.lat), np.where(negative, 1.0, np.nan),
                          cmap=ListedColormap([NEGATIVE_COLOR]), vmin=0, vmax=1,
                          shading="flat", rasterized=True, edgecolors="none", zorder=1)
            mesh = ax.pcolormesh(edges(ds.lon), edges(ds.lat), np.where(positive, raw, np.nan),
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
            ax.set_title(f"Mean ≥0 = {mean:.2f}", loc="right", fontsize=12.5, pad=11)
            counts = f"n≥0: {positive.sum()}   n<0: {negative.sum()}   NA: {missing.sum()}"
            ax.text(.025, .045, counts, transform=ax.transAxes, fontsize=11,
                    ha="left", va="bottom", zorder=12,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": .90, "pad": 2})
            panels.append({
                "panel": "abcdef"[row*3+col], "triplet": "A" if row == 0 else "B",
                "product": product, "sample_eligible_cells": int(eligible.sum()),
                "nonnegative_cells": int(positive.sum()), "negative_cells": int(negative.sum()),
                "unavailable_cells": int(missing.sum()),
                "nonnegative_but_other_admissibility_failed": int(other_invalid.sum()),
                "area_weighted_mean_nonnegative_error_variance": mean,
                "nonnegative_weighted_watershed_area_percent": coverage,
                "minimum_finite_estimate": float(np.nanmin(raw[eligible])),
                "maximum_finite_estimate": float(np.nanmax(raw[eligible])),
                "other_failure_coordinates": [
                    {"latitude": float(ds.lat.values[yy]), "longitude": float(ds.lon.values[xx])}
                    for yy, xx in zip(*np.where(other_invalid))
                ],
            })
    cax = fig.add_axes([.893, .219, .018, .590])
    cb = fig.colorbar(mesh, cax=cax, boundaries=BOUNDARIES, ticks=BOUNDARIES, spacing="uniform")
    cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    cb.ax.tick_params(labelsize=12, length=4, width=.6, pad=5)
    cb.outline.set_linewidth(.7)
    cb.set_label("Random-error variance [(mm day$^{-1}$)$^2$]", fontsize=15, labelpad=14)
    fig.text(.4625, .126, "Longitude", ha="center", fontsize=15)
    fig.text(.011, .500, "Latitude", ha="center", va="center", rotation=90, fontsize=15)
    fig.legend(handles=[
        Patch(facecolor=NEGATIVE_COLOR, edgecolor="#8D969C", label="Negative estimate"),
        Patch(facecolor="white", edgecolor="#8D969C", label="Unavailable estimate (within study cells)"),
    ], loc="lower center", bbox_to_anchor=(.4625, .066), frameon=False,
       ncol=2, fontsize=11.5, handlelength=1.3, handleheight=.8, columnspacing=1.4)
    fig.text(.4625, .046,
             "Means use each panel’s finite nonnegative estimates; spatial support differs between panels. Negative estimates are not valid physical variances.",
             ha="center", fontsize=11.5, color="#53616A")
    fig.text(.4625, .020,
             "Diagnostic display • No joint-triplet or bootstrap screening • Same 0.25° grid and color scale • Reference estimates kept separate by triplet",
             ha="center", fontsize=11.5, color="#53616A")
    stem = out / f"triplet_error_variance_{window}_all_available_{dpi}dpi"
    fig.savefig(stem.with_suffix(".png"), dpi=dpi)
    fig.savefig(stem.with_suffix(".svg"), dpi=dpi)
    fig.savefig(stem.with_suffix(".tif"), dpi=dpi, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(out / "preview.png", dpi=145)
    plt.close(fig)
    metadata = {
        "source_netcdfs": [str(p) for p in SOURCES], "source_variable": "error_variance_unmasked",
        "window": window, "transform": "raw", "units": "(mm day-1)^2",
        "estimation_rerun": False, "source_data_modified": False,
        "support_rule": "Each product independently: sample_eligible AND study_domain_mask AND finite error variance",
        "color_rule": "Finite >=0 colored; finite <0 gray; nonfinite white; other product-validity failures not separately marked",
        "other_admissibility_markers_shown": False,
        "joint_admissibility_mask_applied": False, "bootstrap_filter_applied": False,
        "means_rule": "Area-weighted mean over the individual panel's finite nonnegative estimates; includes flagged nonnegative values",
        "weight_rule": "study_watershed_fraction * cos(latitude)",
        "caution": "Diagnostic maps; colored/nonnegative does not guarantee full admissibility. Means from differing spatial support are not matched-support rankings.",
        "dpi": dpi, "figure_inches": [16, 9.3], "raster_dimensions": [16*dpi, round(9.3*dpi)],
        "color_boundaries": BOUNDARIES, "extent_lonlat": list(EXTENT), "panels": panels,
    }
    (out / "figure_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    caption = (
        f"Figure. Product-specific available algebraic random-error variance estimates for raw daily precipitation during {window} event windows. "
        "The upper row uses the CONUS404–MRMS–PERSIANN-CDR triplet; the lower row uses the CONUS404BA–MRMS–PERSIANN-CDR triplet. "
        "Each panel independently displays all finite nonnegative values stored in error_variance_unmasked on sample-eligible study cells, "
        "without a common or joint-triplet admissibility mask and without bootstrap screening. Negative estimates are gray, "
        "and unavailable/nonfinite estimates are white within the watershed outlines. Nonnegative estimates are shown regardless of "
        "other individual-product admissibility conditions; consequently, colored cells are not necessarily fully admissible. "
        "These diagnostic flags remain in the source data and figure metadata but are not separately marked. "
        "Panel counts distinguish nonnegative, negative, and unavailable estimates. Mean ≥0 is the area-weighted mean of the colored "
        "nonnegative estimates, including values failing other individual-product conditions; weights equal the fractional watershed overlap multiplied by cosine latitude. "
        "Because these means use different spatial supports, they are not matched-support comparisons or domain-wide product rankings. "
        "The nonlinear color intervals are identical across products, triplets, and windows and do not transform the raw estimates. "
        "MRMS and PERSIANN-CDR estimates remain separate for the two triplets. Negative estimates are diagnostic failures, "
        "not physically meaningful negative variances, and no negative values are clipped to zero or converted to absolute values.\n"
    )
    (out / "figure_caption.md").write_text(caption, encoding="utf-8")
    print(json.dumps({"window": window, "output": str(out), "panels": panels}, indent=2), flush=True)
    return metadata


def main(dpi=600):
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 12, "text.color": "#202A33", "axes.labelcolor": "#202A33",
        "xtick.color": "#39454E", "ytick.color": "#39454E", "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })
    datasets = [xr.open_dataset(p, engine="netcdf4").load() for p in SOURCES]
    for window in ["core", "expanded"]:
        draw(datasets, window, dpi)
    for ds in datasets:
        ds.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpi", type=int, default=600, choices=[300, 400, 600])
    main(dpi=parser.parse_args().dpi)
