"""Draw the core or expanded six-panel error-variance figure at publication resolution.

This is a presentation-only change. No estimation is rerun and neither source
NetCDF is modified. The original mask, means, class boundaries, and geographic
extent are verified and retained.
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
from map_overlays import add_map_boundaries


ROOT = RESULTS_ROOT
OUT = ROOT / "triplet_error_variance_large"
ORIGINAL_METADATA = ROOT / "hurricane_results_word" / "figures" / "triplet_metric_figures_metadata.json"
EXCLUDED_COLOR = "#D9DDE0"
BOUNDARIES = [0, 5, 10, 25, 50, 100, 200, 400, 800, 1600, 3200, 6400]


def main(dpi=400, window="core"):
    out_root = OUT if window == "core" else ROOT / "triplet_error_variance_expanded_large"
    out = out_root if dpi == 400 else out_root / f"{dpi}dpi"
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 12, "text.color": "#202A33", "axes.labelcolor": "#202A33",
        "xtick.color": "#39454E", "ytick.color": "#39454E", "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })
    with ORIGINAL_METADATA.open(encoding="utf-8") as handle:
        original = json.load(handle)
    original_means = [r for r in original["metric_rows"] if r["window"] == window]
    assert len(original_means) == 6
    assert original["figures"][0]["boundaries"] == BOUNDARIES
    datasets = [xr.open_dataset(p, engine="netcdf4").load() for p in SOURCES]
    mask = datasets[0].valid_both_triplets.sel(window=window).values.astype(bool)
    np.testing.assert_array_equal(mask, datasets[1].valid_both_triplets.sel(window=window).values.astype(bool))
    cells = int(mask.sum())
    assert cells == {"core": 314, "expanded": 382}[window]
    weights = datasets[0].study_watershed_fraction.values.astype(float) * np.cos(np.deg2rad(datasets[0].lat.values))[:, None]
    coverage = float(weights[mask].sum() / weights.sum() * 100)
    assert abs(coverage - original_means[0]["weighted_area_coverage_percent"]) < 1e-7
    domain = datasets[0].study_domain_mask.values.astype(bool)
    assert int(domain.sum()) == 683
    excluded = domain & ~mask
    assert not np.any(mask & ~domain)

    fig = plt.figure(figsize=(16, 9.3), facecolor="white")
    fig.text(.465, .977, "Pixelwise precipitation random-error variance", ha="center", va="top",
             fontsize=22, fontweight="bold")
    fig.text(.465, .935, f"{window.capitalize()} event windows  |  Raw precipitation  |  Separate matched triplets",
             ha="center", fontsize=14, color="#53616A")
    row_titles = ["Triplet A: CONUS404 – MRMS – PERSIANN-CDR",
                  "Triplet B: CONUS404BA – MRMS – PERSIANN-CDR"]
    row_header_y = [.897, .515]
    y_positions = [.575, .192]
    x_positions = [.055, .335, .615]
    panel_width = .255
    panel_height = panel_width * 16 * ((EXTENT[3] - EXTENT[2]) / (EXTENT[1] - EXTENT[0])) / 9.3
    cmap = plt.get_cmap("YlGnBu", len(BOUNDARIES) - 1)
    norm = BoundaryNorm(BOUNDARIES, cmap.N)
    panel_records = []
    for row, ds in enumerate(datasets):
        fig.text(.4625, row_header_y[row], row_titles[row], ha="center", fontsize=16,
                 fontweight="bold")
        for col, product_value in enumerate(ds.product.values):
            product = str(product_value)
            ax = fig.add_axes([x_positions[col], y_positions[row], panel_width, panel_height])
            values = ds.error_variance.sel(window=window, product=product).values
            retained = np.where(mask, values, np.nan)
            assert np.isfinite(retained[mask]).all() and (retained[mask] >= 0).all()
            assert np.max(retained[mask]) <= BOUNDARIES[-1], "Shared scale would clip valid estimates"
            mean = float(np.sum(retained[mask] * weights[mask]) / weights[mask].sum())
            original_panel = next(p for p in original_means
                                  if p["triplet"] == ("A" if row == 0 else "B") and p["product"] == product)
            np.testing.assert_allclose(mean, original_panel["error_variance"], atol=1e-9)
            ax.pcolormesh(edges(ds.lon), edges(ds.lat), np.where(excluded, 1.0, np.nan),
                          cmap=ListedColormap([EXCLUDED_COLOR]), vmin=0, vmax=1,
                          shading="flat", rasterized=True, zorder=1)
            mesh = ax.pcolormesh(edges(ds.lon), edges(ds.lat), retained, cmap=cmap,
                                 norm=norm, shading="flat", rasterized=True,
                                 edgecolors="none", antialiased=False, zorder=2)
            ax.set_xlim(EXTENT[:2])
            ax.set_ylim(EXTENT[2:])
            add_map_boundaries(ax)
            # Avoid obscuring 0.25-degree pixel colors with thick polygon lines.
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
            ax.set_title(f"Mean = {mean:.2f}", loc="right", fontsize=13, pad=11)
            panel_records.append({
                "panel": "abcdef"[row*3+col], "triplet": "A" if row == 0 else "B",
                "product": product, "area_weighted_mean_error_variance": mean,
                "cells": int(mask.sum()), "minimum": float(np.min(retained[mask])),
                "maximum": float(np.max(retained[mask])),
            })

    cax = fig.add_axes([.893, .219, .018, .590])
    cb = fig.colorbar(mesh, cax=cax, boundaries=BOUNDARIES, ticks=BOUNDARIES,
                      spacing="uniform")
    cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    cb.ax.tick_params(labelsize=12, length=4, width=.6, pad=5)
    cb.outline.set_linewidth(.7)
    cb.set_label("Random-error variance [(mm day$^{-1}$)$^2$]", fontsize=15, labelpad=14)
    fig.text(.4625, .126, "Longitude", ha="center", fontsize=15)
    fig.text(.011, .500, "Latitude", ha="center", va="center", rotation=90, fontsize=15)
    fig.legend(handles=[Patch(facecolor=EXCLUDED_COLOR, edgecolor="#8D969C",
                              label="Excluded study cells (not zero variance)")],
               loc="lower center", bbox_to_anchor=(.4625, .065), frameon=False,
               fontsize=12, handlelength=1.3, handleheight=.8)
    fig.text(.4625, .046,
             f"Identical support in all panels: {cells} admissible cells ({coverage:.2f}% of weighted watershed area). Means are area-weighted.",
             ha="center", fontsize=12, color="#53616A")
    fig.text(.4625, .020,
             "Common 0.25° grid • Nonlinear color intervals • No bootstrap screening • MRMS and PERSIANN-CDR estimates kept separate by triplet",
             ha="center", fontsize=11.5, color="#53616A")
    stem_name = f"triplet_error_variance_{window}_raw_large" + (f"_{dpi}dpi" if dpi != 400 else "")
    stem = out / stem_name
    fig.savefig(stem.with_suffix(".png"), dpi=dpi)
    fig.savefig(stem.with_suffix(".svg"), dpi=dpi)
    fig.savefig(stem.with_suffix(".tif"), dpi=dpi, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(out / "preview.png", dpi=145)
    plt.close(fig)
    metadata = {
        "source_netcdfs": [str(p) for p in SOURCES],
        "original_figure_metadata": str(ORIGINAL_METADATA),
        "analysis_rerun": False, "original_data_modified": False,
        "metric": "error_variance", "units": "(mm day-1)^2", "window": window,
        "transform": "raw", "support": "valid_both_triplets", "common_cells": cells,
        "weighted_watershed_area_coverage_percent": coverage,
        "bootstrap_filter": False, "reference_averaging_between_triplets": False,
        "figure_size_inches": [16, 9.3], "raster_export_dpi": dpi,
        "raster_dimensions_pixels": [16 * dpi, round(9.3 * dpi)],
        "color_boundaries": BOUNDARIES, "color_spacing": "uniform display of nonlinear intervals",
        "extent_lonlat": list(EXTENT), "gray_definition": "study_domain_mask AND NOT valid_both_triplets",
        "panels": panel_records,
    }
    (out / "figure_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    caption = (
        f"Figure. Pixelwise random-error variance for raw daily precipitation during {window} tropical-cyclone event windows. "
        "The upper row shows the CONUS404–MRMS–PERSIANN-CDR triplet; the lower row shows the "
        "CONUS404BA–MRMS–PERSIANN-CDR triplet. "
        f"All six panels use the same {cells} cells with admissible estimates for all three products in both triplets, "
        f"covering {coverage:.2f}% of the weighted watershed area. Panel annotations report area-weighted mean pixelwise "
        "error variance in (mm day−1)². MRMS and PERSIANN-CDR estimates are kept separate by triplet and are not averaged. "
        "The shared color intervals, geographic extent, figure size, labels, and boundary styling match the companion "
        "core/expanded figure. Nonlinear color intervals do not represent a transformation of the raw estimates. "
        "Gray cells are excluded study cells, not zero error variance. Black outlines identify the 23 study watersheds; "
        "lighter lines show state boundaries, with the CONUS boundary for context. No bootstrap screening is applied. "
        "Core and expanded figures use their respective common admissibility masks, which are not identical across windows.\n"
    )
    (out / "figure_caption.md").write_text(caption, encoding="utf-8")
    for ds in datasets:
        ds.close()
    print(json.dumps({"output_directory": str(out), "cells": int(mask.sum()),
                      "panel_means": [p["area_weighted_mean_error_variance"] for p in panel_records]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpi", type=int, default=400, choices=[300, 400, 600])
    parser.add_argument("--window", choices=["core", "expanded"], default="core")
    args = parser.parse_args()
    main(dpi=args.dpi, window=args.window)
