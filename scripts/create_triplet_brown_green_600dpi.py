"""Restyle six existing raw triplet maps without changing their estimates.

The watershed reference map is a palette/cartography reference. Scientific
figures are regenerated from original NetCDF estimates, not recolored bitmaps.
"""

from __future__ import annotations

from study_paths import PROJECT_ROOT, RESULTS_ROOT, METADATA_ROOT, REPO_ROOT, STATES_FILE, WATERSHEDS_FILE, DAILY_INPUT, EC_DIRECTORY, EC_INPUT, LOSO_INPUT, AGREEMENT_DIRECTORY, AGREEMENT_INPUT


import hashlib
import json
from pathlib import Path
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np
from PIL import Image
import xarray as xr

from brown_green_cartography import PALETTE_ANCHORS, add_cartography
from build_word_triplet_metric_figures import EXTENT, LABELS, SOURCES, edges
from create_all_available_triplet_sd_r2_figures import METRICS, metric_fields
from create_large_triplet_error_variance_figure import BOUNDARIES
from map_overlays import add_map_boundaries


HERE = RESULTS_ROOT
OUT = HERE / "triplet_brown_green_600dpi"
DPI = 600
SIZE = (16, 9.3)
GRAY = "#B9BEC3"
TEXT = "#202A33"
MUTED = "#53616A"
REFERENCE_IMAGE = (METADATA_ROOT / 'palette_reference_not_redistributed.png')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fields(ds: xr.Dataset, window: str, product: str, metric: str):
    if metric != "error_variance":
        return metric_fields(ds, window, product, metric)
    domain = ds.study_domain_mask.values.astype(bool)
    eligible = domain & ds.sample_eligible.sel(window=window).values.astype(bool)
    values = ds.error_variance_unmasked.sel(window=window, product=product).values
    colored = eligible & np.isfinite(values) & (values >= 0)
    gray = eligible & np.isfinite(values) & (values < 0)
    missing = domain & ~(colored | gray)
    assert np.array_equal(colored | gray | missing, domain)
    return values, colored, gray, missing, eligible


def config_for(metric: str) -> dict:
    if metric == "error_variance":
        return {
            "title": "Pixelwise precipitation random-error variance",
            "label": "Random-error variance [(mm day$^{-1}$)$^2$]",
            "gray_label": "Negative error-variance estimate",
            "digits": 2,
        }
    return METRICS[metric].copy()


def original_metadata_path(metric: str, window: str) -> Path:
    if metric == "error_variance":
        return HERE / "triplet_error_variance_all_available" / window / "600dpi" / "figure_metadata.json"
    return HERE / "triplet_sd_r2_all_available" / metric / window / "600dpi" / "figure_metadata.json"


def verify_panel(original: dict, calculated: dict, metric: str) -> None:
    if metric == "error_variance":
        keys = (("nonnegative_cells", "colored_cells"), ("negative_cells", "gray_cells"),
                ("unavailable_cells", "unavailable_cells"))
        mean_key = "area_weighted_mean_nonnegative_error_variance"
    else:
        keys = (("colored_cells", "colored_cells"), ("gray_cells", "gray_cells"),
                ("unavailable_cells", "unavailable_cells"))
        mean_key = "area_weighted_mean_colored_values"
    for old_key, new_key in keys:
        assert original[old_key] == calculated[new_key], (old_key, original, calculated)
    np.testing.assert_allclose(original[mean_key], calculated["area_weighted_mean"], rtol=1e-12, atol=1e-12)


def draw(datasets: list[xr.Dataset], metric: str, window: str, hashes: dict) -> dict:
    out = OUT / metric / window
    out.mkdir(parents=True, exist_ok=True)
    config = config_for(metric)
    original_file = original_metadata_path(metric, window)
    original = json.loads(original_file.read_text(encoding="utf-8"))
    weights = (datasets[0].study_watershed_fraction.values.astype(float)
               * np.cos(np.deg2rad(datasets[0].lat.values))[:, None])
    ramp = LinearSegmentedColormap.from_list("watershed_brown_cream_green", PALETTE_ANCHORS, N=1024)
    # NaNs in the colored layer must stay transparent: an opaque white bad
    # color would conceal the gray invalid-estimate layer underneath.
    ramp.set_bad((1.0, 1.0, 1.0, 0.0))
    if metric == "error_variance":
        cmap = ramp.resampled(len(BOUNDARIES)-1)
        norm = BoundaryNorm(BOUNDARIES, cmap.N)
    else:
        cmap = ramp
        norm = Normalize(*config["limits"])
    assert cmap(np.nan)[3] == 0.0, "The colored layer must not hide invalid gray cells"
    fig = plt.figure(figsize=SIZE, facecolor="white")
    fig.text(.465, .977, config["title"], ha="center", va="top", fontsize=22, fontweight="bold")
    fig.text(.465, .935, f"{window.capitalize()} event windows  |  Raw precipitation  |  All available product-specific estimates",
             ha="center", fontsize=14, color=MUTED)
    row_titles = ("Triplet A: CONUS404 – MRMS – PERSIANN-CDR",
                  "Triplet B: CONUS404BA – MRMS – PERSIANN-CDR")
    x_positions, y_positions = (.055, .335, .615), (.575, .192)
    width = .255
    height = width * SIZE[0] * (EXTENT[3]-EXTENT[2]) / (EXTENT[1]-EXTENT[0]) / SIZE[1]
    panels = []
    for row, ds in enumerate(datasets):
        fig.text(.4625, (.897, .515)[row], row_titles[row], ha="center", fontsize=16, fontweight="bold")
        for col, product_value in enumerate(ds.product.values):
            product = str(product_value)
            values, colored, gray, missing, eligible = fields(ds, window, product, metric)
            assert int(eligible.sum()) == 683
            mean = float(np.sum(values[colored]*weights[colored])/weights[colored].sum())
            panel = {
                "panel": "abcdef"[row*3+col], "triplet": "A" if row == 0 else "B", "product": product,
                "colored_cells": int(colored.sum()), "gray_cells": int(gray.sum()),
                "unavailable_cells": int(missing.sum()), "area_weighted_mean": mean,
                "minimum_colored_value": float(values[colored].min()),
                "maximum_colored_value": float(values[colored].max()),
            }
            verify_panel(original["panels"][row*3+col], panel, metric)
            ax = fig.add_axes([x_positions[col], y_positions[row], width, height])
            ax.pcolormesh(edges(ds.lon), edges(ds.lat), np.where(gray, 1.0, np.nan),
                          cmap=ListedColormap([GRAY]), vmin=0, vmax=1, shading="flat",
                          rasterized=True, edgecolors="none", antialiased=False, zorder=1)
            mesh = ax.pcolormesh(edges(ds.lon), edges(ds.lat), np.where(colored, values, np.nan),
                                 cmap=cmap, norm=norm, shading="flat", rasterized=True,
                                 edgecolors="none", antialiased=False, zorder=2)
            ax.set_xlim(EXTENT[:2])
            ax.set_ylim(EXTENT[2:])
            add_map_boundaries(ax)
            for collection in ax.collections[2:]:
                if hasattr(collection, "get_linewidths"):
                    collection.set_linewidth(collection.get_linewidths()*.85)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(False)
            ax.xaxis.set_major_locator(FixedLocator([-100, -90, -80]))
            ax.yaxis.set_major_locator(FixedLocator([30, 35, 40]))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{abs(v):.0f}°W"))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}°N"))
            ax.tick_params(labelsize=12, length=3.5, width=.6, pad=3, labelleft=col == 0)
            for spine in ax.spines.values():
                spine.set_color("#8B959B")
                spine.set_linewidth(.65)
            ax.set_title(f"({panel['panel']}) {LABELS[product]}", loc="left", fontsize=16, fontweight="bold", pad=10)
            prefix = "Mean ≥0" if metric == "error_variance" else "Mean"
            ax.set_title(f"{prefix} = {mean:.{config['digits']}f}", loc="right", fontsize=12.5, pad=11)
            counts = (f"n≥0: {colored.sum()}   n<0: {gray.sum()}   NA: {missing.sum()}"
                      if metric == "error_variance"
                      else f"Color: {colored.sum()}   Gray: {gray.sum()}   NA: {missing.sum()}")
            ax.text(.025, .970, counts, transform=ax.transAxes, fontsize=9.5, ha="left", va="top",
                     zorder=25, bbox={"facecolor": "white", "edgecolor": "none", "alpha": .93, "pad": 2})
            panel["cartography"] = add_cartography(ax)
            panels.append(panel)
    cax = fig.add_axes([.893, .219, .018, .590])
    if metric == "error_variance":
        cb = fig.colorbar(mesh, cax=cax, boundaries=BOUNDARIES, ticks=BOUNDARIES, spacing="uniform")
        cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    else:
        ticks = np.arange(0, 71, 10) if metric == "error_sd" else np.linspace(0, 1, 6)
        cb = fig.colorbar(mesh, cax=cax, ticks=ticks)
    cb.ax.tick_params(labelsize=12, length=4, width=.6, pad=5)
    cb.outline.set_linewidth(.7)
    cb.set_label(config["label"], fontsize=15, labelpad=14)
    fig.text(.4625, .126, "Longitude", ha="center", fontsize=15)
    fig.text(.011, .500, "Latitude", ha="center", va="center", rotation=90, fontsize=15)
    fig.legend(handles=[Patch(facecolor=GRAY, edgecolor="#8D969C", label=config["gray_label"]),
                        Patch(facecolor="white", edgecolor="#8D969C", label="Unavailable estimate (within study cells)")],
               loc="lower center", bbox_to_anchor=(.4625, .066), frameon=False, ncol=2,
               fontsize=11.5, handlelength=1.3, handleheight=.8, columnspacing=1.4)
    fig.text(.4625, .046, "Same original values, masks and color limits; each panel mean uses its own colored cells. Gray cells retain their original meaning.",
             ha="center", fontsize=11.5, color=MUTED)
    fig.text(.4625, .020, "Brown-to-green palette  •  North-up geographic grid  •  Scale bars in statute miles, calibrated at their latitude",
             ha="center", fontsize=11.5, color=MUTED)
    stem = f"triplet_{metric}_{window}_brown_green_600dpi"
    paths = {}
    for suffix in ("png", "tif", "svg"):
        target = out / f"{stem}.{suffix}"
        options = {"pil_kwargs": {"compression": "tiff_lzw"}} if suffix == "tif" else {}
        fig.savefig(target, dpi=DPI, **options)
        paths[suffix] = str(target)
    preview = out / "preview.png"
    fig.savefig(preview, dpi=145)
    plt.close(fig)
    for suffix in ("png", "tif"):
        with Image.open(paths[suffix]) as im:
            assert im.size == (9600, 5580), im.size
            assert all(abs(float(dpi)-DPI) < .1 for dpi in im.info["dpi"])
    metadata = {
        "metric": metric, "window": window, "transform": "raw", "panels": panels,
        "source_netcdf_sha256": hashes, "original_figure_metadata": str(original_file),
        "numeric_panel_means_and_counts_match_original": True, "source_data_modified": False,
        "estimation_rerun": False, "palette_reference": str(REFERENCE_IMAGE),
        "palette_anchors": PALETTE_ANCHORS, "palette_direction": "low brown through cream to high green/teal",
        "normalization": "Original unequal variance bins" if metric == "error_variance" else "Original linear limits; not zero-centered",
        "color_boundaries": BOUNDARIES if metric == "error_variance" else None,
        "color_limits": [0, 6400] if metric == "error_variance" else config["limits"],
        "gray_color": GRAY, "grid_crs": "EPSG:4326", "extent_lonlat": list(EXTENT),
        "figure_inches": list(SIZE), "raster_dimensions": [9600, 5580], "dpi": DPI,
        "north_compasses": 6, "scale_bars": 6, "outputs": paths, "preview": str(preview),
        "tiff_note": "Publication figure TIFF, not a georeferenced data raster.",
    }
    (out / "figure_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    old_caption = original_file.with_name("figure_caption.md").read_text(encoding="utf-8").rstrip()
    caption = (old_caption + "\n\nUpdated cartography: colors progress from brown at low values through cream to green at high values; "
               "the original color normalization, estimates, masks and statistics are unchanged. Every subplot has a true-north compass "
               "and a WGS84 geodesically calibrated 0–125–250–500 statute-mile scale bar at the documented reference latitude. "
               "Scale varies with latitude on the geographic grid. The palette midpoint does not indicate zero error or a significance threshold.\n")
    (out / "figure_caption.md").write_text(caption, encoding="utf-8")
    print(json.dumps({"metric": metric, "window": window, "outputs": paths,
                      "means": [p["area_weighted_mean"] for p in panels]}, indent=2), flush=True)
    return metadata


def overview(records: list[dict]) -> Path:
    fig, axes = plt.subplots(3, 2, figsize=(16, 20), facecolor="white")
    fig.subplots_adjust(left=.015, right=.985, bottom=.015, top=.965, wspace=.02, hspace=.06)
    for ax, record in zip(axes.flat, records):
        with Image.open(record["preview"]) as preview:
            ax.imshow(np.asarray(preview))
        ax.axis("off")
        ax.set_title(f"{record['metric'].replace('_', ' ').title()} — {record['window'].capitalize()}", fontsize=17, pad=4)
    target = OUT / "all_six_brown_green_preview.png"
    fig.savefig(target, dpi=120)
    plt.close(fig)
    return target


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 12, "text.color": TEXT, "axes.labelcolor": TEXT,
        "xtick.color": "#39454E", "ytick.color": "#39454E", "svg.fonttype": "none",
        "mathtext.fontset": "stix", "savefig.facecolor": "white",
    })
    hashes = {str(path): sha(path) for path in SOURCES}
    datasets = [xr.open_dataset(path, engine="netcdf4").load() for path in SOURCES]
    records = []
    try:
        for metric in ("latent_r2", "error_sd", "error_variance"):
            for window in ("core", "expanded"):
                records.append(draw(datasets, metric, window, hashes))
    finally:
        for dataset in datasets:
            dataset.close()
    assert hashes == {str(path): sha(path) for path in SOURCES}
    preview = overview(records)
    manifest = {"figures": records, "overview": str(preview), "all_means_and_counts_verified_unchanged": True}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    archive = OUT / "all_six_brown_green_figures_600dpi.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for record in records:
            folder = Path(record["preview"]).parent
            for path in sorted(folder.iterdir()):
                if path.is_file():
                    bundle.write(path, path.relative_to(OUT))
        bundle.write(OUT / "manifest.json", "manifest.json")
        bundle.write(preview, preview.name)
    print(json.dumps({"archive": str(archive), "overview": str(preview), "figures": len(records)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
