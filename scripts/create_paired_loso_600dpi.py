"""Publish the verified, fixed-support paired EC leave-one-storm-out estimates.

This is a display-only script: it does not recompute, alter, or bootstrap estimates.
"""

from __future__ import annotations

from study_paths import PROJECT_ROOT, RESULTS_ROOT, METADATA_ROOT, REPO_ROOT, STATES_FILE, WATERSHEDS_FILE, DAILY_INPUT, EC_DIRECTORY, EC_INPUT, LOSO_INPUT, AGREEMENT_DIRECTORY, AGREEMENT_INPUT


import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import numpy as np
from PIL import Image


ROOT = RESULTS_ROOT
SOURCE = ROOT / "hurricane_results_word" / "verified_results_metrics.json"
DEST = ROOT / "paired_uncertainty_figures" / "loso" / "600dpi"
NAME = "paired_ec_leave_one_storm_out_600dpi"
FIGSIZE = (14.8, 7.8)
DPI = 600
COLORS = {"core": "#238B78", "expanded": "#A66A27"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    before_hash = sha256(SOURCE)
    metrics = json.loads(SOURCE.read_text(encoding="utf-8"))
    font_path = font_manager.findfont("Times New Roman", fallback_to_default=True)
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 13,
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 13,
            "text.color": "#262626",
            "axes.labelcolor": "#262626",
            "axes.edgecolor": "#545454",
            "xtick.color": "#454545",
            "ytick.color": "#454545",
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "mathtext.fontset": "stix",
        }
    )
    ordered_storms = list(metrics["storm_id_name_map"])
    supports = {"core": 527, "expanded": 660}
    panels = {}
    for window in ("core", "expanded"):
        info = metrics["windows"][window]
        loso = info["loso"]
        assert loso["fixed_support_same_as_primary"] is True
        assert loso["fixed_support"]["cells"] == supports[window]
        rows = loso["omissions"]
        assert len(rows) == 10
        assert [row["storm_id"] for row in rows] == ordered_storms
        assert len({row["storm_id"] for row in rows}) == 10
        assert all(row["all_primary_screen_cells_still_valid"] for row in rows)
        values = np.array([row["mean_delta_sd_ba_minus_conus404"] for row in rows])
        assert np.isfinite(values).all() and (values > 0).all()
        assert np.isclose(
            loso["full_sample_delta_sd_on_fixed_support"],
            info["mean_delta_sd_ba_minus_conus404"],
            atol=1e-12,
        )
        assert np.isclose(
            values.min(),
            {"core": 0.585369017937665, "expanded": 0.4095486261025448}[window],
            atol=1e-12,
        )
        assert np.isclose(
            values.max(),
            {"core": 1.0835174603145197, "expanded": 0.7251754253734328}[window],
            atol=1e-12,
        )
        panels[window] = {
            "fixed_support": loso["fixed_support"],
            "full_sample_mean_delta_sd": loso["full_sample_delta_sd_on_fixed_support"],
            "omissions": rows,
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "minimum_omitted_storm": rows[int(values.argmin())]["storm"],
            "maximum_omitted_storm": rows[int(values.argmax())]["storm"],
            "all_omissions_positive": bool((values > 0).all()),
        }

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, sharey=True)
    fig.subplots_adjust(left=0.148, right=0.978, bottom=0.165, top=0.785, wspace=0.10)
    fig.suptitle("Sensitivity to individual tropical cyclones", y=0.965, fontsize=21)
    fig.text(
        0.5,
        0.911,
        "Leave-one-storm-out estimates; not bootstrap confidence intervals",
        ha="center",
        fontsize=14,
        color="#4C4C4C",
    )

    labels = [
        f"{metrics['storm_id_name_map'][storm]} ({storm[-4:]})"
        for storm in ordered_storms
    ]
    ypos = np.arange(10)
    for idx, (window, ax) in enumerate(zip(("core", "expanded"), axes)):
        panel = panels[window]
        color = COLORS[window]
        values = np.array(
            [r["mean_delta_sd_ba_minus_conus404"] for r in panel["omissions"]]
        )
        full = panel["full_sample_mean_delta_sd"]
        ax.set_xlim(-0.035, 1.23)
        ax.set_ylim(9.6, -0.65)
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.set_yticks(ypos, labels)
        ax.grid(False)
        ax.tick_params(axis="y", length=0, pad=9)
        ax.tick_params(axis="x", length=4, width=0.8)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linestyle("-")
        ax.axvline(0, color="#777777", linewidth=0.8, zorder=1)
        ax.axvline(full, color=color, linestyle=(0, (5, 3)), linewidth=1.5, zorder=1)
        ax.scatter(values, ypos, s=67, color=color, edgecolor="#FFFFFF", linewidth=0.9, zorder=4)
        for val, yy in zip(values, ypos):
            ax.text(
                val + 0.032,
                yy,
                f"+{val:.3f}",
                ha="left",
                va="center",
                fontsize=12,
                zorder=5,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8},
            )
        ax.set_title(f"({chr(97+idx)}) {window.capitalize()}", loc="left", pad=36, fontsize=17)
        ax.text(
            1,
            1.087,
            f"{panel['fixed_support']['cells']} cells | "
            f"{panel['fixed_support']['weighted_watershed_area_pct']:.2f}% area",
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=12,
            color="#4C4C4C",
        )
        ax.plot([0.025, 0.094], [1.03, 1.03], transform=ax.transAxes,
                color=color, linestyle=(0, (5, 3)), linewidth=1.5, clip_on=False)
        ax.text(
            0.107,
            1.03,
            f"All ten storms: +{full:.3f} mm day⁻¹",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=12,
        )
    axes[0].set_ylabel("Omitted storm", labelpad=14)
    axes[1].tick_params(axis="y", labelleft=False)
    fig.text(
        0.565,
        0.092,
        "Area-weighted mean random-error SD difference: CONUS404BA − CONUS404 (mm day⁻¹)",
        ha="center",
        va="center",
        fontsize=14,
    )
    fig.text(
        0.565,
        0.039,
        "Positive values indicate higher estimated random-error SD in CONUS404BA.",
        ha="center",
        fontsize=12,
        color="#4C4C4C",
    )

    png = DEST / f"{NAME}.png"
    tiff = DEST / f"{NAME}.tif"
    svg = DEST / f"{NAME}.svg"
    preview = DEST / "preview.png"
    fig.savefig(png, dpi=DPI)
    fig.savefig(svg, metadata={"Title": "Paired EC leave-one-storm-out sensitivity"})
    fig.savefig(preview, dpi=150)
    plt.close(fig)
    with Image.open(png) as image:
        image.convert("RGB").save(tiff, compression="tiff_lzw", dpi=(DPI, DPI))
        assert image.size == (8880, 4680), image.size
        assert all(abs(v - DPI) < 0.1 for v in image.info["dpi"])
        png_validation = {"dimensions_px": list(image.size), "dpi": list(image.info["dpi"])}
    with Image.open(tiff) as image:
        assert image.size == (8880, 4680)
        assert image.info["compression"] == "tiff_lzw"
        tiff_validation = {
            "dimensions_px": list(image.size),
            "dpi": [float(v) for v in image.info["dpi"]],
            "compression": image.info["compression"],
        }
    after_hash = sha256(SOURCE)
    assert before_hash == after_hash
    metadata = {
        "title": "Sensitivity to individual tropical cyclones",
        "method": "Leave-one-storm-out sensitivity, not bootstrap confidence intervals",
        "metric": "Area-weighted arithmetic mean of pixelwise paired random-error SD differences",
        "difference_definition": "CONUS404BA minus CONUS404",
        "units": "mm day^-1",
        "source": str(SOURCE),
        "source_sha256_before": before_hash,
        "source_sha256_after": after_hash,
        "source_unchanged": before_hash == after_hash,
        "original_loso_netcdf": metrics["sources"]["loso"],
        "original_primary_netcdf": metrics["sources"]["ec"],
        "area_weighting": metrics["area_weighting"],
        "fixed_support_note": "Each omission retains the corresponding full-sample paired primary support. Core and expanded supports differ; all omitted-storm solutions remained valid on those supports.",
        "primary_support_definition": metrics["primary_support_definition"],
        "sample_mode": metrics["sample_mode"],
        "no_confidence_intervals_drawn": True,
        "panels": panels,
        "style": {"font": "Times New Roman", "font_path": font_path, "window_colors": COLORS,
                  "no_inner_grid": True, "continuous_panel_spines": True,
                  "points": "circles", "all_storm_reference": "dashed vertical line",
                  "zero_reference": "thin solid vertical line"},
        "exports": {"png": str(png), "tiff": str(tiff), "svg": str(svg), "preview": str(preview)},
        "png_validation": png_validation,
        "tiff_validation": tiff_validation,
    }
    (DEST / "figure_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    caption = (
        "**Figure X.** Leave-one-storm-out sensitivity of the area-weighted mean paired "
        "random-error standard deviation (SD) difference between CONUS404BA and CONUS404 "
        "during (a) core and (b) expanded tropical-cyclone event windows. Each point is the "
        "estimate after excluding the labeled storm; numbers report BA-minus-CONUS404 SD "
        "differences in mm day⁻¹. Dashed vertical lines show the corresponding all-ten-storm "
        "estimates (+0.889 and +0.623 mm day⁻¹). Estimates use fixed spatial supports of "
        "527 core-window cells (70.81% of the weighted watershed area) and 660 expanded-window "
        "cells (89.25%); the same support is retained for all ten omissions within each window. "
        "Positive differences indicate higher estimated random-error SD in CONUS404BA. "
        "The points show sensitivity to individual storm omissions, not bootstrap confidence "
        "intervals. Spatial weights are watershed overlap fraction multiplied by the cosine "
        "of latitude.\n"
    )
    (DEST / "figure_caption.md").write_text(caption, encoding="utf-8")
    print(json.dumps({"outputs": metadata["exports"], "png_validation": png_validation,
                      "source_unchanged": metadata["source_unchanged"],
                      "panels": {key: {"minimum": value["minimum"], "maximum": value["maximum"],
                                          "fixed_support": value["fixed_support"]}
                                 for key, value in panels.items()}}, indent=2))


if __name__ == "__main__":
    main()
