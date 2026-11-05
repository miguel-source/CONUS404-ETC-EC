"""Separate, unaveraged three-product collocation estimates and six-panel maps.

Run with the project Python environment using %run followed by this script's path.
Defaults read the existing aligned daily archive and write a NEW workspace
result folder. Existing EC/GeoTIFF packages are never modified.
"""

from __future__ import annotations

from study_paths import PROJECT_ROOT, RESULTS_ROOT, METADATA_ROOT, REPO_ROOT, STATES_FILE, WATERSHEDS_FILE, DAILY_INPUT, EC_DIRECTORY, EC_INPUT, LOSO_INPUT, AGREEMENT_DIRECTORY, AGREEMENT_INPUT


import argparse
import json
import os
from pathlib import Path

HERE = RESULTS_ROOT
os.environ.setdefault("MPLCONFIGDIR", str(HERE / "_triplet_mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
import numpy as np
import rasterio
from rasterio.transform import from_origin
import xarray as xr

from map_overlays import add_map_boundaries
from run_gridded_extended_collocation import (
    DEFAULT_INPUT,
    PRODUCTS,
    _observation_indices,
    _point_statistics,
    _require_input_schema,
    _storm_sufficient_statistics,
    _valid_positive_cross,
)

ROOT = DEFAULT_INPUT.parent.parent
DEFAULT_EC = EC_INPUT
DEFAULT_OUTPUT = RESULTS_ROOT / 'separate_triplet_results'
TRIPLETS = {
    "CONUS404_MRMS_PERSIANN_CDR": (0, 2, 3),
    "CONUS404BA_MRMS_PERSIANN_CDR": (1, 2, 3),
}
LABELS = {
    "conus404": "CONUS404",
    "conus404ba": "CONUS404BA",
    "mrms": "MRMS",
    "persiann_cdr": "PERSIANN-CDR",
}
WINDOWS = ("core", "expanded")
TRANSFORMS = ("raw", "log1p")
EXTENT = (-105.5, -71.5, 26.0, 45.0)


def triplet_metrics(covariance):
    """TC variance equations; no averaging or four-system validity mask.

    A finite zero error variance is retained as a boundary solution. Negative
    variances are never abs-transformed/clipped. SNR is undefined at zero error.
    """
    c = np.asarray(covariance, dtype=np.float64)
    if c.shape[-2:] != (3, 3):
        raise ValueError("Expected trailing covariance dimensions (3, 3).")
    variance = np.diagonal(c, axis1=-2, axis2=-1)
    error = np.full(variance.shape, np.nan)
    signal = np.full(variance.shape, np.nan)
    denominators = np.zeros(variance.shape, dtype=bool)
    for i in range(3):
        j, k = [n for n in range(3) if n != i]
        supported = _valid_positive_cross(c[..., j, k], variance[..., j], variance[..., k])
        denominators[..., i] = supported
        with np.errstate(divide="ignore", invalid="ignore"):
            estimate = c[..., i, j] * c[..., i, k] / c[..., j, k]
        signal[..., i] = np.where(supported, estimate, np.nan)
        error[..., i] = variance[..., i] - signal[..., i]
    finite_covariance = np.isfinite(c).all(axis=(-2, -1))
    valid_product = (
        finite_covariance[..., None]
        & denominators
        & np.isfinite(variance) & (variance > 0)
        & np.isfinite(error) & (error >= 0)
        & np.isfinite(signal) & (signal >= 0)
    )
    valid_triplet = valid_product.all(axis=-1)
    sd = np.full(error.shape, np.nan)
    np.sqrt(error, out=sd, where=np.isfinite(error) & (error >= 0))
    snr = np.full(error.shape, np.nan)
    np.divide(signal, error, out=snr, where=valid_product & (error > 0))
    r2 = np.full(error.shape, np.nan)
    np.divide(signal, variance, out=r2, where=valid_product)
    return {
        "error_variance_unmasked": error,
        "signal_variance_unmasked": signal,
        "observed_variance": variance,
        "error_sd_unmasked": sd,
        "snr_unmasked": snr,
        "estimated_r2_to_truth_unmasked": r2,
        "valid_product_estimate": valid_product,
        "valid_triplet": valid_triplet,
        "denominator_valid": denominators,
        "zero_error_boundary": valid_product & (error == 0),
    }


def self_test():
    beta = np.array([1.0, 1.1, 1.2, 0.8])
    expected = np.array([4.0, 25.0, 9.0, 16.0])
    c = 100.0 * np.outer(beta, beta) + np.diag(expected)
    c[0, 1] += 5.0
    c[1, 0] += 5.0
    for indices in TRIPLETS.values():
        sub = c[np.ix_(indices, indices)]
        metrics = triplet_metrics(sub)
        np.testing.assert_allclose(metrics["error_variance_unmasked"], expected[list(indices)], atol=1e-12)
        assert metrics["valid_triplet"]
    impossible = np.array([[1, 0.8, 0.8], [0.8, 1, 0.3], [0.8, 0.3, 1]])
    assert np.linalg.eigvalsh(impossible).min() > 0
    bad = triplet_metrics(impossible)
    assert not bad["valid_triplet"] and bad["error_variance_unmasked"][0] < 0
    assert not triplet_metrics(np.eye(3))["valid_triplet"]
    zero = triplet_metrics(4 * np.ones((3, 3)) + np.diag([0.0, 1.0, 2.0]))
    assert zero["valid_triplet"] and zero["zero_error_boundary"][0]
    assert np.isnan(zero["snr_unmasked"][0])
    return {"exact_known_error_recovery": True, "negative_variance_rejected": True,
            "zero_denominators_rejected": True, "zero_error_boundary_retained": True}


def edges(values):
    a = np.asarray(values, dtype=float)
    if len(a) < 2 or not np.allclose(np.diff(a), np.diff(a)[0]):
        raise ValueError("Expected regular latitude/longitude coordinates.")
    return np.r_[a[0] - (a[1] - a[0]) / 2, (a[1:] + a[:-1]) / 2,
                 a[-1] + (a[-1] - a[-2]) / 2]


def weighted_mean(values, mask, weights):
    keep = mask & np.isfinite(values)
    return float(np.sum(values[keep] * weights[keep]) / weights[keep].sum()) if keep.any() else None


def support_summary(values, mask, weights, products):
    return {
        "cells": int(mask.sum()),
        "weighted_watershed_area_pct": float(100 * weights[mask].sum() / weights.sum()),
        "area_weighted_mean_pixel_error_variance": {
            p: weighted_mean(values[..., i], mask, weights) for i, p in enumerate(products)
        },
        "area_weighted_mean_pixel_error_sd": {
            p: weighted_mean(np.sqrt(np.where(values[..., i] >= 0, values[..., i], np.nan)), mask, weights)
            for i, p in enumerate(products)
        },
    }


def write_tiff(path, values, lat, lon, tags):
    path.parent.mkdir(parents=True, exist_ok=True)
    iy, ix = np.argsort(lat)[::-1], np.argsort(lon)
    xe, ye = edges(lon[ix]), edges(lat[iy])
    raster = np.asarray(values[np.ix_(iy, ix)], dtype=np.float32)
    nodata = -9999.0
    raster = np.where(np.isfinite(raster), raster, nodata)
    transform = from_origin(xe[0], ye[0], abs(xe[1] - xe[0]), abs(ye[1] - ye[0]))
    with rasterio.open(path, "w", driver="GTiff", height=len(lat), width=len(lon),
                       count=1, dtype="float32", crs="EPSG:4326", transform=transform,
                       nodata=nodata, compress="deflate") as dst:
        dst.write(raster, 1)
        dst.update_tags(**{k: str(v) for k, v in tags.items()})
        dst.set_band_description(1, "Triplet-specific random-error variance")
    with rasterio.open(path) as src:
        assert src.crs.to_epsg() == 4326
        np.testing.assert_array_equal(src.read(1), raster)
        assert src.transform == transform


def make_figures(data, daily, weights, output_dir):
    figures = {}
    domain = daily.study_domain_mask.values.astype(bool)
    xe, ye = edges(daily.lon.values), edges(daily.lat.values)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})
    for transform in TRANSFORMS:
        finite = []
        for window in WINDOWS:
            for entry in data[(window, transform)].values():
                values = entry["metrics"]["error_variance_unmasked"]
                mask = entry["valid"]
                if mask.any():
                    finite.append(float(np.max(values[mask])))
        upper = max(finite, default=1.0)
        bounds = [0, 5, 10, 25, 50, 100, 200, 400, 800, 1600, 3200, 6400]
        while bounds[-1] < upper:
            bounds.append(2 * bounds[-1])
        bounds = bounds[:next(i for i, x in enumerate(bounds) if x >= upper and i > 0) + 1]
        if transform == "raw":
            cmap = plt.get_cmap("viridis", len(bounds) - 1)
            norm = BoundaryNorm(bounds, cmap.N)
            units = "(mm day⁻¹)²"
        else:
            cmap, norm, units = plt.get_cmap("viridis"), Normalize(0, upper), "dimensionless²"
        for window in WINDOWS:
            entries = data[(window, transform)]
            shared = np.logical_and.reduce([d["valid"] for d in entries.values()])
            for scope in ("triplet_admissible", "common_admissible"):
                fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
                fig.subplots_adjust(left=0.075, right=0.89, top=0.84, bottom=0.12,
                                    wspace=0.10, hspace=0.35)
                for row, (triplet, entry) in enumerate(entries.items()):
                    mask = entry["valid"] if scope == "triplet_admissible" else shared
                    summary = support_summary(entry["metrics"]["error_variance_unmasked"], mask, weights, entry["products"])
                    for col, product in enumerate(entry["products"]):
                        ax = axes[row, col]
                        values = entry["metrics"]["error_variance_unmasked"][..., col]
                        ax.set_xlim(EXTENT[:2]); ax.set_ylim(EXTENT[2:])
                        ax.pcolormesh(xe, ye, np.where(domain, 1.0, np.nan),
                                      cmap=ListedColormap(["#e1e7ec"]), vmin=0, vmax=1,
                                      shading="flat", zorder=1, rasterized=True)
                        artist = ax.pcolormesh(xe, ye, np.where(mask, values, np.nan),
                                               cmap=cmap, norm=norm, shading="flat", zorder=3,
                                               rasterized=True)
                        add_map_boundaries(ax)
                        mean = summary["area_weighted_mean_pixel_error_variance"][product]
                        mean_text = f"{mean:.2f}" if mean is not None else "unavailable"
                        ax.set_title(f"({chr(97 + row * 3 + col)}) {LABELS[product]}\nArea-weighted mean variance: {mean_text}", fontsize=12)
                        ax.set_xlabel("Longitude (°)")
                        if col == 0:
                            ax.set_ylabel("Latitude (°)")
                        ax.tick_params(labelbottom=True, labelleft=(col == 0), labelsize=9)
                    row_y = 0.882 if row == 0 else 0.475
                    members = " + ".join(LABELS[p] for p in entry["products"])
                    fig.text(0.075, row_y, f"Triplet {'A' if row == 0 else 'B'}: {members}    |    "
                             f"{summary['cells']} cells; {summary['weighted_watershed_area_pct']:.2f}% of weighted watershed area",
                             fontsize=12, weight="bold")
                color_ax = fig.add_axes([0.915, 0.185, 0.016, 0.61])
                kwargs = dict(boundaries=bounds, ticks=bounds, spacing="uniform") if transform == "raw" else {}
                bar = fig.colorbar(artist, cax=color_ax, **kwargs)
                bar.set_label(f"Triplet-specific random-error variance ({units})", fontsize=11)
                subtitle = "Each row uses its own jointly admissible pixels" if scope == "triplet_admissible" else "The same admissible pixels in all six panels"
                fig.suptitle(f"{window.capitalize()} windows | {transform} precipitation\nSeparate triplet error variances — {subtitle}", fontsize=18, y=0.97)
                fig.text(0.5, 0.060, "No averaging between triplets. No bootstrap-validity filter. Gray = study cells excluded by the plotted admissibility mask.", ha="center", fontsize=10)
                scale_note = "Identical absolute-value color bins across all raw panels and windows; unequal bin widths are labeled." if transform == "raw" else "Identical linear color scale across all log1p panels and windows."
                fig.text(0.5, 0.032, scale_note, ha="center", fontsize=10, color="#404040")
                stem = output_dir / "figures" / f"{window}_{transform}_error_variance_{scope}"
                stem.parent.mkdir(parents=True, exist_ok=True)
                figures[f"{window}_{transform}_{scope}"] = {}
                for extension in ("png", "pdf"):
                    path = stem.with_suffix("." + extension)
                    fig.savefig(path, dpi=240)
                    figures[f"{window}_{transform}_{scope}"][extension] = str(path)
                plt.close(fig)
    return figures


def run_analysis(input_path=DEFAULT_INPUT, output_dir=DEFAULT_OUTPUT, ec_path=DEFAULT_EC):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checks = self_test()
    with xr.open_dataset(input_path, engine="netcdf4") as source:
        _require_input_schema(source, "all_windows")
        daily = source.load()
    original = None
    if ec_path is not None and Path(ec_path).exists():
        with xr.open_dataset(ec_path, engine="netcdf4") as source:
            original = source.load()
        xr.testing.assert_equal(daily.lat, original.lat)
        xr.testing.assert_equal(daily.lon, original.lon)
    attrs = original.attrs if original is not None else {}
    min_storms = int(attrs.get("minimum_storms", 8))
    min_days = int(attrs.get("minimum_valid_days_per_storm", 3))
    minimum = {"core": int(attrs.get("minimum_core_observations", 60)),
               "expanded": int(attrs.get("minimum_expanded_observations", 100))}
    domain = daily.study_domain_mask.values.astype(bool)
    weights = daily.study_watershed_fraction.values.astype(float) * np.cos(np.deg2rad(daily.lat.values))[:, None]
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("Missing or invalid watershed-area weights.")
    ny, nx = domain.shape
    data, supports, sampling, numerical_checks = {}, {}, {}, {}
    for window in WINDOWS:
        selected, duplicates = _observation_indices(daily, "all_windows", window)
        sample = daily.isel(observation=selected)
        raw = np.stack([sample[p].transpose("observation", "lat", "lon").values for p in PRODUCTS], axis=1).astype(np.float64)
        base = np.isfinite(raw).all(axis=1) & (raw >= 0).all(axis=1) & domain[None]
        if "valid_all4" in sample:
            base &= sample.valid_all4.transpose("observation", "lat", "lon").values.astype(bool)
        storm_index = sample.storm_index.values.astype(int)
        sampling[window] = {"unique_days": len(selected), "duplicate_records_discarded": duplicates}
        for transform in TRANSFORMS:
            values = np.log1p(raw) if transform == "log1p" else raw
            n, sums, cross, _, _ = _storm_sufficient_statistics(
                values.reshape(len(selected), 4, -1), base.reshape(len(selected), -1),
                storm_index, daily.sizes["storm"], min_days)
            counts, storms, covariance, _ = _point_statistics(n, sums, cross)
            eligible = domain.ravel() & (counts >= minimum[window]) & (storms >= min_storms)
            covariance = covariance.reshape(ny, nx, 4, 4)
            entries = {}
            for triplet, indices in TRIPLETS.items():
                metrics = triplet_metrics(covariance[..., indices, :][..., :, indices])
                valid = eligible.reshape(ny, nx) & metrics["valid_triplet"]
                entries[triplet] = {"metrics": metrics, "valid": valid,
                                    "products": [PRODUCTS[i] for i in indices],
                                    "eligible": eligible.reshape(ny, nx),
                                    "counts": counts.reshape(ny, nx), "storms": storms.reshape(ny, nx)}
            data[(window, transform)] = entries
            common = np.logical_and.reduce([e["valid"] for e in entries.values()])
            supports[f"{window}_{transform}"] = {}
            for triplet, entry in entries.items():
                supports[f"{window}_{transform}"][triplet] = {
                    scope: support_summary(entry["metrics"]["error_variance_unmasked"], mask, weights, entry["products"])
                    for scope, mask in (("triplet_admissible", entry["valid"]), ("common_admissible", common))
                }
            if original is not None:
                old = original.sel(window=window, transform=transform)
                np.testing.assert_array_equal(counts.reshape(ny, nx), old.n_observations.values)
                np.testing.assert_array_equal(storms.reshape(ny, nx), old.n_contributing_storms.values)
                for triplet, entry in entries.items():
                    name = entry["products"][0]
                    old_values = old.error_variance.sel(product=name).values
                    new_values = entry["metrics"]["error_variance_unmasked"][..., 0]
                    keep = eligible.reshape(ny, nx) & np.isfinite(old_values) & np.isfinite(new_values)
                    np.testing.assert_allclose(new_values[keep], old_values[keep], rtol=2e-6, atol=2e-5)
                for col, product in ((1, "mrms"), (2, "persiann_cdr")):
                    averaged = np.mean([e["metrics"]["error_variance_unmasked"][..., col] for e in entries.values()], axis=0)
                    old_values = old.error_variance.sel(product=product).values
                    keep = eligible.reshape(ny, nx) & np.isfinite(old_values) & np.isfinite(averaged)
                    np.testing.assert_allclose(averaged[keep], old_values[keep], rtol=2e-6, atol=2e-5)
                numerical_checks[f"{window}_{transform}"] = "Sample counts, model variances, and reconstruction of old averaged reference variances agree."
            print(f"{window}/{transform}: " + "; ".join(f"{t}={int(e['valid'].sum())} cells" for t, e in entries.items()) + f"; common={int(common.sum())}", flush=True)

    exports, netcdfs = [], []
    for triplet in TRIPLETS:
        for transform in TRANSFORMS:
            products = data[("core", transform)][triplet]["products"]
            out = xr.Dataset(coords={"window": list(WINDOWS), "product": products, "lat": daily.lat, "lon": daily.lon})
            dims = ("window", "product", "lat", "lon")
            for key in ("error_variance_unmasked", "signal_variance_unmasked", "observed_variance", "error_sd_unmasked", "snr_unmasked", "estimated_r2_to_truth_unmasked", "valid_product_estimate", "denominator_valid", "zero_error_boundary"):
                a = np.stack([np.moveaxis(data[(w, transform)][triplet]["metrics"][key], -1, 0) for w in WINDOWS])
                eligible = np.stack([data[(w, transform)][triplet]["eligible"] for w in WINDOWS])[:, None]
                a = (a & eligible).astype("uint8") if a.dtype == bool else np.where(eligible, a, np.nan)
                out[key] = (dims, a)
            out["valid_triplet"] = (("window", "lat", "lon"), np.stack([data[(w, transform)][triplet]["valid"] for w in WINDOWS]).astype("uint8"))
            common = np.stack([np.logical_and.reduce([e["valid"] for e in data[(w, transform)].values()]) for w in WINDOWS])
            out["valid_both_triplets"] = (("window", "lat", "lon"), common.astype("uint8"))
            out["sample_eligible"] = (("window", "lat", "lon"), np.stack([data[(w, transform)][triplet]["eligible"] for w in WINDOWS]).astype("uint8"))
            for key in ("counts", "storms"):
                out["n_observations" if key == "counts" else "n_contributing_storms"] = (("window", "lat", "lon"), np.stack([data[(w, transform)][triplet][key] for w in WINDOWS]).astype("int16"))
            for base_name in ("error_variance", "signal_variance", "error_sd", "snr", "estimated_r2_to_truth"):
                out[base_name] = out[base_name + "_unmasked"].where(out.valid_triplet == 1)
            out["study_domain_mask"] = daily.study_domain_mask
            out["study_watershed_fraction"] = daily.study_watershed_fraction
            out["crs"] = daily.crs if "crs" in daily else xr.DataArray(0, attrs={"grid_mapping_name": "latitude_longitude", "epsg_code": "EPSG:4326"})
            variance_units = "mm2 day-2" if transform == "raw" else "1"
            sd_units = "mm day-1" if transform == "raw" else "1"
            for key in out.data_vars:
                if "lat" in out[key].dims and "lon" in out[key].dims:
                    out[key].attrs["grid_mapping"] = "crs"
                if "variance" in key:
                    out[key].attrs["units"] = variance_units
                elif "error_sd" in key:
                    out[key].attrs["units"] = sd_units
                elif key.startswith("snr") or key.startswith("estimated_r2"):
                    out[key].attrs["units"] = "1"
            out.error_variance_unmasked.attrs["description"] = "Unscreened algebraic TC estimate on sample-eligible cells; may be negative. Use valid_triplet for interpretation."
            out.error_variance.attrs["description"] = "Unaveraged error variance using this triplet only; identical all-three admissibility mask for every product."
            out.attrs.update(title=f"Separate triplet collocation: {' / '.join(products)}", transform=transform,
                             source_daily=str(input_path), error_independence="All three errors assumed mutually uncorrelated within this triplet.",
                             averaging_between_triplets="none", bootstrap_screen_applied="false",
                             sample_policy="Same four-product coincident daily sample in both triplets; all_windows calendar de-duplication.",
                             admissibility="All three errors finite and nonnegative; all three signals finite and nonnegative; observed variances positive; positive numerically supported denominators; sample eligibility.",
                             zero_error_policy="Exact zero retained as boundary solution; SNR unavailable there. Negative estimates never clipped.",
                             log1p_definition="ln(1 + daily precipitation / (1 mm day-1))" if transform == "log1p" else "none",
                             minimum_storms=min_storms, minimum_days_per_storm=min_days,
                             minimum_core_observations=minimum["core"], minimum_expanded_observations=minimum["expanded"],
                             interpretation="Three-product TC/extended-TC diagnostics, not an estimate of CONUS404-CONUS404BA cross-error covariance.")
            folder = output_dir / triplet
            folder.mkdir(parents=True, exist_ok=True)
            nc_path = folder / f"triplet_error_estimates_{transform}.nc"
            encoding = {v: {"zlib": True, "complevel": 4} for v in out.data_vars if out[v].ndim >= 2}
            out.to_netcdf(nc_path, engine="netcdf4", encoding=encoding)
            with xr.open_dataset(nc_path, engine="netcdf4") as saved:
                xr.testing.assert_equal(saved.error_variance, out.error_variance)
                xr.testing.assert_equal(saved.valid_triplet, out.valid_triplet)
            netcdfs.append(str(nc_path))
            for window in WINDOWS:
                entry = data[(window, transform)][triplet]
                shared = out.valid_both_triplets.sel(window=window).values.astype(bool)
                for scope, mask in (("triplet_admissible", entry["valid"]), ("common_admissible", shared)):
                    for i, product in enumerate(products):
                        values = np.where(mask, entry["metrics"]["error_variance_unmasked"][..., i], np.nan)
                        tiff = folder / "GeoTIFF" / scope / "error_variance" / f"win-{window}__xform-{transform}__prod-{product}.tif"
                        tags = dict(triplet=triplet, product=product, window=window, transform=transform,
                                    support=scope, units=variance_units, bootstrap_screen="none", reference_averaging="none")
                        write_tiff(tiff, values, daily.lat.values, daily.lon.values, tags)
                        exports.append({**tags, "path": str(tiff), "valid_cells": int(mask.sum())})
    figures = make_figures(data, daily, weights, output_dir)
    manifest = {
        "input_daily": str(input_path), "original_ec_used_for_reproduction_check": str(ec_path),
        "triplets": {name: [PRODUCTS[i] for i in ids] for name, ids in TRIPLETS.items()},
        "reference_averaging": "none", "bootstrap_screen_applied": False,
        "sample_mode": "all_windows", "sampling": sampling,
        "eligibility": {"minimum_storms": min_storms, "minimum_days_per_storm": min_days, "minimum_observations": minimum},
        "support_definitions": {"triplet_admissible": "All three members admissible within that triplet only.",
                                "common_admissible": "Intersection of the two triplet-valid masks; same cells across all six panels."},
        "weighting": "study_watershed_fraction*cos(latitude); full stored watershed fraction is the area denominator",
        "synthetic_checks": checks, "reproduction_checks": numerical_checks,
        "summaries": supports, "netcdf": netcdfs, "figures": figures,
        "notes": ["Point estimates without bootstrap filtering do not establish sampling stability or statistical significance.",
                  "Reference-product errors from each triplet are retained separately, including constituent negative estimates in unmasked NetCDF diagnostic variables.",
                  "Different support masks change spatial averages; use common_admissible panels for direct numerical comparison.",
                  "Existing averaged-reference EC files and existing agreement files have not been altered; their agreement scores have not been recomputed here."],
    }
    manifest_path = output_dir / "separate_triplet_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    (output_dir / "GeoTIFF_manifest.json").write_text(json.dumps(exports, indent=2), encoding="utf-8")
    print(f"Finished. Manifest: {manifest_path}", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ec-source", type=Path, default=DEFAULT_EC)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
    else:
        run_analysis(args.input, args.output_dir, args.ec_source)


if __name__ == "__main__":
    main()
