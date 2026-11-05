"""Render the window explainer with real San Jacinto/Harvey geography.

The 500-km outline is the contour of spherical point-to-polygon distance,
calculated using the same radius and segment-distance equations as the original
window generator. The rendered contour is discretized on a 0.08-degree grid;
qualifying track timestamps are checked using the full polygon, not the grid.
"""

from __future__ import annotations

from study_paths import PROJECT_ROOT, RESULTS_ROOT, METADATA_ROOT, REPO_ROOT, STATES_FILE, WATERSHEDS_FILE, DAILY_INPUT, EC_DIRECTORY, EC_INPUT, LOSO_INPUT, AGREEMENT_DIRECTORY, AGREEMENT_INPUT


import csv
from datetime import datetime, timedelta
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import shapely
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import create_hurdat2_window_figure as base


ROOT = REPO_ROOT
OUT = RESULTS_ROOT / "hurdat2_event_window_figure" / "san_jacinto_harvey_actual_map"
SHAPEFILE = WATERSHEDS_FILE
TRACK_FILE = METADATA_ROOT / "hurdat2_atl_1851_2025.txt"
STATES = STATES_FILE
RADIUS_KM = 6371.0088


def parse_coord(s):
    return float(s[:-1]) * (-1 if s[-1] in "SW" else 1)


def load_track():
    lines = TRACK_FILE.read_text(encoding="utf-8").splitlines()
    header_i = next(i for i, line in enumerate(lines) if line.startswith("AL092017,"))
    count = int(lines[header_i].split(",")[2])
    records = []
    for line in lines[header_i + 1:header_i + count + 1]:
        p = [x.strip() for x in line.split(",")]
        records.append({
            "time": datetime.strptime(p[0] + p[1].zfill(4), "%Y%m%d%H%M"),
            "lat": parse_coord(p[4]), "lon": parse_coord(p[5]),
        })
    samples = []
    for i, current in enumerate(records):
        samples.append(current.copy())
        if i == len(records) - 1:
            continue
        nxt = records[i + 1]
        hours = (nxt["time"] - current["time"]).total_seconds() / 3600
        if hours <= 0 or hours > 18:
            continue
        p1, l1, p2, l2 = np.radians([current["lat"], current["lon"], nxt["lat"], nxt["lon"]])
        v1 = np.array([np.cos(p1) * np.cos(l1), np.cos(p1) * np.sin(l1), np.sin(p1)])
        v2 = np.array([np.cos(p2) * np.cos(l2), np.cos(p2) * np.sin(l2), np.sin(p2)])
        delta = np.arccos(np.clip(v1 @ v2, -1, 1))
        for offset in range(1, int(np.floor(hours)) + 1):
            if offset >= hours:
                break
            fraction = offset / hours
            vec = v1 if delta < 1e-12 else (
                np.sin((1 - fraction) * delta) * v1 + np.sin(fraction * delta) * v2
            ) / np.sin(delta)
            vec /= np.linalg.norm(vec)
            lat = np.degrees(np.arctan2(vec[2], np.hypot(vec[0], vec[1])))
            lon = np.degrees(np.arctan2(vec[1], vec[0]))
            samples.append({"time": current["time"] + timedelta(hours=offset),
                            "lat": float(lat), "lon": float(lon)})
    return records, sorted(samples, key=lambda s: s["time"])


def haversine(phi1, lam1, phi2, lam2):
    a = np.sin((phi2 - phi1) / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin((lam2 - lam1) / 2)**2
    return 2 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(0, 1 - a)))


def bearing(phi1, lam1, phi2, lam2):
    return np.arctan2(np.sin(lam2 - lam1) * np.cos(phi2),
                      np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(lam2 - lam1))


def spherical_distances(lon, lat, polygon):
    """Match the original exterior-ring spherical segment-distance algorithm."""
    coords = np.asarray(polygon.exterior.coords)
    lam1, phi1 = np.radians(coords[:-1]).T
    lam2, phi2 = np.radians(coords[1:]).T
    delta12 = haversine(phi1, lam1, phi2, lam2)
    theta12 = bearing(phi1, lam1, phi2, lam2)
    x, y = np.asarray(lon).ravel(), np.asarray(lat).ravel()
    result = np.empty(x.size)
    for start in range(0, x.size, 160):
        stop = min(x.size, start + 160)
        phi3, lam3 = np.radians(y[start:stop])[:, None], np.radians(x[start:stop])[:, None]
        delta13 = haversine(phi1, lam1, phi3, lam3)
        theta13 = bearing(phi1, lam1, phi3, lam3)
        diff = theta13 - theta12
        delta_xt = np.arcsin(np.clip(np.sin(delta13) * np.sin(diff), -1, 1))
        delta_at = np.arctan2(np.sin(delta13) * np.cos(diff), np.cos(delta13))
        delta23 = haversine(phi2, lam2, phi3, lam3)
        candidates = np.where((delta_at >= 0) & (delta_at <= delta12),
                              np.abs(delta_xt), np.minimum(delta13, delta23))
        result[start:stop] = np.min(candidates, axis=1) * RADIUS_KM
    result[shapely.intersects_xy(polygon, x, y)] = 0
    return result.reshape(np.shape(lon))


def draw_actual_map(fig, sj, show_harvey_watersheds=False):
    frame = gpd.read_file(SHAPEFILE).to_crs(4326)
    basin = frame.loc[frame["HUC_CODES"].astype(str) == "120401"].copy()
    assert len(basin) == 1 and basin.iloc[0].HUC_NAME == "San Jacinto"
    harvey_basins = frame.loc[
        frame["STORMS"].fillna("").str.contains("Harvey (2017)", regex=False)
    ].copy()
    with base.SOURCE.open(encoding="utf-8-sig", newline="") as handle:
        harvey_ids = {r["huc_code"] for r in csv.DictReader(handle)
                      if r["storm_id"] == "AL092017"}
    assert len(harvey_basins) == 8
    assert set(harvey_basins["HUC_CODES"].astype(str)) == harvey_ids
    polygon = basin.geometry.iloc[0]
    records, track = load_track()
    distances = spherical_distances([s["lon"] for s in track], [s["lat"] for s in track], polygon)
    qualifying = np.flatnonzero(distances <= 500)
    first, last = track[qualifying[0]], track[qualifying[-1]]
    assert first["time"] == sj["first"], (first, sj["first"])
    assert last["time"] == sj["last"], (last, sj["last"])
    nearest_i = int(np.argmin(distances))
    assert abs(distances[nearest_i] - 135.8) < 0.1

    # Latitude/longitude map with geographic aspect correction at central latitude.
    ax = fig.add_axes([0.055, 0.594, 0.445, 0.282])
    fig.text(0.055, 0.884, "a  San Jacinto watershed and Harvey track", fontsize=12.5,
             fontweight="bold", ha="left")
    extent = (-104.5 if show_harvey_watersheds else -103, 23, -86.5, 36.5)
    if show_harvey_watersheds:
        minx, miny, maxx, maxy = harvey_basins.total_bounds
        assert extent[0] < minx and extent[1] < miny and extent[2] > maxx and extent[3] > maxy
    states = gpd.read_file(STATES, bbox=extent).to_crs(4326)
    states.plot(ax=ax, facecolor="#F6F5F0", edgecolor="#A9B0B4", linewidth=0.45, zorder=1)
    ax.set_facecolor("#F6FAFC")
    glon, glat = np.meshgrid(np.arange(extent[0], extent[2] + .08, .08),
                             np.arange(extent[1], extent[3] + .08, .08))
    print("Calculating spherical 500-km outline on map grid...", flush=True)
    grid_d = spherical_distances(glon, glat, polygon)
    ax.contourf(glon, glat, grid_d, levels=[-0.1, 500], colors=[base.PALE_BLUE], alpha=0.7, zorder=2)
    ax.contour(glon, glat, grid_d, levels=[500], colors=[base.BLUE], linewidths=1.0,
               linestyles=[(0, (4, 3))], zorder=3)
    if show_harvey_watersheds:
        other_basins = harvey_basins.loc[harvey_basins["HUC_CODES"].astype(str) != "120401"]
        other_basins.plot(ax=ax, facecolor="#BEC2C5", edgecolor="#71797F",
                          linewidth=0.65, zorder=3.5)
    basin.plot(ax=ax, facecolor="#92BDB0", edgecolor=base.GREEN, linewidth=1.1, zorder=7)

    displayed = [s for s in track if datetime(2017, 8, 23) <= s["time"]]
    observed = [s for s in records if datetime(2017, 8, 23) <= s["time"]]
    ax.plot([s["lon"] for s in displayed], [s["lat"] for s in displayed], color=base.GRAY,
            lw=1.0, zorder=4)
    inds = [i for i, s in enumerate(track) if datetime(2017, 8, 23) <= s["time"]]
    mask = distances[inds] <= 500
    tx = np.array([s["lon"] for s in displayed])
    ty = np.array([s["lat"] for s in displayed])
    ax.scatter(tx[mask], ty[mask], s=4, c=base.NAVY, zorder=5)
    ax.scatter([s["lon"] for s in observed], [s["lat"] for s in observed], s=12,
               facecolors="white", edgecolors=base.INK, linewidths=0.55, zorder=6)
    ax.scatter(first["lon"], first["lat"], marker="^", s=65, color=base.GREEN,
               edgecolors="white", linewidths=.6, zorder=8)
    ax.scatter(last["lon"], last["lat"], marker="s", s=47, color=base.ORANGE,
               edgecolors="white", linewidths=.6, zorder=8)
    ax.annotate("First ≤500 km\nAug 25, 02:00", (first["lon"], first["lat"]),
                (-101.8, 24.0), ha="left", va="center", fontsize=8.8,
                arrowprops={"arrowstyle": "-", "lw": .8, "color": base.GREEN})
    ax.annotate("Last ≤500 km\nAug 31, 17:00", (last["lon"], last["lat"]),
                (-89.0, 34.5), ha="center", va="center", fontsize=8.8,
                arrowprops={"arrowstyle": "-", "lw": .8, "color": base.ORANGE})
    centroid = polygon.representative_point()
    ax.annotate("San Jacinto\nHUC6 120401", (centroid.x, centroid.y),
                (-101.0, 30.9) if show_harvey_watersheds else (-99.2, 31.2),
                ha="center", va="center", fontsize=9, color=base.GREEN, zorder=9,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.2}
                     if show_harvey_watersheds else None,
                arrowprops={"arrowstyle": "-", "lw": .8, "color": base.GREEN})
    ax.text(-100.9, 29.1, "TEXAS", fontsize=8, color=base.GRAY)
    ax.text(-92.6, 31.5, "LOUISIANA", fontsize=7.5, color=base.GRAY, ha="center")
    ax.text(-91.0, 25.3, "Gulf of Mexico", fontsize=10, fontstyle="italic",
            color="#6B8A9D", ha="center")
    ax.set(xlim=(extent[0], extent[2]), ylim=(extent[1], extent[3]))
    ax.set_aspect(1 / np.cos(np.radians(30)))
    xticks = [-104, -100, -96, -92, -88] if show_harvey_watersheds else [-102, -98, -94, -90]
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{abs(t)}°W" for t in xticks])
    ax.set_yticks([24, 28, 32, 36])
    ax.set_yticklabels(["24°N", "28°N", "32°N", "36°N"])
    ax.tick_params(labelsize=8.8, length=2.5)
    for spine in ax.spines.values():
        spine.set_color(base.LIGHT)
    legend_handles = [
        Patch(facecolor=base.PALE_BLUE, edgecolor=base.BLUE, linestyle="--",
              label="500 km to San Jacinto" if show_harvey_watersheds else "500-km zone"),
        Line2D([], [], color=base.GRAY, marker="o", markerfacecolor="white", markeredgecolor=base.INK,
               markersize=3.5, lw=.8, label="Harvey best track"),
    ]
    if show_harvey_watersheds:
        legend_handles.append(Patch(facecolor="#BEC2C5", edgecolor="#71797F",
                                    label="Other Harvey watersheds"))
    ax.legend(handles=legend_handles, loc="upper left", frameon=False,
       fontsize=8.0 if show_harvey_watersheds else 8.5, handlelength=1.6,
       borderpad=.25, labelspacing=.35)
    fig.text(0.285, 0.562, "Actual geography • Hurricane Harvey (2017) • All times UTC",
             ha="center", fontsize=9, color=base.GRAY)

    # Enlarged exact basin boundary so its shape remains identifiable at regional scale.
    inset = fig.add_axes([0.444, 0.746, 0.061, 0.102])
    basin.plot(ax=inset, facecolor="#92BDB0", edgecolor=base.GREEN, linewidth=1.0)
    inset.set_title("Basin detail", fontsize=8.5, pad=3)
    inset.set_axis_off()
    return {
        "description": "Actual San Jacinto HUC6 polygon and Harvey best track; no schematic geometry.",
        "watershed_shapefile": str(SHAPEFILE), "watershed_huc": "120401",
        "hurdat2_file": str(TRACK_FILE), "basemap_states": str(STATES),
        "distance_method": "Original spherical segment-distance equations, R=6371.0088 km; exact original polygon.",
        "display_contour_grid_degrees": 0.08,
        "display_track_start_utc": "2017-08-23T00:00Z",
        "first_500km_contact": first, "last_500km_contact": last,
        "first_distance_km": float(distances[qualifying[0]]),
        "last_distance_km": float(distances[qualifying[-1]]),
        "nearest_position": track[nearest_i], "nearest_distance_km": float(distances[nearest_i]),
        "harvey_watershed_overlay": bool(show_harvey_watersheds),
        "gray_watersheds": [
            {"huc": str(r.HUC_CODES), "name": str(r.HUC_NAME)}
            for _, r in harvey_basins.iterrows() if str(r.HUC_CODES) != "120401"
        ] if show_harvey_watersheds else [],
        "highlighted_watershed": "San Jacinto (120401); green",
        "proximity_zone_reference": "San Jacinto polygon only, not the eight-watershed union",
    }


if __name__ == "__main__":
    base.main(map_drawer=draw_actual_map, output_dir=OUT,
              stem_name="sanjacinto_harvey_event_window_definition")
