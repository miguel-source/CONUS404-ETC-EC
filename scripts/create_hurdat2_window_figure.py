"""Create a verified event-window explainer using Hurricane Harvey examples.

Panel a is explicitly schematic, not an observed storm track or geographic map.
Panels b/c use the existing spherical-distance basin-window inventory. No data
or results in the source-data directory are modified.
"""

from __future__ import annotations

from study_paths import PROJECT_ROOT, RESULTS_ROOT, METADATA_ROOT, REPO_ROOT, STATES_FILE, WATERSHEDS_FILE, DAILY_INPUT, EC_DIRECTORY, EC_INPUT, LOSO_INPUT, AGREEMENT_DIRECTORY, AGREEMENT_INPUT


import csv
import json
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as PlotPolygon
from matplotlib.patches import Rectangle
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points


ROOT = REPO_ROOT
OUT = RESULTS_ROOT / "hurdat2_event_window_figure"
SOURCE = METADATA_ROOT / "tc_event_windows_by_watershed_spherical.csv"

NAVY = "#20557C"
BLUE = "#5F9EC0"
PALE_BLUE = "#DCEBF3"
ORANGE = "#B65D27"
PALE_ORANGE = "#F5DFCA"
INK = "#24313C"
GRAY = "#6D767E"
LIGHT = "#D9DFE3"
GREEN = "#256D56"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10.5,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    }
)


def date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def number(value: datetime) -> float:
    return float(mdates.date2num(value))


def short(value: datetime) -> str:
    return f"{value:%b} {value.day}"


def daybar(ax, start, end, y, color, height=0.5, **kwargs):
    """Each daily bar includes both endpoint days; right edge = end + 1 day."""
    left = number(start)
    width = (end - start).days + 1
    ax.add_patch(Rectangle((left, y - height / 2), width, height,
                           facecolor=color, edgecolor="none", **kwargs))


def span(ax, left, right, y, text, color=INK, fontsize=10):
    ax.annotate("", (right, y), (left, y),
                arrowprops={"arrowstyle": "|-|", "color": color, "lw": 0.9})
    ax.text((left + right) / 2, y + 0.1, text, color=color,
            ha="center", va="bottom", fontsize=fontsize)


def setup_date_axis(ax, start, end):
    ax.set_xlim(number(start), number(end + timedelta(days=1)))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    ax.tick_params(axis="x", labelsize=10, length=3)
    ax.tick_params(axis="y", length=0, pad=9)
    for side in ["left", "right", "top"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(LIGHT)
    ax.set_axisbelow(True)
    ax.grid(axis="x", which="minor", color=LIGHT, alpha=0.35, lw=0.55)
    ax.grid(axis="x", which="major", color=LIGHT, alpha=0.6, lw=0.6)


def main(map_drawer=None, output_dir=OUT, stem_name="hurdat2_event_window_definition_harvey"):
    output_dir.mkdir(parents=True, exist_ok=True)
    with SOURCE.open(encoding="utf-8-sig", newline="") as handle:
        records = [r for r in csv.DictReader(handle) if r["storm_id"] == "AL092017"]
    assert len(records) == 8
    for r in records:
        r["start"] = date(r["analysis_start_date"])
        r["end"] = date(r["analysis_end_date"])
        r["first"] = datetime.strptime(r["first_500km_contact_utc"], "%Y-%m-%dT%H:%MZ")
        r["last"] = datetime.strptime(r["last_500km_contact_utc"], "%Y-%m-%dT%H:%MZ")
        assert r["start"] == r["first"].replace(hour=0, minute=0) - timedelta(days=1)
        assert r["end"] == r["last"].replace(hour=0, minute=0) + timedelta(days=2)
    core_start, core_end = min(r["start"] for r in records), max(r["end"] for r in records)
    expanded_start, expanded_end = core_start - timedelta(days=4), core_end + timedelta(days=1)
    assert (core_start, core_end) == (date("2017-08-23"), date("2017-09-03"))
    assert (expanded_end - expanded_start).days + 1 == 17
    sj = next(r for r in records if r["watershed"] == "San Jacinto")

    fig = plt.figure(figsize=(13.8, 11.0), facecolor="white")
    fig.text(0.5, 0.963, "Track-based definition of tropical-cyclone event windows",
             ha="center", va="top", fontsize=17, fontweight="bold")
    fig.text(0.5, 0.927, "Spatial proximity selects dates; precipitation magnitude does not select dates",
             ha="center", va="top", fontsize=12, color=GRAY)

    # a: synthetic geometry explicitly separated from the observed examples.
    ax = fig.add_axes([0.055, 0.595, 0.425, 0.275])
    ax.set_title("a  Spatial criterion (schematic)", loc="left", pad=10, fontsize=12.5,
                 fontweight="bold")
    watershed = Polygon([(-170, -90), (-80, -175), (65, -100), (155, 35),
                         (70, 180), (-25, 145), (-165, 65)])
    zone = watershed.buffer(500)
    ax.add_patch(PlotPolygon(np.asarray(zone.exterior.coords), closed=True,
                             facecolor=PALE_BLUE, edgecolor=BLUE,
                             linestyle=(0, (4, 3)), lw=1.1))
    ax.add_patch(PlotPolygon(np.asarray(watershed.exterior.coords), closed=True,
                             facecolor="#BDD8CE", edgecolor=GREEN, lw=1.5))
    tx = np.linspace(-950, 950, 25)
    ty = np.full(tx.shape, 325.0)
    inside = np.array([watershed.distance(Point(x, y)) <= 500 for x, y in zip(tx, ty)])
    ax.plot(tx, ty, color=GRAY, lw=1.2, zorder=3)
    ax.scatter(tx[~inside], ty[~inside], s=17, c=GRAY, zorder=4)
    ax.scatter(tx[inside], ty[inside], s=20, c=NAVY, zorder=4)
    synoptic = np.arange(0, 25, 6)
    ax.scatter(tx[synoptic], ty[synoptic], s=55, facecolors="white",
               edgecolors=INK, linewidths=1.1, zorder=5)
    first_i, last_i = np.flatnonzero(inside)[[0, -1]]
    ax.scatter([tx[first_i]], [ty[first_i]], marker="^", s=92, c=GREEN,
               edgecolors="white", lw=0.6, zorder=6)
    ax.scatter([tx[last_i]], [ty[last_i]], marker="s", s=66, c=ORANGE,
               edgecolors="white", lw=0.6, zorder=6)
    ax.annotate("First hourly position\nwithin 500 km", (tx[first_i], 325),
                (-850, 625), ha="center", va="center", fontsize=10,
                arrowprops={"arrowstyle": "-", "lw": 0.9, "color": GRAY})
    ax.annotate("Last hourly position\nwithin 500 km", (tx[last_i], 325),
                (805, 625), ha="center", va="center", fontsize=10,
                arrowprops={"arrowstyle": "-", "lw": 0.9, "color": GRAY})
    ax.annotate("", (930, 325), (790, 325),
                arrowprops={"arrowstyle": "->", "color": GRAY, "lw": 1.1})
    p = Point(0, 325)
    nearest = nearest_points(p, watershed)[1]
    ax.annotate("", (p.x, p.y), (nearest.x, nearest.y),
                arrowprops={"arrowstyle": "<->", "color": INK, "lw": 0.9})
    ax.text(220, 135, "Shortest distance\nto the polygon", fontsize=10,
            ha="left", va="center")
    ax.text(-10, -15, "Watershed", ha="center", va="center", fontsize=10,
            color=GREEN)
    ax.text(0, -485, "500-km proximity zone", ha="center", color=NAVY, fontsize=11)
    ax.text(0, -725, "Not a rainfall footprint; not a geographic map", ha="center",
            fontsize=9.5, color=GRAY)
    ax.legend(handles=[
        Line2D([], [], marker="o", color="none", markerfacecolor="white",
               markeredgecolor=INK, markersize=6, label="Best-track fix"),
        Line2D([], [], marker="o", color="none", markerfacecolor=NAVY,
               markersize=4, label="Interpolated hourly position"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2,
       frameon=False, fontsize=9.5, handletextpad=0.5, columnspacing=1.5)
    ax.set(xlim=(-1000, 1000), ylim=(-760, 730))
    ax.set_aspect("equal")
    ax.axis("off")
    map_metadata = None
    if map_drawer is not None:
        ax.remove()
        map_metadata = map_drawer(fig, sj)

    # b: a single real basin, with calendar-day edges and hourly contact markers.
    bx = fig.add_axes([0.635, 0.608, 0.325, 0.235])
    fig.text(0.535, 0.884, "b  One basin: San Jacinto during Harvey", fontsize=12.5,
             fontweight="bold", ha="left")
    setup_date_axis(bx, date("2017-08-19"), date("2017-09-04"))
    bx.set_ylim(-0.65, 3.1)
    bx.set_yticks([2.5, 1.3, 0.1], ["Contact\nenvelope", "Basin\ncore", "Basin\nexpanded"])
    bx.tick_params(axis="y", labelsize=10)
    bx.plot([number(sj["first"]), number(sj["last"])], [2.5, 2.5], lw=4,
            color=GRAY, solid_capstyle="butt")
    bx.scatter([number(sj["first"])], [2.5], c=GREEN, s=60, marker="^", zorder=5)
    bx.scatter([number(sj["last"])], [2.5], c=ORANGE, s=44, marker="s", zorder=5)
    daybar(bx, sj["start"], sj["end"], 1.3, NAVY, 0.48)
    sj_exp_start, sj_exp_end = sj["start"] - timedelta(days=4), sj["end"] + timedelta(days=1)
    daybar(bx, sj_exp_start, sj_exp_end, 0.1, PALE_ORANGE, 0.48)
    bx.text(number(sj["first"]), 2.76, "Aug 25, 02:00", fontsize=9.5, ha="left")
    bx.text(number(sj["last"]), 2.76, "Aug 31, 17:00", fontsize=9.5, ha="right")
    bx.text((number(sj["start"]) + number(sj["end"]) + 1) / 2, 1.3,
            "Aug 24–Sep 2  |  10 days", color="white", ha="center", va="center", fontsize=10)
    bx.text((number(sj_exp_start) + number(sj_exp_end) + 1) / 2, 0.1,
            "Aug 20–Sep 3  |  15 days", ha="center", va="center", fontsize=10)
    span(bx, number(sj["start"]), number(date("2017-08-25")), 1.72, "−1 day", fontsize=9.5)
    span(bx, number(date("2017-09-01")), number(sj["end"] + timedelta(days=1)),
         1.72, "+2 days", fontsize=9.5)
    span(bx, number(sj_exp_start), number(sj["start"]), 0.48, "−4 days", color=ORANGE, fontsize=9.5)
    span(bx, number(sj["end"] + timedelta(days=1)), number(sj_exp_end + timedelta(days=1)),
         0.48, "+1 day", color=ORANGE, fontsize=9.5)
    bx.set_xlabel("UTC calendar days, 2017", labelpad=9)

    # c: all real basin core intervals and the storm-wide envelope/expanded range.
    cx = fig.add_axes([0.245, 0.125, 0.705, 0.365])
    fig.text(0.055, 0.535, "c  Harvey: combine the eight assigned basin windows", fontsize=12.5,
             fontweight="bold", ha="left")
    setup_date_axis(cx, expanded_start, expanded_end)
    for i, r in enumerate(records):
        y = 9.2 - i * 0.87
        daybar(cx, r["start"], r["end"], y, PALE_BLUE, height=0.52)
        cx.text((number(r["start"]) + number(r["end"]) + 1) / 2, y,
                f"{short(r['start'])}–{short(r['end'])}", ha="center", va="center", fontsize=10)
    ys = [9.2 - i * 0.87 for i in range(8)]
    labels = [r["watershed"].replace("Middle Brazos-Clear Fork", "Middle Brazos–Clear Fork")
              .replace("Middle Brazos-Bosque", "Middle Brazos–Bosque") for r in records]
    storm_y, expanded_y = 1.25, 0.12
    daybar(cx, core_start, core_end, storm_y, NAVY, height=0.68)
    daybar(cx, expanded_start, expanded_end, expanded_y, PALE_ORANGE, height=0.68)
    cx.text((number(core_start) + number(core_end) + 1) / 2, storm_y,
            "Aug 23–Sep 3  |  12 days", color="white", ha="center", va="center", fontsize=11)
    cx.text((number(expanded_start) + number(expanded_end) + 1) / 2, expanded_y,
            "Aug 19–Sep 4  |  17 days", ha="center", va="center", fontsize=11)
    cx.axhline(1.94, color=LIGHT, lw=0.9)
    cx.set_yticks(ys + [storm_y, expanded_y], labels + ["Storm core", "Storm expanded"])
    cx.tick_params(axis="y", labelsize=10.5)
    cx.set_ylim(-0.55, 9.83)
    for edge in [number(core_start), number(core_end + timedelta(days=1))]:
        cx.plot([edge, edge], [-0.35, 9.6], color=NAVY, lw=0.7,
                ls=(0, (3, 3)), alpha=0.65, zorder=1)
    cx.set_xlabel("UTC calendar days, 2017 (both endpoint days included)", labelpad=9)
    fig.text(0.245, 0.059,
             "Storm core = earliest basin-core start through latest basin-core end; expanded = another −4 / +1 days.",
             fontsize=10, ha="left", color=GRAY)
    fig.text(0.055, 0.031,
             "Dates: existing HURDAT2-derived spherical-distance inventory. Core denotes a temporal window, not the physical cyclone core.",
             fontsize=9.5, ha="left", color=GRAY)
    fig.text(0.055, 0.014,
             "The proximity rule and temporal padding are study-defined choices; they do not establish the origin of every rainfall observation.",
             fontsize=9.5, ha="left", color=GRAY)

    stem = output_dir / stem_name
    fig.savefig(stem.with_suffix(".png"), dpi=400)
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".tif"), dpi=400, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(output_dir / "preview.png", dpi=135)
    plt.close(fig)
    manifest = {
        "source_inventory": str(SOURCE),
        "spherical_distance_code": str(METADATA_ROOT / "build_tc_event_windows_spherical.py"),
        "panel_a": map_metadata or "Illustrative planar geometry only, not an observed track or geographic map.",
        "panel_b": "Actual San Jacinto / Harvey contact timestamps and derived daily windows.",
        "panel_c": "Actual eight Harvey basin core windows and combined event windows.",
        "event_core": [core_start.isoformat(), core_end.isoformat()],
        "event_expanded": [expanded_start.isoformat(), expanded_end.isoformat()],
        "all_windows_caveat": "Final all_windows EC uses the union of selected storm calendar windows at each study pixel; not a local 500-km pixel filter.",
        "records": records,
    }
    (output_dir / "figure_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"Created figure files in {output_dir}")
    print(f"Harvey core: {short(core_start)}–{short(core_end)}; expanded: {short(expanded_start)}–{short(expanded_end)}")


if __name__ == "__main__":
    main()
