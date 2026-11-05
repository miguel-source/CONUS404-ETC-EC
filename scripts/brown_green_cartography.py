"""Shared brown-to-green palette and vector map adornments for TC figures.

The scale bar is calibrated to WGS84 geodesic distances between tick points
at 27.35 degrees north.  It is not an arbitrary degree-length scale, and the
labels refer to international statute miles (exactly 1609.344 metres).
"""

from __future__ import annotations

import math

from matplotlib.patches import Circle, Polygon, Rectangle
from pyproj import Geod
from scipy.optimize import brentq


PALETTE_ANCHORS = (
    "#A6611A", "#DFC27D", "#F5F5F5", "#80CDC1", "#238B78", "#01665E",
)
BROWN_GREEN_ANCHORS = PALETTE_ANCHORS
STATUTE_MILE_METRES = 1609.344
SCALE_ORIGIN = (-103.5, 27.35)
GEOD = Geod(ellps="WGS84")


def scale_tick_longitudes(mile_values=(0, 125, 250, 375, 500)):
    """Return geodesic-from-origin tick locations on the scale parallel."""
    lon0, lat0 = SCALE_ORIGIN
    ticks = []
    for miles in mile_values:
        target_metres = miles * STATUTE_MILE_METRES
        lon = lon0 if miles == 0 else brentq(
            lambda candidate: GEOD.inv(lon0, lat0, candidate, lat0)[2]
            - target_metres,
            lon0, lon0 + 20.0, xtol=1e-12,
        )
        measured_metres = GEOD.inv(lon0, lat0, lon, lat0)[2]
        assert abs(measured_metres - target_metres) < 1e-5
        ticks.append({
            "miles": int(miles) if float(miles).is_integer() else float(miles),
            "longitude": float(lon),
            "latitude": float(lat0), "distance_metres": float(measured_metres),
            "distance_error_metres": float(measured_metres - target_metres),
        })
    return ticks


def _north_rose(ax):
    """Draw a compact, north-up eight-point black-and-white compass rose."""
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xspan, yspan = xmax - xmin, ymax - ymin
    cx = xmin + .10 * xspan
    cy = ymin + .77 * yspan
    # Equal data-coordinate radii make a round rose on the equal-aspect
    # geographic maps used by the figure generators.
    radius = .025 * xspan
    inner_radius = .23 * radius
    kwargs = {"zorder": 24, "clip_on": True}
    ax.add_patch(Circle((cx, cy), radius * 1.06,
                        facecolor="white", edgecolor="none", **kwargs))
    ax.add_patch(Circle((cx, cy), radius * .65, facecolor="none",
                        edgecolor="#202020", linewidth=.38, **kwargs))
    for direction in range(8):
        angle = math.pi / 2 - direction * math.pi / 4
        spike_radius = radius if direction % 2 == 0 else radius * .72
        tip = (cx + spike_radius * math.cos(angle),
               cy + spike_radius * math.sin(angle))
        left_angle, right_angle = angle + math.pi / 8, angle - math.pi / 8
        left = (cx + inner_radius * math.cos(left_angle),
                cy + inner_radius * math.sin(left_angle))
        right = (cx + inner_radius * math.cos(right_angle),
                 cy + inner_radius * math.sin(right_angle))
        ax.add_patch(Polygon([(cx, cy), left, tip], closed=True,
                             facecolor="#202020", edgecolor="#202020",
                             linewidth=.30, **kwargs))
        ax.add_patch(Polygon([(cx, cy), tip, right], closed=True,
                             facecolor="white", edgecolor="#202020",
                             linewidth=.30, **kwargs))
    labels = {
        "N": (cx, cy + radius * 1.70),
        "S": (cx, cy - radius * 1.70),
        "E": (cx + radius * 1.70, cy),
        "W": (cx - radius * 1.70, cy),
    }
    for label, (xx, yy) in labels.items():
        ax.text(xx, yy, label, fontsize=8.5, ha="center", va="center",
                color="#202020", fontweight="normal", zorder=25,
                clip_on=True)
    return {
        "type": "eight-point black-and-white vector compass rose",
        "center_axes_fraction": [.10, .77],
        "center_lonlat": [float(cx), float(cy)],
        "outer_radius_degrees": float(radius),
        "orientation": "North is vertically upward; unrotated EPSG:4326 axes",
    }


def add_cartography(ax):
    """Add the common scale bar and compass; return serializable geometry.

    Call after assigning the map's geographic limits and equal aspect.  The
    caller should keep counts at the top of the map because the scale bar
    occupies the otherwise blank southwest corner.
    """
    ticks = scale_tick_longitudes((0, 62.5, 125, 187.5, 250, 312.5, 375, 437.5, 500))
    lon0, lat0 = SCALE_ORIGIN
    end_lon = ticks[-1]["longitude"]
    ax.add_patch(Rectangle((lon0 - .42, lat0 - 1.10),
                           end_lon - lon0 + 4.0, 1.32,
                           facecolor="white", edgecolor="none",
                           zorder=20, clip_on=True))
    # We use a thin ruler: one unfilled horizontal rule,
    # longer 125-mile ticks, and shorter intermediate 62.5-mile ticks.
    ax.plot([lon0, end_lon], [lat0, lat0], color="#202020", linewidth=.60,
            solid_capstyle="butt", zorder=22, clip_on=True)
    for tick in ticks:
        xx = tick["longitude"]
        is_major = tick["miles"] % 125 == 0
        tick_depth = .22 if is_major else .12
        ax.plot([xx, xx], [lat0, lat0 - tick_depth],
                color="#202020", linewidth=.55, solid_capstyle="butt",
                zorder=23, clip_on=True)
        if tick["miles"] in (0, 125, 250, 500):
            ax.text(xx, lat0 - .30, str(tick["miles"]), fontsize=7.7,
                    ha="center", va="top", color="#202020",
                    zorder=24, clip_on=True)
    ax.text(end_lon + 1.45, lat0 - .30, "Miles", fontsize=7.7,
            ha="left", va="top", color="#202020", zorder=24, clip_on=True)
    compass = _north_rose(ax)
    return {
        "scale_bar": {
            "style": "thin unfilled ruler with downward major and minor ticks",
            "baseline_linewidth_points": .60,
            "major_tick_interval_miles": 125,
            "minor_tick_interval_miles": 62.5,
            "units": "international statute miles",
            "metres_per_mile": STATUTE_MILE_METRES,
            "ellipsoid": "WGS84", "reference_latitude": lat0,
            "origin_lonlat": [lon0, lat0],
            "tick_labels_miles": [0, 125, 250, 500],
            "ticks": ticks,
            "geometry_rule": (
                "All tick points share the reference latitude. Each longitude "
                "is solved so its WGS84 geodesic distance from the origin "
                "equals the indicated cumulative number of statute miles."
            ),
            "scope": (
                "Scale is calibrated at the bar latitude; geographic map "
                "distance scale varies with latitude away from the bar."
            ),
        },
        "north_symbol": compass,
    }
