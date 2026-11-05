from __future__ import annotations

import csv
import math
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from osgeo import gdal, ogr


EARTH_RADIUS_M = 6_371_008.8
STORM_RE = re.compile(r"([A-Za-z][A-Za-z -]+?)\s*\((\d{4})\)")


def parse_coord(value: str) -> float:
    value = value.strip()
    number = float(value[:-1])
    return -number if value[-1].upper() in {"S", "W"} else number


def parse_hurdat(path: Path):
    storms = {}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    i = 0
    while i < len(lines):
        fields = [part.strip() for part in lines[i].split(",")]
        if len(fields) < 3 or not re.fullmatch(r"AL\d{6}", fields[0]):
            i += 1
            continue
        storm_id, name, count = fields[0], fields[1].upper(), int(fields[2])
        storm_year = int(storm_id[-4:])
        if storm_year < 1983:
            i += count + 1
            continue
        records = []
        for line in lines[i + 1 : i + 1 + count]:
            parts = [part.strip() for part in line.split(",")]
            when = datetime.strptime(parts[0] + parts[1].zfill(4), "%Y%m%d%H%M").replace(
                tzinfo=timezone.utc
            )
            records.append(
                {
                    "time": when,
                    "record_id": parts[2],
                    "status": parts[3],
                    "lat": parse_coord(parts[4]),
                    "lon": parse_coord(parts[5]),
                    "wind_kt": int(parts[6]),
                    "pressure_mb": int(parts[7]),
                }
            )
        storms[(storm_year, name)] = {
            "storm_id": storm_id,
            "name": name,
            "year": storm_year,
            "records": sorted(records, key=lambda item: item["time"]),
        }
        i += count + 1
    return storms


def interpolate_great_circle(lat1, lon1, lat2, lon2, fraction):
    phi1, lam1, phi2, lam2 = map(math.radians, (lat1, lon1, lat2, lon2))
    v1 = np.array([math.cos(phi1) * math.cos(lam1), math.cos(phi1) * math.sin(lam1), math.sin(phi1)])
    v2 = np.array([math.cos(phi2) * math.cos(lam2), math.cos(phi2) * math.sin(lam2), math.sin(phi2)])
    dot = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    delta = math.acos(dot)
    if delta < 1e-12:
        vec = v1
    else:
        vec = math.sin((1.0 - fraction) * delta) / math.sin(delta) * v1
        vec += math.sin(fraction * delta) / math.sin(delta) * v2
    vec /= np.linalg.norm(vec)
    lat = math.degrees(math.atan2(vec[2], math.hypot(vec[0], vec[1])))
    lon = math.degrees(math.atan2(vec[1], vec[0]))
    return lat, lon


def densify_track(records, step_hours=1.0):
    samples = []
    for idx, current in enumerate(records):
        samples.append({"time": current["time"], "lat": current["lat"], "lon": current["lon"]})
        if idx == len(records) - 1:
            continue
        nxt = records[idx + 1]
        hours = (nxt["time"] - current["time"]).total_seconds() / 3600.0
        if hours <= 0 or hours > 18:
            continue
        for step in range(1, int(math.floor(hours / step_hours)) + 1):
            offset = step * step_hours
            if offset >= hours:
                break
            lat, lon = interpolate_great_circle(
                current["lat"], current["lon"], nxt["lat"], nxt["lon"], offset / hours
            )
            samples.append({"time": current["time"] + timedelta(hours=offset), "lat": lat, "lon": lon})
    return sorted(samples, key=lambda item: item["time"])


def haversine_angle(phi1, lam1, phi2, lam2):
    dphi = phi2 - phi1
    dlam = lam2 - lam1
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    return 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(0.0, 1.0 - a)))


def bearing(phi1, lam1, phi2, lam2):
    dlam = lam2 - lam1
    y = np.sin(dlam) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlam)
    return np.arctan2(y, x)


def polygon_segments(geometry):
    if geometry.GetGeometryName().upper() == "MULTIPOLYGON":
        polygons = [geometry.GetGeometryRef(i) for i in range(geometry.GetGeometryCount())]
    else:
        polygons = [geometry]
    segments = []
    for polygon in polygons:
        ring = polygon.GetGeometryRef(0)
        coords = np.array([(ring.GetY(i), ring.GetX(i)) for i in range(ring.GetPointCount())], dtype=float)
        if len(coords) < 2:
            continue
        phi1 = np.radians(coords[:-1, 0])
        lam1 = np.radians(coords[:-1, 1])
        phi2 = np.radians(coords[1:, 0])
        lam2 = np.radians(coords[1:, 1])
        delta12 = haversine_angle(phi1, lam1, phi2, lam2)
        theta12 = bearing(phi1, lam1, phi2, lam2)
        segments.append((phi1, lam1, phi2, lam2, delta12, theta12))
    return segments


def spherical_point_polygon_distance_m(lat, lon, geometry, segments):
    point = ogr.Geometry(ogr.wkbPoint)
    point.AddPoint_2D(lon, lat)
    if geometry.Contains(point) or geometry.Touches(point):
        return 0.0

    phi3 = math.radians(lat)
    lam3 = math.radians(lon)
    best = math.inf
    for phi1, lam1, phi2, lam2, delta12, theta12 in segments:
        phi3a = np.full_like(phi1, phi3)
        lam3a = np.full_like(lam1, lam3)
        delta13 = haversine_angle(phi1, lam1, phi3a, lam3a)
        theta13 = bearing(phi1, lam1, phi3a, lam3a)
        bearing_diff = theta13 - theta12
        delta_xt = np.arcsin(np.clip(np.sin(delta13) * np.sin(bearing_diff), -1.0, 1.0))
        delta_at = np.arctan2(np.sin(delta13) * np.cos(bearing_diff), np.cos(delta13))
        delta23 = haversine_angle(phi2, lam2, phi3a, lam3a)
        on_segment = (delta_at >= 0.0) & (delta_at <= delta12)
        candidates = np.where(on_segment, np.abs(delta_xt), np.minimum(delta13, delta23))
        best = min(best, float(np.min(candidates)) * EARTH_RADIUS_M)
    return best


def iso_date(value):
    return value.strftime("%Y-%m-%d") if value else ""


def iso_time(value):
    return value.strftime("%Y-%m-%dT%H:%MZ") if value else ""


def main():
    shp_path = Path(sys.argv[1])
    hurdat_path = Path(sys.argv[2])
    output_path = Path(sys.argv[3])
    storms = parse_hurdat(hurdat_path)

    gdal.UseExceptions()
    ds = gdal.OpenEx(str(shp_path), gdal.OF_VECTOR | gdal.OF_READONLY)
    layer = ds.GetLayer(0)
    rows = []
    track_cache = {}
    listed_unique = set()

    for feature in layer:
        storms_text = feature.GetField("STORMS") or ""
        if "atmospheric" in storms_text.lower():
            continue
        geometry = feature.GetGeometryRef().Clone()
        segments = polygon_segments(geometry)
        for match in STORM_RE.finditer(storms_text):
            name = match.group(1).strip().upper()
            year = int(match.group(2))
            listed_unique.add((year, name))
            storm = storms.get((year, name))
            if not storm:
                continue
            if storm["storm_id"] not in track_cache:
                track_cache[storm["storm_id"]] = densify_track(storm["records"])
            samples = track_cache[storm["storm_id"]]
            nearest = None
            contacts = []
            for sample in samples:
                distance_m = spherical_point_polygon_distance_m(
                    sample["lat"], sample["lon"], geometry, segments
                )
                if nearest is None or distance_m < nearest[0]:
                    nearest = (distance_m, sample)
                if distance_m <= 500_000.0:
                    contacts.append(sample)

            first = contacts[0]["time"] if contacts else None
            last = contacts[-1]["time"] if contacts else None
            analysis_start = first.date() - timedelta(days=1) if first else None
            analysis_end = last.date() + timedelta(days=2) if last else None
            download_start = analysis_start - timedelta(days=4) if analysis_start else None
            download_end = analysis_end + timedelta(days=1) if analysis_end else None
            nearest_distance, nearest_sample = nearest
            rows.append(
                {
                    "huc_code": feature.GetField("HUC_CODES"),
                    "huc_level": feature.GetField("HUC_LEVEL"),
                    "watershed": feature.GetField("HUC_NAME"),
                    "basin_id": feature.GetField("BASIN_ID"),
                    "storm_name": name.title(),
                    "storm_year": year,
                    "storm_id": storm["storm_id"],
                    "track_start_utc": iso_time(storm["records"][0]["time"]),
                    "track_end_utc": iso_time(storm["records"][-1]["time"]),
                    "nearest_time_utc": iso_time(nearest_sample["time"]),
                    "nearest_distance_km": round(nearest_distance / 1000.0, 1),
                    "first_500km_contact_utc": iso_time(first),
                    "last_500km_contact_utc": iso_time(last),
                    "analysis_start_date": iso_date(analysis_start),
                    "analysis_end_date": iso_date(analysis_end),
                    "download_start_date": iso_date(download_start),
                    "download_end_date": iso_date(download_end),
                    "auto_window_status": "ok" if contacts else "manual_review_no_500km_contact",
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    counts = defaultdict(int)
    for row in rows:
        counts[row["auto_window_status"]] += 1
    print(f"Wrote {len(rows)} watershed-storm rows to {output_path}")
    print(f"Unique listed storms: {len(listed_unique)}")
    print(dict(counts))


if __name__ == "__main__":
    main()
