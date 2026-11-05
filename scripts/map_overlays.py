"""Shared CONUS and watershed overlays for hurricane-result maps and ArcGIS."""

from __future__ import annotations

from study_paths import PROJECT_ROOT, RESULTS_ROOT, METADATA_ROOT, REPO_ROOT, STATES_FILE, WATERSHEDS_FILE, DAILY_INPUT, EC_DIRECTORY, EC_INPUT, LOSO_INPUT, AGREEMENT_DIRECTORY, AGREEMENT_INPUT


from functools import lru_cache
from pathlib import Path

try:
    import geopandas as gpd
except ModuleNotFoundError:  # Plotting still works in the lean ETC kernel.
    gpd = None

from matplotlib.collections import LineCollection
try:
    import shapefile
except ModuleNotFoundError:  # GeoPandas-backed plotting does not need pyshp.
    shapefile = None
import shapely
from shapely.geometry import shape as shapely_shape
from shapely.ops import unary_union


WATERSHED_FILE = WATERSHEDS_FILE
NATURAL_EARTH_STATES = STATES_FILE
HURRICANES = (
    "Matthew",
    "Harvey",
    "Irma",
    "Florence",
    "Imelda",
    "Laura",
    "Zeta",
    "Eta",
    "Ida",
    "Ian",
)


def _require_geopandas() -> None:
    if gpd is None:
        raise ModuleNotFoundError(
            "GeoPackage creation requires geopandas. Use the CONUS404 kernel "
            "for export_gridded_results_to_geotiff.py. Boundary plotting also "
            "works in CONUS404_ETC2 through the built-in pyshp fallback."
        )


@lru_cache(maxsize=1)
def load_watershed_layers() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    _require_geopandas()
    if not WATERSHED_FILE.exists():
        raise FileNotFoundError(WATERSHED_FILE)
    all_watersheds = gpd.read_file(WATERSHED_FILE).to_crs("EPSG:4326")
    # The supplied WILLAMETTE polygon has one ring self-intersection.  A
    # zero-width buffer repairs it without changing its reported area and
    # keeps the ArcGIS boundary package polygonal (make_valid would return a
    # GeometryCollection for this feature).
    invalid = ~all_watersheds.geometry.is_valid
    if invalid.any():
        all_watersheds.loc[invalid, "geometry"] = all_watersheds.loc[
            invalid, "geometry"
        ].buffer(0)
    storm_text = all_watersheds["STORMS"].fillna("").astype(str)
    analysis_mask = False
    for event_name in HURRICANES:
        analysis_mask = analysis_mask | storm_text.str.contains(
            event_name, case=False, regex=False
        )
    analysis = all_watersheds.loc[analysis_mask].copy()
    return all_watersheds, analysis


@lru_cache(maxsize=1)
def load_conus_layers() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load lower-48 states and their dissolved exterior boundary."""
    _require_geopandas()
    if not NATURAL_EARTH_STATES.exists():
        raise FileNotFoundError(
            "Cached Natural Earth states were not found. Expected: "
            f"{NATURAL_EARTH_STATES}"
        )
    states = gpd.read_file(NATURAL_EARTH_STATES)
    if 'admin' in states.columns:
        states = states.loc[
            states["admin"].eq("United States of America")
            & ~states["postal"].isin(["AK", "HI"])
        ].copy()
    else:
        # We also accept our already-filtered public conus_states GeoPackage.
        if not {'name','postal','geometry'}.issubset(states.columns):
            raise ValueError('The state layer must contain name, postal and geometry.')
        states = states.loc[~states['postal'].isin(['AK','HI'])].copy()
    states = states.to_crs("EPSG:4326")[["name", "postal", "geometry"]]
    conus_geometry = shapely.make_valid(shapely.union_all(states.geometry.array))
    conus = gpd.GeoDataFrame(
        {"name": ["Conterminous United States"]},
        geometry=[conus_geometry],
        crs="EPSG:4326",
    )
    return states, conus


@lru_cache(maxsize=2)
def _pyshp_records(path: str) -> tuple[tuple[object, dict], ...]:
    """Read geometry/attribute pairs without requiring geopandas."""
    if shapefile is None:
        raise ModuleNotFoundError("Boundary plotting requires either geopandas or pyshp.")
    reader = shapefile.Reader(path, encodingErrors="replace")
    output = []
    for item in reader.iterShapeRecords():
        geometry = shapely_shape(item.shape.__geo_interface__)
        if not geometry.is_valid:
            geometry = geometry.buffer(0)
        output.append((geometry, item.record.as_dict()))
    return tuple(output)


@lru_cache(maxsize=1)
def _fallback_conus_geometries() -> tuple[tuple[object, ...], object]:
    records = _pyshp_records(str(NATURAL_EARTH_STATES))
    states = tuple(
        geometry
        for geometry, attributes in records
        if attributes.get("admin") == "United States of America"
        and attributes.get("postal") not in {"AK", "HI"}
    )
    return states, unary_union(states)


@lru_cache(maxsize=2)
def _fallback_watershed_geometries(scope: str) -> tuple[object, ...]:
    records = _pyshp_records(str(WATERSHED_FILE))
    if scope != "analysis":
        return tuple(geometry for geometry, _ in records)
    names = tuple(name.casefold() for name in HURRICANES)
    return tuple(
        geometry
        for geometry, attributes in records
        if any(
            name in str(attributes.get("STORMS", "")).casefold()
            for name in names
        )
    )


def _line_segments(geometry) -> list[list[tuple[float, float]]]:
    boundary = geometry.boundary
    if boundary.geom_type in {"LineString", "LinearRing"}:
        return [list(boundary.coords)]
    segments: list[list[tuple[float, float]]] = []
    for part in getattr(boundary, "geoms", ()):
        if part.geom_type in {"LineString", "LinearRing"}:
            segments.append(list(part.coords))
        else:
            segments.extend(_line_segments(part))
    return segments


def _add_geometry_boundaries(ax, geometries, *, color, linewidth, alpha, zorder):
    segments = []
    for geometry in geometries:
        segments.extend(_line_segments(geometry))
    ax.add_collection(
        LineCollection(
            segments,
            colors=color,
            linewidths=linewidth,
            alpha=alpha,
            zorder=zorder,
        )
    )


def add_map_boundaries(
    ax,
    *,
    watershed_scope: str = "analysis",
    show_states: bool = True,
    show_conus: bool = True,
    show_watersheds: bool = True,
) -> None:
    """Overlay boundaries without changing the raster-derived map extent."""
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    if gpd is None:
        if not NATURAL_EARTH_STATES.exists():
            raise FileNotFoundError(NATURAL_EARTH_STATES)
        if not WATERSHED_FILE.exists():
            raise FileNotFoundError(WATERSHED_FILE)
        if show_states or show_conus:
            states, conus = _fallback_conus_geometries()
            if show_states:
                _add_geometry_boundaries(
                    ax,
                    states,
                    color="#969696",
                    linewidth=0.35,
                    alpha=0.65,
                    zorder=5,
                )
            if show_conus:
                _add_geometry_boundaries(
                    ax,
                    (conus,),
                    color="#252525",
                    linewidth=0.8,
                    alpha=0.9,
                    zorder=6,
                )
        if show_watersheds:
            _add_geometry_boundaries(
                ax,
                _fallback_watershed_geometries(watershed_scope),
                color="#111111",
                linewidth=0.65,
                alpha=0.95,
                zorder=7,
            )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        return
    if show_states or show_conus:
        states, conus = load_conus_layers()
        if show_states:
            states.boundary.plot(
                ax=ax, color="#969696", linewidth=0.35, alpha=0.65, zorder=5
            )
        if show_conus:
            conus.boundary.plot(
                ax=ax, color="#252525", linewidth=0.8, alpha=0.9, zorder=6
            )
    if show_watersheds:
        all_watersheds, analysis = load_watershed_layers()
        watersheds = analysis if watershed_scope == "analysis" else all_watersheds
        watersheds.boundary.plot(
            ax=ax, color="#111111", linewidth=0.65, alpha=0.95, zorder=7
        )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def write_boundary_geopackage(
    output_path: Path | str, *, overwrite: bool = False
) -> Path:
    """Write ArcGIS-ready CONUS and watershed overlay layers to one GeoPackage."""
    _require_geopandas()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        expected_layers = {
            "conus_states",
            "conus_boundary",
            "watersheds_all_27",
            "watersheds_hurricane_analysis",
        }
        try:
            existing_layers = set(gpd.list_layers(output_path)["name"])
        except Exception as exc:
            raise RuntimeError(
                f"Could not validate existing boundary package: {output_path}. "
                "Rerun with overwrite=True to replace it."
            ) from exc
        if expected_layers.issubset(existing_layers):
            return output_path
        raise RuntimeError(
            f"Boundary package is incomplete: {output_path}. "
            "Rerun with overwrite=True to replace it."
        )
    if output_path.exists():
        output_path.unlink()
    states, conus = load_conus_layers()
    all_watersheds, analysis = load_watershed_layers()
    layers = (
        (states, "conus_states"),
        (conus, "conus_boundary"),
        (all_watersheds, "watersheds_all_27"),
        (analysis, "watersheds_hurricane_analysis"),
    )
    for frame, layer_name in layers:
        frame.to_file(
            output_path,
            layer=layer_name,
            driver="GPKG",
            # A unique GPKG layer can be created in write mode without
            # deleting the other layers.  This works with both Fiona and
            # pyogrio; Fiona append mode expects the named layer to exist.
            mode="w",
        )
    return output_path
