from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import as_file, files
from typing import Iterable
import warnings

import numpy as np
from matplotlib.collections import LineCollection

from ..coordinates import EARTH_RADIUS_KM, latlon_to_local_km
from .common import theme_values


_BUNDLED_KINDS = ("coastlines", "countries", "states", "counties")


def _bundled_resource(kind: str):
    return files("lmas.resources").joinpath("maps", f"{kind}.npz")


def _bundled_available() -> bool:
    try:
        return all(_bundled_resource(kind).is_file() for kind in _BUNDLED_KINDS)
    except Exception:
        return False


def cartography_backend() -> str | None:
    """Return the dependable map source used by this installation."""
    if _bundled_available():
        return "bundled"
    try:
        import cartopy.io.shapereader  # noqa: F401
        return "cartopy"
    except Exception:
        try:
            from mpl_toolkits.basemap import Basemap  # noqa: F401
            return "basemap"
        except Exception:
            return None


def _iter_geometry_lines(geometry) -> Iterable[np.ndarray]:
    if geometry is None or getattr(geometry, "is_empty", False):
        return
    kind = getattr(geometry, "geom_type", "")
    if kind in {"LineString", "LinearRing"}:
        values = np.asarray(geometry.coords, dtype=float)
        if values.ndim == 2 and values.shape[0] >= 2:
            yield values[:, :2]
    elif kind == "Polygon":
        yield from _iter_geometry_lines(geometry.exterior)
        for ring in geometry.interiors:
            yield from _iter_geometry_lines(ring)
    elif hasattr(geometry, "geoms"):
        for item in geometry.geoms:
            yield from _iter_geometry_lines(item)


@lru_cache(maxsize=8)
def _bundled_segments(kind: str) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    resource = _bundled_resource(kind)
    with as_file(resource) as path:
        with np.load(path, allow_pickle=False) as payload:
            points = np.asarray(payload["points"], dtype=float)
            offsets = np.asarray(payload["offsets"], dtype=np.int64)
            bounds = np.asarray(payload["bounds"], dtype=float)
    segments = tuple(points[offsets[i] : offsets[i + 1]] for i in range(len(offsets) - 1))
    return segments, bounds


@lru_cache(maxsize=16)
def _cartopy_segments(kind: str, scale: str = "50m") -> tuple[np.ndarray, ...]:
    import cartopy.io.shapereader as shpreader

    datasets = {
        "coastlines": ("physical", "coastline"),
        "countries": ("cultural", "admin_0_boundary_lines_land"),
        "states": ("cultural", "admin_1_states_provinces_lines"),
    }
    if kind == "counties":
        category, name, scale = "cultural", "admin_2_counties", "10m"
    else:
        category, name = datasets[kind]
    path = shpreader.natural_earth(resolution=scale, category=category, name=name)
    reader = shpreader.Reader(path)
    lines: list[np.ndarray] = []
    for geometry in reader.geometries():
        lines.extend(_iter_geometry_lines(geometry))
    return tuple(lines)


@lru_cache(maxsize=8)
def _basemap_segments(kind: str) -> tuple[np.ndarray, ...]:
    from matplotlib.figure import Figure
    from mpl_toolkits.basemap import Basemap

    fig = Figure(figsize=(2, 2))
    axis = fig.add_subplot(111)
    basemap = Basemap(
        projection="cyl",
        llcrnrlon=-180,
        urcrnrlon=180,
        llcrnrlat=-89.9,
        urcrnrlat=89.9,
        resolution="i",
        ax=axis,
    )
    draw = {
        "coastlines": basemap.drawcoastlines,
        "countries": basemap.drawcountries,
        "states": basemap.drawstates,
        "counties": basemap.drawcounties,
    }.get(kind)
    if draw is None:
        return ()
    collection = draw(linewidth=0.1)
    if hasattr(collection, "get_segments"):
        raw = collection.get_segments()
    elif hasattr(collection, "get_paths"):
        raw = [path.vertices for path in collection.get_paths()]
    else:
        raw = ()
    result = tuple(
        np.asarray(item, dtype=float)[:, :2]
        for item in raw
        if np.asarray(item).ndim == 2 and np.asarray(item).shape[0] >= 2
    )
    fig.clear()
    return result


def _computed_bounds(segments: tuple[np.ndarray, ...]) -> np.ndarray:
    return np.asarray(
        [
            (np.nanmin(item[:, 0]), np.nanmax(item[:, 0]), np.nanmin(item[:, 1]), np.nanmax(item[:, 1]))
            for item in segments
        ],
        dtype=float,
    ) if segments else np.empty((0, 4), dtype=float)


@lru_cache(maxsize=16)
def _segment_store(kind: str) -> tuple[tuple[np.ndarray, ...], np.ndarray, str | None]:
    backend = cartography_backend()
    if backend == "bundled":
        segments, bounds = _bundled_segments(kind)
        return segments, bounds, "bundled"
    if backend == "cartopy":
        try:
            segments = _cartopy_segments(kind)
            return segments, _computed_bounds(segments), "cartopy"
        except Exception as exc:
            warnings.warn(f"Cartopy map data unavailable for {kind}: {exc}", RuntimeWarning)
    if backend in {"cartopy", "basemap"}:
        try:
            segments = _basemap_segments(kind)
            return segments, _computed_bounds(segments), "basemap"
        except Exception as exc:
            warnings.warn(f"Basemap data unavailable for {kind}: {exc}", RuntimeWarning)
    return (), np.empty((0, 4), dtype=float), None


@dataclass
class MapUnderlay:
    axis: object
    coordinate_system: str
    reference_longitude: float
    reference_latitude: float
    east_sign: float = 1.0
    north_sign: float = 1.0
    theme: str = "dark"
    show_coastlines: bool = True
    show_countries: bool = True
    show_states: bool = True
    show_counties: bool = True

    def __post_init__(self) -> None:
        values = theme_values(self.theme)
        color = values["text"]
        self.backend = cartography_backend()
        self.collections: dict[str, LineCollection] = {}
        self.visible_counts: dict[str, int] = {}
        specifications = (
            ("coastlines", self.show_coastlines, 1.15, 0.82),
            ("countries", self.show_countries, 1.00, 0.76),
            ("states", self.show_states, 1.00, 0.78),
            ("counties", self.show_counties, 0.70, 0.62),
        )
        for kind, enabled, width, alpha in specifications:
            if not enabled:
                continue
            collection = LineCollection(
                [], colors=color, linewidths=width, alpha=alpha, zorder=0.25
            )
            collection.set_gid(f"lmas-map-{kind}")
            self.axis.add_collection(collection)
            self.collections[kind] = collection
        self.update()

    def _geodetic_extent(self) -> tuple[float, float, float, float]:
        xlim = tuple(sorted(float(value) for value in self.axis.get_xlim()))
        ylim = tuple(sorted(float(value) for value in self.axis.get_ylim()))
        if self.coordinate_system == "geodetic":
            west, east = xlim
            south, north = ylim
        else:
            east_km = np.asarray(xlim, dtype=float) / float(self.east_sign)
            north_km = np.asarray(ylim, dtype=float) / float(self.north_sign)
            west = self.reference_longitude + np.rad2deg(
                np.min(east_km)
                / (
                    EARTH_RADIUS_KM
                    * max(np.cos(np.deg2rad(self.reference_latitude)), 1e-8)
                )
            )
            east = self.reference_longitude + np.rad2deg(
                np.max(east_km)
                / (
                    EARTH_RADIUS_KM
                    * max(np.cos(np.deg2rad(self.reference_latitude)), 1e-8)
                )
            )
            south = self.reference_latitude + np.rad2deg(
                np.min(north_km) / EARTH_RADIUS_KM
            )
            north = self.reference_latitude + np.rad2deg(
                np.max(north_km) / EARTH_RADIUS_KM
            )
            west, east = sorted((west, east))
            south, north = sorted((south, north))
        lon_pad = max((east - west) * 0.12, 0.05)
        lat_pad = max((north - south) * 0.12, 0.05)
        return west - lon_pad, east + lon_pad, south - lat_pad, north + lat_pad

    def update(self) -> None:
        extent = self._geodetic_extent()
        west, east, south, north = extent
        for kind, collection in self.collections.items():
            if kind == "counties" and (east - west > 8.0 or north - south > 6.0):
                collection.set_segments([])
                self.visible_counts[kind] = 0
                continue
            segments, bounds, backend = _segment_store(kind)
            if backend is not None:
                self.backend = backend
            if bounds.size:
                keep = (
                    (bounds[:, 1] >= west)
                    & (bounds[:, 0] <= east)
                    & (bounds[:, 3] >= south)
                    & (bounds[:, 2] <= north)
                )
                candidates = (segments[index] for index in np.flatnonzero(keep))
            else:
                candidates = iter(segments)
            visible: list[np.ndarray] = []
            for segment in candidates:
                if self.coordinate_system == "geodetic":
                    converted = segment
                else:
                    x, y = latlon_to_local_km(
                        segment[:, 0],
                        segment[:, 1],
                        self.reference_longitude,
                        self.reference_latitude,
                    )
                    converted = np.column_stack(
                        (self.east_sign * x, self.north_sign * y)
                    )
                finite = np.all(np.isfinite(converted), axis=1)
                if np.count_nonzero(finite) >= 2:
                    visible.append(converted[finite])
            collection.set_segments(visible)
            self.visible_counts[kind] = len(visible)

    @property
    def available(self) -> bool:
        return any(self.visible_counts.values())

    @property
    def status(self) -> str:
        backend = self.backend or "unavailable"
        drawn = ", ".join(
            f"{kind} {count}" for kind, count in self.visible_counts.items() if count
        )
        if drawn:
            return f"Map underlay: {backend}; {drawn}"
        return f"Map underlay: {backend}; no boundary crosses the current view"


def add_map_underlay(
    axis,
    *,
    coordinate_system: str,
    reference_longitude: float,
    reference_latitude: float,
    east_sign: float = 1.0,
    north_sign: float = 1.0,
    theme: str = "dark",
) -> MapUnderlay | None:
    if cartography_backend() is None:
        return None
    return MapUnderlay(
        axis=axis,
        coordinate_system=str(coordinate_system),
        reference_longitude=float(reference_longitude),
        reference_latitude=float(reference_latitude),
        east_sign=float(east_sign),
        north_sign=float(north_sign),
        theme=str(theme),
    )


__all__ = ["MapUnderlay", "add_map_underlay", "cartography_backend"]
