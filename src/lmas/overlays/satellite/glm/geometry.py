"""Fast GLM event-footprint geometry for the native NumPy data model.

The implementation preserves the established glmtools lookup-table method:
L2 event centroids are mapped to GLM fixed-grid angles on the lightning
ellipsoid, pixel-corner offsets are interpolated from the packaged lookup
resource, and the corner rays are intersected with the same lightning
ellipsoid to recover geographic polygons.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
import pickle
from typing import Sequence

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .model import GLMDataError

# Lightning ellipsoid revisions used by glmtools.
_LIGHTNING_ELLIPSOIDS_M = {
    0: (6.394140e6, 6.362755e6),
    1: (6.378137e6 + 14.0e3, 6.362755e6),
}
_GRS80_RE_M = 6.378137e6
_GRS80_RP_M = 6.35675231414e6
_DEFAULT_SAT_HEIGHT_M = 35.786023e6


def lightning_ellipse_revision(observation_time_ns: int) -> int:
    """Return the historical lightning-ellipsoid revision for an observation."""
    boundary = np.datetime64("2018-10-15T00:00:00", "ns").astype(np.int64)
    return 0 if int(observation_time_ns) < int(boundary) else 1


def lightning_ellipse_radii(revision: int) -> tuple[float, float]:
    try:
        return _LIGHTNING_ELLIPSOIDS_M[int(revision)]
    except KeyError as exc:
        raise ValueError(f"Unsupported GLM lightning-ellipsoid revision: {revision}") from exc


def lon_lat_to_fixed_grid(
    longitude_deg: np.ndarray,
    latitude_deg: np.ndarray,
    *,
    satellite_longitude_deg: float,
    ellipse_revision: int,
    satellite_height_m: float = _DEFAULT_SAT_HEIGHT_M,
    grs80_re_m: float = _GRS80_RE_M,
    grs80_rp_m: float = _GRS80_RP_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert GLM L2 lightning-ellipsoid lon/lat to fixed-grid radians.

    This is the NumPy-native equivalent of
    ``glmtools.io.lightning_ellipse.ltg_ellps_lon_lat_to_fixed_grid``.
    """
    lon = np.asarray(longitude_deg, dtype=np.float64)
    lat = np.asarray(latitude_deg, dtype=np.float64)
    re_ltg, rp_ltg = lightning_ellipse_radii(ellipse_revision)

    ff_ltg = (re_ltg - rp_ltg) / re_ltg
    ff_grs80 = (grs80_re_m - grs80_rp_m) / grs80_re_m
    sat_h_from_center = float(satellite_height_m) + float(grs80_re_m)

    dlon = lon - float(satellite_longitude_deg)
    dlon = (dlon + 180.0) % 360.0 - 180.0
    lon_rad = np.radians(dlon)
    lat_rad = np.radians(lat)
    lat_geocentric = np.arctan((1.0 - ff_grs80) ** 2 * np.tan(lat_rad))

    cos_lat = np.cos(lat_geocentric)
    sin_lat = np.sin(lat_geocentric)
    radius = (re_ltg * (1.0 - ff_ltg)) / np.sqrt(
        1.0 - ff_ltg * (2.0 - ff_ltg) * cos_lat * cos_lat
    )
    vx = radius * cos_lat * np.cos(lon_rad) - sat_h_from_center
    vy = radius * cos_lat * np.sin(lon_rad)
    vz = radius * sin_lat
    magnitude = np.sqrt(vx * vx + vy * vy + vz * vz)
    vx = vx / -magnitude
    vy = vy / -magnitude
    vz = vz / magnitude

    y_rad = np.arctan2(vz, vx)
    x_rad = -np.arcsin(np.clip(vy, -1.0, 1.0))
    return x_rad, y_rad


def fixed_grid_to_lon_lat(
    x_rad: np.ndarray,
    y_rad: np.ndarray,
    *,
    satellite_longitude_deg: float,
    ellipse_revision: int,
    satellite_height_m: float = _DEFAULT_SAT_HEIGHT_M,
    grs80_re_m: float = _GRS80_RE_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Intersect fixed-grid viewing rays with the GLM lightning ellipsoid."""
    x = np.asarray(x_rad, dtype=np.float64)
    y = np.asarray(y_rad, dtype=np.float64)
    re_ltg, rp_ltg = lightning_ellipse_radii(ellipse_revision)
    h = float(satellite_height_m) + float(grs80_re_m)

    sin_x, cos_x = np.sin(x), np.cos(x)
    sin_y, cos_y = np.sin(y), np.cos(y)
    ratio = (re_ltg * re_ltg) / (rp_ltg * rp_ltg)
    a = sin_x * sin_x + cos_x * cos_x * (cos_y * cos_y + ratio * sin_y * sin_y)
    b = -2.0 * h * cos_x * cos_y
    c = h * h - re_ltg * re_ltg
    discriminant = b * b - 4.0 * a * c

    valid = discriminant >= 0.0
    rs = np.full(np.broadcast(x, y).shape, np.nan, dtype=np.float64)
    rs[valid] = (-b[valid] - np.sqrt(discriminant[valid])) / (2.0 * a[valid])

    sx = rs * cos_x * cos_y
    sy = -rs * sin_x
    sz = rs * cos_x * sin_y
    lon = np.degrees(
        np.radians(float(satellite_longitude_deg)) - np.arctan2(sy, h - sx)
    )
    # GLM L2 geographic coordinates are expressed as GRS80 geodetic
    # longitude/latitude even though the viewing ray is intersected with the
    # elevated lightning ellipsoid. Recover the geocentric direction of that
    # intersection and convert it back to GRS80 geodetic latitude.
    geocentric_lat = np.arctan2(sz, np.sqrt((h - sx) ** 2 + sy * sy))
    grs80_flattening = (_GRS80_RE_M - 6.35675231414e6) / _GRS80_RE_M
    lat = np.degrees(np.arctan(np.tan(geocentric_lat) / (1.0 - grs80_flattening) ** 2))
    lon = (lon + 180.0) % 360.0 - 180.0
    return lon, lat


@lru_cache(maxsize=1)
def _corner_interpolators():
    resource = resources.files("lmas.resources").joinpath("G16_corner_lut_fixedgrid.pickle")
    with resource.open("rb") as stream:
        x_lut_urad, y_lut_urad, corner_offsets_urad = pickle.load(stream)

    x_axis = np.asarray(x_lut_urad[0, :], dtype=np.float64) * 1.0e-6
    y_axis = np.asarray(y_lut_urad[:, 0], dtype=np.float64) * 1.0e-6
    offsets = np.asarray(corner_offsets_urad, dtype=np.float64) * 1.0e-6
    linear = tuple(
        RegularGridInterpolator(
            (y_axis, x_axis),
            offsets[:, :, corner, coordinate],
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        for corner in range(4)
        for coordinate in range(2)
    )
    nearest = tuple(
        RegularGridInterpolator(
            (y_axis, x_axis),
            offsets[:, :, corner, coordinate],
            method="nearest",
            bounds_error=False,
            fill_value=None,
        )
        for corner in range(4)
        for coordinate in range(2)
    )
    return linear, nearest


def _interpolate_corner_offsets(x_rad: np.ndarray, y_rad: np.ndarray) -> np.ndarray:
    linear, nearest = _corner_interpolators()
    points = np.column_stack((np.asarray(y_rad).reshape(-1), np.asarray(x_rad).reshape(-1)))
    result = np.empty((points.shape[0], 4, 2), dtype=np.float64)
    for corner in range(4):
        for coordinate in range(2):
            idx = corner * 2 + coordinate
            values = np.asarray(linear[idx](points), dtype=np.float64)
            missing = ~np.isfinite(values)
            if np.any(missing):
                values[missing] = nearest[idx](points[missing])
            result[:, corner, coordinate] = values
    return result


@dataclass(slots=True)
class GLMEventGeometry:
    """Lazy, reusable geometry cache attached to one GLM observation."""

    observation: object
    ellipse_revision: int | None = None
    _center_x_rad: np.ndarray | None = field(default=None, init=False, repr=False)
    _center_y_rad: np.ndarray | None = field(default=None, init=False, repr=False)
    _corner_fixed_grid: dict[float, np.ndarray] = field(default_factory=dict, init=False, repr=False)
    _corner_lonlat: dict[float, np.ndarray] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.ellipse_revision is None:
            self.ellipse_revision = lightning_ellipse_revision(
                self.observation.identity.observation_start_ns
            )

    @property
    def satellite_longitude_deg(self) -> float:
        projection = self.observation.identity.projection
        value = projection.field_of_view_lon_deg
        if value is None:
            value = projection.nominal_subpoint_lon_deg
        if value is None or not np.isfinite(value):
            raise GLMDataError("GLM product lacks usable field-of-view/subpoint longitude")
        return float(value)

    @property
    def satellite_height_m(self) -> float:
        value = self.observation.identity.projection.nominal_height_km
        if value is None or not np.isfinite(value):
            return _DEFAULT_SAT_HEIGHT_M
        return float(value) * 1000.0

    def event_centers_fixed_grid(self, indices: Sequence[int] | np.ndarray | None = None) -> np.ndarray:
        if self._center_x_rad is None or self._center_y_rad is None:
            events = self.observation.events
            self._center_x_rad, self._center_y_rad = lon_lat_to_fixed_grid(
                events.longitude_deg,
                events.latitude_deg,
                satellite_longitude_deg=self.satellite_longitude_deg,
                ellipse_revision=int(self.ellipse_revision),
                satellite_height_m=self.satellite_height_m,
            )
        idx = _normalize_indices(indices, self._center_x_rad.size)
        return np.column_stack((self._center_x_rad[idx], self._center_y_rad[idx]))

    def event_corners_fixed_grid(
        self,
        indices: Sequence[int] | np.ndarray | None = None,
        *,
        inflate: float = 1.0,
    ) -> np.ndarray:
        """Return fixed-grid corners, computing only requested event rows.

        dev2/dev3 eagerly generated every footprint on the first request. That
        is fine for a two-minute fixture but unnecessarily blocks the Qt thread
        for longer records. The dense cache is still NumPy-fast, while rows are
        populated lazily as linked views request them.
        """
        key = float(inflate)
        size = len(self.observation.events)
        idx = _normalize_indices(indices, size)
        cache = self._corner_fixed_grid.get(key)
        if cache is None:
            cache = np.full((size, 4, 2), np.nan, dtype=np.float64)
            self._corner_fixed_grid[key] = cache
        if idx.size:
            ready = np.all(np.isfinite(cache[idx]), axis=(1, 2))
            missing_idx = idx[~ready]
            if missing_idx.size:
                centers = self.event_centers_fixed_grid(missing_idx)
                offsets = _interpolate_corner_offsets(centers[:, 0], centers[:, 1])
                corners = np.empty_like(offsets)
                corners[:, :, 0] = centers[:, None, 0] + offsets[:, :, 0] * key
                corners[:, :, 1] = centers[:, None, 1] + offsets[:, :, 1] * key
                cache[missing_idx] = corners
        return cache[idx]

    def event_corners_lonlat(
        self,
        indices: Sequence[int] | np.ndarray | None = None,
        *,
        inflate: float = 1.0,
    ) -> np.ndarray:
        """Return geographic corners, lazily filling the requested cache rows."""
        key = float(inflate)
        size = len(self.observation.events)
        idx = _normalize_indices(indices, size)
        cache = self._corner_lonlat.get(key)
        if cache is None:
            cache = np.full((size, 4, 2), np.nan, dtype=np.float64)
            self._corner_lonlat[key] = cache
        if idx.size:
            ready = np.all(np.isfinite(cache[idx]), axis=(1, 2))
            missing_idx = idx[~ready]
            if missing_idx.size:
                fixed = self.event_corners_fixed_grid(missing_idx, inflate=key)
                lon, lat = fixed_grid_to_lon_lat(
                    fixed[:, :, 0],
                    fixed[:, :, 1],
                    satellite_longitude_deg=self.satellite_longitude_deg,
                    ellipse_revision=int(self.ellipse_revision),
                    satellite_height_m=self.satellite_height_m,
                )
                cache[missing_idx] = np.stack((lon, lat), axis=-1)
        return cache[idx]

    def cache_status(self) -> dict[str, object]:
        return {
            "ellipse_revision": int(self.ellipse_revision),
            "centers_cached": self._center_x_rad is not None,
            "fixed_grid_corner_scales": sorted(self._corner_fixed_grid),
            "geographic_corner_scales": sorted(self._corner_lonlat),
        }

    def clear_cache(self) -> None:
        self._center_x_rad = None
        self._center_y_rad = None
        self._corner_fixed_grid.clear()
        self._corner_lonlat.clear()


def _normalize_indices(indices: Sequence[int] | np.ndarray | None, size: int) -> np.ndarray:
    if indices is None:
        return np.arange(size, dtype=np.int64)
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    if np.any((idx < 0) | (idx >= size)):
        raise IndexError("GLM event geometry index is out of range")
    return idx
