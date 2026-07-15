"""Precision Mode measurement helpers.

Precision Mode is the official LMAS name for the linked two-cursor workflow
that users may colloquially call "scope mode".  This module intentionally has
no Qt dependency so the scientific measurements can be tested independently
of the desktop application.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class PrecisionSource:
    """One solved LMA source selected by a Precision Mode cursor."""

    index: int
    source_id: int
    time: np.datetime64
    latitude: float
    longitude: float
    altitude_km: float
    east_km: float
    north_km: float
    power_dbw: float | None = None
    chi2: float | None = None
    stations: int | None = None

    @classmethod
    def from_values(cls, values: Mapping[str, Any], index: int) -> "PrecisionSource":
        """Construct a source from figure metadata arrays."""

        idx = int(index)

        def finite_optional(name: str) -> float | None:
            raw = np.asarray(values.get(name, ()), dtype=float)
            if idx >= raw.size:
                return None
            value = float(raw[idx])
            return value if math.isfinite(value) else None

        def integer_optional(name: str) -> int | None:
            raw = np.asarray(values.get(name, ()))
            if idx >= raw.size:
                return None
            try:
                value = float(raw[idx])
            except (TypeError, ValueError):
                return None
            return int(round(value)) if math.isfinite(value) else None

        times = np.asarray(values.get("time", ())).astype("datetime64[ns]")
        source_ids = np.asarray(values.get("source_id", ()), dtype=np.int64)
        if idx < 0 or idx >= times.size:
            raise IndexError(f"Precision source index {idx} is outside the source array")

        def required(name: str) -> float:
            raw = np.asarray(values.get(name, ()), dtype=float)
            if idx >= raw.size:
                return float("nan")
            return float(raw[idx])

        return cls(
            index=idx,
            source_id=int(source_ids[idx]) if idx < source_ids.size else idx,
            time=times[idx],
            latitude=required("latitude"),
            longitude=required("longitude"),
            altitude_km=required("altitude_km"),
            east_km=required("east_km"),
            north_km=required("north_km"),
            power_dbw=finite_optional("power"),
            chi2=finite_optional("chi2"),
            stations=integer_optional("stations"),
        )


@dataclass(frozen=True)
class PrecisionDifference:
    """Cursor-B minus cursor-A measurements."""

    delta_time_ms: float
    delta_east_km: float
    delta_north_km: float
    delta_altitude_km: float
    horizontal_distance_km: float
    distance_3d_km: float
    bearing_deg: float
    apparent_horizontal_speed_km_s: float | None
    apparent_3d_speed_km_s: float | None


def precision_difference(a: PrecisionSource, b: PrecisionSource) -> PrecisionDifference:
    """Return B-minus-A displacement and apparent cursor-derived speeds.

    Bearing is clockwise from geographic north.  Apparent speeds use the
    absolute cursor time separation and are undefined when the two source
    timestamps are identical.
    """

    delta_time_ms = float((b.time - a.time) / np.timedelta64(1, "ms"))
    delta_east = float(b.east_km - a.east_km)
    delta_north = float(b.north_km - a.north_km)
    delta_altitude = float(b.altitude_km - a.altitude_km)
    horizontal = float(math.hypot(delta_east, delta_north))
    distance_3d = float(math.sqrt(horizontal * horizontal + delta_altitude * delta_altitude))
    bearing = float(math.degrees(math.atan2(delta_east, delta_north)) % 360.0)
    elapsed_s = abs(delta_time_ms) / 1000.0
    if elapsed_s > 0.0:
        horizontal_speed = horizontal / elapsed_s
        speed_3d = distance_3d / elapsed_s
    else:
        horizontal_speed = None
        speed_3d = None
    return PrecisionDifference(
        delta_time_ms=delta_time_ms,
        delta_east_km=delta_east,
        delta_north_km=delta_north,
        delta_altitude_km=delta_altitude,
        horizontal_distance_km=horizontal,
        distance_3d_km=distance_3d,
        bearing_deg=bearing,
        apparent_horizontal_speed_km_s=horizontal_speed,
        apparent_3d_speed_km_s=speed_3d,
    )



def format_apparent_speed(value_km_s: float | None) -> str:
    """Format an apparent cursor speed in metres per second.

    Scientific notation is used for the large values common in lightning
    propagation measurements, while moderate speeds remain easy to scan.
    """

    if value_km_s is None:
        return "—"
    try:
        value_m_s = float(value_km_s) * 1000.0
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(value_m_s):
        return "—"
    magnitude = abs(value_m_s)
    if magnitude >= 1_000.0 or (0.0 < magnitude < 0.01):
        return f"{value_m_s:.3e} m s⁻¹"
    if magnitude >= 100.0:
        return f"{value_m_s:.1f} m s⁻¹"
    return f"{value_m_s:.2f} m s⁻¹"

def utc_text(value: np.datetime64) -> str:
    """Format a nanosecond UTC timestamp without a timezone ambiguity."""

    timestamp = np.datetime64(value, "ns")
    if np.isnat(timestamp):
        return "—"
    text = np.datetime_as_string(timestamp, unit="us")
    return text.replace("T", " ") + " UTC"


@dataclass(frozen=True)
class PrecisionCoordinateDifference:
    """B-minus-A measurements for free or source-backed cursor coordinates.

    Every field is optional because a free cursor may initially define only
    the two coordinates represented by the panel on which it was placed.
    """

    delta_time_ms: float | None = None
    delta_east_km: float | None = None
    delta_north_km: float | None = None
    delta_altitude_km: float | None = None
    horizontal_distance_km: float | None = None
    distance_3d_km: float | None = None
    bearing_deg: float | None = None
    apparent_horizontal_speed_km_s: float | None = None
    apparent_3d_speed_km_s: float | None = None


def canonical_coordinate_name(name: str) -> tuple[str, float]:
    """Return canonical dimension name and display-to-canonical sign."""

    value = str(name).strip().casefold()
    mapping = {
        "west": ("east", -1.0),
        "east": ("east", 1.0),
        "south": ("north", -1.0),
        "north": ("north", 1.0),
        "longitude": ("longitude", 1.0),
        "latitude": ("latitude", 1.0),
        "altitude": ("altitude", 1.0),
        "time": ("time", 1.0),
    }
    return mapping.get(value, (value, 1.0))


def to_canonical_coordinate(name: str, value: float) -> tuple[str, float]:
    canonical, sign = canonical_coordinate_name(name)
    return canonical, float(value) * sign


def from_canonical_coordinate(name: str, value: float) -> float:
    _, sign = canonical_coordinate_name(name)
    return float(value) / sign


def precision_coordinate_difference(
    a: Mapping[str, float], b: Mapping[str, float]
) -> PrecisionCoordinateDifference:
    """Return differential measurements for potentially partial cursors.

    ``time`` is represented in Matplotlib date-number days. Horizontal
    coordinates may be local ``east``/``north`` kilometres or geographic
    ``longitude``/``latitude`` degrees. Geographic differences use a local
    tangent approximation with wrapped longitude, which is appropriate for
    cursor-scale measurements and remains well behaved across ±180°.
    """

    def finite(mapping: Mapping[str, float], key: str) -> float | None:
        try:
            value = float(mapping[key])
        except (KeyError, TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    ta, tb = finite(a, "time"), finite(b, "time")
    za, zb = finite(a, "altitude"), finite(b, "altitude")
    delta_time_ms = None if ta is None or tb is None else (tb - ta) * 86_400_000.0
    delta_altitude = None if za is None or zb is None else zb - za

    ea, eb = finite(a, "east"), finite(b, "east")
    na, nb = finite(a, "north"), finite(b, "north")
    delta_east = None if ea is None or eb is None else eb - ea
    delta_north = None if na is None or nb is None else nb - na

    if delta_east is None or delta_north is None:
        lona, lonb = finite(a, "longitude"), finite(b, "longitude")
        lata, latb = finite(a, "latitude"), finite(b, "latitude")
        if None not in (lona, lonb, lata, latb):
            dlon = ((float(lonb) - float(lona) + 180.0) % 360.0) - 180.0
            mean_lat = math.radians((float(lata) + float(latb)) * 0.5)
            delta_east = dlon * 111.195 * math.cos(mean_lat)
            delta_north = (float(latb) - float(lata)) * 111.195

    horizontal = (
        None
        if delta_east is None or delta_north is None
        else float(math.hypot(delta_east, delta_north))
    )
    bearing = (
        None
        if delta_east is None or delta_north is None
        else float(math.degrees(math.atan2(delta_east, delta_north)) % 360.0)
    )
    distance_3d = (
        None
        if horizontal is None or delta_altitude is None
        else float(math.hypot(horizontal, delta_altitude))
    )

    elapsed_s = None if delta_time_ms is None else abs(delta_time_ms) / 1000.0
    horizontal_speed = (
        None
        if elapsed_s is None or elapsed_s <= 0.0 or horizontal is None
        else horizontal / elapsed_s
    )
    speed_3d = (
        None
        if elapsed_s is None or elapsed_s <= 0.0 or distance_3d is None
        else distance_3d / elapsed_s
    )
    return PrecisionCoordinateDifference(
        delta_time_ms=delta_time_ms,
        delta_east_km=delta_east,
        delta_north_km=delta_north,
        delta_altitude_km=delta_altitude,
        horizontal_distance_km=horizontal,
        distance_3d_km=distance_3d,
        bearing_deg=bearing,
        apparent_horizontal_speed_km_s=horizontal_speed,
        apparent_3d_speed_km_s=speed_3d,
    )


__all__ = [
    "PrecisionCoordinateDifference",
    "PrecisionDifference",
    "PrecisionSource",
    "canonical_coordinate_name",
    "format_apparent_speed",
    "from_canonical_coordinate",
    "precision_coordinate_difference",
    "precision_difference",
    "to_canonical_coordinate",
    "utc_text",
]

