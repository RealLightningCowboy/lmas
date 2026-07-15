"""NumPy-first ground lightning-location-network observations.

The normalized object deliberately contains no LMA source-group or GLM group
semantics.  Each row is one independent report from a ground-based lightning
location system such as ENTLN, NLDN, GLD360, or a generic event CSV.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


NAT_NS = np.iinfo(np.int64).min


@dataclass(frozen=True, slots=True)
class NetworkSourceFile:
    path: Path
    row_count: int


@dataclass(frozen=True, slots=True)
class NetworkIdentity:
    provider_id: str
    display_name: str
    product_name: str
    observation_start_ns: int
    observation_end_ns: int
    source_files: tuple[NetworkSourceFile, ...]
    reader_name: str = "LMAS native network CSV"
    reader_version: str = "1"
    schema: Mapping[str, str] = field(default_factory=dict)

    @property
    def observation_start(self) -> np.datetime64:
        return np.datetime64(int(self.observation_start_ns), "ns")

    @property
    def observation_end(self) -> np.datetime64:
        return np.datetime64(int(self.observation_end_ns), "ns")


@dataclass(frozen=True, slots=True)
class NetworkEvents:
    time_ns: np.ndarray
    longitude_deg: np.ndarray
    latitude_deg: np.ndarray
    altitude_m: np.ndarray
    event_type: np.ndarray
    original_event_type: np.ndarray
    polarity: np.ndarray
    peak_current_ka: np.ndarray
    amplitude: np.ndarray
    sensor_count: np.ndarray
    quality: np.ndarray
    ellipse_major_km: np.ndarray
    ellipse_minor_km: np.ndarray
    ellipse_angle_deg: np.ndarray
    original_id: np.ndarray

    def __len__(self) -> int:
        return int(self.time_ns.size)

    def validate(self) -> "NetworkEvents":
        n = len(self)
        for name in self.__dataclass_fields__:
            value = np.asarray(getattr(self, name))
            if value.ndim != 1 or value.size != n:
                raise ValueError(f"Network field {name!r} must be one-dimensional with {n} rows")
        return self


@dataclass(frozen=True, slots=True)
class NetworkSelection:
    event_indices: np.ndarray

    def __len__(self) -> int:
        return int(self.event_indices.size)


@dataclass(slots=True)
class NetworkObservation:
    identity: NetworkIdentity
    events: NetworkEvents
    _time_order: np.ndarray = field(init=False, repr=False)
    _sorted_time_ns: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.events.validate()
        times = np.asarray(self.events.time_ns, dtype=np.int64)
        valid = times != NAT_NS
        order = np.flatnonzero(valid)
        if order.size:
            order = order[np.argsort(times[order], kind="stable")]
        self._time_order = order.astype(np.int64, copy=False)
        self._sorted_time_ns = times[self._time_order]

    def select(
        self,
        *,
        time_range_ns: tuple[int, int] | None = None,
        geographic_bounds: tuple[float, float, float, float] | None = None,
        event_types: Sequence[str] | None = None,
        polarities: Sequence[int] | None = None,
        minimum_absolute_peak_current_ka: float | None = None,
        minimum_sensor_count: int | None = None,
        quality_values: Sequence[str] | None = None,
    ) -> NetworkSelection:
        n = len(self.events)
        if time_range_ns is None:
            indices = np.arange(n, dtype=np.int64)
        else:
            low, high = sorted((int(time_range_ns[0]), int(time_range_ns[1])))
            left = int(np.searchsorted(self._sorted_time_ns, low, side="left"))
            right = int(np.searchsorted(self._sorted_time_ns, high, side="right"))
            indices = self._time_order[left:right]
        if not indices.size:
            return NetworkSelection(indices)

        keep = np.ones(indices.size, dtype=bool)
        events = self.events
        if geographic_bounds is not None:
            west, east, south, north = map(float, geographic_bounds)
            lon = events.longitude_deg[indices]
            lat = events.latitude_deg[indices]
            keep &= np.isfinite(lon) & np.isfinite(lat)
            keep &= (lon >= min(west, east)) & (lon <= max(west, east))
            keep &= (lat >= min(south, north)) & (lat <= max(south, north))
        if event_types:
            allowed = np.asarray([str(value).upper() for value in event_types], dtype="U24")
            keep &= np.isin(np.char.upper(events.event_type[indices].astype("U24")), allowed)
        if polarities:
            keep &= np.isin(events.polarity[indices], np.asarray(tuple(polarities), dtype=np.int8))
        if minimum_absolute_peak_current_ka is not None:
            current = np.abs(events.peak_current_ka[indices])
            keep &= np.isfinite(current) & (current >= float(minimum_absolute_peak_current_ka))
        if minimum_sensor_count is not None:
            sensors = events.sensor_count[indices]
            keep &= (sensors >= int(minimum_sensor_count))
        if quality_values:
            allowed_quality = np.asarray([str(value).upper() for value in quality_values], dtype="U32")
            keep &= np.isin(np.char.upper(events.quality[indices].astype("U32")), allowed_quality)
        return NetworkSelection(indices[keep])

    def subset(self, indices: np.ndarray) -> "NetworkObservation":
        indices = np.asarray(indices, dtype=np.int64)
        values = {
            name: np.asarray(getattr(self.events, name))[indices].copy()
            for name in self.events.__dataclass_fields__
        }
        events = NetworkEvents(**values)
        finite_time = events.time_ns[events.time_ns != NAT_NS]
        start = int(np.min(finite_time)) if finite_time.size else NAT_NS
        end = int(np.max(finite_time)) if finite_time.size else NAT_NS
        identity = NetworkIdentity(
            provider_id=self.identity.provider_id,
            display_name=self.identity.display_name,
            product_name=self.identity.product_name,
            observation_start_ns=start,
            observation_end_ns=end,
            source_files=self.identity.source_files,
            reader_name=self.identity.reader_name,
            reader_version=self.identity.reader_version,
            schema=dict(self.identity.schema),
        )
        return NetworkObservation(identity, events)


__all__ = [
    "NAT_NS",
    "NetworkEvents",
    "NetworkIdentity",
    "NetworkObservation",
    "NetworkSelection",
    "NetworkSourceFile",
]
