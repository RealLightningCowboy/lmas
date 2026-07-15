from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .coordinates import station_local_coordinates
from .model import LMAProject


@dataclass(frozen=True)
class StationRecord:
    index: int
    code: str
    latitude: float
    longitude: float
    altitude_km: float
    east_km: float
    north_km: float


@dataclass(frozen=True)
class BaselineRecord:
    station_a: str
    station_b: str
    horizontal_length_km: float
    three_d_length_km: float
    azimuth_deg: float


@dataclass(frozen=True)
class BaselineStatistics:
    count: int
    minimum_km: float
    first_quartile_km: float
    median_km: float
    mean_km: float
    third_quartile_km: float
    maximum_km: float
    standard_deviation_km: float


def _statistics(values) -> BaselineStatistics | None:
    finite = np.asarray(tuple(values), dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    q1, median, q3 = np.percentile(finite, [25.0, 50.0, 75.0])
    return BaselineStatistics(
        count=int(finite.size),
        minimum_km=float(np.min(finite)),
        first_quartile_km=float(q1),
        median_km=float(median),
        mean_km=float(np.mean(finite)),
        third_quartile_km=float(q3),
        maximum_km=float(np.max(finite)),
        standard_deviation_km=float(np.std(finite, ddof=0)),
    )


@dataclass(frozen=True)
class ArrayInformation:
    network_name: str
    reference_latitude: float
    reference_longitude: float
    stations: tuple[StationRecord, ...]
    baselines: tuple[BaselineRecord, ...]

    @property
    def longest_horizontal_baseline_km(self) -> float | None:
        if not self.baselines:
            return None
        return max(item.horizontal_length_km for item in self.baselines)

    @property
    def horizontal_baseline_statistics(self) -> BaselineStatistics | None:
        return _statistics(item.horizontal_length_km for item in self.baselines)

    @property
    def three_d_baseline_statistics(self) -> BaselineStatistics | None:
        return _statistics(item.three_d_length_km for item in self.baselines)


def _station_altitude_km(project: LMAProject, count: int) -> np.ndarray:
    dataset = project.dataset
    if "station_altitude" not in dataset:
        return np.full(count, np.nan, dtype=float)
    values = np.asarray(dataset["station_altitude"].values, dtype=float).reshape(-1)
    units = str(dataset["station_altitude"].attrs.get("units", "")).strip().lower()
    if units in {"m", "meter", "meters", "metre", "metres"}:
        values = values / 1000.0
    elif units not in {"km", "kilometer", "kilometers", "kilometre", "kilometres"}:
        finite = np.abs(values[np.isfinite(values)])
        if finite.size and np.nanmedian(finite) > 100.0:
            values = values / 1000.0
    result = np.full(count, np.nan, dtype=float)
    result[: min(count, values.size)] = values[:count]
    return result


def _station_codes(project: LMAProject, count: int) -> np.ndarray:
    dataset = project.dataset
    for field in ("station_code", "station_name", "station_id"):
        if field in dataset:
            values = np.asarray(dataset[field].values).reshape(-1).astype(str)
            result = np.asarray([f"S{index:02d}" for index in range(count)], dtype=object)
            for index, value in enumerate(values[:count]):
                cleaned = str(value).strip()
                if cleaned:
                    result[index] = cleaned
            return result
    return np.asarray([f"S{index:02d}" for index in range(count)], dtype=object)


def build_array_information(project: LMAProject) -> ArrayInformation:
    """Return station and baseline geometry for the loaded LMA array.

    Baseline azimuth is measured clockwise from north from station A toward
    station B. Only station pairs with finite horizontal coordinates are
    included in the baseline table.
    """

    dataset = project.dataset
    if "station_latitude" not in dataset or "station_longitude" not in dataset:
        return ArrayInformation(
            network_name=str(dataset.attrs.get("network_name") or project.name),
            reference_latitude=float(project.reference_latitude),
            reference_longitude=float(project.reference_longitude),
            stations=(),
            baselines=(),
        )

    latitude = np.asarray(dataset["station_latitude"].values, dtype=float).reshape(-1)
    longitude = np.asarray(dataset["station_longitude"].values, dtype=float).reshape(-1)
    count = min(latitude.size, longitude.size)
    latitude = latitude[:count]
    longitude = longitude[:count]
    altitude = _station_altitude_km(project, count)
    codes = _station_codes(project, count)

    local = station_local_coordinates(
        dataset,
        float(project.reference_longitude),
        float(project.reference_latitude),
    )
    if local is None:
        east = np.full(count, np.nan, dtype=float)
        north = np.full(count, np.nan, dtype=float)
    else:
        east_values, north_values = local
        east = np.full(count, np.nan, dtype=float)
        north = np.full(count, np.nan, dtype=float)
        east_raw = np.asarray(east_values, dtype=float).reshape(-1)
        north_raw = np.asarray(north_values, dtype=float).reshape(-1)
        east[: min(count, east_raw.size)] = east_raw[:count]
        north[: min(count, north_raw.size)] = north_raw[:count]

    stations = tuple(
        StationRecord(
            index=index,
            code=str(codes[index]),
            latitude=float(latitude[index]),
            longitude=float(longitude[index]),
            altitude_km=float(altitude[index]),
            east_km=float(east[index]),
            north_km=float(north[index]),
        )
        for index in range(count)
    )

    baselines: list[BaselineRecord] = []
    for left, right in combinations(stations, 2):
        if not all(
            np.isfinite(value)
            for value in (left.east_km, left.north_km, right.east_km, right.north_km)
        ):
            continue
        delta_east = right.east_km - left.east_km
        delta_north = right.north_km - left.north_km
        horizontal = float(np.hypot(delta_east, delta_north))
        if np.isfinite(left.altitude_km) and np.isfinite(right.altitude_km):
            three_d = float(np.hypot(horizontal, right.altitude_km - left.altitude_km))
        else:
            three_d = float("nan")
        azimuth = float((np.degrees(np.arctan2(delta_east, delta_north)) + 360.0) % 360.0)
        baselines.append(
            BaselineRecord(
                station_a=left.code,
                station_b=right.code,
                horizontal_length_km=horizontal,
                three_d_length_km=three_d,
                azimuth_deg=azimuth,
            )
        )

    return ArrayInformation(
        network_name=str(dataset.attrs.get("network_name") or project.name),
        reference_latitude=float(project.reference_latitude),
        reference_longitude=float(project.reference_longitude),
        stations=stations,
        baselines=tuple(baselines),
    )


__all__ = [
    "ArrayInformation",
    "BaselineStatistics",
    "BaselineRecord",
    "StationRecord",
    "build_array_information",
]
