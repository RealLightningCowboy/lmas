from __future__ import annotations

import numpy as np

from lmas.precision import PrecisionSource, format_apparent_speed, precision_difference, utc_text


def _source(index: int, time: str, east: float, north: float, altitude: float) -> PrecisionSource:
    return PrecisionSource(
        index=index,
        source_id=100 + index,
        time=np.datetime64(time, "ns"),
        latitude=34.0,
        longitude=-106.0,
        altitude_km=altitude,
        east_km=east,
        north_km=north,
    )


def test_precision_difference_b_minus_a() -> None:
    a = _source(0, "2026-07-06T21:18:39.000000", 1.0, 2.0, 3.0)
    b = _source(1, "2026-07-06T21:18:39.010000", 4.0, 6.0, 15.0)
    result = precision_difference(a, b)
    assert result.delta_time_ms == 10.0
    assert result.delta_east_km == 3.0
    assert result.delta_north_km == 4.0
    assert result.delta_altitude_km == 12.0
    assert result.horizontal_distance_km == 5.0
    assert result.distance_3d_km == 13.0
    assert round(result.bearing_deg, 6) == round(np.degrees(np.arctan2(3.0, 4.0)), 6)
    assert result.apparent_horizontal_speed_km_s == 500.0
    assert result.apparent_3d_speed_km_s == 1300.0


def test_precision_source_from_metadata() -> None:
    values = {
        "time": np.array([np.datetime64("2026-07-06T21:18:39", "ns")]),
        "source_id": np.array([42]),
        "latitude": np.array([34.1]),
        "longitude": np.array([-106.2]),
        "altitude_km": np.array([7.5]),
        "east_km": np.array([1.2]),
        "north_km": np.array([-0.8]),
        "power": np.array([12.5]),
        "chi2": np.array([0.4]),
        "stations": np.array([9]),
    }
    source = PrecisionSource.from_values(values, 0)
    assert source.source_id == 42
    assert source.power_dbw == 12.5
    assert source.chi2 == 0.4
    assert source.stations == 9
    assert utc_text(source.time).endswith(" UTC")

from lmas.precision import (
    canonical_coordinate_name,
    precision_coordinate_difference,
    to_canonical_coordinate,
)


def test_precision_free_coordinate_difference_local() -> None:
    result = precision_coordinate_difference(
        {"time": 100.0, "east": 1.0, "north": 2.0, "altitude": 3.0},
        {"time": 100.0 + 0.010 / 86400.0, "east": 4.0, "north": 6.0, "altitude": 15.0},
    )
    assert round(result.delta_time_ms, 4) == 10.0
    assert result.delta_east_km == 3.0
    assert result.delta_north_km == 4.0
    assert result.horizontal_distance_km == 5.0
    assert result.distance_3d_km == 13.0


def test_precision_coordinate_viewpoint_mapping() -> None:
    assert canonical_coordinate_name("west") == ("east", -1.0)
    assert to_canonical_coordinate("west", 5.0) == ("east", -5.0)
    assert canonical_coordinate_name("south") == ("north", -1.0)


def test_apparent_speed_uses_metres_per_second_scientific_notation() -> None:
    assert format_apparent_speed(500.0) == "5.000e+05 m s⁻¹"
    assert format_apparent_speed(0.5) == "500.0 m s⁻¹"
    assert format_apparent_speed(None) == "—"
