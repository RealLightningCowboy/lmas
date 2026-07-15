from __future__ import annotations

import numpy as np
import xarray as xr

EARTH_RADIUS_KM = 6371.0088


def altitude_values_km(values: np.ndarray, units: str = "") -> np.ndarray:
    result = np.asarray(values, dtype=float)
    normalized_units = str(units).strip().lower()
    if normalized_units in {"m", "meter", "meters", "metre", "metres"}:
        return result / 1000.0
    if normalized_units in {"km", "kilometer", "kilometers", "kilometre", "kilometres"}:
        return result
    finite = np.abs(result[np.isfinite(result)])
    if finite.size and np.nanmedian(finite) > 100.0:
        return result / 1000.0
    return result


def altitude_km(dataset: xr.Dataset) -> np.ndarray:
    return altitude_values_km(
        dataset["event_altitude"].values,
        str(dataset["event_altitude"].attrs.get("units", "")),
    )


def latlon_to_local_km(
    longitude: np.ndarray,
    latitude: np.ndarray,
    reference_longitude: float,
    reference_latitude: float,
) -> tuple[np.ndarray, np.ndarray]:
    lon = np.asarray(longitude, dtype=float)
    lat = np.asarray(latitude, dtype=float)
    ref_lon = float(reference_longitude)
    ref_lat = float(reference_latitude)
    x = EARTH_RADIUS_KM * np.cos(np.deg2rad(ref_lat)) * np.deg2rad(lon - ref_lon)
    y = EARTH_RADIUS_KM * np.deg2rad(lat - ref_lat)
    return x, y


def event_local_coordinates(dataset: xr.Dataset, reference_longitude: float, reference_latitude: float) -> tuple[np.ndarray, np.ndarray]:
    return latlon_to_local_km(
        dataset["event_longitude"].values,
        dataset["event_latitude"].values,
        reference_longitude,
        reference_latitude,
    )


def station_local_coordinates(dataset: xr.Dataset, reference_longitude: float, reference_latitude: float) -> tuple[np.ndarray, np.ndarray] | None:
    if "station_longitude" not in dataset or "station_latitude" not in dataset:
        return None
    return latlon_to_local_km(
        dataset["station_longitude"].values,
        dataset["station_latitude"].values,
        reference_longitude,
        reference_latitude,
    )

def station_center_latlon(dataset: xr.Dataset) -> tuple[float, float] | None:
    """Return the finite arithmetic mean station longitude/latitude."""
    if "station_longitude" not in dataset or "station_latitude" not in dataset:
        return None
    lon = np.asarray(dataset["station_longitude"].values, dtype=float)
    lat = np.asarray(dataset["station_latitude"].values, dtype=float)
    valid = np.isfinite(lon) & np.isfinite(lat)
    if not np.any(valid):
        return None
    return float(np.mean(lon[valid])), float(np.mean(lat[valid]))


def station_center_local_km(
    dataset: xr.Dataset,
    reference_longitude: float,
    reference_latitude: float,
) -> tuple[float, float] | None:
    """Return the finite arithmetic mean station location in local km."""
    station = station_local_coordinates(dataset, reference_longitude, reference_latitude)
    if station is None:
        return None
    east, north = (np.asarray(values, dtype=float) for values in station)
    valid = np.isfinite(east) & np.isfinite(north)
    if not np.any(valid):
        return None
    return float(np.mean(east[valid])), float(np.mean(north[valid]))

