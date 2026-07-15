from __future__ import annotations

import numpy as np
import xarray as xr

from .coordinates import altitude_values_km, latlon_to_local_km
from .errors import DatasetError
from .model import FilterSpec, LMAProject
from .source_store import LmaSourceStore


def _require_field(store: LmaSourceStore, field: str, purpose: str) -> np.ndarray:
    if field not in store:
        raise DatasetError(f"Cannot apply {purpose}: dataset has no {field} field")
    return store.event_array(field)


def event_selection_mask(project: LMAProject, filters: FilterSpec) -> np.ndarray:
    spec = filters.validated()
    store = project.source_store
    mask = np.ones(store.event_count, dtype=bool)
    times = np.asarray(store.event_array("event_time")).astype("datetime64[ns]")
    mask &= ~np.isnat(times)
    if spec.start_time is not None:
        mask &= times >= np.datetime64(spec.start_time, "ns")
    if spec.end_time is not None:
        mask &= times <= np.datetime64(spec.end_time, "ns")

    alt = altitude_values_km(
        store.event_array("event_altitude"),
        str(store.field_attrs("event_altitude").get("units", "")),
    )
    mask &= np.isfinite(alt)
    if spec.minimum_altitude_km is not None:
        mask &= alt >= spec.minimum_altitude_km
    if spec.maximum_altitude_km is not None:
        mask &= alt <= spec.maximum_altitude_km

    if spec.minimum_stations is not None:
        values = np.asarray(
            _require_field(store, "event_stations", "minimum-station filtering"),
            dtype=float,
        )
        mask &= np.isfinite(values) & (values >= spec.minimum_stations)
    if spec.maximum_chi2 is not None:
        values = np.asarray(
            _require_field(store, "event_chi2", "chi-squared filtering"),
            dtype=float,
        )
        mask &= np.isfinite(values) & (values < spec.maximum_chi2)
    if spec.minimum_power is not None or spec.maximum_power is not None:
        values = np.asarray(
            _require_field(store, "event_power", "power filtering"), dtype=float
        )
        mask &= np.isfinite(values)
        if spec.minimum_power is not None:
            mask &= values >= spec.minimum_power
        if spec.maximum_power is not None:
            mask &= values <= spec.maximum_power

    if any(
        value is not None
        for value in (
            spec.minimum_x_km,
            spec.maximum_x_km,
            spec.minimum_y_km,
            spec.maximum_y_km,
        )
    ):
        x, y = latlon_to_local_km(
            store.event_array("event_longitude"),
            store.event_array("event_latitude"),
            project.reference_longitude,
            project.reference_latitude,
        )
        mask &= np.isfinite(x) & np.isfinite(y)
        if spec.minimum_x_km is not None:
            mask &= x >= spec.minimum_x_km
        if spec.maximum_x_km is not None:
            mask &= x <= spec.maximum_x_km
        if spec.minimum_y_km is not None:
            mask &= y >= spec.minimum_y_km
        if spec.maximum_y_km is not None:
            mask &= y <= spec.maximum_y_km
    return mask


def select_event_store(project: LMAProject, filters: FilterSpec) -> LmaSourceStore:
    return project.source_store.select_events(event_selection_mask(project, filters))


def select_events(project: LMAProject, filters: FilterSpec) -> xr.Dataset:
    """Return the selected xarray compatibility view."""

    return select_event_store(project, filters).to_xarray()


__all__ = ["event_selection_mask", "select_event_store", "select_events"]
