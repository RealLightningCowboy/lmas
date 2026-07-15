"""Xarray and glmtools interoperability for the native GLM object."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .model import (
    GLMDataError,
    GLMDatasetIdentity,
    GLMEventTable,
    GLMFlashTable,
    GLMGroupTable,
    GLMObservation,
    GLMProjectionMetadata,
    GLMSourceFile,
)


def to_glmtools_compatible_xarray(
    observation: GLMObservation,
    *,
    glmtools_units: bool = True,
):
    """Return an xarray dataset using LCFA/glmtools names and hierarchy."""
    import xarray as xr

    ev = observation.events
    gr = observation.groups
    fl = observation.flashes
    origin_ns = observation.identity.observation_start_ns
    origin = np.datetime64(origin_ns, "ns")

    event_energy = ev.energy_j * (1.0e9 if glmtools_units else 1.0)
    group_energy = gr.energy_j * (1.0e9 if glmtools_units else 1.0)
    flash_energy = fl.energy_j * (1.0e9 if glmtools_units else 1.0)
    group_area = gr.area_m2 * (1.0e-6 if glmtools_units else 1.0)
    flash_area = fl.area_m2 * (1.0e-6 if glmtools_units else 1.0)
    energy_units = "nJ" if glmtools_units else "J"
    area_units = "km2" if glmtools_units else "m2"

    event_parent_flash_id = np.zeros(len(ev), dtype=np.uint64)
    event_parent_flash_id[:] = np.iinfo(np.uint64).max
    valid_event_flash = ev.parent_flash_index >= 0
    event_parent_flash_id[valid_event_flash] = fl.flash_id[ev.parent_flash_index[valid_event_flash]]

    coords = {
        "flash_id": ("number_of_flashes", fl.flash_id),
        "group_id": ("number_of_groups", gr.group_id),
        "group_parent_flash_id": ("number_of_groups", gr.parent_flash_id),
        "event_id": ("number_of_events", ev.event_id),
        "event_parent_group_id": ("number_of_events", ev.parent_group_id),
        "event_parent_flash_id": ("number_of_events", event_parent_flash_id),
    }
    data_vars = {
        "event_time_offset": (
            "number_of_events",
            (ev.time_ns - origin_ns).astype("timedelta64[ns]"),
        ),
        "event_lat": ("number_of_events", ev.latitude_deg),
        "event_lon": ("number_of_events", ev.longitude_deg),
        "event_energy": ("number_of_events", event_energy),
        "event_source_file_index": ("number_of_events", ev.source_file_index),
        "group_time_offset": (
            "number_of_groups",
            (gr.time_ns - origin_ns).astype("timedelta64[ns]"),
        ),
        "group_frame_time_offset": (
            "number_of_groups",
            (gr.frame_time_ns - origin_ns).astype("timedelta64[ns]"),
        ),
        "group_lat": ("number_of_groups", gr.latitude_deg),
        "group_lon": ("number_of_groups", gr.longitude_deg),
        "group_area": ("number_of_groups", group_area),
        "group_energy": ("number_of_groups", group_energy),
        "group_quality_flag": ("number_of_groups", gr.quality_flag),
        "group_child_event_count": ("number_of_groups", gr.child_event_count),
        "group_source_file_index": ("number_of_groups", gr.source_file_index),
        "flash_time_offset_of_first_event": (
            "number_of_flashes",
            (fl.first_event_time_ns - origin_ns).astype("timedelta64[ns]"),
        ),
        "flash_time_offset_of_last_event": (
            "number_of_flashes",
            (fl.last_event_time_ns - origin_ns).astype("timedelta64[ns]"),
        ),
        "flash_frame_time_offset_of_first_event": (
            "number_of_flashes",
            (fl.first_frame_time_ns - origin_ns).astype("timedelta64[ns]"),
        ),
        "flash_frame_time_offset_of_last_event": (
            "number_of_flashes",
            (fl.last_frame_time_ns - origin_ns).astype("timedelta64[ns]"),
        ),
        "flash_lat": ("number_of_flashes", fl.latitude_deg),
        "flash_lon": ("number_of_flashes", fl.longitude_deg),
        "flash_area": ("number_of_flashes", flash_area),
        "flash_energy": ("number_of_flashes", flash_energy),
        "flash_quality_flag": ("number_of_flashes", fl.quality_flag),
        "flash_child_group_count": ("number_of_flashes", fl.child_group_count),
        "flash_child_event_count": ("number_of_flashes", fl.child_event_count),
        "flash_source_file_index": ("number_of_flashes", fl.source_file_index),
        "product_time": origin,
        "product_time_bounds": (
            "number_of_time_bounds",
            np.asarray(
                [
                    np.datetime64(observation.identity.observation_start_ns, "ns"),
                    np.datetime64(observation.identity.observation_end_ns, "ns"),
                ],
                dtype="datetime64[ns]",
            ),
        ),
        "nominal_satellite_subpoint_lat": _or_nan(
            observation.identity.projection.nominal_subpoint_lat_deg
        ),
        "nominal_satellite_subpoint_lon": _or_nan(
            observation.identity.projection.nominal_subpoint_lon_deg
        ),
        "nominal_satellite_height": _or_nan(
            observation.identity.projection.nominal_height_km
        ),
        "lat_field_of_view": _or_nan(
            observation.identity.projection.field_of_view_lat_deg
        ),
        "lon_field_of_view": _or_nan(
            observation.identity.projection.field_of_view_lon_deg
            if observation.identity.projection.field_of_view_lon_deg is not None
            else observation.identity.projection.nominal_subpoint_lon_deg
        ),
    }
    attrs = {
        "title": "GLM L2 Lightning Detections: Events, Groups, and Flashes",
        "instrument_family": observation.identity.instrument_family,
        "platform_ID": observation.identity.platform_id,
        "orbital_slot": f"GOES-{observation.identity.operational_role.title()}",
        "operational_role_source": observation.identity.operational_role_source,
        "processing_level": observation.identity.product_level,
        "time_coverage_start": str(observation.identity.observation_start),
        "time_coverage_end": str(observation.identity.observation_end),
        "source_files": tuple(str(record.path) for record in observation.identity.source_files),
        "lmas_native_glm_schema": "1.0",
    }
    for key, value in observation.identity.attributes.items():
        if key not in attrs and _xarray_safe_attr(value):
            attrs[key] = value

    dataset = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
    for name in ("event_time_offset", "group_time_offset", "group_frame_time_offset"):
        dataset[name].attrs["units"] = f"seconds since {str(origin).replace('T', ' ')}"
    for name in (
        "flash_time_offset_of_first_event",
        "flash_time_offset_of_last_event",
        "flash_frame_time_offset_of_first_event",
        "flash_frame_time_offset_of_last_event",
    ):
        dataset[name].attrs["units"] = f"seconds since {str(origin).replace('T', ' ')}"
    for name in ("event_energy", "group_energy", "flash_energy"):
        dataset[name].attrs["units"] = energy_units
    for name in ("group_area", "flash_area"):
        dataset[name].attrs["units"] = area_units
    dataset["event_lat"].attrs["units"] = "degrees_north"
    dataset["event_lon"].attrs["units"] = "degrees_east"
    dataset["group_lat"].attrs["units"] = "degrees_north"
    dataset["group_lon"].attrs["units"] = "degrees_east"
    dataset["flash_lat"].attrs["units"] = "degrees_north"
    dataset["flash_lon"].attrs["units"] = "degrees_east"
    return dataset


def from_xarray(dataset) -> GLMObservation:
    """Create the native object from an LCFA/glmtools-style xarray dataset."""
    required = (
        "event_id",
        "event_time_offset",
        "event_lat",
        "event_lon",
        "event_energy",
        "event_parent_group_id",
        "group_id",
        "group_time_offset",
        "group_lat",
        "group_lon",
        "group_area",
        "group_energy",
        "group_parent_flash_id",
        "flash_id",
        "flash_time_offset_of_first_event",
        "flash_time_offset_of_last_event",
        "flash_lat",
        "flash_lon",
        "flash_area",
        "flash_energy",
    )
    missing = [name for name in required if name not in dataset]
    if missing:
        raise GLMDataError(f"Xarray dataset is missing GLM variables: {missing}")

    origin_ns = _dataset_origin_ns(dataset)
    event_id = _values(dataset, "event_id", np.uint64)
    group_id = _values(dataset, "group_id", np.uint64)
    flash_id = _values(dataset, "flash_id", np.uint64)
    event_parent_group_id = _values(dataset, "event_parent_group_id", np.uint64)
    group_parent_flash_id = _values(dataset, "group_parent_flash_id", np.uint64)
    parent_group_index = _map_ids(event_parent_group_id, group_id)
    parent_flash_index = _map_ids(group_parent_flash_id, flash_id)
    event_parent_flash_index = np.full(event_id.size, -1, dtype=np.int64)
    valid = parent_group_index >= 0
    event_parent_flash_index[valid] = parent_flash_index[parent_group_index[valid]]

    event_time_ns = _time_values_ns(dataset["event_time_offset"], origin_ns)
    group_time_ns = _time_values_ns(dataset["group_time_offset"], origin_ns)
    group_frame_time_ns = (
        _time_values_ns(dataset["group_frame_time_offset"], origin_ns)
        if "group_frame_time_offset" in dataset
        else group_time_ns.copy()
    )
    first_event_time_ns = _time_values_ns(
        dataset["flash_time_offset_of_first_event"], origin_ns
    )
    last_event_time_ns = _time_values_ns(
        dataset["flash_time_offset_of_last_event"], origin_ns
    )
    first_frame_time_ns = (
        _time_values_ns(dataset["flash_frame_time_offset_of_first_event"], origin_ns)
        if "flash_frame_time_offset_of_first_event" in dataset
        else first_event_time_ns.copy()
    )
    last_frame_time_ns = (
        _time_values_ns(dataset["flash_frame_time_offset_of_last_event"], origin_ns)
        if "flash_frame_time_offset_of_last_event" in dataset
        else last_event_time_ns.copy()
    )

    events = GLMEventTable(
        event_id=event_id,
        source_file_index=_optional_values(dataset, "event_source_file_index", event_id.size, np.int32),
        time_ns=event_time_ns,
        latitude_deg=_values(dataset, "event_lat", np.float64),
        longitude_deg=_values(dataset, "event_lon", np.float64),
        energy_j=_convert_energy_to_j(dataset["event_energy"]),
        parent_group_id=event_parent_group_id,
        parent_group_index=parent_group_index,
        parent_flash_index=event_parent_flash_index,
    )
    groups = GLMGroupTable(
        group_id=group_id,
        source_file_index=_optional_values(dataset, "group_source_file_index", group_id.size, np.int32),
        time_ns=group_time_ns,
        frame_time_ns=group_frame_time_ns,
        latitude_deg=_values(dataset, "group_lat", np.float64),
        longitude_deg=_values(dataset, "group_lon", np.float64),
        area_m2=_convert_area_to_m2(dataset["group_area"]),
        energy_j=_convert_energy_to_j(dataset["group_energy"]),
        parent_flash_id=group_parent_flash_id,
        parent_flash_index=parent_flash_index,
        quality_flag=_optional_values(dataset, "group_quality_flag", group_id.size, np.uint16),
        child_event_count=_safe_bincount(parent_group_index, group_id.size),
    )
    flashes = GLMFlashTable(
        flash_id=flash_id,
        source_file_index=_optional_values(dataset, "flash_source_file_index", flash_id.size, np.int32),
        first_event_time_ns=first_event_time_ns,
        last_event_time_ns=last_event_time_ns,
        first_frame_time_ns=first_frame_time_ns,
        last_frame_time_ns=last_frame_time_ns,
        latitude_deg=_values(dataset, "flash_lat", np.float64),
        longitude_deg=_values(dataset, "flash_lon", np.float64),
        area_m2=_convert_area_to_m2(dataset["flash_area"]),
        energy_j=_convert_energy_to_j(dataset["flash_energy"]),
        quality_flag=_optional_values(dataset, "flash_quality_flag", flash_id.size, np.uint16),
        child_group_count=_safe_bincount(parent_flash_index, flash_id.size),
        child_event_count=_safe_bincount(event_parent_flash_index, flash_id.size),
    )

    platform = str(dataset.attrs.get("platform_ID", dataset.attrs.get("platform_id", "UNKNOWN")))
    role = str(dataset.attrs.get("orbital_slot", dataset.attrs.get("operational_role", "unknown")))
    role = role.lower().replace("goes-", "").strip()
    projection = GLMProjectionMetadata(
        nominal_subpoint_lat_deg=_optional_scalar(dataset, "nominal_satellite_subpoint_lat"),
        nominal_subpoint_lon_deg=_optional_scalar(dataset, "nominal_satellite_subpoint_lon"),
        nominal_height_km=_optional_scalar(dataset, "nominal_satellite_height"),
        field_of_view_lat_deg=_optional_scalar(dataset, "lat_field_of_view"),
        field_of_view_lon_deg=_optional_scalar(dataset, "lon_field_of_view"),
    )
    source_paths = dataset.attrs.get("source_files", ())
    if isinstance(source_paths, str):
        source_paths = (source_paths,)
    source_files = tuple(
        GLMSourceFile(
            path=Path(path),
            dataset_name=Path(path).name,
            platform_id=platform,
            operational_role=role,
            time_coverage_start_ns=int(min(event_time_ns)) if event_time_ns.size else origin_ns,
            time_coverage_end_ns=int(max(event_time_ns)) if event_time_ns.size else origin_ns,
            event_count=int(np.count_nonzero(events.source_file_index == index)),
            group_count=int(np.count_nonzero(groups.source_file_index == index)),
            flash_count=int(np.count_nonzero(flashes.source_file_index == index)),
            file_size_bytes=0,
        )
        for index, path in enumerate(source_paths)
    )
    identity = GLMDatasetIdentity(
        instrument_family="GLM",
        platform_id=platform,
        operational_role=role,
        operational_role_source=str(dataset.attrs.get("operational_role_source", "xarray")),
        product_level=str(dataset.attrs.get("processing_level", "L2_LCFA")),
        observation_start_ns=int(min(event_time_ns)) if event_time_ns.size else origin_ns,
        observation_end_ns=int(max(event_time_ns)) if event_time_ns.size else origin_ns,
        projection=projection,
        source_files=source_files,
        attributes=dict(dataset.attrs),
    )
    observation = GLMObservation(identity, events, groups, flashes)
    observation.validate_hierarchy(raise_on_error=True)
    return observation


def _values(dataset, name: str, dtype) -> np.ndarray:
    return np.asarray(dataset[name].values, dtype=dtype)


def _optional_values(dataset, name: str, size: int, dtype) -> np.ndarray:
    if name not in dataset:
        return np.zeros(size, dtype=dtype)
    return np.asarray(dataset[name].values, dtype=dtype)


def _dataset_origin_ns(dataset) -> int:
    if "product_time" in dataset:
        value = np.asarray(dataset["product_time"].values)
        if np.issubdtype(value.dtype, np.datetime64):
            return int(value.astype("datetime64[ns]").astype(np.int64))
    for attr_name in ("time_coverage_start", "observation_start"):
        value = dataset.attrs.get(attr_name)
        if value is not None:
            return int(np.datetime64(str(value), "ns").astype(np.int64))
    raise GLMDataError("Xarray GLM dataset has no product time origin")


def _time_values_ns(data_array, origin_ns: int) -> np.ndarray:
    values = np.asarray(data_array.values)
    if np.issubdtype(values.dtype, np.datetime64):
        return values.astype("datetime64[ns]").astype(np.int64)
    if np.issubdtype(values.dtype, np.timedelta64):
        return origin_ns + values.astype("timedelta64[ns]").astype(np.int64)
    units = str(data_array.attrs.get("units", "seconds"))
    multiplier = 1_000_000_000.0
    if units.startswith("millisecond"):
        multiplier = 1_000_000.0
    elif units.startswith("microsecond"):
        multiplier = 1_000.0
    return origin_ns + np.rint(values.astype(np.float64) * multiplier).astype(np.int64)


def _convert_energy_to_j(data_array) -> np.ndarray:
    values = np.asarray(data_array.values, dtype=np.float64)
    units = str(data_array.attrs.get("units", "J")).lower().replace(" ", "")
    if units in {"nj", "nanojoule", "nanojoules"}:
        return values * 1.0e-9
    if units in {"uj", "microjoule", "microjoules"}:
        return values * 1.0e-6
    return values


def _convert_area_to_m2(data_array) -> np.ndarray:
    values = np.asarray(data_array.values, dtype=np.float64)
    units = str(data_array.attrs.get("units", "m2")).lower().replace("^", "").replace(" ", "")
    if units in {"km2", "squarekilometer", "squarekilometers"}:
        return values * 1.0e6
    return values


def _map_ids(parent_ids: np.ndarray, entity_ids: np.ndarray) -> np.ndarray:
    if entity_ids.size == 0:
        return np.full(parent_ids.shape, -1, dtype=np.int64)
    order = np.argsort(entity_ids, kind="stable")
    sorted_ids = entity_ids[order]
    positions = np.searchsorted(sorted_ids, parent_ids)
    valid = positions < sorted_ids.size
    matched = np.zeros(parent_ids.shape, dtype=bool)
    matched[valid] = sorted_ids[positions[valid]] == parent_ids[valid]
    result = np.full(parent_ids.shape, -1, dtype=np.int64)
    result[matched] = order[positions[matched]]
    return result


def _safe_bincount(indices: np.ndarray, size: int) -> np.ndarray:
    valid = indices[indices >= 0]
    return np.bincount(valid, minlength=size).astype(np.int32, copy=False)


def _optional_scalar(dataset, name: str) -> float | None:
    if name not in dataset:
        return None
    values = np.asarray(dataset[name].values).reshape(-1)
    return float(values[0]) if values.size and np.isfinite(values[0]) else None


def _or_nan(value: float | None) -> float:
    return np.nan if value is None else float(value)


def _xarray_safe_attr(value: Any) -> bool:
    return isinstance(value, (str, bytes, int, float, np.number, tuple, list, np.ndarray))
