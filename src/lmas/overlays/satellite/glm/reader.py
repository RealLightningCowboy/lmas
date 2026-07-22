"""Native HDF5 reader for GOES-R Series GLM L2 LCFA products."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Iterable, Sequence

import numpy as np

from .roles import resolve_operational_role, role_consistent_with_longitude
from .xarray_compat import from_xarray

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

_REQUIRED_VARIABLES = (
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
    "group_quality_flag",
    "flash_id",
    "flash_time_offset_of_first_event",
    "flash_time_offset_of_last_event",
    "flash_lat",
    "flash_lon",
    "flash_area",
    "flash_energy",
    "flash_quality_flag",
)


def read_glm(
    paths: str | Path | Sequence[str | Path],
    *,
    backend: str = "native",
    validate: bool = True,
) -> GLMObservation:
    """Read one platform's GLM products into the fast native object.

    Parameters
    ----------
    backend
        ``"native"`` uses LMAS's fast NumPy/HDF5 reader. ``"glmtools"``
        reads each LCFA file through :class:`glmtools.io.glm.GLMDataset` and
        converts the resulting xarray datasets into the same NumPy-native LMAS
        storage object. ``"auto"`` prefers the native reader and falls back to
        glmtools only when the native reader cannot decode the collection.
    """
    normalized = str(backend).strip().lower()
    if normalized not in {"auto", "native", "glmtools"}:
        raise ValueError(
            f"Unsupported GLM backend {backend!r}; use 'auto', 'native', or 'glmtools'"
        )
    if normalized == "native":
        observation = read_glm_l2_lcfa(paths, validate=validate)
        observation.identity.attributes["lmas_reader_backend"] = "native"
        return observation
    if normalized == "glmtools":
        return read_glm_with_glmtools(paths, validate=validate)
    try:
        observation = read_glm_l2_lcfa(paths, validate=validate)
        observation.identity.attributes["lmas_reader_backend"] = "native"
        return observation
    except Exception as native_error:
        try:
            return read_glm_with_glmtools(paths, validate=validate)
        except Exception as glmtools_error:
            raise GLMDataError(
                "GLM auto reader failed with both backends. "
                f"Native: {native_error}; glmtools: {glmtools_error}"
            ) from native_error


def read_glm_with_glmtools(
    paths: str | Path | Sequence[str | Path],
    *,
    validate: bool = True,
) -> GLMObservation:
    """Read LCFA products through glmtools, then normalize to LMAS arrays.

    glmtools remains optional. Selecting this backend without glmtools installed
    produces a clear dependency error rather than silently changing readers.
    """
    try:
        from glmtools.io.glm import GLMDataset
    except Exception as exc:  # optional dependency
        raise GLMDataError(
            "The glmtools GLM backend was selected, but glmtools is not installed "
            "or could not be imported in this environment."
        ) from exc

    file_paths = _normalize_paths(paths)
    observations: list[GLMObservation] = []
    for path in file_paths:
        try:
            glm_dataset = GLMDataset(str(path))
            dataset = glm_dataset.dataset.copy(deep=False)
        except Exception as direct_error:
            # Some installations contain glmtools but lack an xarray NetCDF
            # backend.  LMAS can still provide the decoded LCFA dataset to
            # glmtools in memory, preserving glmtools hierarchy/unit handling
            # without requiring netCDF4 or h5netcdf solely for this option.
            try:
                native_transport = read_glm_l2_lcfa(path, validate=False)
                glm_dataset = GLMDataset(
                    native_transport.to_glmtools_compatible_xarray(),
                    fix_bad_DO07_times=False,
                    check_area_units=False,
                    change_energy_units=False,
                )
                dataset = glm_dataset.dataset.copy(deep=False)
            except Exception as exc:
                raise GLMDataError(
                    f"glmtools could not read or normalize {path}: {exc}"
                ) from direct_error
        attrs = dict(dataset.attrs)
        attrs["source_files"] = (str(path),)
        attrs["lmas_reader_backend"] = "glmtools"
        dataset.attrs = attrs
        observations.append(from_xarray(dataset))
    return _concatenate_glm_observations(
        observations, backend="glmtools", validate=validate
    )


def _concatenate_glm_observations(
    observations: Sequence[GLMObservation],
    *,
    backend: str,
    validate: bool,
) -> GLMObservation:
    if not observations:
        raise GLMDataError("No GLM observations were supplied")
    ordered = sorted(observations, key=lambda item: item.identity.observation_start_ns)
    platform = ordered[0].identity.platform_id
    if any(item.identity.platform_id != platform for item in ordered[1:]):
        raise GLMDataError("GLM files from different spacecraft cannot be combined")

    event_ids = np.concatenate([item.events.event_id for item in ordered]).astype(np.uint64, copy=False)
    group_ids = np.concatenate([item.groups.group_id for item in ordered]).astype(np.uint64, copy=False)
    flash_ids = np.concatenate([item.flashes.flash_id for item in ordered]).astype(np.uint64, copy=False)
    _require_unique_ids("event", event_ids)
    _require_unique_ids("group", group_ids)
    _require_unique_ids("flash", flash_ids)

    def cat(table_name: str, field_name: str, dtype):
        return np.concatenate([getattr(getattr(item, table_name), field_name) for item in ordered]).astype(dtype, copy=False)

    event_source = np.concatenate([
        np.full(len(item.events), index, dtype=np.int32) for index, item in enumerate(ordered)
    ])
    group_source = np.concatenate([
        np.full(len(item.groups), index, dtype=np.int32) for index, item in enumerate(ordered)
    ])
    flash_source = np.concatenate([
        np.full(len(item.flashes), index, dtype=np.int32) for index, item in enumerate(ordered)
    ])
    event_parent_group_id = cat("events", "parent_group_id", np.uint64)
    group_parent_flash_id = cat("groups", "parent_flash_id", np.uint64)
    parent_group_index = _map_parent_ids(event_parent_group_id, group_ids)
    parent_flash_index = _map_parent_ids(group_parent_flash_id, flash_ids)
    event_parent_flash_index = np.full(event_ids.size, -1, dtype=np.int64)
    valid = parent_group_index >= 0
    event_parent_flash_index[valid] = parent_flash_index[parent_group_index[valid]]

    events = GLMEventTable(
        event_id=event_ids, source_file_index=event_source,
        time_ns=cat("events", "time_ns", np.int64),
        latitude_deg=cat("events", "latitude_deg", np.float64),
        longitude_deg=cat("events", "longitude_deg", np.float64),
        energy_j=cat("events", "energy_j", np.float64),
        parent_group_id=event_parent_group_id,
        parent_group_index=parent_group_index,
        parent_flash_index=event_parent_flash_index,
    )
    groups = GLMGroupTable(
        group_id=group_ids, source_file_index=group_source,
        time_ns=cat("groups", "time_ns", np.int64),
        frame_time_ns=cat("groups", "frame_time_ns", np.int64),
        latitude_deg=cat("groups", "latitude_deg", np.float64),
        longitude_deg=cat("groups", "longitude_deg", np.float64),
        area_m2=cat("groups", "area_m2", np.float64),
        energy_j=cat("groups", "energy_j", np.float64),
        parent_flash_id=group_parent_flash_id,
        parent_flash_index=parent_flash_index,
        quality_flag=cat("groups", "quality_flag", np.uint16),
        child_event_count=_safe_bincount(parent_group_index, group_ids.size),
    )
    flashes = GLMFlashTable(
        flash_id=flash_ids, source_file_index=flash_source,
        first_event_time_ns=cat("flashes", "first_event_time_ns", np.int64),
        last_event_time_ns=cat("flashes", "last_event_time_ns", np.int64),
        first_frame_time_ns=cat("flashes", "first_frame_time_ns", np.int64),
        last_frame_time_ns=cat("flashes", "last_frame_time_ns", np.int64),
        latitude_deg=cat("flashes", "latitude_deg", np.float64),
        longitude_deg=cat("flashes", "longitude_deg", np.float64),
        area_m2=cat("flashes", "area_m2", np.float64),
        energy_j=cat("flashes", "energy_j", np.float64),
        quality_flag=cat("flashes", "quality_flag", np.uint16),
        child_group_count=_safe_bincount(parent_flash_index, flash_ids.size),
        child_event_count=_safe_bincount(event_parent_flash_index, flash_ids.size),
    )

    first = ordered[0].identity
    role = first.operational_role
    role_source = first.operational_role_source
    source_files: list[GLMSourceFile] = []
    for index, item in enumerate(ordered):
        identity = item.identity
        path = (
            identity.source_files[0].path
            if identity.source_files
            else Path(f"{platform}_{index:04d}.nc")
        )
        source_files.append(GLMSourceFile(
            path=Path(path), dataset_name=Path(path).name, platform_id=platform,
            operational_role=identity.operational_role,
            time_coverage_start_ns=identity.observation_start_ns,
            time_coverage_end_ns=identity.observation_end_ns,
            event_count=len(item.events), group_count=len(item.groups),
            flash_count=len(item.flashes),
            file_size_bytes=Path(path).stat().st_size if Path(path).is_file() else 0,
        ))
    attrs = dict(first.attributes)
    attrs.update({"source_file_count": len(source_files), "lmas_reader_backend": backend})
    identity = GLMDatasetIdentity(
        instrument_family="GLM", platform_id=platform, operational_role=role,
        operational_role_source=role_source, product_level="L2_LCFA",
        observation_start_ns=min(item.identity.observation_start_ns for item in ordered),
        observation_end_ns=max(item.identity.observation_end_ns for item in ordered),
        projection=first.projection, source_files=tuple(source_files), attributes=attrs,
    )
    observation = GLMObservation(identity, events, groups, flashes)
    if validate:
        observation.validate_hierarchy(raise_on_error=True)
    return observation


def read_glm_l2_lcfa(
    paths: str | Path | Sequence[str | Path],
    *,
    validate: bool = True,
) -> GLMObservation:
    """Read and concatenate GLM L2 LCFA files for one spacecraft.

    Files are sorted by their declared observation start.  Platform identity is
    kept separate from the operational East/West role.  The operational role is
    read from each product when possible and otherwise inferred from the nominal
    satellite subpoint longitude.
    """
    file_paths = _normalize_paths(paths)
    records = [_read_single_file(path) for path in file_paths]
    records.sort(key=lambda record: record["coverage_start_ns"])
    _validate_collection(records)

    platform = records[0]["platform_id"]
    role, role_source = _resolve_collection_role(records)
    projection = _merge_projection(records)

    events = GLMEventTable(
        event_id=_concat(records, "event_id", np.uint64),
        source_file_index=_source_indices(records, "event_id"),
        time_ns=_concat(records, "event_time_ns", np.int64),
        latitude_deg=_concat(records, "event_latitude_deg", np.float64),
        longitude_deg=_concat(records, "event_longitude_deg", np.float64),
        energy_j=_concat(records, "event_energy_j", np.float64),
        parent_group_id=_concat(records, "event_parent_group_id", np.uint64),
        parent_group_index=np.empty(0, dtype=np.int64),
        parent_flash_index=np.empty(0, dtype=np.int64),
    )
    groups = GLMGroupTable(
        group_id=_concat(records, "group_id", np.uint64),
        source_file_index=_source_indices(records, "group_id"),
        time_ns=_concat(records, "group_time_ns", np.int64),
        frame_time_ns=_concat(records, "group_frame_time_ns", np.int64),
        latitude_deg=_concat(records, "group_latitude_deg", np.float64),
        longitude_deg=_concat(records, "group_longitude_deg", np.float64),
        area_m2=_concat(records, "group_area_m2", np.float64),
        energy_j=_concat(records, "group_energy_j", np.float64),
        parent_flash_id=_concat(records, "group_parent_flash_id", np.uint64),
        parent_flash_index=np.empty(0, dtype=np.int64),
        quality_flag=_concat(records, "group_quality_flag", np.uint16),
        child_event_count=np.empty(0, dtype=np.int32),
    )
    flashes = GLMFlashTable(
        flash_id=_concat(records, "flash_id", np.uint64),
        source_file_index=_source_indices(records, "flash_id"),
        first_event_time_ns=_concat(records, "flash_first_event_time_ns", np.int64),
        last_event_time_ns=_concat(records, "flash_last_event_time_ns", np.int64),
        first_frame_time_ns=_concat(records, "flash_first_frame_time_ns", np.int64),
        last_frame_time_ns=_concat(records, "flash_last_frame_time_ns", np.int64),
        latitude_deg=_concat(records, "flash_latitude_deg", np.float64),
        longitude_deg=_concat(records, "flash_longitude_deg", np.float64),
        area_m2=_concat(records, "flash_area_m2", np.float64),
        energy_j=_concat(records, "flash_energy_j", np.float64),
        quality_flag=_concat(records, "flash_quality_flag", np.uint16),
        child_group_count=np.empty(0, dtype=np.int32),
        child_event_count=np.empty(0, dtype=np.int32),
    )

    _require_unique_ids("event", events.event_id)
    _require_unique_ids("group", groups.group_id)
    _require_unique_ids("flash", flashes.flash_id)

    events.parent_group_index = _map_parent_ids(events.parent_group_id, groups.group_id)
    groups.parent_flash_index = _map_parent_ids(groups.parent_flash_id, flashes.flash_id)
    events.parent_flash_index = np.full(len(events), -1, dtype=np.int64)
    valid_event_parent = events.parent_group_index >= 0
    events.parent_flash_index[valid_event_parent] = groups.parent_flash_index[
        events.parent_group_index[valid_event_parent]
    ]

    groups.child_event_count = _safe_bincount(events.parent_group_index, len(groups))
    flashes.child_group_count = _safe_bincount(groups.parent_flash_index, len(flashes))
    flashes.child_event_count = _safe_bincount(events.parent_flash_index, len(flashes))

    source_files = tuple(
        GLMSourceFile(
            path=record["path"],
            dataset_name=record["dataset_name"],
            platform_id=record["platform_id"],
            operational_role=record["operational_role"],
            time_coverage_start_ns=record["coverage_start_ns"],
            time_coverage_end_ns=record["coverage_end_ns"],
            event_count=int(record["event_id"].size),
            group_count=int(record["group_id"].size),
            flash_count=int(record["flash_id"].size),
            file_size_bytes=record["path"].stat().st_size,
        )
        for record in records
    )
    attrs = dict(records[0]["global_attributes"])
    attrs["source_file_count"] = len(records)
    identity = GLMDatasetIdentity(
        instrument_family="GLM",
        platform_id=platform,
        operational_role=role,
        operational_role_source=role_source,
        product_level="L2_LCFA",
        observation_start_ns=min(record["coverage_start_ns"] for record in records),
        observation_end_ns=max(record["coverage_end_ns"] for record in records),
        projection=projection,
        source_files=source_files,
        attributes=attrs,
    )
    observation = GLMObservation(identity, events, groups, flashes)
    if validate:
        observation.validate_hierarchy(raise_on_error=True)
    return observation


def _read_single_file(path: Path) -> dict[str, object]:
    import h5py

    try:
        handle = h5py.File(path, "r")
    except OSError as exc:
        raise GLMDataError(f"Unable to open GLM file {path}: {exc}") from exc
    with handle as dataset:
        missing = [name for name in _REQUIRED_VARIABLES if name not in dataset]
        if missing:
            raise GLMDataError(f"{path.name} is missing required LCFA variables: {missing}")

        attrs = {key: _decode_attribute(value) for key, value in dataset.attrs.items()}
        dataset_name = str(attrs.get("dataset_name") or path.name)
        platform_id = _normalize_platform(attrs.get("platform_ID"), dataset_name)
        coverage_start_ns = _parse_utc_ns(attrs.get("time_coverage_start"))
        coverage_end_ns = _parse_utc_ns(attrs.get("time_coverage_end"))
        subpoint_lon = _read_scalar(dataset, "nominal_satellite_subpoint_lon")
        role, role_source = _resolve_role(
            attrs.get("orbital_slot"),
            subpoint_lon,
            platform_id=platform_id,
            observation_time_ns=coverage_start_ns,
        )

        out: dict[str, object] = {
            "path": path,
            "dataset_name": dataset_name,
            "platform_id": platform_id,
            "operational_role": role,
            "operational_role_source": role_source,
            "coverage_start_ns": coverage_start_ns,
            "coverage_end_ns": coverage_end_ns,
            "projection": _read_projection(dataset),
            "global_attributes": attrs,
        }

        out["event_id"] = _read_unsigned_integer(dataset["event_id"], np.uint64)
        out["event_time_ns"] = _read_time_ns(dataset["event_time_offset"])
        out["event_latitude_deg"] = _read_scaled_float(dataset["event_lat"])
        out["event_longitude_deg"] = _read_scaled_float(dataset["event_lon"])
        out["event_energy_j"] = _read_scaled_float(dataset["event_energy"])
        out["event_parent_group_id"] = _read_unsigned_integer(
            dataset["event_parent_group_id"], np.uint64
        )

        out["group_id"] = _read_unsigned_integer(dataset["group_id"], np.uint64)
        out["group_time_ns"] = _read_time_ns(dataset["group_time_offset"])
        if "group_frame_time_offset" in dataset:
            out["group_frame_time_ns"] = _read_time_ns(dataset["group_frame_time_offset"])
        else:
            out["group_frame_time_ns"] = np.asarray(out["group_time_ns"]).copy()
        out["group_latitude_deg"] = _read_scaled_float(dataset["group_lat"])
        out["group_longitude_deg"] = _read_scaled_float(dataset["group_lon"])
        out["group_area_m2"] = _read_scaled_float(dataset["group_area"])
        out["group_energy_j"] = _read_scaled_float(dataset["group_energy"])
        out["group_parent_flash_id"] = _read_unsigned_integer(
            dataset["group_parent_flash_id"], np.uint64
        )
        out["group_quality_flag"] = _read_unsigned_integer(
            dataset["group_quality_flag"], np.uint16
        )

        out["flash_id"] = _read_unsigned_integer(dataset["flash_id"], np.uint64)
        out["flash_first_event_time_ns"] = _read_time_ns(
            dataset["flash_time_offset_of_first_event"]
        )
        out["flash_last_event_time_ns"] = _read_time_ns(
            dataset["flash_time_offset_of_last_event"]
        )
        first_frame = dataset.get("flash_frame_time_offset_of_first_event")
        last_frame = dataset.get("flash_frame_time_offset_of_last_event")
        out["flash_first_frame_time_ns"] = (
            _read_time_ns(first_frame)
            if first_frame is not None
            else np.asarray(out["flash_first_event_time_ns"]).copy()
        )
        out["flash_last_frame_time_ns"] = (
            _read_time_ns(last_frame)
            if last_frame is not None
            else np.asarray(out["flash_last_event_time_ns"]).copy()
        )
        out["flash_latitude_deg"] = _read_scaled_float(dataset["flash_lat"])
        out["flash_longitude_deg"] = _read_scaled_float(dataset["flash_lon"])
        out["flash_area_m2"] = _read_scaled_float(dataset["flash_area"])
        out["flash_energy_j"] = _read_scaled_float(dataset["flash_energy"])
        out["flash_quality_flag"] = _read_unsigned_integer(
            dataset["flash_quality_flag"], np.uint16
        )
        return out


def _normalize_paths(paths: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(paths, (str, Path)):
        raw = [paths]
    else:
        raw = list(paths)
    if not raw:
        raise ValueError("At least one GLM file is required")
    normalized = [Path(path).expanduser().resolve() for path in raw]
    missing = [str(path) for path in normalized if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"GLM source files not found: {missing}")
    return normalized


def _validate_collection(records: Sequence[dict[str, object]]) -> None:
    platforms = {record["platform_id"] for record in records}
    if len(platforms) != 1:
        raise GLMDataError(
            "GLM files from different spacecraft must remain independent datasets; "
            f"received {sorted(platforms)}"
        )
    for left, right in zip(records, records[1:]):
        if int(right["coverage_start_ns"]) < int(left["coverage_start_ns"]):
            raise GLMDataError("GLM files are not chronologically sortable")
    projections = [record["projection"] for record in records]
    first = projections[0]
    for current in projections[1:]:
        for field_name in (
            "nominal_subpoint_lat_deg",
            "nominal_subpoint_lon_deg",
            "nominal_height_km",
            "semi_major_axis_m",
            "semi_minor_axis_m",
        ):
            a = getattr(first, field_name)
            b = getattr(current, field_name)
            if a is None or b is None:
                continue
            tolerance = 1e-5 if "deg" in field_name else 1e-2
            if not np.isclose(a, b, atol=tolerance, rtol=0):
                raise GLMDataError(
                    f"Incompatible projection metadata across files: {field_name} {a} != {b}"
                )


def _resolve_collection_role(records: Sequence[dict[str, object]]) -> tuple[str, str]:
    roles = {str(record["operational_role"]) for record in records}
    if len(roles) == 1:
        role = next(iter(roles))
    elif roles <= {"unknown", "east", "west"}:
        known = roles - {"unknown"}
        if len(known) == 1:
            role = next(iter(known))
        else:
            role = "unknown"
    else:
        raise GLMDataError(f"Conflicting operational roles across GLM files: {sorted(roles)}")
    sources = {str(record["operational_role_source"]) for record in records}
    return role, next(iter(sources)) if len(sources) == 1 else "mixed"


def _resolve_role(
    slot: object,
    subpoint_lon: float | None,
    *,
    platform_id: str,
    observation_time_ns: int,
) -> tuple[str, str]:
    if slot is not None:
        text = str(slot).strip().lower().replace("_", "-")
        if "east" in text:
            return "east", "file:orbital_slot"
        if "west" in text:
            return "west", "file:orbital_slot"
        if text:
            return text.removeprefix("goes-"), "file:orbital_slot"

    role, source = resolve_operational_role(platform_id, observation_time_ns)
    if role != "unknown":
        consistent = role_consistent_with_longitude(role, subpoint_lon)
        if consistent is False:
            return role, source + ":projection-warning"
        return role, source

    if subpoint_lon is not None and np.isfinite(subpoint_lon):
        return ("west" if subpoint_lon <= -100.0 else "east"), "inferred:subpoint_longitude"
    return "unknown", "unknown"


def _merge_projection(records: Sequence[dict[str, object]]) -> GLMProjectionMetadata:
    return records[0]["projection"]  # collection compatibility was checked already


def _read_projection(dataset: h5py.File) -> GLMProjectionMetadata:
    projection = dataset.get("goes_lat_lon_projection")
    attrs = projection.attrs if projection is not None else {}
    return GLMProjectionMetadata(
        nominal_subpoint_lat_deg=_read_scalar(dataset, "nominal_satellite_subpoint_lat"),
        nominal_subpoint_lon_deg=_read_scalar(dataset, "nominal_satellite_subpoint_lon"),
        nominal_height_km=_read_scalar(dataset, "nominal_satellite_height"),
        semi_major_axis_m=_attribute_float(attrs, "semi_major_axis"),
        semi_minor_axis_m=_attribute_float(attrs, "semi_minor_axis"),
        inverse_flattening=_attribute_float(attrs, "inverse_flattening"),
        longitude_of_prime_meridian_deg=_attribute_float(attrs, "longitude_of_prime_meridian"),
        field_of_view_lat_deg=_read_scalar(dataset, "lat_field_of_view"),
        field_of_view_lon_deg=_read_scalar(dataset, "lon_field_of_view"),
        field_of_view_lat_bounds_deg=_read_pair(dataset, "lat_field_of_view_bounds"),
        field_of_view_lon_bounds_deg=_read_pair(dataset, "lon_field_of_view_bounds"),
    )


def _read_pair(dataset: h5py.File, name: str) -> tuple[float, float] | None:
    variable = dataset.get(name)
    if variable is None:
        return None
    values = np.asarray(variable[...], dtype=np.float64).reshape(-1)
    if values.size < 2 or np.any(~np.isfinite(values[:2])):
        return None
    return float(values[0]), float(values[1])


def _read_scalar(dataset: h5py.File, name: str) -> float | None:
    variable = dataset.get(name)
    if variable is None:
        return None
    value = np.asarray(variable[...]).reshape(-1)
    if value.size == 0:
        return None
    result = float(value[0])
    fill = variable.attrs.get("_FillValue")
    if fill is not None and np.isclose(result, float(np.asarray(fill).reshape(-1)[0])):
        return None
    return result


def _attribute_float(attrs, name: str) -> float | None:
    value = attrs.get(name)
    if value is None:
        return None
    array = np.asarray(value).reshape(-1)
    return float(array[0]) if array.size else None


def _read_unsigned_integer(variable: h5py.Dataset, dtype) -> np.ndarray:
    raw = np.asarray(variable[...])
    if raw.dtype.kind == "i" and _is_unsigned(variable):
        raw = raw.view(np.dtype(raw.dtype.str.replace("i", "u")))
    return np.asarray(raw, dtype=dtype)


def _read_scaled_float(variable: h5py.Dataset) -> np.ndarray:
    signed_raw = np.asarray(variable[...])
    missing = np.zeros(signed_raw.shape, dtype=bool)
    fill = variable.attrs.get("_FillValue")
    if fill is not None:
        missing = signed_raw == np.asarray(fill, dtype=signed_raw.dtype).reshape(-1)[0]
    raw = signed_raw
    if raw.dtype.kind == "i" and _is_unsigned(variable):
        raw = raw.view(np.dtype(raw.dtype.str.replace("i", "u")))
    values = raw.astype(np.float64)
    scale = _attribute_float(variable.attrs, "scale_factor")
    offset = _attribute_float(variable.attrs, "add_offset")
    if scale is not None:
        values *= scale
    if offset is not None:
        values += offset
    values[missing] = np.nan
    return values


def _read_time_ns(variable: h5py.Dataset | None) -> np.ndarray:
    if variable is None:
        return np.empty(0, dtype=np.int64)
    seconds = _read_scaled_float(variable)
    units = _decode_attribute(variable.attrs.get("units"))
    if not isinstance(units, str) or " since " not in units:
        raise GLMDataError(f"Unsupported GLM time units for {variable.name}: {units!r}")
    unit_name, origin_text = units.split(" since ", 1)
    if unit_name.strip().lower() not in {"second", "seconds", "s"}:
        raise GLMDataError(f"Unsupported GLM time unit {unit_name!r}")
    origin_ns = _parse_utc_ns(origin_text)
    if np.any(~np.isfinite(seconds)):
        raise GLMDataError(f"Missing time values in {variable.name}")
    return origin_ns + np.rint(seconds * 1_000_000_000.0).astype(np.int64)


def _is_unsigned(variable: h5py.Dataset) -> bool:
    value = _decode_attribute(variable.attrs.get("_Unsigned"))
    return str(value).strip().lower() == "true"


def _decode_attribute(value):
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return _decode_attribute(value.reshape(-1)[0])
        return tuple(_decode_attribute(item) for item in value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _parse_utc_ns(value: object) -> int:
    if value is None:
        raise GLMDataError("GLM product is missing required UTC coverage metadata")
    text = str(_decode_attribute(value)).strip().replace("Z", "+00:00")
    if re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", text):
        text = text.replace(" ", "T", 1)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise GLMDataError(f"Unable to parse GLM UTC timestamp {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    parsed = parsed.astimezone(timezone.utc)
    return int(round(parsed.timestamp() * 1_000_000_000.0))


def _normalize_platform(value: object, dataset_name: str) -> str:
    if value is not None:
        text = str(value).strip().upper().replace("GOES-", "G")
        if re.fullmatch(r"G\d{2}", text):
            return text
    match = re.search(r"_(G\d{2})_", dataset_name.upper())
    if match:
        return match.group(1)
    raise GLMDataError(f"Unable to identify GLM spacecraft for {dataset_name}")


def _concat(records: Sequence[dict[str, object]], name: str, dtype) -> np.ndarray:
    arrays = [np.asarray(record[name]) for record in records]
    if not arrays:
        return np.empty(0, dtype=dtype)
    return np.concatenate(arrays).astype(dtype, copy=False)


def _source_indices(records: Sequence[dict[str, object]], name: str) -> np.ndarray:
    parts = [np.full(np.asarray(record[name]).size, index, dtype=np.int32) for index, record in enumerate(records)]
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.int32)


def _require_unique_ids(entity: str, values: np.ndarray) -> None:
    if values.size == np.unique(values).size:
        return
    unique, counts = np.unique(values, return_counts=True)
    duplicate = unique[counts > 1]
    preview = ", ".join(str(int(value)) for value in duplicate[:8])
    raise GLMDataError(
        f"Duplicate {entity} IDs across the requested file collection ({preview}). "
        "Load shorter independent collections or inspect overlapping products."
    )


def _map_parent_ids(parent_ids: np.ndarray, entity_ids: np.ndarray) -> np.ndarray:
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
