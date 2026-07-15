"""Canonical LMAS manual-polarity products and interoperable exports.

The NetCDF/xarray product is the authoritative round-trip format.  It carries
all loaded LMA variables plus source-level polarity, named source groups,
sparse group membership, display metadata, and provenance.  The CSV/DataFrame
view is a flat one-row-per-source interchange table derived from the same
canonical classification logic.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from . import __version__
from .coordinates import altitude_values_km, latlon_to_local_km
from .errors import ConfigurationError, DatasetError
from .model import EVENT_DIM, LMAProject
from .selection import event_selection_mask
from .source_selection import (
    CHARGE_NUMERIC_VALUES,
    SourceSelectionGroup,
    charge_values_for_source_ids,
)

POLARITY_PRODUCT_SCHEMA = "lmas-polarity-v1"
POLARITY_TABLE_SCHEMA = "lmas-polarity-table-v1"
POLARITY_CODES = {"negative": -1, "unassigned": 0, "positive": 1}
POLARITY_LABELS = {-1: "Negative", 0: "Unassigned", 1: "Positive"}
EXPORT_SCOPES = ("all", "filtered", "assigned", "active_group")
PolarityExportScope = Literal["all", "filtered", "assigned", "active_group"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.datetime64):
        return str(value.astype("datetime64[ns]"))
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default)


def _group_id(group: SourceSelectionGroup) -> str:
    token = f"{group.created_utc}\0{group.name}\0{group.created_with_lmas_version}".encode("utf-8")
    return "grp-" + hashlib.sha256(token).hexdigest()[:16]


def _source_ids(project: LMAProject) -> np.ndarray:
    values = np.asarray(project.source_store.event_array("event_source_index"), dtype=np.int64)
    if values.ndim != 1 or values.size != project.event_count:
        raise DatasetError("event_source_index is not aligned with the loaded LMA sources")
    if np.unique(values).size != values.size:
        raise DatasetError("event_source_index must be unique before polarity export")
    return values


def _update_hash_with_array(digest: "hashlib._Hash", values: np.ndarray, *, kind: str) -> None:
    array = np.asarray(values)
    digest.update(kind.encode("ascii") + b"\0")
    digest.update(str(array.shape).encode("ascii") + b"\0")
    if np.issubdtype(array.dtype, np.datetime64):
        canonical = array.astype("datetime64[ns]").astype("<i8", copy=False)
        digest.update(np.ascontiguousarray(canonical).tobytes())
        return
    if np.issubdtype(array.dtype, np.integer):
        canonical = array.astype("<i8", copy=False)
        digest.update(np.ascontiguousarray(canonical).tobytes())
        return
    canonical = np.asarray(array, dtype="<f8")
    finite = np.isfinite(canonical)
    digest.update(np.ascontiguousarray(finite.astype(np.uint8)).tobytes())
    digest.update(np.ascontiguousarray(np.where(finite, canonical, 0.0)).tobytes())


def dataset_fingerprint(project: LMAProject) -> str:
    """Return a portable source-identity fingerprint for import verification."""

    store = project.source_store
    digest = hashlib.sha256()
    digest.update(POLARITY_PRODUCT_SCHEMA.encode("ascii") + b"\0")
    digest.update(str(project.event_count).encode("ascii") + b"\0")
    for name, kind in (
        ("event_source_index", "source_id"),
        ("event_time", "time_ns"),
        ("event_latitude", "latitude"),
        ("event_longitude", "longitude"),
        ("event_altitude", "altitude"),
    ):
        if name not in store:
            raise DatasetError(f"Cannot fingerprint polarity dataset: missing {name}")
        _update_hash_with_array(digest, store.event_array(name), kind=kind)
    return "sha256:" + digest.hexdigest()


def _selection_state(project: LMAProject) -> dict[str, Any]:
    return dict(project.source_selection_state or {})


def _groups(project: LMAProject) -> tuple[SourceSelectionGroup, ...]:
    raw_groups = _selection_state(project).get("groups") or ()
    return tuple(
        item if isinstance(item, SourceSelectionGroup) else SourceSelectionGroup.from_dict(item)
        for item in raw_groups
    )


def _scope_mask(project: LMAProject, scope: str) -> np.ndarray:
    requested = str(scope).strip().lower()
    if requested not in EXPORT_SCOPES:
        raise ConfigurationError(
            f"Unsupported polarity export scope {scope!r}; choose one of {', '.join(EXPORT_SCOPES)}"
        )
    ids = _source_ids(project)
    if requested == "all":
        return np.ones(ids.shape, dtype=bool)
    if requested == "filtered":
        mask = event_selection_mask(project, project.filters)
        mask &= event_selection_mask(project, project.view_filters)
        if project.selected_source_ids is not None:
            mask &= np.isin(ids, np.asarray(project.selected_source_ids, dtype=np.int64))
        return mask
    groups = _groups(project)
    if requested == "active_group":
        active = str(_selection_state(project).get("active_group") or "")
        groups = tuple(group for group in groups if group.name == active)
    selected: set[int] = set()
    for group in groups:
        selected.update(group.source_ids)
    return np.isin(ids, np.fromiter(selected, dtype=np.int64) if selected else np.array([], dtype=np.int64))


def _source_classification(
    source_ids: np.ndarray,
    groups: Sequence[SourceSelectionGroup],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list[str]], list[list[str]], list[list[str]]]:
    state = {"groups": [group.to_dict() for group in groups]}
    values, conflicts = charge_values_for_source_ids(source_ids, state)
    codes = values.astype(np.int8, copy=False)
    group_count = np.zeros(source_ids.shape, dtype=np.int32)
    ids_by_source: list[list[str]] = [[] for _ in range(source_ids.size)]
    names_by_source: list[list[str]] = [[] for _ in range(source_ids.size)]
    categories_by_source: list[list[str]] = [[] for _ in range(source_ids.size)]
    index_by_id = {int(value): index for index, value in enumerate(source_ids)}
    for group in groups:
        gid = _group_id(group)
        for source_id in group.source_ids:
            index = index_by_id.get(int(source_id))
            if index is None:
                continue
            group_count[index] += 1
            ids_by_source[index].append(gid)
            names_by_source[index].append(group.name)
            categories_by_source[index].append(group.charge_category)
    return codes, conflicts, group_count, ids_by_source, names_by_source, categories_by_source


def polarity_dataframe(
    project: LMAProject,
    *,
    scope: PolarityExportScope | str = "all",
) -> pd.DataFrame:
    """Return a one-row-per-source polarity table with all scalar source fields."""

    all_ids = _source_ids(project)
    mask = _scope_mask(project, str(scope))
    indices = np.flatnonzero(mask)
    ids = all_ids[indices]
    groups = _groups(project)
    codes, conflicts, group_count, group_ids, group_names, group_categories = _source_classification(ids, groups)

    data: dict[str, Any] = {}
    dataset = project.dataset
    for name, variable in dataset.variables.items():
        if variable.dims == (EVENT_DIM,):
            values = np.asarray(variable.values)[indices]
            if np.issubdtype(values.dtype, np.datetime64):
                data[str(name)] = pd.to_datetime(values.astype("datetime64[ns]"), utc=True)
            else:
                data[str(name)] = values

    if "event_source_index" in data:
        data["source_id"] = np.asarray(data["event_source_index"], dtype=np.int64)
    else:
        data["source_id"] = ids
    if "event_time" in data:
        data["time_utc"] = data["event_time"]
    if "event_latitude" in data:
        data["latitude_deg"] = np.asarray(data["event_latitude"], dtype=float)
    if "event_longitude" in data:
        data["longitude_deg"] = np.asarray(data["event_longitude"], dtype=float)
    if "event_altitude" in data:
        altitude_units = str(dataset["event_altitude"].attrs.get("units", ""))
        data["altitude_m_msl"] = altitude_values_km(data["event_altitude"], altitude_units) * 1000.0
    if "event_latitude" in data and "event_longitude" in data:
        east_km, north_km = latlon_to_local_km(
            np.asarray(data["event_longitude"], dtype=float),
            np.asarray(data["event_latitude"], dtype=float),
            float(project.reference_longitude),
            float(project.reference_latitude),
        )
        data["east_km"] = east_km
        data["north_km"] = north_km
    if "event_power" in data:
        data["power_dbw"] = np.asarray(data["event_power"], dtype=float)
    if "event_chi2" in data:
        data["reduced_chi2"] = np.asarray(data["event_chi2"], dtype=float)
    if "event_stations" in data:
        data["station_count"] = np.asarray(data["event_stations"])

    data["polarity_code"] = codes
    data["polarity"] = [POLARITY_LABELS[int(value)] for value in codes]
    data["polarity_conflict"] = conflicts
    data["group_count"] = group_count
    data["group_ids_json"] = [_json_dumps(value) for value in group_ids]
    data["group_names_json"] = [_json_dumps(value) for value in group_names]
    data["group_categories_json"] = [_json_dumps(value) for value in group_categories]

    fingerprint = dataset_fingerprint(project)
    created = _utc_now()
    count = ids.size
    data["lmas_polarity_schema"] = np.repeat(POLARITY_TABLE_SCHEMA, count)
    data["dataset_fingerprint"] = np.repeat(fingerprint, count)
    data["export_scope"] = np.repeat(str(scope), count)
    data["export_created_utc"] = np.repeat(created, count)
    data["lmas_version"] = np.repeat(__version__, count)

    frame = pd.DataFrame(data)
    preferred = [
        "lmas_polarity_schema",
        "dataset_fingerprint",
        "export_scope",
        "export_created_utc",
        "lmas_version",
        "source_id",
        "time_utc",
        "latitude_deg",
        "longitude_deg",
        "altitude_m_msl",
        "east_km",
        "north_km",
        "power_dbw",
        "reduced_chi2",
        "station_count",
        "polarity_code",
        "polarity",
        "polarity_conflict",
        "group_count",
        "group_ids_json",
        "group_names_json",
        "group_categories_json",
    ]
    ordered = [name for name in preferred if name in frame.columns]
    ordered.extend(name for name in frame.columns if name not in ordered)
    frame = frame.loc[:, ordered]
    frame.attrs.update(
        {
            "schema": POLARITY_TABLE_SCHEMA,
            "dataset_fingerprint": fingerprint,
            "export_scope": str(scope),
            "lmas_version": __version__,
            "created_utc": created,
            "project_name": project.name,
        }
    )
    return frame


def _safe_netcdf_attr(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (str, bytes, int, float, np.integer, np.floating)):
        return value
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    return _json_dumps(value)


def polarity_dataset(
    project: LMAProject,
    *,
    scope: PolarityExportScope | str = "all",
) -> xr.Dataset:
    """Build the complete LMAS polarity product as an xarray Dataset."""

    requested_scope = str(scope).strip().lower()
    all_ids = _source_ids(project)
    mask = _scope_mask(project, requested_scope)
    indices = np.flatnonzero(mask)
    product = project.dataset.isel({EVENT_DIM: indices}).copy(deep=True)
    if EVENT_DIM != "source":
        if "source" in product.dims:
            raise DatasetError("Cannot create polarity product because the dataset already has a non-event 'source' dimension")
        product = product.rename({EVENT_DIM: "source"})

    ids = all_ids[indices]
    groups = _groups(project)
    codes, conflicts, group_count, _, _, _ = _source_classification(ids, groups)
    product = product.assign_coords(source_id=("source", ids.astype(np.int64)))
    product["polarity_code"] = xr.DataArray(
        codes.astype(np.int8), dims=("source",), attrs={"flag_values": [-1, 0, 1], "flag_meanings": "negative unassigned positive"}
    )
    product["polarity_label"] = xr.DataArray(
        np.asarray([POLARITY_LABELS[int(value)] for value in codes], dtype="U10"), dims=("source",)
    )
    product["polarity_conflict"] = xr.DataArray(conflicts.astype(np.int8), dims=("source",))
    product["polarity_group_count"] = xr.DataArray(group_count.astype(np.int32), dims=("source",))

    group_ids = np.asarray([_group_id(group) for group in groups], dtype="U20")
    product = product.assign_coords(polarity_group=np.arange(len(groups), dtype=np.int32))
    product["polarity_group_id"] = xr.DataArray(group_ids, dims=("polarity_group",))
    product["polarity_group_name"] = xr.DataArray(
        np.asarray([group.name for group in groups], dtype="U"), dims=("polarity_group",)
    )
    product["polarity_group_code"] = xr.DataArray(
        np.asarray([POLARITY_CODES[group.charge_category] for group in groups], dtype=np.int8),
        dims=("polarity_group",),
        attrs={"flag_values": [-1, 0, 1], "flag_meanings": "negative unassigned positive"},
    )
    product["polarity_group_category"] = xr.DataArray(
        np.asarray([group.charge_category for group in groups], dtype="U10"), dims=("polarity_group",)
    )
    product["polarity_group_color"] = xr.DataArray(
        np.asarray([group.color for group in groups], dtype="U16"), dims=("polarity_group",)
    )
    product["polarity_group_display_style"] = xr.DataArray(
        np.asarray([group.display_style for group in groups], dtype="U24"), dims=("polarity_group",)
    )
    product["polarity_group_visible"] = xr.DataArray(
        np.asarray([group.visible for group in groups], dtype=np.int8), dims=("polarity_group",)
    )
    product["polarity_group_locked"] = xr.DataArray(
        np.asarray([group.locked for group in groups], dtype=np.int8), dims=("polarity_group",)
    )
    product["polarity_group_created_utc"] = xr.DataArray(
        np.asarray([group.created_utc for group in groups], dtype="U32"), dims=("polarity_group",)
    )
    product["polarity_group_modified_utc"] = xr.DataArray(
        np.asarray([group.modified_utc for group in groups], dtype="U32"), dims=("polarity_group",)
    )
    product["polarity_group_created_with_lmas_version"] = xr.DataArray(
        np.asarray([group.created_with_lmas_version for group in groups], dtype="U24"), dims=("polarity_group",)
    )

    source_index_by_id = {int(value): index for index, value in enumerate(ids)}
    membership_source_index: list[int] = []
    membership_group_index: list[int] = []
    membership_source_id: list[int] = []
    for group_index, group in enumerate(groups):
        for source_id in sorted(group.source_ids):
            source_index = source_index_by_id.get(int(source_id))
            if source_index is None:
                continue
            membership_source_index.append(source_index)
            membership_group_index.append(group_index)
            membership_source_id.append(int(source_id))
    membership_count = len(membership_source_index)
    product = product.assign_coords(polarity_membership=np.arange(membership_count, dtype=np.int32))
    product["polarity_membership_source_index"] = xr.DataArray(
        np.asarray(membership_source_index, dtype=np.int32), dims=("polarity_membership",)
    )
    product["polarity_membership_group_index"] = xr.DataArray(
        np.asarray(membership_group_index, dtype=np.int32), dims=("polarity_membership",)
    )
    product["polarity_membership_source_id"] = xr.DataArray(
        np.asarray(membership_source_id, dtype=np.int64), dims=("polarity_membership",)
    )

    state = _selection_state(project)
    full_fingerprint = dataset_fingerprint(project)
    product.attrs = {str(key): _safe_netcdf_attr(value) for key, value in project.dataset.attrs.items()}
    product.attrs.update(
        {
            "lmas_polarity_schema": POLARITY_PRODUCT_SCHEMA,
            "lmas_version": __version__,
            "created_utc": _utc_now(),
            "project_name": str(project.name),
            "export_scope": requested_scope,
            "full_dataset_source_count": int(project.event_count),
            "exported_source_count": int(ids.size),
            "dataset_fingerprint": full_fingerprint,
            "source_identity_fields": "event_source_index,event_time,event_latitude,event_longitude,event_altitude",
            "polarity_encoding": "negative=-1;unassigned=0;positive=1",
            "active_group": str(state.get("active_group") or ""),
            "category_visibility_json": _json_dumps(state.get("category_visibility") or {}),
            "selection_scope": str(state.get("selection_scope") or "filtered"),
            "member_display_scope": str(state.get("member_display_scope") or "filtered"),
            "charge_region_label": str(state.get("charge_region_label") or "leader_polarity"),
            "reference_latitude": float(project.reference_latitude),
            "reference_longitude": float(project.reference_longitude),
            "source_files_json": _json_dumps([Path(path).name for path in project.source_files]),
            "filters_json": _json_dumps(project.filters.to_dict()),
            "view_filters_json": _json_dumps(project.view_filters.to_dict()),
            "reader_json": _json_dumps(
                {
                    "backend": project.reader_backend,
                    "version": project.reader_backend_version,
                    "details": project.reader_details,
                }
            ),
        }
    )
    return product


def export_polarity_csv(
    project: LMAProject,
    path: str | Path,
    *,
    scope: PolarityExportScope | str = "all",
) -> Path:
    destination = Path(path).expanduser()
    if destination.suffix.lower() != ".csv":
        destination = destination.with_suffix(".csv")
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    polarity_dataframe(project, scope=scope).to_csv(destination, index=False, date_format="%Y-%m-%dT%H:%M:%S.%fZ")
    return destination


def _choose_netcdf_engine(engine: str | None) -> str:
    if engine:
        return str(engine)
    try:
        import h5netcdf  # noqa: F401
    except Exception:
        return "scipy"
    return "h5netcdf"


def _scipy_compatible_dataset(dataset: xr.Dataset) -> xr.Dataset:
    """Return a NetCDF3-safe view while preserving ordinary LMAS source IDs."""

    result = dataset.copy(deep=False)
    int32 = np.iinfo(np.int32)
    for name in tuple(result.variables):
        variable = result[name]
        if np.issubdtype(variable.dtype, np.integer) and variable.dtype.itemsize > 4:
            values = np.asarray(variable.values)
            if values.size and (np.nanmin(values) < int32.min or np.nanmax(values) > int32.max):
                raise ConfigurationError(
                    f"{name} contains 64-bit integers outside the NetCDF3 range. "
                    "Install h5netcdf or netCDF4 and export again."
                )
            result[name] = xr.DataArray(values.astype(np.int32), dims=variable.dims, attrs=variable.attrs)
    return result


def export_polarity_netcdf(
    project: LMAProject,
    path: str | Path,
    *,
    scope: PolarityExportScope | str = "all",
    engine: str | None = None,
) -> Path:
    destination = Path(path).expanduser()
    if destination.suffix.lower() not in {".nc", ".netcdf"}:
        destination = destination.with_suffix(".nc")
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    chosen_engine = _choose_netcdf_engine(engine)
    dataset = polarity_dataset(project, scope=scope)
    if chosen_engine == "scipy":
        dataset = _scipy_compatible_dataset(dataset)
        dataset.to_netcdf(destination, engine="scipy", format="NETCDF3_64BIT")
    else:
        dataset.to_netcdf(destination, engine=chosen_engine)
    return destination


def load_polarity_dataset(path: str | Path) -> xr.Dataset:
    source = Path(path).expanduser().resolve()
    try:
        dataset = xr.load_dataset(source)
    except Exception as exc:
        raise ConfigurationError(f"Could not read LMAS polarity product {source}: {exc}") from exc
    schema = str(dataset.attrs.get("lmas_polarity_schema") or "")
    if schema != POLARITY_PRODUCT_SCHEMA:
        raise ConfigurationError(
            f"Unsupported polarity-product schema {schema!r}; expected {POLARITY_PRODUCT_SCHEMA!r}"
        )
    return dataset


def selection_state_from_polarity_dataset(
    project: LMAProject,
    dataset: xr.Dataset,
    *,
    allow_partial: bool = False,
) -> dict[str, Any]:
    schema = str(dataset.attrs.get("lmas_polarity_schema") or "")
    if schema != POLARITY_PRODUCT_SCHEMA:
        raise ConfigurationError(f"Unsupported polarity-product schema {schema!r}")
    expected = dataset_fingerprint(project)
    actual = str(dataset.attrs.get("dataset_fingerprint") or "")
    if actual != expected:
        raise ConfigurationError(
            "Polarity product does not match the loaded LMA dataset. "
            "The dataset fingerprint differs; assignments were not imported."
        )
    scope = str(dataset.attrs.get("export_scope") or "all")
    if scope != "all" and not allow_partial:
        raise ConfigurationError(
            f"This polarity product contains scope={scope!r}, not the complete loaded dataset. "
            "Import it with allow_partial=True only when partial group restoration is intended."
        )

    required = (
        "source_id",
        "polarity_group_name",
        "polarity_group_category",
        "polarity_group_color",
        "polarity_group_display_style",
        "polarity_membership_source_id",
        "polarity_membership_group_index",
    )
    missing = [name for name in required if name not in dataset]
    if missing:
        raise ConfigurationError("Polarity product is missing required variables: " + ", ".join(missing))

    names = [str(value) for value in np.asarray(dataset["polarity_group_name"].values)]
    categories = [str(value) for value in np.asarray(dataset["polarity_group_category"].values)]
    colors = [str(value) for value in np.asarray(dataset["polarity_group_color"].values)]
    styles = [str(value) for value in np.asarray(dataset["polarity_group_display_style"].values)]
    visible = np.asarray(dataset.get("polarity_group_visible", xr.DataArray(np.ones(len(names), dtype=np.int8))).values, dtype=bool)
    locked = np.asarray(dataset.get("polarity_group_locked", xr.DataArray(np.zeros(len(names), dtype=np.int8))).values, dtype=bool)
    created = [str(value) for value in np.asarray(dataset.get("polarity_group_created_utc", xr.DataArray(np.repeat("", len(names)))).values)]
    modified = [str(value) for value in np.asarray(dataset.get("polarity_group_modified_utc", xr.DataArray(np.repeat("", len(names)))).values)]
    created_versions = [str(value) for value in np.asarray(dataset.get("polarity_group_created_with_lmas_version", xr.DataArray(np.repeat("", len(names)))).values)]

    membership_source_ids = np.asarray(dataset["polarity_membership_source_id"].values, dtype=np.int64)
    membership_group_indices = np.asarray(dataset["polarity_membership_group_index"].values, dtype=np.int64)
    if membership_source_ids.shape != membership_group_indices.shape:
        raise ConfigurationError("Polarity membership arrays have inconsistent lengths")
    valid_project_ids = set(int(value) for value in _source_ids(project))
    members: list[set[int]] = [set() for _ in names]
    for source_id, group_index in zip(membership_source_ids, membership_group_indices, strict=False):
        if group_index < 0 or group_index >= len(names):
            raise ConfigurationError("Polarity membership refers to an invalid group index")
        if int(source_id) not in valid_project_ids:
            raise ConfigurationError("Polarity membership refers to a source absent from the loaded dataset")
        members[int(group_index)].add(int(source_id))

    groups: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        group = SourceSelectionGroup(
            name=name,
            source_ids=frozenset(members[index]),
            visible=bool(visible[index]) if index < visible.size else True,
            locked=bool(locked[index]) if index < locked.size else False,
            color=colors[index],
            display_style=styles[index],
            charge_category=categories[index],
            created_utc=created[index] if index < len(created) else "",
            modified_utc=modified[index] if index < len(modified) else "",
            created_with_lmas_version=created_versions[index] if index < len(created_versions) else "",
        )
        groups.append(group.to_dict())

    try:
        category_visibility = json.loads(str(dataset.attrs.get("category_visibility_json") or "{}"))
    except json.JSONDecodeError:
        category_visibility = {}
    active = str(dataset.attrs.get("active_group") or "")
    if active not in names:
        active = names[0] if names else None
    return {
        "active_group": active,
        "groups": groups,
        "category_visibility": category_visibility,
        "selection_scope": str(dataset.attrs.get("selection_scope") or "filtered"),
        "member_display_scope": str(dataset.attrs.get("member_display_scope") or "filtered"),
        "charge_region_label": str(dataset.attrs.get("charge_region_label") or "leader_polarity"),
        "import_provenance": {
            "schema": POLARITY_PRODUCT_SCHEMA,
            "imported_utc": _utc_now(),
            "product_created_utc": str(dataset.attrs.get("created_utc") or ""),
            "product_lmas_version": str(dataset.attrs.get("lmas_version") or ""),
            "dataset_fingerprint": actual,
            "export_scope": scope,
        },
    }


def import_polarity_netcdf(
    project: LMAProject,
    path: str | Path,
    *,
    allow_partial: bool = False,
) -> dict[str, Any]:
    dataset = load_polarity_dataset(path)
    return selection_state_from_polarity_dataset(project, dataset, allow_partial=allow_partial)


__all__ = [
    "EXPORT_SCOPES",
    "POLARITY_CODES",
    "POLARITY_PRODUCT_SCHEMA",
    "POLARITY_TABLE_SCHEMA",
    "dataset_fingerprint",
    "export_polarity_csv",
    "export_polarity_netcdf",
    "import_polarity_netcdf",
    "load_polarity_dataset",
    "polarity_dataframe",
    "polarity_dataset",
    "selection_state_from_polarity_dataset",
]
