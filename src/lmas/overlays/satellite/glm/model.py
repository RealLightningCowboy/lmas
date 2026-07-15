"""Fast NumPy-native data model for GLM lightning observations.

The interactive representation is deliberately NumPy-first.  Xarray objects are
created only at import/export boundaries so linked-view filtering and future
rendering can operate on compact contiguous arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Mapping, Sequence

import numpy as np

EntityName = Literal["event", "group", "flash"]


class GLMDataError(ValueError):
    """Raised when a GLM product cannot be represented safely."""


@dataclass(frozen=True, slots=True)
class GLMProjectionMetadata:
    """Spacecraft and Earth-figure metadata carried by a GLM product."""

    nominal_subpoint_lat_deg: float | None = None
    nominal_subpoint_lon_deg: float | None = None
    nominal_height_km: float | None = None
    semi_major_axis_m: float | None = None
    semi_minor_axis_m: float | None = None
    inverse_flattening: float | None = None
    longitude_of_prime_meridian_deg: float | None = None
    field_of_view_lat_deg: float | None = None
    field_of_view_lon_deg: float | None = None
    field_of_view_lat_bounds_deg: tuple[float, float] | None = None
    field_of_view_lon_bounds_deg: tuple[float, float] | None = None


@dataclass(frozen=True, slots=True)
class GLMSourceFile:
    """Provenance record for one input product."""

    path: Path
    dataset_name: str
    platform_id: str
    operational_role: str
    time_coverage_start_ns: int
    time_coverage_end_ns: int
    event_count: int
    group_count: int
    flash_count: int
    file_size_bytes: int


@dataclass(frozen=True, slots=True)
class GLMDatasetIdentity:
    """Identity shared by one independent spacecraft/product collection."""

    instrument_family: str
    platform_id: str
    operational_role: str
    operational_role_source: str
    product_level: str
    observation_start_ns: int
    observation_end_ns: int
    projection: GLMProjectionMetadata
    source_files: tuple[GLMSourceFile, ...]
    attributes: Mapping[str, object] = field(default_factory=dict)

    @property
    def observation_start(self) -> np.datetime64:
        return np.datetime64(self.observation_start_ns, "ns")

    @property
    def observation_end(self) -> np.datetime64:
        return np.datetime64(self.observation_end_ns, "ns")

    @property
    def spacecraft_name(self) -> str:
        platform = self.platform_id.strip().upper()
        if platform.startswith("GOES-"):
            return platform
        if platform.startswith("G") and platform[1:].isdigit():
            return f"GOES-{int(platform[1:]):02d}"
        return self.platform_id.strip() or "Unknown spacecraft"

    @property
    def position_name(self) -> str:
        role = self.operational_role.strip().title()
        return role if role and role.lower() not in {"unknown", "none"} else "Unknown"

    @property
    def legend_prefix(self) -> str:
        if self.position_name != "Unknown":
            return f"{self.spacecraft_name} ({self.position_name})"
        return self.spacecraft_name

    @property
    def display_name(self) -> str:
        role = self.position_name
        if role != "Unknown":
            return f"GLM {role} — {self.platform_id}"
        return f"GLM — {self.platform_id}"


@dataclass(slots=True)
class GLMEventTable:
    event_id: np.ndarray
    source_file_index: np.ndarray
    time_ns: np.ndarray
    latitude_deg: np.ndarray
    longitude_deg: np.ndarray
    energy_j: np.ndarray
    parent_group_id: np.ndarray
    parent_group_index: np.ndarray
    parent_flash_index: np.ndarray

    def __len__(self) -> int:
        return int(self.event_id.size)


@dataclass(slots=True)
class GLMGroupTable:
    group_id: np.ndarray
    source_file_index: np.ndarray
    time_ns: np.ndarray
    frame_time_ns: np.ndarray
    latitude_deg: np.ndarray
    longitude_deg: np.ndarray
    area_m2: np.ndarray
    energy_j: np.ndarray
    parent_flash_id: np.ndarray
    parent_flash_index: np.ndarray
    quality_flag: np.ndarray
    child_event_count: np.ndarray

    def __len__(self) -> int:
        return int(self.group_id.size)


@dataclass(slots=True)
class GLMFlashTable:
    flash_id: np.ndarray
    source_file_index: np.ndarray
    first_event_time_ns: np.ndarray
    last_event_time_ns: np.ndarray
    first_frame_time_ns: np.ndarray
    last_frame_time_ns: np.ndarray
    latitude_deg: np.ndarray
    longitude_deg: np.ndarray
    area_m2: np.ndarray
    energy_j: np.ndarray
    quality_flag: np.ndarray
    child_group_count: np.ndarray
    child_event_count: np.ndarray

    def __len__(self) -> int:
        return int(self.flash_id.size)


@dataclass(frozen=True, slots=True)
class GLMHierarchyReport:
    event_count: int
    group_count: int
    flash_count: int
    orphan_event_count: int
    orphan_group_count: int
    duplicate_event_id_count: int
    duplicate_group_id_count: int
    duplicate_flash_id_count: int

    @property
    def valid(self) -> bool:
        return not any(
            (
                self.orphan_event_count,
                self.orphan_group_count,
                self.duplicate_event_id_count,
                self.duplicate_group_id_count,
                self.duplicate_flash_id_count,
            )
        )


@dataclass(slots=True)
class GLMSelection:
    """Zero-copy view indices into one :class:`GLMObservation`."""

    observation: "GLMObservation"
    event_indices: np.ndarray
    group_indices: np.ndarray
    flash_indices: np.ndarray

    @property
    def events(self) -> GLMEventTable:
        return self.observation.events

    @property
    def groups(self) -> GLMGroupTable:
        return self.observation.groups

    @property
    def flashes(self) -> GLMFlashTable:
        return self.observation.flashes

    def materialize(self) -> "GLMObservation":
        """Return an independent compact observation containing this selection."""
        return self.observation._materialize_selection(self)


@dataclass(slots=True)
class GLMObservation:
    """One platform's normalized L2 LCFA observation collection."""

    identity: GLMDatasetIdentity
    events: GLMEventTable
    groups: GLMGroupTable
    flashes: GLMFlashTable
    event_time_order: np.ndarray | None = None
    group_time_order: np.ndarray | None = None
    flash_time_order: np.ndarray | None = None
    _geometry: object | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._validate_lengths()
        if self.event_time_order is None:
            self.event_time_order = np.argsort(self.events.time_ns, kind="stable")
        if self.group_time_order is None:
            self.group_time_order = np.argsort(self.groups.time_ns, kind="stable")
        if self.flash_time_order is None:
            self.flash_time_order = np.argsort(self.flashes.first_event_time_ns, kind="stable")

    @property
    def geometry(self):
        """Lazy event-footprint geometry cache for this observation."""
        if self._geometry is None:
            from .geometry import GLMEventGeometry

            self._geometry = GLMEventGeometry(self)
        return self._geometry

    @property
    def dataset(self):
        """A glmtools-style xarray dataset, generated explicitly at the boundary."""
        return self.to_glmtools_compatible_xarray()

    @property
    def filename(self) -> str | tuple[str, ...]:
        paths = tuple(str(record.path) for record in self.identity.source_files)
        return paths[0] if len(paths) == 1 else paths

    def _validate_lengths(self) -> None:
        def check(table_name: str, arrays: Iterable[np.ndarray]) -> None:
            lengths = {int(np.asarray(arr).shape[0]) for arr in arrays}
            if len(lengths) > 1:
                raise GLMDataError(f"{table_name} arrays have inconsistent lengths: {sorted(lengths)}")

        check(
            "event",
            (
                self.events.event_id,
                self.events.source_file_index,
                self.events.time_ns,
                self.events.latitude_deg,
                self.events.longitude_deg,
                self.events.energy_j,
                self.events.parent_group_id,
                self.events.parent_group_index,
                self.events.parent_flash_index,
            ),
        )
        check(
            "group",
            (
                self.groups.group_id,
                self.groups.source_file_index,
                self.groups.time_ns,
                self.groups.frame_time_ns,
                self.groups.latitude_deg,
                self.groups.longitude_deg,
                self.groups.area_m2,
                self.groups.energy_j,
                self.groups.parent_flash_id,
                self.groups.parent_flash_index,
                self.groups.quality_flag,
                self.groups.child_event_count,
            ),
        )
        check(
            "flash",
            (
                self.flashes.flash_id,
                self.flashes.source_file_index,
                self.flashes.first_event_time_ns,
                self.flashes.last_event_time_ns,
                self.flashes.first_frame_time_ns,
                self.flashes.last_frame_time_ns,
                self.flashes.latitude_deg,
                self.flashes.longitude_deg,
                self.flashes.area_m2,
                self.flashes.energy_j,
                self.flashes.quality_flag,
                self.flashes.child_group_count,
                self.flashes.child_event_count,
            ),
        )

    def validate_hierarchy(self, *, raise_on_error: bool = False) -> GLMHierarchyReport:
        report = GLMHierarchyReport(
            event_count=len(self.events),
            group_count=len(self.groups),
            flash_count=len(self.flashes),
            orphan_event_count=int(np.count_nonzero(self.events.parent_group_index < 0)),
            orphan_group_count=int(np.count_nonzero(self.groups.parent_flash_index < 0)),
            duplicate_event_id_count=_duplicate_count(self.events.event_id),
            duplicate_group_id_count=_duplicate_count(self.groups.group_id),
            duplicate_flash_id_count=_duplicate_count(self.flashes.flash_id),
        )
        if raise_on_error and not report.valid:
            raise GLMDataError(f"Invalid GLM hierarchy: {report}")
        return report

    def select(
        self,
        *,
        time_range_ns: tuple[int | np.datetime64, int | np.datetime64] | None = None,
        geographic_bounds: tuple[float, float, float, float] | None = None,
        event_energy_min_j: float | None = None,
        group_quality_flags: Sequence[int] | None = None,
    ) -> GLMSelection:
        """Select events and retain their parent groups and flashes.

        Parameters
        ----------
        time_range_ns
            Inclusive UTC range as integer nanoseconds or ``numpy.datetime64``.
        geographic_bounds
            ``(west, east, south, north)`` in degrees.
        event_energy_min_j
            Minimum event energy in joules.
        group_quality_flags
            Optional accepted group quality values.  Events whose parent group is
            not accepted are excluded.
        """
        event_candidates = self._time_candidates(
            self.events.time_ns,
            self.event_time_order,
            time_range_ns,
        )
        if event_candidates.size == 0:
            return GLMSelection(
                self,
                event_candidates.astype(np.int64, copy=False),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
            )

        keep = np.ones(event_candidates.size, dtype=bool)
        ev = self.events
        if geographic_bounds is not None:
            west, east, south, north = map(float, geographic_bounds)
            lon = ev.longitude_deg[event_candidates]
            lat = ev.latitude_deg[event_candidates]
            if west <= east:
                lon_keep = (lon >= west) & (lon <= east)
            else:  # antimeridian-spanning bounds
                lon_keep = (lon >= west) | (lon <= east)
            keep &= lon_keep & (lat >= south) & (lat <= north)
        if event_energy_min_j is not None:
            keep &= ev.energy_j[event_candidates] >= float(event_energy_min_j)
        if group_quality_flags is not None:
            parent = ev.parent_group_index[event_candidates]
            parent_valid = parent >= 0
            quality_keep = np.zeros(parent.shape, dtype=bool)
            quality_keep[parent_valid] = np.isin(
                self.groups.quality_flag[parent[parent_valid]],
                np.asarray(group_quality_flags),
            )
            keep &= quality_keep

        event_idx = event_candidates[keep].astype(np.int64, copy=False)
        valid_group_idx = ev.parent_group_index[event_idx]
        valid_group_idx = valid_group_idx[valid_group_idx >= 0]
        group_idx = np.unique(valid_group_idx).astype(np.int64, copy=False)
        valid_flash_idx = self.groups.parent_flash_index[group_idx]
        valid_flash_idx = valid_flash_idx[valid_flash_idx >= 0]
        flash_idx = np.unique(valid_flash_idx).astype(np.int64, copy=False)
        return GLMSelection(self, event_idx, group_idx, flash_idx)

    def reduce_to_entities(self, entity_id: str, entity_ids: Sequence[int]) -> "GLMObservation":
        """Compatibility helper similar to ``glmtools.GLMDataset.reduce_to_entities``."""
        ids = np.asarray(entity_ids)
        name = entity_id.lower()
        if name in {"flash", "flash_id"}:
            flash_idx = np.flatnonzero(np.isin(self.flashes.flash_id, ids))
            group_idx = np.flatnonzero(np.isin(self.groups.parent_flash_index, flash_idx))
            event_idx = np.flatnonzero(np.isin(self.events.parent_group_index, group_idx))
        elif name in {"group", "group_id"}:
            group_idx = np.flatnonzero(np.isin(self.groups.group_id, ids))
            event_idx = np.flatnonzero(np.isin(self.events.parent_group_index, group_idx))
            flash_idx = np.unique(self.groups.parent_flash_index[group_idx])
            flash_idx = flash_idx[flash_idx >= 0]
        elif name in {"event", "event_id"}:
            event_idx = np.flatnonzero(np.isin(self.events.event_id, ids))
            group_idx = np.unique(self.events.parent_group_index[event_idx])
            group_idx = group_idx[group_idx >= 0]
            flash_idx = np.unique(self.groups.parent_flash_index[group_idx])
            flash_idx = flash_idx[flash_idx >= 0]
        else:
            raise KeyError(f"Unknown GLM entity: {entity_id!r}")
        return GLMSelection(self, event_idx, group_idx, flash_idx).materialize()

    def get_flashes(self, flash_ids: Sequence[int]) -> "GLMObservation":
        return self.reduce_to_entities("flash_id", flash_ids)

    def subset_flashes(self, flash_ids: Sequence[int]) -> "GLMObservation":
        return self.get_flashes(flash_ids)

    def _time_candidates(
        self,
        times: np.ndarray,
        order: np.ndarray,
        time_range_ns: tuple[int | np.datetime64, int | np.datetime64] | None,
    ) -> np.ndarray:
        if time_range_ns is None:
            return np.arange(times.size, dtype=np.int64)
        start = _as_ns(time_range_ns[0])
        end = _as_ns(time_range_ns[1])
        if end < start:
            raise ValueError("time range end precedes start")
        ordered_times = times[order]
        lo = int(np.searchsorted(ordered_times, start, side="left"))
        hi = int(np.searchsorted(ordered_times, end, side="right"))
        return order[lo:hi].astype(np.int64, copy=False)

    def _materialize_selection(self, selection: GLMSelection) -> "GLMObservation":
        ev_idx = np.asarray(selection.event_indices, dtype=np.int64)
        gr_idx = np.asarray(selection.group_indices, dtype=np.int64)
        fl_idx = np.asarray(selection.flash_indices, dtype=np.int64)

        group_remap = np.full(len(self.groups), -1, dtype=np.int64)
        group_remap[gr_idx] = np.arange(gr_idx.size, dtype=np.int64)
        flash_remap = np.full(len(self.flashes), -1, dtype=np.int64)
        flash_remap[fl_idx] = np.arange(fl_idx.size, dtype=np.int64)

        events = GLMEventTable(
            event_id=self.events.event_id[ev_idx].copy(),
            source_file_index=self.events.source_file_index[ev_idx].copy(),
            time_ns=self.events.time_ns[ev_idx].copy(),
            latitude_deg=self.events.latitude_deg[ev_idx].copy(),
            longitude_deg=self.events.longitude_deg[ev_idx].copy(),
            energy_j=self.events.energy_j[ev_idx].copy(),
            parent_group_id=self.events.parent_group_id[ev_idx].copy(),
            parent_group_index=group_remap[self.events.parent_group_index[ev_idx]],
            parent_flash_index=flash_remap[self.events.parent_flash_index[ev_idx]],
        )
        groups = GLMGroupTable(
            group_id=self.groups.group_id[gr_idx].copy(),
            source_file_index=self.groups.source_file_index[gr_idx].copy(),
            time_ns=self.groups.time_ns[gr_idx].copy(),
            frame_time_ns=self.groups.frame_time_ns[gr_idx].copy(),
            latitude_deg=self.groups.latitude_deg[gr_idx].copy(),
            longitude_deg=self.groups.longitude_deg[gr_idx].copy(),
            area_m2=self.groups.area_m2[gr_idx].copy(),
            energy_j=self.groups.energy_j[gr_idx].copy(),
            parent_flash_id=self.groups.parent_flash_id[gr_idx].copy(),
            parent_flash_index=flash_remap[self.groups.parent_flash_index[gr_idx]],
            quality_flag=self.groups.quality_flag[gr_idx].copy(),
            child_event_count=np.bincount(events.parent_group_index, minlength=gr_idx.size).astype(np.int32),
        )
        flashes = GLMFlashTable(
            flash_id=self.flashes.flash_id[fl_idx].copy(),
            source_file_index=self.flashes.source_file_index[fl_idx].copy(),
            first_event_time_ns=self.flashes.first_event_time_ns[fl_idx].copy(),
            last_event_time_ns=self.flashes.last_event_time_ns[fl_idx].copy(),
            first_frame_time_ns=self.flashes.first_frame_time_ns[fl_idx].copy(),
            last_frame_time_ns=self.flashes.last_frame_time_ns[fl_idx].copy(),
            latitude_deg=self.flashes.latitude_deg[fl_idx].copy(),
            longitude_deg=self.flashes.longitude_deg[fl_idx].copy(),
            area_m2=self.flashes.area_m2[fl_idx].copy(),
            energy_j=self.flashes.energy_j[fl_idx].copy(),
            quality_flag=self.flashes.quality_flag[fl_idx].copy(),
            child_group_count=np.bincount(groups.parent_flash_index, minlength=fl_idx.size).astype(np.int32),
            child_event_count=np.bincount(events.parent_flash_index, minlength=fl_idx.size).astype(np.int32),
        )
        identity = GLMDatasetIdentity(
            instrument_family=self.identity.instrument_family,
            platform_id=self.identity.platform_id,
            operational_role=self.identity.operational_role,
            operational_role_source=self.identity.operational_role_source,
            product_level=self.identity.product_level,
            observation_start_ns=int(events.time_ns.min()) if len(events) else self.identity.observation_start_ns,
            observation_end_ns=int(events.time_ns.max()) if len(events) else self.identity.observation_end_ns,
            projection=self.identity.projection,
            source_files=self.identity.source_files,
            attributes=dict(self.identity.attributes),
        )
        return GLMObservation(identity, events, groups, flashes)

    def to_glmtools_compatible_xarray(self, *, glmtools_units: bool = True):
        """Export an xarray dataset using glmtools/LCFA variable conventions."""
        from .xarray_compat import to_glmtools_compatible_xarray

        return to_glmtools_compatible_xarray(self, glmtools_units=glmtools_units)

    @classmethod
    def from_xarray(cls, dataset) -> "GLMObservation":
        from .xarray_compat import from_xarray

        return from_xarray(dataset)


def _duplicate_count(values: np.ndarray) -> int:
    if values.size < 2:
        return 0
    return int(values.size - np.unique(values).size)


def _as_ns(value: int | np.datetime64) -> int:
    if isinstance(value, np.datetime64):
        return int(value.astype("datetime64[ns]").astype(np.int64))
    return int(value)
