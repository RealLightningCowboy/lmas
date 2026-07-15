from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
import time
from typing import Any, Iterable
from uuid import uuid4

import numpy as np

from ..coordinates import altitude_values_km, latlon_to_local_km
from ..errors import DatasetError
from ..model import FilterSpec, LMAProject, PlotSpec
from ..source_selection import (
    charge_region_label,
    charge_values_for_source_ids,
    group_values_for_source_ids,
)
from ..source_store import LmaSourceStore

SNAPSHOT_FORMAT = "lmas-3d-snapshot-v1"


@dataclass(frozen=True)
class VisualizationSnapshot:
    path: Path
    title: str
    event_timestamp: str
    time_utc_ns: np.ndarray
    time_ms: np.ndarray
    points_km: np.ndarray
    source_ids: np.ndarray
    color_values: np.ndarray
    color_label: str
    color_by: str
    logarithmic_color: bool
    reference_latitude: float
    reference_longitude: float
    vertical_reference: str = "MSL"
    ground_subtraction_applied: bool = False
    categorical_colors: tuple[str, ...] = ()
    categorical_labels: tuple[str, ...] = ()

    @property
    def source_count(self) -> int:
        return int(self.time_ms.size)

    @property
    def time_limits(self) -> tuple[float, float]:
        if self.time_ms.size == 0:
            raise DatasetError("The 3D visualization snapshot contains no sources")
        return float(self.time_ms[0]), float(self.time_ms[-1])

    @property
    def bounds(self) -> tuple[float, float, float, float, float, float]:
        values = np.asarray(self.points_km, dtype=float)
        if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] == 0:
            raise DatasetError("The 3D visualization snapshot has invalid point coordinates")
        return (
            float(np.nanmin(values[:, 0])),
            float(np.nanmax(values[:, 0])),
            float(np.nanmin(values[:, 1])),
            float(np.nanmax(values[:, 1])),
            float(np.nanmin(values[:, 2])),
            float(np.nanmax(values[:, 2])),
        )


def _subset_by_source_ids(
    store: LmaSourceStore, source_ids: Iterable[int] | None
) -> LmaSourceStore:
    if source_ids is None:
        return store
    requested = np.asarray(tuple(source_ids), dtype=np.int64)
    if requested.size == 0:
        return store.select_events(np.empty(0, dtype=np.int64))
    available = np.asarray(store.event_array("event_source_index"), dtype=np.int64)
    mask = np.isin(available, requested, assume_unique=False)
    return store.select_events(mask)


def _snapshot_color_values(
    store: LmaSourceStore, plot: PlotSpec, time_ms: np.ndarray, project: LMAProject
) -> tuple[np.ndarray, str, tuple[str, ...], tuple[str, ...]]:
    mode = str(plot.color_by).lower().replace("_", "-")
    if mode == "time":
        return time_ms.copy(), "Source time (ms)", (), ()
    if mode == "altitude":
        return (
            altitude_values_km(
                store.event_array("event_altitude"),
                str(store.field_attrs("event_altitude").get("units", "")),
            ),
            "Altitude (km MSL)",
            (),
            (),
        )
    if mode == "charge":
        source_ids = np.asarray(store.event_array("event_source_index"), dtype=np.int64)
        values, _conflicts = charge_values_for_source_ids(
            source_ids, project.source_selection_state
        )
        return (
            values,
            charge_region_label(project.source_selection_state),
            ("#0077ff", "#8a8a8a", "#d62728"),
            ("Negative", "Unassigned", "Positive"),
        )
    if mode == "group":
        source_ids = np.asarray(store.event_array("event_source_index"), dtype=np.int64)
        values, colors, labels, _overlaps = group_values_for_source_ids(
            source_ids, project.source_selection_state
        )
        return values, "Source group", colors, labels
    mapping = {
        "power": ("event_power", "VHF Source Power (dBW)"),
        "stations": ("event_stations", "Contributing stations"),
        "chi2": ("event_chi2", "Reduced χ²"),
    }
    try:
        field, label = mapping[mode]
    except KeyError as exc:
        raise DatasetError(f"Unsupported 3D color quantity: {plot.color_by!r}") from exc
    if field not in store:
        raise DatasetError(f"Cannot color by {mode}: dataset has no {field}")
    if plot.log_color_scale and mode == "chi2":
        label = "log₁₀(χ²)"
    return np.asarray(store.event_array(field), dtype=float), label, (), ()


def build_visualization_snapshot(
    project: LMAProject,
    *,
    filters: FilterSpec,
    plot: PlotSpec,
    selected_source_ids: Iterable[int] | None = None,
    output_path: str | Path | None = None,
) -> VisualizationSnapshot:
    """Create an LMA-native 3D snapshot from the current quality and linked subset."""

    filters = filters.validated()
    plot = plot.validated()
    selected = project.selected_source_store(filters)
    selected = _subset_by_source_ids(selected, selected_source_ids)
    if selected.event_count == 0:
        raise DatasetError("No sources remain in the current LMAS view")

    time_utc = np.asarray(selected.event_array("event_time")).astype("datetime64[ns]")
    east, north = latlon_to_local_km(
        selected.event_array("event_longitude"),
        selected.event_array("event_latitude"),
        project.reference_longitude,
        project.reference_latitude,
    )
    altitude = altitude_values_km(
        selected.event_array("event_altitude"),
        str(selected.field_attrs("event_altitude").get("units", "")),
    )
    source_ids = np.asarray(selected.event_array("event_source_index"), dtype=np.int64)

    finite = (
        (~np.isnat(time_utc))
        & np.isfinite(east)
        & np.isfinite(north)
        & np.isfinite(altitude)
    )
    if not np.any(finite):
        raise DatasetError("The current LMAS view has no finite 3D sources")
    selected = selected.select_events(finite)
    time_utc = time_utc[finite]
    east = np.asarray(east, dtype=float)[finite]
    north = np.asarray(north, dtype=float)[finite]
    altitude = np.asarray(altitude, dtype=float)[finite]
    source_ids = source_ids[finite]

    order = np.argsort(time_utc.astype(np.int64), kind="stable")
    selected = selected.select_events(order)
    time_utc = time_utc[order]
    east = east[order]
    north = north[order]
    altitude = altitude[order]
    source_ids = source_ids[order]

    first_time = time_utc[0]
    time_ms = np.asarray((time_utc - first_time) / np.timedelta64(1, "ms"), dtype=float)
    display_colors, color_label, categorical_colors, categorical_labels = _snapshot_color_values(
        selected, plot, time_ms, project
    )
    display_colors = np.asarray(display_colors, dtype=float)
    valid_color = np.isfinite(display_colors)
    if plot.log_color_scale:
        valid_color &= display_colors > 0
    if not np.all(valid_color):
        time_utc = time_utc[valid_color]
        time_ms = time_ms[valid_color]
        east = east[valid_color]
        north = north[valid_color]
        altitude = altitude[valid_color]
        source_ids = source_ids[valid_color]
        display_colors = display_colors[valid_color]
    if time_ms.size == 0:
        raise DatasetError("No sources remain for the selected 3D color quantity")

    points = np.column_stack((east, north, altitude)).astype(float, copy=False)
    timestamp = np.datetime_as_string(time_utc[0], unit="ms").replace("T", " ") + " UTC"
    title = str(plot.title or project.data_source_stem)
    destination = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else temporary_snapshot_path(title)
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "format": SNAPSHOT_FORMAT,
        "title": title,
        "event_timestamp": timestamp,
        "color_label": color_label,
        "color_by": plot.color_by,
        "logarithmic_color": bool(plot.log_color_scale),
        "reference_latitude": float(project.reference_latitude),
        "reference_longitude": float(project.reference_longitude),
        "vertical_reference": "MSL",
        "ground_subtraction_applied": False,
        "categorical_colors": list(categorical_colors),
        "categorical_labels": list(categorical_labels),
    }
    with destination.open("wb") as stream:
        np.savez_compressed(
            stream,
            metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
            time_utc_ns=time_utc.astype("datetime64[ns]").astype(np.int64),
            time_ms=time_ms,
            points_km=points,
            source_ids=source_ids,
            color_values=display_colors,
        )
    return load_visualization_snapshot(destination)


def load_visualization_snapshot(path: str | Path) -> VisualizationSnapshot:
    source = Path(path).expanduser().resolve()
    try:
        with np.load(source, allow_pickle=False) as archive:
            metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
            time_ns = np.asarray(archive["time_utc_ns"], dtype=np.int64)
            time_ms = np.asarray(archive["time_ms"], dtype=float)
            points = np.asarray(archive["points_km"], dtype=float)
            source_ids = np.asarray(archive["source_ids"], dtype=np.int64)
            colors = np.asarray(archive["color_values"], dtype=float)
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise DatasetError(f"Could not read LMAS 3D snapshot {source}: {exc}") from exc
    if metadata.get("format") != SNAPSHOT_FORMAT:
        raise DatasetError(f"Unsupported LMAS 3D snapshot format in {source}")
    count = time_ms.size
    if (
        time_ns.size != count
        or source_ids.size != count
        or colors.size != count
        or points.shape != (count, 3)
        or count == 0
    ):
        raise DatasetError(f"LMAS 3D snapshot arrays are inconsistent in {source}")
    return VisualizationSnapshot(
        path=source,
        title=str(metadata.get("title") or source.stem),
        event_timestamp=str(metadata.get("event_timestamp") or ""),
        time_utc_ns=time_ns,
        time_ms=time_ms,
        points_km=points,
        source_ids=source_ids,
        color_values=colors,
        color_label=str(metadata.get("color_label") or "Source time (ms)"),
        color_by=str(metadata.get("color_by") or "time"),
        logarithmic_color=bool(metadata.get("logarithmic_color", False)),
        reference_latitude=float(metadata.get("reference_latitude", np.nan)),
        reference_longitude=float(metadata.get("reference_longitude", np.nan)),
        vertical_reference=str(metadata.get("vertical_reference") or "MSL"),
        ground_subtraction_applied=bool(metadata.get("ground_subtraction_applied", False)),
        categorical_colors=tuple(str(value) for value in metadata.get("categorical_colors") or ()),
        categorical_labels=tuple(str(value) for value in metadata.get("categorical_labels") or ()),
    )


def temporary_snapshot_path(title: str = "lma") -> Path:
    root = Path(tempfile.gettempdir()) / "lmas" / "visualization"
    root.mkdir(parents=True, exist_ok=True)
    cleanup_temporary_snapshots(root)
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in title).strip("_")
    safe = (safe or "lma")[:60]
    return root / f"{safe}_{uuid4().hex[:10]}.lmas3d.npz"


def cleanup_temporary_snapshots(root: str | Path | None = None, *, maximum_age_hours: float = 48.0) -> None:
    directory = Path(root) if root is not None else Path(tempfile.gettempdir()) / "lmas" / "visualization"
    if not directory.is_dir():
        return
    threshold = time.time() - max(1.0, float(maximum_age_hours)) * 3600.0
    for path in directory.glob("*.lmas3d.npz"):
        try:
            if path.stat().st_mtime < threshold:
                path.unlink()
        except OSError:
            continue


__all__ = [
    "SNAPSHOT_FORMAT",
    "VisualizationSnapshot",
    "build_visualization_snapshot",
    "cleanup_temporary_snapshots",
    "load_visualization_snapshot",
    "temporary_snapshot_path",
]
