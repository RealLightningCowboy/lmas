from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import xarray as xr

from .errors import ConfigurationError, DatasetError
from .source_store import LmaSourceStore

EVENT_DIM = "number_of_events"
REQUIRED_EVENT_FIELDS = (
    "event_time",
    "event_latitude",
    "event_longitude",
    "event_altitude",
)


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    result = float(value)
    if not np.isfinite(result):
        raise ConfigurationError("Numeric filter values must be finite")
    return result


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _optional_datetime64(value: Any) -> np.datetime64 | None:
    if value is None or value == "":
        return None
    try:
        result = np.datetime64(value, "ns")
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"Invalid UTC timestamp: {value!r}") from exc
    if np.isnat(result):
        raise ConfigurationError(f"Invalid UTC timestamp: {value!r}")
    return result


@dataclass(frozen=True)
class FilterSpec:
    """Display-only selection criteria for LMA source events."""

    start_time: str | None = None
    end_time: str | None = None
    minimum_stations: int | None = 6
    maximum_chi2: float | None = 1.0
    minimum_altitude_km: float | None = None
    maximum_altitude_km: float | None = None
    minimum_power: float | None = None
    maximum_power: float | None = None
    minimum_x_km: float | None = None
    maximum_x_km: float | None = None
    minimum_y_km: float | None = None
    maximum_y_km: float | None = None

    def validated(self) -> "FilterSpec":
        values = {
            "start_time": None if self.start_time in (None, "") else str(_optional_datetime64(self.start_time)),
            "end_time": None if self.end_time in (None, "") else str(_optional_datetime64(self.end_time)),
            "minimum_stations": _optional_int(self.minimum_stations),
            "maximum_chi2": _optional_float(self.maximum_chi2),
            "minimum_altitude_km": _optional_float(self.minimum_altitude_km),
            "maximum_altitude_km": _optional_float(self.maximum_altitude_km),
            "minimum_power": _optional_float(self.minimum_power),
            "maximum_power": _optional_float(self.maximum_power),
            "minimum_x_km": _optional_float(self.minimum_x_km),
            "maximum_x_km": _optional_float(self.maximum_x_km),
            "minimum_y_km": _optional_float(self.minimum_y_km),
            "maximum_y_km": _optional_float(self.maximum_y_km),
        }
        if values["minimum_stations"] is not None and values["minimum_stations"] < 0:
            raise ConfigurationError("Minimum stations cannot be negative")
        pairs = (
            ("start_time", "end_time"),
            ("minimum_altitude_km", "maximum_altitude_km"),
            ("minimum_power", "maximum_power"),
            ("minimum_x_km", "maximum_x_km"),
            ("minimum_y_km", "maximum_y_km"),
        )
        for low_name, high_name in pairs:
            low, high = values[low_name], values[high_name]
            if low is not None and high is not None:
                if "time" in low_name:
                    if _optional_datetime64(low) >= _optional_datetime64(high):
                        raise ConfigurationError("Filter start time must be earlier than end time")
                elif float(low) >= float(high):
                    raise ConfigurationError(f"{low_name} must be less than {high_name}")
        return FilterSpec(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self.validated())

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None) -> "FilterSpec":
        return cls(**(values or {})).validated()


@dataclass(frozen=True)
class PlotSpec:
    """Figure settings shared by the GUI and CLI."""

    layout: str = "intfs"
    coordinate_system: str = "local"
    show_histogram: bool = False
    text_size_preset: str = "normal"
    color_by: str = "time"
    cmap: str = "turbo"
    theme: str = "dark"
    point_size: float = 3.0
    show_stations: bool = True
    show_station_labels: bool = False
    show_stations_in_vertical_projections: bool = False
    show_colorbar: bool = True
    show_grid: bool = True
    show_legend: bool = False
    show_panel_labels: bool = False
    relative_time_from_window_start: bool = False
    true_aspect: bool = False
    show_map_underlay: bool = False
    reverse_cmap: bool = False
    log_color_scale: bool = False
    auto_fit_spatial: bool = True
    remap_time_colors: bool = True
    north_south_viewpoint: str = "south"
    east_west_viewpoint: str = "east"
    show_north_south_title: bool = False
    show_east_west_title: bool = False
    depth_mode: str = "spatial"
    title: str | None = None
    dpi: int = 100
    saved_figure_dpi: int = 300
    preview_point_limit: int = 12000
    three_d_display_mode: str = "cumulative"
    three_d_trail_ms: float = 30.0
    three_d_afterimage_ms: float = 30.0
    three_d_playback_fps: int = 30
    three_d_playback_duration_s: float = 15.0
    three_d_hold_end_s: float = 5.0
    three_d_orbit_speed_deg_s: float = 14.0
    three_d_interaction_mode: str = "z-orbit"
    three_d_show_grid_and_labels: bool = True

    def validated(self) -> "PlotSpec":
        layout = str(self.layout).strip().lower().replace("_", "-")
        aliases = {
            "intfs": "intfs",
            "intfs-landscape": "intfs",
            "landscape": "intfs",
            "xlma": "xlma",
            "xlma-portrait": "xlma",
            "portrait": "xlma",
        }
        if layout not in aliases:
            raise ConfigurationError(f"Unsupported plot layout: {self.layout!r}")
        coordinate_system = str(self.coordinate_system).strip().lower().replace("_", "-")
        coordinate_aliases = {
            "local": "local", "local-km": "local", "km": "local",
            "geodetic": "geodetic", "lat-lon": "geodetic", "latlon": "geodetic",
        }
        if coordinate_system not in coordinate_aliases:
            raise ConfigurationError(f"Unsupported coordinate system: {self.coordinate_system!r}")
        text_size_preset = str(self.text_size_preset).strip().lower()
        if text_size_preset not in {"normal", "publication", "poster"}:
            raise ConfigurationError(f"Unsupported text-size preset: {self.text_size_preset!r}")
        color_by = str(self.color_by).strip().lower().replace("_", "-")
        if color_by not in {"time", "altitude", "power", "stations", "chi2", "charge", "group"}:
            raise ConfigurationError(f"Unsupported color quantity: {self.color_by!r}")
        theme = str(self.theme).strip().lower()
        if theme not in {"dark", "light", "space"}:
            raise ConfigurationError(f"Unsupported figure theme: {self.theme!r}")
        point_size = float(self.point_size)
        if not np.isfinite(point_size) or point_size < 0:
            raise ConfigurationError("Point size must be zero (automatic) or positive")
        north_south_viewpoint = str(self.north_south_viewpoint).strip().lower()
        if north_south_viewpoint not in {"north", "south"}:
            raise ConfigurationError(
                f"Unsupported north/south viewpoint: {self.north_south_viewpoint!r}"
            )
        east_west_viewpoint = str(self.east_west_viewpoint).strip().lower()
        if east_west_viewpoint not in {"east", "west"}:
            raise ConfigurationError(
                f"Unsupported east/west viewpoint: {self.east_west_viewpoint!r}"
            )
        depth_mode = str(self.depth_mode).strip().lower()
        if depth_mode not in {"time", "spatial"}:
            raise ConfigurationError(f"Unsupported depth mode: {self.depth_mode!r}")
        dpi = int(self.dpi)
        if dpi < 50 or dpi > 1200:
            raise ConfigurationError("DPI must be between 50 and 1200")
        saved_figure_dpi = int(self.saved_figure_dpi)
        if saved_figure_dpi < 72 or saved_figure_dpi > 1200:
            raise ConfigurationError("Saved-figure DPI must be between 72 and 1200")
        preview_point_limit = int(self.preview_point_limit)
        if preview_point_limit < 0 or preview_point_limit > 5_000_000:
            raise ConfigurationError("Preview point limit must be zero (disabled) or between 1 and 5,000,000")
        three_d_display_mode = str(self.three_d_display_mode).strip().lower().replace("_", "-")
        # v0.3.6 removes the visually redundant cumulative-afterimage mode.
        # Older projects/profiles migrate cleanly to ordinary cumulative display.
        if three_d_display_mode == "cumulative-afterimage":
            three_d_display_mode = "cumulative"
        if three_d_display_mode not in {
            "full", "cumulative", "trail", "trail-afterimage"
        }:
            raise ConfigurationError(
                f"Unsupported 3D display mode: {self.three_d_display_mode!r}"
            )
        three_d_trail_ms = float(self.three_d_trail_ms)
        three_d_afterimage_ms = float(self.three_d_afterimage_ms)
        if not np.isfinite(three_d_trail_ms) or three_d_trail_ms <= 0:
            raise ConfigurationError("3D trail duration must be positive")
        if not np.isfinite(three_d_afterimage_ms) or three_d_afterimage_ms <= 0:
            raise ConfigurationError("3D afterimage duration must be positive")
        three_d_playback_fps = int(self.three_d_playback_fps)
        if three_d_playback_fps < 1 or three_d_playback_fps > 240:
            raise ConfigurationError("3D playback FPS must be between 1 and 240")
        three_d_playback_duration_s = float(self.three_d_playback_duration_s)
        if not np.isfinite(three_d_playback_duration_s) or three_d_playback_duration_s <= 0:
            raise ConfigurationError("3D playback duration must be positive")
        three_d_hold_end_s = float(self.three_d_hold_end_s)
        if not np.isfinite(three_d_hold_end_s) or three_d_hold_end_s < 0:
            raise ConfigurationError("Animation final-frame hold cannot be negative")
        three_d_orbit_speed_deg_s = float(self.three_d_orbit_speed_deg_s)
        if not np.isfinite(three_d_orbit_speed_deg_s):
            raise ConfigurationError("3D orbit speed must be finite")
        three_d_interaction_mode = str(self.three_d_interaction_mode).strip().lower().replace("_", "-")
        if three_d_interaction_mode not in {"z-orbit", "full-3d"}:
            raise ConfigurationError(
                f"Unsupported 3D interaction mode: {self.three_d_interaction_mode!r}"
            )
        return PlotSpec(
            layout=aliases[layout],
            coordinate_system=coordinate_aliases[coordinate_system],
            show_histogram=bool(self.show_histogram),
            text_size_preset=text_size_preset,
            color_by=color_by,
            cmap=str(self.cmap),
            theme=theme,
            point_size=point_size,
            show_stations=bool(self.show_stations),
            show_station_labels=bool(self.show_station_labels),
            show_stations_in_vertical_projections=bool(self.show_stations_in_vertical_projections),
            show_colorbar=bool(self.show_colorbar),
            show_grid=bool(self.show_grid),
            show_legend=bool(self.show_legend),
            show_panel_labels=bool(self.show_panel_labels),
            relative_time_from_window_start=bool(self.relative_time_from_window_start),
            true_aspect=bool(self.true_aspect or self.show_map_underlay),
            show_map_underlay=bool(self.show_map_underlay),
            reverse_cmap=bool(self.reverse_cmap),
            log_color_scale=bool(self.log_color_scale and color_by not in {"charge", "group", "power"}),
            auto_fit_spatial=bool(self.auto_fit_spatial),
            remap_time_colors=bool(self.remap_time_colors),
            north_south_viewpoint=north_south_viewpoint,
            east_west_viewpoint=east_west_viewpoint,
            show_north_south_title=bool(self.show_north_south_title),
            show_east_west_title=bool(self.show_east_west_title),
            depth_mode=depth_mode,
            title=None if self.title in (None, "") else str(self.title),
            dpi=dpi,
            saved_figure_dpi=saved_figure_dpi,
            preview_point_limit=preview_point_limit,
            three_d_display_mode=three_d_display_mode,
            three_d_trail_ms=three_d_trail_ms,
            three_d_afterimage_ms=three_d_afterimage_ms,
            three_d_playback_fps=three_d_playback_fps,
            three_d_playback_duration_s=three_d_playback_duration_s,
            three_d_hold_end_s=three_d_hold_end_s,
            three_d_orbit_speed_deg_s=three_d_orbit_speed_deg_s,
            three_d_interaction_mode=three_d_interaction_mode,
            three_d_show_grid_and_labels=bool(self.three_d_show_grid_and_labels),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self.validated())

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None) -> "PlotSpec":
        return cls(**(values or {})).validated()


@dataclass
class LMAProject:
    """LMAS project with an immutable NumPy store and xarray compatibility view."""

    dataset: xr.Dataset
    _source_store: LmaSourceStore = field(init=False, repr=False)
    _source_store_signature: tuple[Any, ...] = field(init=False, repr=False)
    source_files: tuple[Path, ...] = field(default_factory=tuple)
    name: str = "LMA project"
    reference_latitude: float | None = None
    reference_longitude: float | None = None
    filters: FilterSpec = field(default_factory=FilterSpec)
    # Persistent linked-view constraints are kept separate from quality filters.
    # This prevents project loading from destructively trimming the source dataset.
    view_filters: FilterSpec = field(
        default_factory=lambda: FilterSpec(minimum_stations=None, maximum_chi2=None)
    )
    # Exact named linked-axis bounds for the Project Home action.  These are
    # stored separately from FilterSpec so viewpoint names and all displayed
    # limits survive save/reload without lossy reconstruction.
    project_home_limits: dict[str, tuple[float, float]] = field(default_factory=dict)
    plot: PlotSpec = field(default_factory=PlotSpec)
    project_path: Path | None = None
    # Exact linked-subset membership and color normalization are project/session
    # state, not reusable profile settings.
    selected_source_ids: tuple[int, ...] | None = None
    color_norm_limits: tuple[float, float] | None = None
    # Project-persisted named source groups, charge-region polarity assignments, and display state.
    source_selection_state: dict[str, Any] = field(default_factory=dict)
    # Satellite overlay source references, visibility, layer, and styling state.
    # The interactive GLM data remain in a separate NumPy-native manager.
    satellite_overlay_state: dict[str, Any] = field(default_factory=dict)
    # Ground lightning-location-network source references, filters, and styles.
    network_overlay_state: dict[str, Any] = field(default_factory=dict)
    reader_backend: str = "native"
    reader_backend_version: str | None = None
    reader_details: dict[str, str] = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self) -> None:
        self.dataset = validate_dataset(self.dataset)
        self._source_store = LmaSourceStore.from_xarray(self.dataset, event_dimension=EVENT_DIM)
        self._source_store_signature = _dataset_store_signature(self.dataset)
        self.source_files = tuple(Path(path).expanduser() for path in self.source_files)
        self.filters = self.filters.validated()
        self.view_filters = self.view_filters.validated()
        normalized_home: dict[str, tuple[float, float]] = {}
        for name, bounds in dict(self.project_home_limits or {}).items():
            try:
                low, high = sorted((float(bounds[0]), float(bounds[1])))
            except (TypeError, ValueError, IndexError):
                continue
            if np.isfinite(low) and np.isfinite(high) and high > low:
                normalized_home[str(name)] = (low, high)
        self.project_home_limits = normalized_home
        self.plot = self.plot.validated()
        self.reader_backend = str(self.reader_backend or "native")
        self.reader_backend_version = (
            None if self.reader_backend_version in (None, "") else str(self.reader_backend_version)
        )
        self.reader_details = {str(key): str(value) for key, value in dict(self.reader_details or {}).items()}
        if self.selected_source_ids is not None:
            self.selected_source_ids = tuple(int(value) for value in self.selected_source_ids)
        self.source_selection_state = dict(self.source_selection_state or {})
        self.satellite_overlay_state = dict(self.satellite_overlay_state or {})
        self.network_overlay_state = dict(self.network_overlay_state or {})
        if self.color_norm_limits is not None:
            low, high = (float(value) for value in self.color_norm_limits)
            if not np.isfinite(low) or not np.isfinite(high) or high <= low:
                raise ConfigurationError("Project color-normalization limits must be finite and ordered")
            self.color_norm_limits = (low, high)
        if self.reference_latitude is None:
            self.reference_latitude = _scalar_or_none(self.dataset.get("network_center_latitude"))
        if self.reference_longitude is None:
            self.reference_longitude = _scalar_or_none(self.dataset.get("network_center_longitude"))
        if self.reference_latitude is None:
            self.reference_latitude = float(np.nanmedian(self.dataset["event_latitude"].values))
        if self.reference_longitude is None:
            self.reference_longitude = float(np.nanmedian(self.dataset["event_longitude"].values))

    @property
    def source_store(self) -> LmaSourceStore:
        """Return the current immutable store, refreshing after xarray reassignment."""

        signature = _dataset_store_signature(self.dataset)
        if signature != self._source_store_signature:
            self.refresh_source_store()
        return self._source_store

    def refresh_source_store(self) -> LmaSourceStore:
        """Rebuild the NumPy store after deliberate direct xarray mutation."""

        self.dataset = validate_dataset(self.dataset)
        self._source_store = LmaSourceStore.from_xarray(
            self.dataset, event_dimension=EVENT_DIM
        )
        self._source_store_signature = _dataset_store_signature(self.dataset)
        return self._source_store

    @property
    def data_source_stem(self) -> str:
        """Return the ordinary display-title stem for the loaded data source."""

        if self.source_files:
            name = Path(self.source_files[0]).name
            for suffix in (".tar.gz", ".dat.gz", ".netcdf", ".dat", ".tgz", ".tar", ".nc"):
                if name.lower().endswith(suffix):
                    return name[: -len(suffix)]
            return Path(name).stem
        return str(self.name)

    @property
    def output_stem(self) -> str:
        """Preferred user-facing stem for figures and animations."""

        if self.project_path is not None:
            name = self.project_path.name
            lower = name.lower()
            for suffix in (".lmas-project.yaml", ".lmas-project.yml", ".lmas.yaml", ".lmas.yml", ".yaml", ".yml"):
                if lower.endswith(suffix):
                    return name[: -len(suffix)]
            return self.project_path.stem
        return self.data_source_stem

    @property
    def output_directory(self) -> Path | None:
        """Visible project/data directory used as the default export location."""

        if self.source_files:
            return self.source_files[0].expanduser().resolve().parent
        if self.project_path is not None:
            return self.project_path.parent
        return None

    @property
    def event_count(self) -> int:
        return self.source_store.event_count

    @property
    def time_limits(self) -> tuple[np.datetime64, np.datetime64]:
        values = np.asarray(self.source_store["event_time"]).astype("datetime64[ns]")
        finite = values[~np.isnat(values)]
        if finite.size == 0:
            raise DatasetError("The dataset has no valid event times")
        return finite.min(), finite.max()

    @property
    def available_color_fields(self) -> tuple[str, ...]:
        fields = ["time", "altitude", "charge", "group"]
        for mode, variable in (("power", "event_power"), ("stations", "event_stations"), ("chi2", "event_chi2")):
            if variable in self.source_store:
                fields.append(mode)
        return tuple(fields)

    def selected_source_store(self, filters: FilterSpec | None = None) -> LmaSourceStore:
        from .selection import select_event_store

        return select_event_store(self, filters or self.filters)

    def selected_dataset(self, filters: FilterSpec | None = None) -> xr.Dataset:
        return self.selected_source_store(filters).to_xarray()

    def polarity_dataframe(self, *, scope: str = "all"):
        """Return the canonical one-row-per-source manual-polarity table."""

        from .polarity_product import polarity_dataframe

        return polarity_dataframe(self, scope=scope)

    def polarity_dataset(self, *, scope: str = "all") -> xr.Dataset:
        """Return the complete xarray manual-polarity product."""

        from .polarity_product import polarity_dataset

        return polarity_dataset(self, scope=scope)

    def with_source_store(
        self, store: LmaSourceStore, *, name: str | None = None
    ) -> "LMAProject":
        return self.with_dataset(store.to_xarray(), name=name)

    def with_dataset(self, dataset: xr.Dataset, *, name: str | None = None) -> "LMAProject":
        return LMAProject(
            dataset=dataset,
            source_files=self.source_files,
            name=name or self.name,
            reference_latitude=self.reference_latitude,
            reference_longitude=self.reference_longitude,
            filters=self.filters,
            view_filters=self.view_filters,
            plot=self.plot,
            project_path=self.project_path,
            selected_source_ids=self.selected_source_ids,
            color_norm_limits=self.color_norm_limits,
            source_selection_state=dict(self.source_selection_state),
            satellite_overlay_state=dict(self.satellite_overlay_state),
            network_overlay_state=dict(self.network_overlay_state),
            reader_backend=self.reader_backend,
            reader_backend_version=self.reader_backend_version,
            reader_details=dict(self.reader_details),
            notes=self.notes,
        )


def _dataset_store_signature(dataset: xr.Dataset) -> tuple[Any, ...]:
    """Cheaply detect ordinary xarray variable replacement without reading arrays."""

    variable_signature = tuple(
        (
            str(name),
            id(variable),
            id(getattr(variable, "_data", None)),
            tuple(variable.dims),
            tuple(variable.shape),
            str(variable.dtype),
            tuple(sorted((str(key), repr(value)) for key, value in variable.attrs.items())),
        )
        for name, variable in dataset.variables.items()
    )
    dataset_attrs = tuple(
        sorted((str(key), repr(value)) for key, value in dataset.attrs.items())
    )
    return variable_signature + (("__dataset_attrs__", dataset_attrs),)


def _scalar_or_none(value: xr.DataArray | None) -> float | None:
    if value is None:
        return None
    array = np.asarray(value.values, dtype=float)
    if array.size != 1 or not np.isfinite(array.ravel()[0]):
        return None
    return float(array.ravel()[0])


def validate_dataset(dataset: xr.Dataset) -> xr.Dataset:
    if not isinstance(dataset, xr.Dataset):
        raise DatasetError("LMA input must be an xarray.Dataset")
    if EVENT_DIM not in dataset.dims:
        inferred = None
        for field in REQUIRED_EVENT_FIELDS:
            if field in dataset and dataset[field].ndim == 1:
                inferred = dataset[field].dims[0]
                break
        if inferred is None:
            raise DatasetError(f"Dataset is missing the {EVENT_DIM!r} dimension")
        dataset = dataset.rename({inferred: EVENT_DIM})
    missing = [name for name in REQUIRED_EVENT_FIELDS if name not in dataset]
    if missing:
        raise DatasetError("Dataset is missing required event fields: " + ", ".join(missing))
    count = dataset.sizes[EVENT_DIM]
    for name in REQUIRED_EVENT_FIELDS:
        variable = dataset[name]
        if EVENT_DIM not in variable.dims or variable.sizes[EVENT_DIM] != count:
            raise DatasetError(f"{name} is not aligned with {EVENT_DIM}")
    try:
        event_time = np.asarray(dataset["event_time"].values).astype("datetime64[ns]")
    except (TypeError, ValueError) as exc:
        raise DatasetError("event_time cannot be converted to datetime64[ns]") from exc
    result = dataset.copy(deep=False)
    result["event_time"] = xr.DataArray(
        event_time,
        dims=dataset["event_time"].dims,
        attrs=dataset["event_time"].attrs,
    )
    # A stable per-project source identifier lets a live linked selection survive
    # quality-filter and display-option redraws even when the filtered array
    # changes length.  Existing identifiers are preserved when supplied by a
    # reader or saved project; otherwise source order is the canonical identity.
    source_id_name = "event_source_index"
    if source_id_name not in result or EVENT_DIM not in result[source_id_name].dims:
        result[source_id_name] = xr.DataArray(
            np.arange(count, dtype=np.int64),
            dims=(EVENT_DIM,),
            attrs={"long_name": "stable LMAS source index"},
        )
    else:
        try:
            source_ids = np.asarray(result[source_id_name].values, dtype=np.int64)
            valid_source_ids = source_ids.size == count and np.unique(source_ids).size == count
        except (TypeError, ValueError):
            valid_source_ids = False
        if not valid_source_ids:
            source_ids = np.arange(count, dtype=np.int64)
        result[source_id_name] = xr.DataArray(
            source_ids,
            dims=(EVENT_DIM,),
            attrs={"long_name": "stable LMAS source index"},
        )
    return result
