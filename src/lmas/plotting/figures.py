from __future__ import annotations

from collections.abc import Callable

import matplotlib.dates as mdates
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, FuncFormatter, MaxNLocator

from ..coordinates import (
    altitude_km,
    event_local_coordinates,
    station_center_latlon,
    station_center_local_km,
    station_local_coordinates,
)
from ..errors import DatasetError
from ..model import FilterSpec, LMAProject, PlotSpec
from ..source_selection import (
    CHARGE_COLORS,
    charge_region_label,
    charge_values_for_source_ids,
    group_values_for_source_ids,
)
from .common import (
    add_aligned_vertical_colorbar,
    apply_figure_theme,
    automatic_point_size,
    centered_span_limits,
    color_values,
    finite_limits,
    resolved_cmap,
    save_figure,
    style_colorbar,
    theme_values,
)
from .projection_order import painter_indices, spatial_depth_keys
from .map_underlay import add_map_underlay
from .time_axis import configure_relative_time_axis, configure_utc_time_axis
from .spatial_aspect import enforce_true_spatial_scale


def _view_title(
    project: LMAProject,
    filters: FilterSpec,
    spec: PlotSpec,
) -> Callable[[int, int], str]:
    """Return a dynamic title formatter for the current linked view.

    A user-supplied title is treated as the scientific/event-name prefix, not
    as a replacement for the live source-count and quality-cut information.
    This keeps saved figures informative while still allowing publication-ready
    event naming.
    """

    base_title = str(spec.title).strip() if spec.title else project.data_source_stem
    chi2 = filters.maximum_chi2
    qualifier = "" if chi2 is None else f" (χ² < {float(chi2):.2f})"

    def formatter(
        visible_count: int,
        in_view_count: int,
        displayed_count: int | None = None,
    ) -> str:
        visible = int(visible_count)
        in_view = int(in_view_count)
        displayed = visible if displayed_count is None else int(displayed_count)
        if displayed < visible:
            return (
                f"{base_title} — {displayed:,} displayed of {visible:,} visible; "
                f"{in_view:,} sources in view{qualifier}"
            )
        return (
            f"{base_title} — {visible:,} visible of "
            f"{in_view:,} sources in view{qualifier}"
        )

    return formatter


def _finite_coordinate_arrays(project: LMAProject, dataset):
    """Return finite loaded-source arrays without applying source-quality cuts."""
    time = np.asarray(dataset["event_time"].values).astype("datetime64[ns]")
    lat = np.asarray(dataset["event_latitude"].values, dtype=float)
    lon = np.asarray(dataset["event_longitude"].values, dtype=float)
    alt = altitude_km(dataset)
    east, north = event_local_coordinates(
        dataset,
        project.reference_longitude,
        project.reference_latitude,
    )
    source_ids = np.asarray(
        dataset.get("event_source_index", np.arange(time.size)).values
        if hasattr(dataset.get("event_source_index", None), "values")
        else np.arange(time.size),
        dtype=np.int64,
    )
    mask = (
        (~np.isnat(time))
        & np.isfinite(lat)
        & np.isfinite(lon)
        & np.isfinite(alt)
        & np.isfinite(east)
        & np.isfinite(north)
    )
    time = time[mask]
    time_num = np.asarray(
        mdates.date2num(time.astype("datetime64[us]").astype(object)), dtype=float
    )
    return (
        time,
        time_num,
        lat[mask],
        lon[mask],
        alt[mask],
        east[mask],
        north[mask],
        source_ids[mask],
    )


def _charge_cmap_and_norm():
    cmap = ListedColormap(
        [
            CHARGE_COLORS["negative"],
            CHARGE_COLORS["unassigned"],
            CHARGE_COLORS["positive"],
        ],
        name="lmas_charge",
    )
    norm = BoundaryNorm((-1.5, -0.5, 0.5, 1.5), cmap.N)
    return cmap, norm


def _style_charge_colorbar(colorbar) -> None:
    if colorbar is None:
        return
    colorbar.set_ticks((-1.0, 0.0, 1.0))
    colorbar.set_ticklabels(("Negative", "Unassigned", "Positive"))
    labels = (
        colorbar.ax.get_yticklabels()
        if colorbar.orientation == "vertical"
        else colorbar.ax.get_xticklabels()
    )
    for label in labels:
        label.set_rotation(45)
        label.set_rotation_mode("anchor")
        if colorbar.orientation == "vertical":
            label.set_horizontalalignment("left")
            label.set_verticalalignment("center")
        else:
            label.set_horizontalalignment("right")
            label.set_verticalalignment("top")


def _group_cmap_and_norm(colors: tuple[str, ...]):
    palette = tuple(colors) or (CHARGE_COLORS["unassigned"],)
    cmap = ListedColormap(palette, name="lmas_source_groups")
    boundaries = np.arange(-0.5, len(palette) + 0.5, 1.0)
    norm = BoundaryNorm(boundaries, cmap.N)
    return cmap, norm


def _style_group_colorbar(colorbar, labels: tuple[str, ...]) -> None:
    if colorbar is None:
        return
    ticks = np.arange(len(labels), dtype=float)
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels(labels)
    if len(labels) > 10:
        colorbar.ax.tick_params(labelsize=max(6.0, 10.0 - 0.22 * len(labels)))


def _count_in_limits(
    coordinates: dict[str, np.ndarray],
    limits: dict[str, tuple[float, float]],
) -> int:
    if not coordinates:
        return 0
    size = len(next(iter(coordinates.values())))
    mask = np.ones(size, dtype=bool)
    for name, bounds in limits.items():
        values = coordinates.get(name)
        if values is None:
            continue
        low, high = sorted((float(bounds[0]), float(bounds[1])))
        values = np.asarray(values, dtype=float)
        mask &= np.isfinite(values) & (values >= low) & (values <= high)
    return int(np.count_nonzero(mask))


def _valid_event_arrays(project: LMAProject, dataset, plot: PlotSpec):
    time = np.asarray(dataset["event_time"].values).astype("datetime64[ns]")
    lat = np.asarray(dataset["event_latitude"].values, dtype=float)
    lon = np.asarray(dataset["event_longitude"].values, dtype=float)
    alt = altitude_km(dataset)
    east, north = event_local_coordinates(
        dataset,
        project.reference_longitude,
        project.reference_latitude,
    )
    source_ids = np.asarray(
        dataset.get("event_source_index", np.arange(time.size)).values
        if hasattr(dataset.get("event_source_index", None), "values")
        else np.arange(time.size),
        dtype=np.int64,
    )
    categorical_colors: tuple[str, ...] = ()
    categorical_labels: tuple[str, ...] = ()
    if plot.color_by == "charge":
        colors, _charge_conflicts = charge_values_for_source_ids(
            source_ids, project.source_selection_state
        )
        color_label = charge_region_label(project.source_selection_state)
        categorical_colors = (
            CHARGE_COLORS["negative"],
            CHARGE_COLORS["unassigned"],
            CHARGE_COLORS["positive"],
        )
        categorical_labels = ("Negative", "Unassigned", "Positive")
        _charge_cmap, norm = _charge_cmap_and_norm()
    elif plot.color_by == "group":
        colors, categorical_colors, categorical_labels, _group_overlaps = (
            group_values_for_source_ids(source_ids, project.source_selection_state)
        )
        color_label = "Source group"
        _group_cmap, norm = _group_cmap_and_norm(categorical_colors)
    else:
        colors, color_label, norm = color_values(
            dataset,
            plot.color_by,
            logarithmic=plot.log_color_scale,
        )
    mask = (
        (~np.isnat(time))
        & np.isfinite(lat)
        & np.isfinite(lon)
        & np.isfinite(alt)
        & np.isfinite(east)
        & np.isfinite(north)
        & np.isfinite(colors)
    )
    if plot.log_color_scale:
        mask &= colors > 0
    if not np.any(mask):
        raise DatasetError("No finite LMA sources remain after filtering")
    time = time[mask]
    time_num = np.asarray(
        mdates.date2num(time.astype("datetime64[us]").astype(object)), dtype=float
    )
    source_time_s = np.asarray(
        (time - time.min()) / np.timedelta64(1, "s"), dtype=float
    )
    return (
        time,
        time_num,
        source_time_s,
        lat[mask],
        lon[mask],
        alt[mask],
        east[mask],
        north[mask],
        source_ids[mask],
        colors[mask],
        color_label,
        norm,
        mask,
        categorical_colors,
        categorical_labels,
    )



def _precision_source_values(
    dataset,
    valid_event_mask: np.ndarray,
    *,
    time: np.ndarray,
    time_num: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    altitude: np.ndarray,
    east: np.ndarray,
    north: np.ndarray,
    source_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return exact filtered source arrays used by Precision Mode."""

    mask = np.asarray(valid_event_mask, dtype=bool)
    size = mask.size

    def optional(name: str) -> np.ndarray:
        if name not in dataset:
            return np.full(time.size, np.nan, dtype=float)
        values = np.asarray(dataset[name].values)
        if values.size != size:
            return np.full(time.size, np.nan, dtype=float)
        try:
            return np.ascontiguousarray(values[mask], dtype=float)
        except (TypeError, ValueError):
            return np.full(time.size, np.nan, dtype=float)

    return {
        "time": np.ascontiguousarray(time.astype("datetime64[ns]")),
        "time_num": np.ascontiguousarray(time_num, dtype=float),
        "latitude": np.ascontiguousarray(latitude, dtype=float),
        "longitude": np.ascontiguousarray(longitude, dtype=float),
        "altitude_km": np.ascontiguousarray(altitude, dtype=float),
        "east_km": np.ascontiguousarray(east, dtype=float),
        "north_km": np.ascontiguousarray(north, dtype=float),
        "source_id": np.ascontiguousarray(source_ids, dtype=np.int64),
        "power": optional("event_power"),
        "chi2": optional("event_chi2"),
        "stations": optional("event_stations"),
    }

def _deterministic_preview_indices(
    source_time_s: np.ndarray,
    count: int,
    limit: int,
) -> np.ndarray:
    """Return a stable time-stratified preview without changing membership.

    The exact scientific subset remains authoritative.  This function only
    chooses which of those sources are sent to Matplotlib for interactive
    display.  Evenly spaced positions through stable time order preserve both
    quiet and dense portions of long storm records and do not reshuffle between
    redraws.
    """

    count = int(count)
    limit = int(limit)
    if count <= 0:
        return np.array([], dtype=np.int64)
    if limit <= 0 or count <= limit:
        return np.arange(count, dtype=np.int64)
    time_values = np.asarray(source_time_s, dtype=float)
    if time_values.size != count:
        order = np.arange(count, dtype=np.int64)
    else:
        order = np.argsort(time_values, kind="stable")
    positions = np.linspace(0, count - 1, limit, dtype=np.int64)
    return np.asarray(order[positions], dtype=np.int64)


def _scatter(axis, x, y, colors, *, cmap, norm, size, order=None):
    if order is None:
        order = np.arange(len(colors), dtype=int)
    order = np.asarray(order, dtype=int)
    return axis.scatter(
        np.asarray(x)[order],
        np.asarray(y)[order],
        c=np.asarray(colors)[order],
        cmap=cmap,
        norm=norm,
        s=size,
        marker="o",
        linewidths=0,
        rasterized=len(colors) > 10000,
    )


def _station_codes(dataset) -> np.ndarray | None:
    if "station_code" not in dataset:
        return None
    return np.asarray(dataset["station_code"].values).astype(str)


def _plot_stations_local(
    project,
    dataset,
    plan_axis,
    north_altitude_axis,
    east_altitude_axis,
    theme: str,
    *,
    show_labels: bool = False,
    show_vertical: bool = False,
    east_sign: float = 1.0,
    north_sign: float = 1.0,
) -> None:
    station = station_local_coordinates(
        dataset,
        project.reference_longitude,
        project.reference_latitude,
    )
    if station is None:
        return
    sx, sy = station
    sx = np.asarray(sx, dtype=float) * float(east_sign)
    sy = np.asarray(sy, dtype=float) * float(north_sign)
    values = theme_values(theme)
    plan_axis.scatter(
        sx,
        sy,
        marker="^",
        s=34,
        facecolors="none",
        edgecolors=values["station"],
        linewidths=0.9,
        zorder=20,
        label="LMA stations",
    )
    codes = _station_codes(dataset)
    if show_labels and codes is not None and codes.size == sx.size and sx.size <= 60:
        for code, x_value, y_value in zip(codes, sx, sy, strict=False):
            plan_axis.annotate(
                str(code),
                (x_value, y_value),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
                color=values["text"],
            )
    if not show_vertical:
        return
    if "station_altitude" in dataset:
        station_altitude = np.asarray(dataset["station_altitude"].values, dtype=float)
        units = str(dataset["station_altitude"].attrs.get("units", "")).lower()
        if units in {"m", "meter", "meters", "metre", "metres"} or (
            np.any(np.isfinite(station_altitude))
            and np.nanmedian(np.abs(station_altitude[np.isfinite(station_altitude)])) > 100.0
        ):
            station_altitude = station_altitude / 1000.0
    else:
        station_altitude = np.full_like(sx, min(north_altitude_axis.get_ylim()))
    if station_altitude.size != sx.size:
        station_altitude = np.full_like(sx, min(north_altitude_axis.get_ylim()))
    north_altitude_axis.scatter(
        sy,
        station_altitude,
        marker="^",
        s=24,
        facecolors="none",
        edgecolors=values["station"],
        linewidths=0.8,
        zorder=20,
    )
    east_altitude_axis.scatter(
        sx,
        station_altitude,
        marker="^",
        s=24,
        facecolors="none",
        edgecolors=values["station"],
        linewidths=0.8,
        zorder=20,
    )


def _landscape_limits(east: np.ndarray, north: np.ndarray, altitude: np.ndarray):
    x_limits = finite_limits(east)
    y_limits = finite_limits(north)
    z_limits = finite_limits(altitude)
    shared_span = max(
        x_limits[1] - x_limits[0],
        y_limits[1] - y_limits[0],
        z_limits[1] - z_limits[0],
    )
    return (
        centered_span_limits(x_limits, shared_span),
        centered_span_limits(y_limits, shared_span),
        centered_span_limits(z_limits, shared_span),
    )


def _set_time_limits(axis, time_num: np.ndarray) -> None:
    start, end = float(np.min(time_num)), float(np.max(time_num))
    if start == end:
        one_ms = 1.0 / 86_400_000.0
        start, end = start - one_ms, end + one_ms
    axis.set_xlim(start, end)


def _apply_intfs_axis_ticks(axis, *, time_x: bool = False) -> None:
    axis.tick_params(
        axis="both",
        which="major",
        length=5.5,
        width=0.9,
        direction="inout",
        top=True,
        bottom=True,
        left=True,
        right=True,
        labeltop=False,
        labelright=False,
    )
    axis.tick_params(
        axis="both",
        which="minor",
        length=3.5,
        width=0.8,
        direction="inout",
        top=True,
        bottom=True,
        left=True,
        right=True,
    )
    if not time_x:
        axis.xaxis.set_minor_locator(AutoMinorLocator(2))
    axis.yaxis.set_minor_locator(AutoMinorLocator(2))


def _local_cardinal_axis_label(axis: str, sign: float = 1.0) -> str:
    """Return a directionally correct local-coordinate axis title.

    ``sign`` describes the plotted coordinate orientation.  Positive east/north
    values increase toward the right/top; negative values reverse the screen
    direction for the selected viewpoint.
    """

    positive = float(sign) >= 0.0
    if axis == "east_west":
        return "W ← (km) → E" if positive else "E ← (km) → W"
    if axis == "north_south":
        return "S ← (km) → N" if positive else "N ← (km) → S"
    raise ValueError(f"Unknown local cardinal axis: {axis}")


class _AdaptiveGeodeticLocator(MaxNLocator):
    """Choose a compact geodetic tick count from the live axes size."""

    def __init__(self, *, orientation: str) -> None:
        super().__init__(nbins=5, min_n_ticks=3, steps=[1, 2, 2.5, 5, 10])
        self.orientation = str(orientation)

    def __call__(self):
        try:
            pixels = (
                float(self.axis.axes.bbox.width)
                if self.orientation == "x"
                else float(self.axis.axes.bbox.height)
            )
            # Negative longitudes need more room than latitude values.
            pixels_per_label = 92.0 if self.orientation == "x" else 58.0
            count = int(np.clip(round(pixels / pixels_per_label), 3, 6))
            self.set_params(nbins=count)
        except Exception:
            self.set_params(nbins=4)
        return super().__call__()


def _geodetic_formatter(axis, orientation: str):
    def formatter(value, _position):
        limits = axis.get_xlim() if orientation == "x" else axis.get_ylim()
        span = abs(float(limits[1]) - float(limits[0]))
        if span >= 20.0:
            digits = 0
        elif span >= 2.0:
            digits = 1
        elif span >= 0.2:
            digits = 2
        elif span >= 0.02:
            digits = 3
        else:
            digits = 4
        text = f"{float(value):.{digits}f}"
        if text.startswith("-0") and abs(float(value)) < 0.5 * 10 ** (-digits):
            text = text[1:]
        return text
    return formatter


def _configure_geodetic_axis(axis, *, x: bool = False, y: bool = False) -> None:
    if x:
        axis.xaxis.set_major_locator(_AdaptiveGeodeticLocator(orientation="x"))
        axis.xaxis.set_major_formatter(FuncFormatter(_geodetic_formatter(axis, "x")))
        # Preserve the clean dev9 major-label cadence while restoring visual
        # scale detail with three unlabeled minor ticks between major ticks.
        axis.xaxis.set_minor_locator(AutoMinorLocator(4))
        axis.xaxis.get_offset_text().set_visible(False)
    if y:
        axis.yaxis.set_major_locator(_AdaptiveGeodeticLocator(orientation="y"))
        axis.yaxis.set_major_formatter(FuncFormatter(_geodetic_formatter(axis, "y")))
        axis.yaxis.set_minor_locator(AutoMinorLocator(4))
        axis.yaxis.get_offset_text().set_visible(False)


def _window_time_reference(project: LMAProject, filters: FilterSpec, time_num: np.ndarray) -> tuple[float, float]:
    """Return the fixed elapsed-time origin and window duration in seconds."""

    starts = (
        getattr(getattr(project, "view_filters", None), "start_time", None),
        getattr(filters, "start_time", None),
    )
    ends = (
        getattr(getattr(project, "view_filters", None), "end_time", None),
        getattr(filters, "end_time", None),
    )
    finite = np.asarray(time_num, dtype=float)
    finite = finite[np.isfinite(finite)]
    fallback_start = float(np.min(finite)) if finite.size else 0.0
    fallback_end = float(np.max(finite)) if finite.size else fallback_start + 1.0 / 86400.0

    def convert(value, fallback):
        if value in (None, ""):
            return fallback
        try:
            stamp = np.datetime64(value, "ns")
            return float(mdates.date2num(stamp.astype("datetime64[us]").astype(object)))
        except Exception:
            return fallback

    origin = fallback_start
    for value in starts:
        if value not in (None, ""):
            origin = convert(value, fallback_start)
            break
    end = fallback_end
    for value in ends:
        if value not in (None, ""):
            end = convert(value, fallback_end)
            break
    return origin, max((end - origin) * 86400.0, 1e-9)


def _refresh_time_axis(
    axis,
    theme: str,
    *,
    relative: bool = False,
    origin: float | None = None,
    window_span_s: float | None = None,
) -> float:
    if relative and origin is not None:
        step, unit = configure_relative_time_axis(
            axis, origin=float(origin), window_span_s=float(window_span_s or 0.0)
        )
        axis.set_xlabel(f"Time from window start ({unit})")
    else:
        step = configure_utc_time_axis(axis)
        axis.set_xlabel("Time (UTC)")
    _apply_intfs_axis_ticks(axis, time_x=True)
    values = theme_values(theme)
    axis.tick_params(
        axis="x", which="major", colors=values["text"], length=8,
        width=1.1, direction="inout", top=True, bottom=True, labeltop=False,
    )
    axis.tick_params(
        axis="x", which="minor", colors=values["text"], length=5,
        width=0.95, direction="inout", top=True, bottom=True,
    )
    for label in axis.get_xticklabels(which="both"):
        label.set_color(values["text"])
    for tick in (*axis.xaxis.get_major_ticks(), *axis.xaxis.get_minor_ticks()):
        tick.tick1line.set_color(values["text"])
        tick.tick2line.set_color(values["text"])
        tick.label1.set_color(values["text"])
        tick.label2.set_color(values["text"])
    offset = axis.xaxis.get_offset_text()
    offset.set_text(axis.xaxis.get_major_formatter().get_offset())
    offset.set_color(values["text"])
    offset.set_fontsize(9)
    offset.set_horizontalalignment("right")
    return step


def _apply_true_spatial_aspects(
    axes,
    coordinate_names,
    *,
    enabled: bool,
    reference_latitude: float,
    anchors,
    axis_indices=None,
) -> tuple[float | None, ...]:
    """Apply True Aspect without changing the designed axes geometry.

    By default every spatial panel participates in one common physical scale,
    preserving the established Landscape behavior.  Portrait may provide a
    narrower set so its square plan view remains true-aspect while the shallow
    altitude projections do not limit horizontal zoom.
    """
    selected = range(1, len(axes)) if axis_indices is None else axis_indices
    return enforce_true_spatial_scale(
        axes[0].figure,
        axes,
        coordinate_names,
        axis_indices=selected if enabled else (),
        reference_latitude=reference_latitude,
        anchors=anchors,
        emit=False,
    )

def _refresh_utc_time_axis(axis, theme: str) -> float:
    """Rebuild adaptive UTC ticks and restore all theme/tick styling.

    The anchored date is provided by the formatter's offset text, matching the
    original LMAS/ConciseDateFormatter placement outside and below the axes.
    """

    step = configure_utc_time_axis(axis)
    _apply_intfs_axis_ticks(axis, time_x=True)
    values = theme_values(theme)
    axis.tick_params(
        axis="x",
        which="major",
        colors=values["text"],
        length=8,
        width=1.1,
        direction="inout",
        top=True,
        bottom=True,
        labeltop=False,
    )
    axis.tick_params(
        axis="x",
        which="minor",
        colors=values["text"],
        length=5,
        width=0.95,
        direction="inout",
        top=True,
        bottom=True,
    )
    for label in axis.get_xticklabels(which="both"):
        label.set_color(values["text"])
    for tick in (*axis.xaxis.get_major_ticks(), *axis.xaxis.get_minor_ticks()):
        tick.tick1line.set_color(values["text"])
        tick.tick2line.set_color(values["text"])
        tick.label1.set_color(values["text"])
        tick.label2.set_color(values["text"])
    offset = axis.xaxis.get_offset_text()
    offset.set_text(axis.xaxis.get_major_formatter().get_offset())
    offset.set_color(values["text"])
    offset.set_fontsize(9)
    offset.set_horizontalalignment("right")
    return step


def _full_color_limits(norm) -> tuple[float, float]:
    return float(norm.vmin), float(norm.vmax)


def _text_size_offsets(plot: PlotSpec) -> tuple[float, float]:
    if plot.text_size_preset == "publication":
        return 2.0, 3.0
    if plot.text_size_preset == "poster":
        return 5.0, 6.0
    return 0.0, 0.0


def _apply_text_sizes(fig, axes, plot: PlotSpec, *, title_artist=None, colorbar=None) -> None:
    ordinary_delta, title_delta = _text_size_offsets(plot)
    if title_artist is not None and title_delta:
        title_artist.set_fontsize(float(title_artist.get_fontsize()) + title_delta)
    for axis in axes:
        for label in (axis.xaxis.label, axis.yaxis.label, axis.title):
            label.set_fontsize(float(label.get_fontsize()) + ordinary_delta)
        axis.tick_params(axis="both", which="major",
                         labelsize=float(axis.get_xticklabels()[0].get_fontsize()) + ordinary_delta
                         if axis.get_xticklabels() else 10.0 + ordinary_delta)
        axis.xaxis.get_offset_text().set_fontsize(
            float(axis.xaxis.get_offset_text().get_fontsize()) + ordinary_delta
        )
        axis.yaxis.get_offset_text().set_fontsize(
            float(axis.yaxis.get_offset_text().get_fontsize()) + ordinary_delta
        )
        for text in axis.texts:
            if text.get_gid() == "lmas-panel-label-e":
                continue
            text.set_fontsize(float(text.get_fontsize()) + ordinary_delta)
    if colorbar is not None:
        current = colorbar.ax.get_yticklabels() or colorbar.ax.get_xticklabels()
        base_tick = float(current[0].get_fontsize()) if current else 10.0
        colorbar.ax.tick_params(labelsize=base_tick + ordinary_delta)
        colorbar.ax.xaxis.label.set_fontsize(
            float(colorbar.ax.xaxis.label.get_fontsize()) + ordinary_delta
        )
        colorbar.ax.yaxis.label.set_fontsize(
            float(colorbar.ax.yaxis.label.get_fontsize()) + ordinary_delta
        )

def _stacked_title(formatter):
    def wrapped(*args, **kwargs):
        text = str(formatter(*args, **kwargs))
        if " — " in text:
            first, second = text.split(" — ", 1)
            return f"{first}\n{second}"
        return text
    return wrapped




def _add_histogram_panel_label(axis, plot: PlotSpec, *, portrait: bool):
    """Create the histogram's persistent ``(e)`` label after a redraw.

    Histogram updates begin with ``axis.cla()``, so this label must be recreated
    as part of the redraw itself rather than added only during initial figure
    construction.
    """
    if not plot.show_panel_labels:
        axis._lmas_histogram_panel_label = None
        return None

    # Avoid duplicates if this helper is called without an intervening clear.
    for text in tuple(axis.texts):
        if text.get_gid() == "lmas-panel-label-e":
            text.remove()

    values = theme_values(plot.theme)
    ordinary_delta, _ = _text_size_offsets(plot)
    y_position = 0.900 if portrait else 0.940
    artist = axis.text(
        0.060, y_position, "(e)",
        transform=axis.transAxes, ha="left", va="top",
        fontsize=11 + ordinary_delta, fontweight="bold", color=values["text"],
        zorder=120, clip_on=False, gid="lmas-panel-label-e",
        bbox={"facecolor": values["axes"], "edgecolor": "none",
              "alpha": 0.78, "pad": 1.2},
    )
    axis._lmas_histogram_panel_label = artist
    return artist


def _add_panel_labels(
    axes,
    plot: PlotSpec,
    *,
    histogram_axis=None,
    portrait: bool = False,
):
    """Add publication-style labels to the four primary scientific panels.

    The histogram's auxiliary ``(e)`` label is managed by its redraw function
    because that axes is cleared whenever the linked scientific subset changes.
    """
    if not plot.show_panel_labels:
        return tuple()
    values = theme_values(plot.theme)
    artists = []

    for index, axis in enumerate(list(axes)):
        if index >= 26:
            break
        y_position = 0.940 if portrait and index in (0, 1) else 0.982
        artists.append(
            axis.text(
                0.018, y_position, f"({chr(ord('a') + index)})",
                transform=axis.transAxes, ha="left", va="top",
                fontsize=11, fontweight="bold", color=values["text"],
                zorder=100,
                bbox={"facecolor": values["axes"], "edgecolor": "none",
                      "alpha": 0.72, "pad": 1.2},
            )
        )

    # The current histogram label may already exist after the initial redraw.
    # Do not create it here: a later ``cla()`` would immediately erase it.
    if histogram_axis is not None:
        histogram_artist = getattr(histogram_axis, "_lmas_histogram_panel_label", None)
        if histogram_artist is not None:
            artists.append(histogram_artist)
    return tuple(artists)


def _add_figure_legend(
    fig,
    source_axes,
    plot: PlotSpec,
    *,
    extra_clearance_inches: float = 0.0,
):
    """Add one deduplicated figure legend beneath the bottom axes.

    Landscape keeps the established grouped arrangement, except that one or
    two GLM spacecraft without a ground-network overlay use one row.

    Portrait uses deterministic three-column rows:

    - ordinary LMA/network/station entries first;
    - one row per GLM spacecraft, event footprints before group centroids;
    - the simple LMA + one-GLM-spacecraft case remains a compact one-row legend.
    """
    if not plot.show_legend:
        return None

    values = theme_values(plot.theme)

    handles = [
        Line2D(
            [],
            [],
            linestyle="None",
            marker="o",
            markersize=5.2,
            markerfacecolor=values["text"],
            markeredgecolor="none",
            label="LMA sources",
        )
    ]
    labels = ["LMA sources"]

    seen = set(labels)
    for axis in source_axes:
        axis_handles, axis_labels = axis.get_legend_handles_labels()
        for handle, label in zip(axis_handles, axis_labels, strict=False):
            text = str(label).strip()
            visible = getattr(handle, "get_visible", lambda: True)()
            if (
                not visible
                or not text
                or text.startswith("_")
                or text in seen
            ):
                continue
            seen.add(text)
            handles.append(handle)
            labels.append(text)

    positions = [axis.get_position() for axis in source_axes]
    left = min(position.x0 for position in positions)
    right = max(position.x1 for position in positions)
    bottom = min(position.y0 for position in positions)
    axes_centre_x = 0.5 * (left + right)

    metadata = getattr(fig, "_lmas_metadata", {}) or {}
    layout_text = " ".join(
        (
            str(getattr(plot, "layout", "")),
            str(metadata.get("layout", "")),
        )
    ).strip().lower()

    is_portrait = "portrait" in layout_text or "xlma" in layout_text
    is_landscape = "landscape" in layout_text or "intfs" in layout_text

    ordinary_delta, _ = _text_size_offsets(plot)

    ordinary_entries = []
    glm_entries = []
    for handle, label in zip(handles, labels, strict=False):
        if "GLM" in label.upper():
            glm_entries.append((handle, label))
        else:
            ordinary_entries.append((handle, label))

    blank_handle = Line2D(
        [],
        [],
        linestyle="None",
        linewidth=0,
        marker=None,
        alpha=0.0,
    )
    blank_entry = (blank_handle, "\u200b")

    def glm_spacecraft_key(label: str) -> str:
        """Return a stable spacecraft/platform key from a GLM legend label."""
        text = str(label).strip()
        upper = text.upper()
        for token in ("GOES-16", "GOES-17", "GOES-18", "GOES-19"):
            if token in upper:
                return token
        if "—" in text:
            return text.split("—", 1)[0].strip()
        if " - " in text:
            return text.split(" - ", 1)[0].strip()
        return "GLM"

    def glm_layer_rank(label: str) -> tuple[int, str]:
        """Place event footprints before group centroids within a spacecraft."""
        upper = str(label).upper()
        if "EVENT FOOTPRINT" in upper:
            return (0, upper)
        if "GROUP CENTROID" in upper:
            return (1, upper)
        if "MESH CENTROID" in upper:
            return (2, upper)
        return (3, upper)

    glm_by_spacecraft: dict[str, list[tuple[object, str]]] = {}
    for entry in glm_entries:
        glm_by_spacecraft.setdefault(glm_spacecraft_key(entry[1]), []).append(entry)
    ordered_glm_groups = []
    for key in sorted(glm_by_spacecraft):
        ordered_glm_groups.append(
            (key, sorted(glm_by_spacecraft[key], key=lambda item: glm_layer_rank(item[1])))
        )

    def pack_two_rows(top_entries, bottom_entries, columns=4):
        """Pack two visual rows into Matplotlib's column-major ordering."""
        top_row = list(top_entries[:columns])
        bottom_row = list(bottom_entries[:columns])
        while len(top_row) < columns:
            top_row.append(blank_entry)
        while len(bottom_row) < columns:
            bottom_row.append(blank_entry)

        packed_handles = []
        packed_labels = []
        for column in range(columns):
            top_handle, top_label = top_row[column]
            bottom_handle, bottom_label = bottom_row[column]
            packed_handles.extend((top_handle, bottom_handle))
            packed_labels.extend((top_label, bottom_label))
        return packed_handles, packed_labels

    def pack_visual_rows(rows, columns):
        """Pack arbitrary visual rows into Matplotlib's column-major order."""
        padded_rows = []
        for row in rows:
            padded = list(row[:columns])
            while len(padded) < columns:
                padded.append(blank_entry)
            padded_rows.append(padded)

        packed_handles = []
        packed_labels = []
        for column in range(columns):
            for row in padded_rows:
                handle, label = row[column]
                packed_handles.append(handle)
                packed_labels.append(label)
        return packed_handles, packed_labels

    legend_handles = handles
    legend_labels = labels

    # Portrait keeps its already-approved compact one-row exception only for
    # LMA plus one GLM spacecraft. Landscape uses the broader user-facing rule:
    # one or two GLM spacecraft and no ground-network legend entries.
    simple_single_glm_portrait = (
        len(ordered_glm_groups) == 1
        and len(ordinary_entries) == 1
        and ordinary_entries[0][1] == "LMA sources"
    )

    network_label_tokens = (
        "ENTLN",
        "NLDN",
        "GLD360",
        "NETWORK",
        "LIGHTNING LOCATION",
    )
    has_network_entries = any(
        any(token in str(label).upper() for token in network_label_tokens)
        for _handle, label in ordinary_entries
    )
    no_network_glm_landscape = (
        len(ordered_glm_groups) in (1, 2)
        and not has_network_entries
    )

    if is_portrait:
        # Preserve the approved physical legend centre from the prior
        # 10.20-inch Portrait canvas when only the right margin is enlarged.
        figure_width = max(float(fig.get_figwidth()), 1.0)
        legend_centre_x = min(0.5, 5.10 / figure_width)
        legend_columns = 3

        if simple_single_glm_portrait:
            # LMA + one spacecraft (normally footprints + centroids) fits
            # cleanly in one centered row.
            combined = ordinary_entries + ordered_glm_groups[0][1]
            legend_columns = max(1, min(3, len(combined)))
            legend_handles = [handle for handle, _ in combined]
            legend_labels = [label for _, label in combined]
            row_count = 1
            legend_fontsize = 8.2 + ordinary_delta
        else:
            visual_rows = []

            if len(ordered_glm_groups) >= 2 and len(ordinary_entries) == 1:
                # With two spacecraft and no network/station entries, place
                # the lone LMA entry in column 1 of the first row.
                visual_rows.append([ordinary_entries[0], blank_entry, blank_entry])
            else:
                for start_index in range(0, len(ordinary_entries), legend_columns):
                    visual_rows.append(
                        ordinary_entries[start_index:start_index + legend_columns]
                    )

            if ordered_glm_groups:
                # One deterministic row per spacecraft. Current spacecraft
                # layers occupy the first two cells; the third remains blank.
                for _key, entries in ordered_glm_groups:
                    visual_rows.append(entries[:legend_columns])
            else:
                for start_index in range(0, len(glm_entries), legend_columns):
                    visual_rows.append(
                        glm_entries[start_index:start_index + legend_columns]
                    )

            if not visual_rows:
                visual_rows = [[blank_entry]]

            legend_handles, legend_labels = pack_visual_rows(
                visual_rows,
                legend_columns,
            )
            row_count = max(1, len(visual_rows))
            if row_count == 1:
                legend_fontsize = 8.4 + ordinary_delta
            elif row_count == 2:
                legend_fontsize = 7.9 + ordinary_delta
            else:
                legend_fontsize = 7.6 + ordinary_delta

        legend_location = "lower center"
        legend_anchor = (legend_centre_x, 0.022)
        borderpad = 0.28
        handletextpad = 0.38
        columnspacing = 0.72
        labelspacing = 0.24
        handlelength = 1.45

    elif is_landscape:
        if no_network_glm_landscape:
            # Keep every entry on one row whenever one or both GLM
            # spacecraft are active without a ground-network overlay.
            # LMA station entries, when enabled, remain on that same row.
            combined = ordinary_entries + [
                entry
                for _spacecraft, entries in ordered_glm_groups
                for entry in entries
            ]
            legend_columns = max(1, len(combined))
            legend_handles = [handle for handle, _ in combined]
            legend_labels = [label for _, label in combined]
            legend_fontsize = 9.0 + ordinary_delta
        elif glm_entries:
            # Preserve the currently working Landscape grouped arrangement for
            # every other base + overlay combination.
            legend_columns = 4
            legend_handles, legend_labels = pack_two_rows(
                ordinary_entries,
                glm_entries,
                columns=legend_columns,
            )
            legend_fontsize = 8.2 + ordinary_delta
        else:
            legend_columns = max(1, len(ordinary_entries))
            legend_handles = [handle for handle, _ in ordinary_entries]
            legend_labels = [label for _, label in ordinary_entries]
            legend_fontsize = 9.0 + ordinary_delta

        legend_clearance = (
            0.62 + max(0.0, float(extra_clearance_inches))
        ) / max(float(fig.get_figheight()), 1.0)
        legend_top = max(0.012, bottom - legend_clearance)

        legend_location = "upper center"
        legend_anchor = (axes_centre_x, legend_top)
        borderpad = 0.35
        handletextpad = 0.42
        columnspacing = 0.78
        labelspacing = 0.30
        handlelength = 1.45

    else:
        legend_columns = max(1, min(len(handles), 4))
        legend_fontsize = 9.0 + ordinary_delta
        legend_clearance = (
            0.62 + max(0.0, float(extra_clearance_inches))
        ) / max(float(fig.get_figheight()), 1.0)
        legend_top = max(0.012, bottom - legend_clearance)

        legend_location = "upper center"
        legend_anchor = (axes_centre_x, legend_top)
        borderpad = 0.45
        handletextpad = 0.55
        columnspacing = 1.2
        labelspacing = 0.50
        handlelength = 2.0

    legend = fig.legend(
        legend_handles,
        legend_labels,
        loc=legend_location,
        bbox_to_anchor=legend_anchor,
        bbox_transform=fig.transFigure,
        ncol=legend_columns,
        fontsize=legend_fontsize,
        frameon=True,
        framealpha=0.86,
        borderpad=borderpad,
        handletextpad=handletextpad,
        columnspacing=columnspacing,
        labelspacing=labelspacing,
        handlelength=handlelength,
    )

    frame = legend.get_frame()
    frame.set_facecolor(values["axes"])
    frame.set_edgecolor(values["text"])
    for text in legend.get_texts():
        text.set_color(values["text"])

    # Keep the legend included in bbox_inches="tight" saved outputs.
    legend.set_in_layout(True)
    legend.set_zorder(110)
    return legend

def refresh_figure_legend(
    fig,
    source_axes,
    plot: PlotSpec,
    *,
    extra_clearance_inches: float = 0.0,
):
    """Rebuild the figure legend after dynamic overlay layers change."""
    metadata = getattr(fig, "_lmas_metadata", {})
    prior = metadata.get("legend") if isinstance(metadata, dict) else None
    if prior is not None:
        try:
            prior.remove()
        except Exception:
            pass
    legend = _add_figure_legend(
        fig, source_axes, plot, extra_clearance_inches=extra_clearance_inches
    )
    if isinstance(metadata, dict):
        metadata = dict(metadata)
        metadata["legend"] = legend
        fig._lmas_metadata = metadata
    return legend


def _plot_stations_local_portrait(
    project, dataset, plan_axis, east_altitude_axis, altitude_north_axis, theme: str,
    *, show_labels: bool = False, show_vertical: bool = False,
) -> None:
    station = station_local_coordinates(
        dataset, project.reference_longitude, project.reference_latitude
    )
    if station is None:
        return
    sx, sy = (np.asarray(value, dtype=float) for value in station)
    values = theme_values(theme)
    plan_axis.scatter(sx, sy, marker="^", s=34, facecolors="none",
                      edgecolors=values["station"], linewidths=0.9, zorder=20,
                      label="LMA stations")
    codes = _station_codes(dataset)
    if show_labels and codes is not None and codes.size == sx.size and sx.size <= 60:
        for code, x_value, y_value in zip(codes, sx, sy, strict=False):
            plan_axis.annotate(str(code), (x_value, y_value), xytext=(3, 3),
                               textcoords="offset points", fontsize=7, color=values["text"])
    if not show_vertical:
        return
    station_altitude = np.zeros_like(sx)
    if "station_altitude" in dataset:
        station_altitude = np.asarray(dataset["station_altitude"].values, dtype=float)
        units = str(dataset["station_altitude"].attrs.get("units", "")).lower()
        finite = station_altitude[np.isfinite(station_altitude)]
        if units in {"m", "meter", "meters", "metre", "metres"} or (
            finite.size and np.nanmedian(np.abs(finite)) > 100.0
        ):
            station_altitude = station_altitude / 1000.0
        if station_altitude.size != sx.size:
            station_altitude = np.zeros_like(sx)
    east_altitude_axis.scatter(sx, station_altitude, marker="^", s=22,
                               facecolors="none", edgecolors=values["station"], linewidths=0.7)
    altitude_north_axis.scatter(station_altitude, sy, marker="^", s=22,
                                facecolors="none", edgecolors=values["station"], linewidths=0.7)

def _plot_stations_geodetic(
    dataset,
    plan_axis,
    horizontal_altitude_axis,
    altitude_vertical_axis,
    theme: str,
    *,
    show_labels: bool = False,
    show_vertical: bool = False,
) -> None:
    if "station_longitude" not in dataset or "station_latitude" not in dataset:
        return
    slon = np.asarray(dataset["station_longitude"].values, dtype=float)
    slat = np.asarray(dataset["station_latitude"].values, dtype=float)
    valid = np.isfinite(slon) & np.isfinite(slat)
    slon, slat = slon[valid], slat[valid]
    if not slon.size:
        return
    values = theme_values(theme)
    plan_axis.scatter(slon, slat, marker="^", s=34, facecolors="none",
                      edgecolors=values["station"], linewidths=0.9, zorder=20,
                      label="LMA stations")
    codes = _station_codes(dataset)
    if show_labels and codes is not None and codes.size == valid.size and slon.size <= 60:
        for code, x_value, y_value in zip(codes[valid], slon, slat, strict=False):
            plan_axis.annotate(str(code), (x_value, y_value), xytext=(3, 3),
                               textcoords="offset points", fontsize=7,
                               color=values["text"])
    if not show_vertical:
        return
    station_altitude = np.zeros_like(slon)
    if "station_altitude" in dataset:
        raw = np.asarray(dataset["station_altitude"].values, dtype=float)
        if raw.size == valid.size:
            station_altitude = raw[valid]
            units = str(dataset["station_altitude"].attrs.get("units", "")).lower()
            finite = station_altitude[np.isfinite(station_altitude)]
            if units in {"m", "meter", "meters", "metre", "metres"} or (
                finite.size and np.nanmedian(np.abs(finite)) > 100.0
            ):
                station_altitude = station_altitude / 1000.0
    horizontal_altitude_axis.scatter(slon, station_altitude, marker="^", s=22,
                                     facecolors="none", edgecolors=values["station"],
                                     linewidths=0.7)
    altitude_vertical_axis.scatter(station_altitude, slat, marker="^", s=22,
                                   facecolors="none", edgecolors=values["station"],
                                   linewidths=0.7)


def _update_altitude_histogram(axis, altitude, mask, altitude_limits, plot: PlotSpec, *, show_y_labels: bool, portrait: bool = False) -> None:
    axis.cla()
    low, high = sorted(float(value) for value in altitude_limits)
    selected_values = np.asarray(altitude[np.asarray(mask, dtype=bool)], dtype=float)
    selected_values = selected_values[np.isfinite(selected_values)]
    selected_values = selected_values[(selected_values >= low) & (selected_values <= high)]
    edges = np.linspace(low, high, 61, dtype=float)
    counts, edges = np.histogram(selected_values, bins=edges)
    total = int(counts.sum())
    fractions = counts.astype(float) / total if total else np.zeros_like(counts, dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    hist_color = "deepskyblue" if plot.theme != "light" else "steelblue"
    axis.barh(centers, fractions, height=np.diff(edges), align="center",
              color=hist_color, alpha=0.85, linewidth=0.0)
    axis.set_ylim(low, high)
    maximum = float(np.max(fractions)) if fractions.size else 0.0
    axis.set_xlim(0.0, max(maximum * 1.08, 0.01))
    axis.set_xlabel("Source fraction")
    axis.set_ylabel("")
    _apply_intfs_axis_ticks(axis)
    axis.tick_params(axis="y", labelleft=bool(show_y_labels), labelright=False)
    theme = theme_values(plot.theme)
    axis.set_facecolor(theme["axes"])
    axis.tick_params(colors=theme["text"], which="both")
    axis.xaxis.label.set_color(theme["text"])
    for spine in axis.spines.values():
        spine.set_color(theme["text"])
    if plot.show_grid:
        axis.grid(True, which="major", linewidth=0.35, alpha=0.24, color=theme["grid"])
    axis._lmas_source_fraction = fractions
    axis._lmas_altitude_bin_edges = edges
    axis._lmas_histogram_source_count = total
    _add_histogram_panel_label(axis, plot, portrait=portrait)


def create_intfs_figure(project: LMAProject, filters: FilterSpec, plot: PlotSpec) -> Figure:
    selected = project.selected_dataset(filters)
    if selected.sizes.get("number_of_events", 0) == 0:
        raise DatasetError("No LMA sources pass the selected filters")
    (time, time_num, source_time_s, lat, lon, alt, east, north, source_ids,
     colors, color_label, norm, valid_event_mask, categorical_colors, categorical_labels) = _valid_event_arrays(project, selected, plot)
    filtered_count = len(time)
    size = plot.point_size or automatic_point_size(filtered_count)
    if plot.color_by == "charge":
        cmap = _charge_cmap_and_norm()[0]
    elif plot.color_by == "group":
        cmap = _group_cmap_and_norm(categorical_colors)[0]
    else:
        cmap = resolved_cmap(plot.cmap, reverse=plot.reverse_cmap)
    is_local = plot.coordinate_system == "local"
    if is_local:
        east_sign = -1.0 if plot.north_south_viewpoint == "north" else 1.0
        north_sign = -1.0 if plot.east_west_viewpoint == "west" else 1.0
        x_values, y_values = east_sign * east, north_sign * north
        x_name = "west" if east_sign < 0 else "east"
        y_name = "south" if north_sign < 0 else "north"
        x_label = _local_cardinal_axis_label("east_west", east_sign)
        y_label = _local_cardinal_axis_label("north_south", north_sign)
    else:
        east_sign = north_sign = 1.0
        x_values, y_values = lon, lat
        x_name, y_name = "longitude", "latitude"
        x_label, y_label = "Longitude (degrees)", "Latitude (degrees)"
    depth_keys = spatial_depth_keys(east, north, alt, source_time_s=source_time_s,
                                    east_west_viewpoint=plot.east_west_viewpoint,
                                    north_south_viewpoint=plot.north_south_viewpoint,
                                    mode=plot.depth_mode)
    raw_time, raw_time_num, raw_lat, raw_lon, raw_alt, raw_east, raw_north, raw_source_ids = _finite_coordinate_arrays(project, project.dataset)
    raw_x = east_sign * raw_east if is_local else raw_lon
    raw_y = north_sign * raw_north if is_local else raw_lat
    filtered_coordinate_values = {"time": time_num, "altitude": alt, x_name: x_values, y_name: y_values}
    unfiltered_coordinate_values = {"time": raw_time_num, "altitude": raw_alt, x_name: raw_x, y_name: raw_y}
    precision_source_values = _precision_source_values(
        selected, valid_event_mask, time=time, time_num=time_num, latitude=lat,
        longitude=lon, altitude=alt, east=east, north=north, source_ids=source_ids,
    )

    fig = Figure(figsize=(15.0, 8.4), dpi=plot.dpi, constrained_layout=False)
    left, right = 0.065, 0.915
    original_left, original_right = 0.090, 0.865
    horizontal_scale = (right - left) / (original_right - original_left)
    scaled_x = lambda value: left + (value - original_left) * horizontal_scale
    panel_width = 0.225 * horizontal_scale
    hist_enabled = bool(plot.show_histogram)
    if hist_enabled:
        hist_height = 0.2325
        hist_width = 0.75 * hist_height * (8.4 / 15.0)
        hist_right = right
        hist_left = hist_right - hist_width
        axis_gap = 0.020
        time_right = hist_left - axis_gap
        ax_time = fig.add_axes([left, 0.690, time_right - left, hist_height])
        ax_hist = fig.add_axes([hist_left, 0.690, hist_width, hist_height], sharey=ax_time)
    else:
        ax_time = fig.add_axes([left, 0.690, right - left, 0.2325])
        ax_hist = None
    ax_nz = fig.add_axes([scaled_x(0.090), 0.105, panel_width, 0.490])
    ax_ez = fig.add_axes([scaled_x(0.340), 0.105, panel_width, 0.490])
    ax_plan = fig.add_axes([scaled_x(0.640), 0.105, panel_width, 0.490])
    axes = [ax_time, ax_nz, ax_ez, ax_plan]
    all_axes = axes + ([ax_hist] if ax_hist is not None else [])
    coordinate_pairs = ((time_num, alt), (y_values, alt), (x_values, alt), (x_values, y_values))
    all_coordinate_pairs = (
        (raw_time_num, raw_alt),
        (raw_y, raw_alt),
        (raw_x, raw_alt),
        (raw_x, raw_y),
    )
    preview_order = np.argsort(source_time_s, kind="stable")
    preview_indices = _deterministic_preview_indices(source_time_s, filtered_count, plot.preview_point_limit)
    scatters = []
    scatter_orders: list[np.ndarray] = []
    for index, (axis, pair) in enumerate(zip(axes, coordinate_pairs)):
        panel_order = preview_indices
        if depth_keys[index] is not None and preview_indices.size:
            key = np.asarray(depth_keys[index], dtype=float)
            panel_order = preview_indices[np.argsort(key[preview_indices], kind="stable")]
        panel_order = np.asarray(panel_order, dtype=np.int64)
        scatter_orders.append(panel_order)
        scatters.append(_scatter(axis, pair[0], pair[1], colors, cmap=cmap, norm=norm, size=size, order=panel_order))

    title_formatter = _view_title(project, filters, plot)
    title_artist = fig.suptitle("", y=0.975, fontsize=15)
    time_origin, window_span_s = _window_time_reference(project, filters, time_num)
    ax_time.set_ylabel("Altitude (km MSL)")
    _set_time_limits(ax_time, time_num)
    _refresh_time_axis(
        ax_time, plot.theme,
        relative=plot.relative_time_from_window_start,
        origin=time_origin, window_span_s=window_span_s,
    )
    date_anchor = ax_time.xaxis.get_offset_text()
    ax_nz.set_xlabel(y_label); ax_nz.set_ylabel("Altitude (km MSL)")
    ax_ez.set_xlabel(x_label); ax_ez.set_ylabel("")
    ax_plan.set_xlabel(x_label); ax_plan.set_ylabel(y_label)

    if is_local:
        centre = station_center_local_km(project.dataset, project.reference_longitude, project.reference_latitude)
        centre_x, centre_y = centre if centre is not None else (0.0, 0.0)
        centre_x *= east_sign; centre_y *= north_sign
        xlim, ylim = (centre_x - 200.0, centre_x + 200.0), (centre_y - 200.0, centre_y + 200.0)
        padding_xy = 0.05
    else:
        centre = station_center_latlon(project.dataset)
        centre_x, centre_y = centre if centre is not None else (project.reference_longitude, project.reference_latitude)
        lat_half = 200.0 / 111.195
        lon_half = 200.0 / max(111.195 * np.cos(np.deg2rad(centre_y)), 1.0e-6)
        xlim, ylim = (centre_x - lon_half, centre_x + lon_half), (centre_y - lat_half, centre_y + lat_half)
        padding_xy = 0.0005
    zlim = (-0.75, 30.0)
    ax_time.set_ylim(zlim); ax_nz.set_xlim(ylim); ax_nz.set_ylim(zlim)
    ax_ez.set_xlim(xlim); ax_ez.set_ylim(zlim); ax_plan.set_xlim(xlim); ax_plan.set_ylim(ylim)
    initial_limits = {"time": tuple(sorted(ax_time.get_xlim())), "altitude": zlim,
                      x_name: tuple(sorted(xlim)), y_name: tuple(sorted(ylim))}
    initial_visible = _count_in_limits(filtered_coordinate_values, initial_limits)
    initial_in_view = _count_in_limits(unfiltered_coordinate_values, initial_limits)
    title_artist.set_text(title_formatter(initial_visible, initial_in_view, min(initial_visible, preview_indices.size)))
    if is_local and plot.show_east_west_title:
        ax_nz.set_title(f"View from {plot.east_west_viewpoint.title()}")
    if is_local and plot.show_north_south_title:
        ax_ez.set_title(f"View from {plot.north_south_viewpoint.title()}")
    for axis in (ax_nz, ax_ez, ax_plan):
        axis.set_box_aspect(1); axis.set_aspect("auto"); axis.set_anchor("N"); _apply_intfs_axis_ticks(axis)
    if not is_local:
        _configure_geodetic_axis(ax_nz, x=True)
        _configure_geodetic_axis(ax_ez, x=True)
        _configure_geodetic_axis(ax_plan, x=True, y=True)
    coordinate_names = (("time", "altitude"), (y_name, "altitude"), (x_name, "altitude"), (x_name, y_name))
    axis_data_aspects = _apply_true_spatial_aspects(
        axes, coordinate_names, enabled=plot.true_aspect,
        reference_latitude=project.reference_latitude,
        anchors=("N", "N", "N", "N"),
    )
    map_underlay = (
        add_map_underlay(
            ax_plan, coordinate_system=plot.coordinate_system,
            reference_longitude=project.reference_longitude,
            reference_latitude=project.reference_latitude,
            east_sign=east_sign, north_sign=north_sign, theme=plot.theme,
        ) if plot.show_map_underlay else None
    )
    ax_ez.tick_params(labelleft=False, labelright=False)

    if plot.show_stations:
        if is_local:
            _plot_stations_local(project, selected, ax_plan, ax_nz, ax_ez, plot.theme,
                                 show_labels=plot.show_station_labels,
                                 show_vertical=plot.show_stations_in_vertical_projections,
                                 east_sign=east_sign, north_sign=north_sign)
        else:
            _plot_stations_geodetic(selected, ax_plan, ax_ez, ax_nz, plot.theme,
                                    show_labels=plot.show_station_labels,
                                    show_vertical=plot.show_stations_in_vertical_projections)
    apply_figure_theme(fig, all_axes, plot.theme, show_grid=plot.show_grid)
    subset_callback = None
    if ax_hist is not None:
        subset_callback = lambda mask: _update_altitude_histogram(ax_hist, alt, mask, ax_time.get_ylim(), plot, show_y_labels=False, portrait=False)
        subset_callback(np.ones(filtered_count, dtype=bool))
    colorbar = None
    if plot.show_colorbar:
        colorbar = add_aligned_vertical_colorbar(fig, scatters[0], axes=all_axes, label=color_label,
                                                 theme=plot.theme, pad=0.010, width=0.012)
        if plot.color_by == "charge":
            _style_charge_colorbar(colorbar)
        elif plot.color_by == "group":
            _style_group_colorbar(colorbar, categorical_labels)
    panel_label_artists = _add_panel_labels(
        axes, plot, histogram_axis=ax_hist, portrait=False
    )
    _apply_text_sizes(fig, all_axes, plot, title_artist=title_artist, colorbar=colorbar)
    legend = _add_figure_legend(fig, all_axes, plot)
    fig._lmas_metadata = {
        "layout": "intfs", "linked_view": True, "theme": plot.theme, "plot_spec": plot,
        "selection_dataset_key": (id(project.dataset), int(project.event_count)),
        "axes": {"time_altitude": ax_time, "north_altitude": ax_nz, "east_altitude": ax_ez, "plan": ax_plan,
                 **({"histogram": ax_hist} if ax_hist is not None else {})},
        "axis_order": tuple(axes), "coordinate_pairs": coordinate_pairs,
        "coordinate_names": coordinate_names,
        "selection_scopes": {
            "filtered": {
                "source_ids": source_ids,
                "coordinate_pairs": coordinate_pairs,
                "time": time,
                "altitude_km": alt,
            },
            "all": {
                "source_ids": raw_source_ids,
                "coordinate_pairs": all_coordinate_pairs,
                "time": raw_time,
                "altitude_km": raw_alt,
            },
        },
        "interactive_limit_axes": {"time": (0, "x"), x_name: (3, "x"), y_name: (3, "y"), "altitude": (1, "y")},
        "axis_padding": ((0.01, 1.0/86_400_000.0, 0.03, 0.05),
                         (0.03, padding_xy, 0.03, 0.05),
                         (0.03, padding_xy, 0.03, 0.05),
                         (0.03, padding_xy, 0.03, padding_xy)),
        "locked_box_axes": (0,1,2,3),
        "equal_scale_axes": (1,2,3) if plot.true_aspect else (),
        "axis_data_aspects": axis_data_aspects,
        "spatial_axes": (1,2,3), "altitude_axes": (0,1,2), "scatters": scatters,
        "scatter_depth_keys": depth_keys, "filtered_count": filtered_count, "loaded_count": project.event_count,
        "filtered_coordinate_values": filtered_coordinate_values, "unfiltered_coordinate_values": unfiltered_coordinate_values,
        "filter_spec": filters.to_dict(), "vertical_reference": "MSL", "ground_subtraction_applied": False,
        "source_count": filtered_count, "source_ids": source_ids,
        "scatter_orders": tuple(scatter_orders),
        "precision_source_values": precision_source_values, "color_values": colors,
        "full_color_limits": _full_color_limits(norm), "norm": norm, "full_norm": norm,
        "color_by": plot.color_by,
        "time_colored": plot.color_by == "time", "categorical_color": plot.color_by in {"charge", "group"}, "colorbar": colorbar,
        "colorbar_update_callback": (
            (lambda: _style_charge_colorbar(colorbar))
            if plot.color_by == "charge"
            else ((lambda: _style_group_colorbar(colorbar, categorical_labels)) if plot.color_by == "group" else None)
        ),
        "legend": legend, "panel_labels": panel_label_artists,
        "auto_fit_spatial": plot.auto_fit_spatial, "remap_time_colors": plot.remap_time_colors,
        "remap_colormap": bool(plot.remap_time_colors and plot.color_by not in {"charge", "group"}), "viewpoints": {"north_south": plot.north_south_viewpoint, "east_west": plot.east_west_viewpoint},
        "time_axis_callback": lambda: _refresh_time_axis(
            ax_time, plot.theme, relative=plot.relative_time_from_window_start,
            origin=time_origin, window_span_s=window_span_s,
        ), "date_anchor_artist": date_anchor,
        "time_origin": time_origin, "window_time_span_s": window_span_s,
        "map_underlay": map_underlay,
        "map_update_callback": (map_underlay.update if map_underlay is not None else None),
        "depth_mode": plot.depth_mode, "title_artist": title_artist, "title_formatter": title_formatter,
        "preview_point_limit": int(plot.preview_point_limit), "preview_order": preview_order,
        "displayed_count": int(preview_indices.size), "subset_callback": subset_callback,
        "coordinate_system": plot.coordinate_system, "reference_latitude": float(project.reference_latitude), "geometry_version": "landscape-linked-v1.0.0",
    }
    return fig


def create_xlma_figure(project: LMAProject, filters: FilterSpec, plot: PlotSpec, *, for_export: bool = False) -> Figure:
    selected = project.selected_dataset(filters)
    if selected.sizes.get("number_of_events", 0) == 0:
        raise DatasetError("No LMA sources pass the selected filters")
    (time, time_num, source_time_s, lat, lon, alt, east, north, source_ids,
     colors, color_label, norm, valid_event_mask, categorical_colors, categorical_labels) = _valid_event_arrays(project, selected, plot)
    filtered_count = len(time)
    size = plot.point_size or automatic_point_size(filtered_count)
    if plot.color_by == "charge":
        cmap = _charge_cmap_and_norm()[0]
    elif plot.color_by == "group":
        cmap = _group_cmap_and_norm(categorical_colors)[0]
    else:
        cmap = resolved_cmap(plot.cmap, reverse=plot.reverse_cmap)
    is_local = plot.coordinate_system == "local"
    if is_local:
        x_values, y_values = east, north
        x_name, y_name = "east", "north"
        x_label = _local_cardinal_axis_label("east_west")
        y_label = _local_cardinal_axis_label("north_south")
    else:
        x_values, y_values = lon, lat
        x_name, y_name = "longitude", "latitude"
        x_label, y_label = "Longitude (degrees)", "Latitude (degrees)"
    raw_time, raw_time_num, raw_lat, raw_lon, raw_alt, raw_east, raw_north, raw_source_ids = _finite_coordinate_arrays(project, project.dataset)
    raw_x, raw_y = (raw_east, raw_north) if is_local else (raw_lon, raw_lat)
    filtered_coordinate_values = {"time": time_num, "altitude": alt, x_name: x_values, y_name: y_values}
    unfiltered_coordinate_values = {"time": raw_time_num, "altitude": raw_alt, x_name: raw_x, y_name: raw_y}
    precision_source_values = _precision_source_values(
        selected, valid_event_mask, time=time, time_num=time_num, latitude=lat,
        longitude=lon, altitude=alt, east=east, north=north, source_ids=source_ids,
    )

    # Reserve a generous left Portrait margin for the optional vertical GLM
    # TOE colorbar, its tick labels, and its full scientific label while
    # preserving the established scientific-panel dimensions.
    figure_width = 10.55 if plot.show_colorbar else 8.55
    fig = Figure(figsize=(figure_width, 11.0), dpi=plot.dpi, constrained_layout=False)

    # Portrait uses one print-correct geometry model for both the embedded
    # preview and saved output.  FigureHost preserves the full canvas ratio in
    # the Qt application, so these inch-based dimensions scale uniformly rather
    # than stretching individual axes.  One common physical altitude length is
    # used everywhere altitude is plotted: the two vertical panels, the square
    # histogram, and the lower-right horizontal altitude axis.
    def box(x, y, w, h):
        return [x / figure_width, y / 11.0, w / figure_width, h / 11.0]

    # Give the portrait page balanced outer whitespace.  With a colorbar,
    # the plotting block and colorbar are centered together on the full
    # 9.25-inch canvas; without one, the science block is centered on 8 inches.
    left = 1.85 if plot.show_colorbar else 1.20
    plan_bottom = 1.35
    plan_size = 5.30
    panel_gap = 0.35
    row_gap = 0.53
    altitude_axis_size = 1.05
    side_left = left + plan_size + panel_gap
    middle_bottom = plan_bottom + plan_size + row_gap
    time_bottom = 8.90
    full_science_width = plan_size + panel_gap + altitude_axis_size

    ax_time = fig.add_axes(box(left, time_bottom, full_science_width, altitude_axis_size))
    ax_x_alt = fig.add_axes(box(left, middle_bottom, plan_size, altitude_axis_size))
    ax_hist = fig.add_axes(box(side_left, middle_bottom, altitude_axis_size, altitude_axis_size), sharey=ax_x_alt)
    ax_plan = fig.add_axes(box(left, plan_bottom, plan_size, plan_size))
    ax_alt_y = fig.add_axes(box(side_left, plan_bottom, altitude_axis_size, plan_size))
    axes = [ax_time, ax_x_alt, ax_plan, ax_alt_y]
    all_axes = [ax_time, ax_x_alt, ax_hist, ax_plan, ax_alt_y]
    coordinate_pairs = ((time_num, alt), (x_values, alt), (x_values, y_values), (alt, y_values))
    all_coordinate_pairs = (
        (raw_time_num, raw_alt),
        (raw_x, raw_alt),
        (raw_x, raw_y),
        (raw_alt, raw_y),
    )
    preview_order = np.argsort(source_time_s, kind="stable")
    preview_indices = _deterministic_preview_indices(source_time_s, filtered_count, plot.preview_point_limit)
    time_order = preview_indices[np.argsort(source_time_s[preview_indices], kind="stable")] if preview_indices.size else preview_indices
    time_order = np.asarray(time_order, dtype=np.int64)
    scatters = [_scatter(axis, pair[0], pair[1], colors, cmap=cmap, norm=norm, size=size, order=time_order)
                for axis, pair in zip(axes, coordinate_pairs)]
    scatter_orders = tuple(time_order.copy() for _axis in axes)

    base_title_formatter = _view_title(project, filters, plot)
    # Portrait-only title wrapping: all text-size presets wrap at the semantic
    # em dash.  This keeps long multi-minute view summaries inside the locked
    # portrait canvas in both preview and export.
    title_formatter = _stacked_title(base_title_formatter)
    # Use one canvas-level title for both the embedded portrait preview and
    # saved portrait figures.  This centers the title against the complete
    # page (including the colorbar gutter) and keeps it clear of the top axes.
    title_artist = fig.suptitle(
        "", x=0.5, y=0.975, ha="center", va="top", fontsize=13, linespacing=1.12
    )
    time_origin, window_span_s = _window_time_reference(project, filters, time_num)
    ax_time.set_ylabel("Altitude (km MSL)")
    _set_time_limits(ax_time, time_num)
    _refresh_time_axis(
        ax_time, plot.theme,
        relative=plot.relative_time_from_window_start,
        origin=time_origin, window_span_s=window_span_s,
    )
    date_anchor = ax_time.xaxis.get_offset_text()
    ax_x_alt.set_xlabel(x_label); ax_x_alt.set_ylabel("Altitude (km MSL)")
    ax_plan.set_xlabel(x_label); ax_plan.set_ylabel(y_label)
    ax_alt_y.set_xlabel("Altitude (km MSL)"); ax_alt_y.tick_params(labelleft=False)
    ax_plan.set_box_aspect(1)
    ax_plan.set_aspect("auto")
    ax_plan.set_anchor("C")

    if is_local:
        centre = station_center_local_km(project.dataset, project.reference_longitude, project.reference_latitude)
        centre_x, centre_y = centre if centre is not None else (0.0, 0.0)
        xlim, ylim = (centre_x - 200.0, centre_x + 200.0), (centre_y - 200.0, centre_y + 200.0)
        padding_xy = 0.05
    else:
        centre = station_center_latlon(project.dataset)
        centre_x, centre_y = centre if centre is not None else (project.reference_longitude, project.reference_latitude)
        lat_half = 200.0 / 111.195
        lon_half = 200.0 / max(111.195 * np.cos(np.deg2rad(centre_y)), 1.0e-6)
        xlim, ylim = (centre_x - lon_half, centre_x + lon_half), (centre_y - lat_half, centre_y + lat_half)
        padding_xy = 0.0005
    zlim = (-0.75, 30.0)
    ax_time.set_ylim(zlim); ax_x_alt.set_xlim(xlim); ax_x_alt.set_ylim(zlim)
    ax_plan.set_xlim(xlim); ax_plan.set_ylim(ylim); ax_alt_y.set_xlim(zlim); ax_alt_y.set_ylim(ylim)
    initial_limits = {"time": tuple(sorted(ax_time.get_xlim())), "altitude": zlim,
                      x_name: tuple(sorted(xlim)), y_name: tuple(sorted(ylim))}
    initial_visible = _count_in_limits(filtered_coordinate_values, initial_limits)
    initial_in_view = _count_in_limits(unfiltered_coordinate_values, initial_limits)
    title_artist.set_text(title_formatter(initial_visible, initial_in_view, min(initial_visible, preview_indices.size)))

    def update_histogram(mask):
        _update_altitude_histogram(ax_hist, alt, mask, ax_x_alt.get_ylim(), plot, show_y_labels=True, portrait=True)
    update_histogram(np.ones(filtered_count, dtype=bool))
    if plot.show_stations:
        if is_local:
            _plot_stations_local_portrait(project, selected, ax_plan, ax_x_alt, ax_alt_y, plot.theme,
                                          show_labels=plot.show_station_labels,
                                          show_vertical=plot.show_stations_in_vertical_projections)
        else:
            _plot_stations_geodetic(selected, ax_plan, ax_x_alt, ax_alt_y, plot.theme,
                                    show_labels=plot.show_station_labels,
                                    show_vertical=plot.show_stations_in_vertical_projections)
    for index, axis in enumerate(all_axes):
        _apply_intfs_axis_ticks(axis, time_x=index == 0)
    if not is_local:
        _configure_geodetic_axis(ax_x_alt, x=True)
        _configure_geodetic_axis(ax_plan, x=True, y=True)
        _configure_geodetic_axis(ax_alt_y, y=True)
    coordinate_names = (("time", "altitude"), (x_name, "altitude"), (x_name, y_name), ("altitude", y_name))
    axis_data_aspects = _apply_true_spatial_aspects(
        axes, coordinate_names, enabled=plot.true_aspect,
        reference_latitude=project.reference_latitude,
        anchors=("N", "N", "C", "N"),
        # Portrait keeps 1 km = 1 km in the square plan view.  The shallow
        # altitude projections retain linked scientific limits but are allowed
        # to stretch vertically/horizontally so they cannot block plan zoom.
        axis_indices=(2,),
    )
    map_underlay = (
        add_map_underlay(
            ax_plan, coordinate_system=plot.coordinate_system,
            reference_longitude=project.reference_longitude,
            reference_latitude=project.reference_latitude,
            east_sign=1.0, north_sign=1.0, theme=plot.theme,
        ) if plot.show_map_underlay else None
    )
    apply_figure_theme(fig, all_axes, plot.theme, show_grid=plot.show_grid)
    colorbar = None
    if plot.show_colorbar:
        # Keep the same print-correct colorbar gutter in preview and export.
        # Shift the source-color colorbar with the scientific block so the
        # expanded left margin remains dedicated to the vertical GLM TOE bar.
        # Match the colorbar exactly to the combined scientific-axis stack:
        # from the bottom of the plan/side-projection row to the top of the
        # time-altitude row.  Keep this portrait-only; Landscape is untouched.
        colorbar_bottom = plan_bottom
        colorbar_top = time_bottom + altitude_axis_size
        ax_color = fig.add_axes(
            box(9.00, colorbar_bottom, 0.18, colorbar_top - colorbar_bottom)
        )
        colorbar = fig.colorbar(scatters[0], cax=ax_color)
        colorbar.set_label(color_label); style_colorbar(colorbar, plot.theme)
        if plot.color_by == "charge":
            _style_charge_colorbar(colorbar)
        elif plot.color_by == "group":
            _style_group_colorbar(colorbar, categorical_labels)
    panel_label_artists = _add_panel_labels(
        axes, plot, histogram_axis=ax_hist, portrait=True
    )
    _apply_text_sizes(fig, all_axes, plot, title_artist=title_artist, colorbar=colorbar)
    legend = _add_figure_legend(fig, all_axes, plot)
    fig._lmas_metadata = {
        "layout": "xlma", "linked_view": True, "theme": plot.theme, "plot_spec": plot,
        "selection_dataset_key": (id(project.dataset), int(project.event_count)),
        "axes": {"time_altitude": ax_time, f"{x_name}_altitude": ax_x_alt, "plan": ax_plan,
                 f"altitude_{y_name}": ax_alt_y, "histogram": ax_hist},
        "axis_order": tuple(axes), "coordinate_pairs": coordinate_pairs,
        "coordinate_names": coordinate_names,
        "selection_scopes": {
            "filtered": {
                "source_ids": source_ids,
                "coordinate_pairs": coordinate_pairs,
                "time": time,
                "altitude_km": alt,
            },
            "all": {
                "source_ids": raw_source_ids,
                "coordinate_pairs": all_coordinate_pairs,
                "time": raw_time,
                "altitude_km": raw_alt,
            },
        },
        "interactive_limit_axes": {"time": (0, "x"), x_name: (2, "x"), y_name: (2, "y"), "altitude": (1, "y")},
        "axis_padding": ((0.01, 1.0/86_400_000.0, 0.03, 0.05),
                         (0.03, padding_xy, 0.03, 0.05),
                         (0.03, padding_xy, 0.03, padding_xy),
                         (0.03, 0.05, 0.03, padding_xy)),
        "locked_box_axes": (0,1,2,3),
        "equal_scale_axes": (2,) if plot.true_aspect else (),
        "axis_data_aspects": axis_data_aspects,
        "spatial_axes": (1,2,3), "altitude_axes": (0,1,3), "scatters": scatters,
        "scatter_depth_keys": (source_time_s,)*4, "filtered_count": filtered_count,
        "loaded_count": project.event_count, "filtered_coordinate_values": filtered_coordinate_values,
        "unfiltered_coordinate_values": unfiltered_coordinate_values, "filter_spec": filters.to_dict(),
        "vertical_reference": "MSL", "ground_subtraction_applied": False, "source_count": filtered_count,
        "source_ids": source_ids, "scatter_orders": scatter_orders,
        "precision_source_values": precision_source_values,
        "color_values": colors, "full_color_limits": _full_color_limits(norm),
        "norm": norm, "full_norm": norm, "time_colored": plot.color_by == "time",
        "color_by": plot.color_by,
        "categorical_color": plot.color_by in {"charge", "group"}, "colorbar": colorbar,
        "colorbar_update_callback": (
            (lambda: _style_charge_colorbar(colorbar))
            if plot.color_by == "charge"
            else ((lambda: _style_group_colorbar(colorbar, categorical_labels)) if plot.color_by == "group" else None)
        ),
        "legend": legend, "panel_labels": panel_label_artists,
        "auto_fit_spatial": plot.auto_fit_spatial, "remap_time_colors": plot.remap_time_colors,
        "remap_colormap": bool(plot.remap_time_colors and plot.color_by not in {"charge", "group"}), "subset_callback": update_histogram,
        "time_axis_callback": lambda: _refresh_time_axis(
            ax_time, plot.theme, relative=plot.relative_time_from_window_start,
            origin=time_origin, window_span_s=window_span_s,
        ), "date_anchor_artist": date_anchor,
        "time_origin": time_origin, "window_time_span_s": window_span_s,
        "map_underlay": map_underlay,
        "map_update_callback": (map_underlay.update if map_underlay is not None else None),
        "title_artist": title_artist, "title_formatter": title_formatter,
        "preview_point_limit": int(plot.preview_point_limit), "preview_order": preview_order,
        "displayed_count": int(preview_indices.size), "coordinate_system": plot.coordinate_system,
        "reference_latitude": float(project.reference_latitude),
        "export_size_inches": (figure_width, 11.0), "for_export": bool(for_export), "geometry_version": "portrait-linked-v1.0.0",
    }
    return fig

def create_lma_figure(
    project: LMAProject,
    *,
    filters: FilterSpec | None = None,
    plot: PlotSpec | None = None,
    for_export: bool = False,
) -> Figure:
    filter_spec = (filters or project.filters).validated()
    plot_spec = (plot or project.plot).validated()
    if plot_spec.layout == "intfs":
        return create_intfs_figure(project, filter_spec, plot_spec)
    return create_xlma_figure(project, filter_spec, plot_spec, for_export=for_export)


__all__ = ["create_lma_figure", "create_intfs_figure", "create_xlma_figure", "save_figure"]
