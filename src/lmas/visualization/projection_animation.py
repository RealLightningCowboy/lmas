from __future__ import annotations

from dataclasses import dataclass, field, replace
import gc
from pathlib import Path
from typing import Literal

import matplotlib as mpl
import matplotlib.dates as mdates
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np

from ..errors import ConfigurationError, DatasetError
from ..model import FilterSpec, LMAProject
from ..plotting import create_lma_figure
from .animation import (
    AnimationProgressReporter,
    animation_frame_times,
    animation_window_bounds_utc,
)
from .pyvista_3d import frame_display_rgba

DisplayMode = Literal["cumulative", "trail", "trail-afterimage"]
_DISPLAY_MODES = {"cumulative", "trail", "trail-afterimage"}


def combined_filter_spec(quality: FilterSpec, view: FilterSpec) -> FilterSpec:
    """Combine scientific-quality criteria with a non-destructive linked view."""

    quality = quality.validated()
    view = view.validated()
    return FilterSpec(
        start_time=view.start_time,
        end_time=view.end_time,
        minimum_stations=quality.minimum_stations,
        maximum_chi2=quality.maximum_chi2,
        minimum_altitude_km=view.minimum_altitude_km,
        maximum_altitude_km=view.maximum_altitude_km,
        minimum_power=quality.minimum_power,
        maximum_power=quality.maximum_power,
        minimum_x_km=view.minimum_x_km,
        maximum_x_km=view.maximum_x_km,
        minimum_y_km=view.minimum_y_km,
        maximum_y_km=view.maximum_y_km,
    ).validated()


def combined_project_filter(project: LMAProject) -> FilterSpec:
    """Combine a Project's quality filtering with its saved linked view."""

    return combined_filter_spec(project.filters, project.view_filters)


def _apply_saved_view(figure, project: LMAProject) -> None:
    metadata = getattr(figure, "_lmas_metadata", {})
    axes = metadata.get("axes", {})
    if not axes:
        return
    view = project.view_filters.validated()
    plot = project.plot.validated()
    if view.start_time and view.end_time:
        values = np.asarray(
            [np.datetime64(view.start_time, "us"), np.datetime64(view.end_time, "us")]
        ).astype(object)
        axes["time_altitude"].set_xlim(tuple(mdates.date2num(values)))
    if view.minimum_altitude_km is not None and view.maximum_altitude_km is not None:
        zlim = (float(view.minimum_altitude_km), float(view.maximum_altitude_km))
        for name in ("time_altitude", "north_altitude", "east_altitude"):
            axes[name].set_ylim(zlim)
    if view.minimum_x_km is not None and view.maximum_x_km is not None:
        xlim = (float(view.minimum_x_km), float(view.maximum_x_km))
        if plot.north_south_viewpoint == "north":
            xlim = (-xlim[1], -xlim[0])
        axes["east_altitude"].set_xlim(xlim)
        axes["plan"].set_xlim(xlim)
    if view.minimum_y_km is not None and view.maximum_y_km is not None:
        ylim = (float(view.minimum_y_km), float(view.maximum_y_km))
        if plot.east_west_viewpoint == "west":
            ylim = (-ylim[1], -ylim[0])
        axes["north_altitude"].set_xlim(ylim)
        axes["plan"].set_ylim(ylim)
    callback = metadata.get("time_axis_callback")
    if callable(callback):
        callback()



def _install_animation_header(figure, metadata: dict, title_artist):
    """Create a compact source-time line that stays clear of the top axes.

    Projection playback previously appended source time as a second suptitle
    line. On short laptop canvases that line extended into the altitude-versus-
    time axes. Keep the main title to one line and anchor a smaller time label
    immediately above the highest science axes instead.
    """

    if title_artist is not None:
        try:
            title_artist.set_y(0.992)
            title_artist.set_va("top")
            title_artist.set_linespacing(1.0)
        except Exception:
            pass

    axes = dict(metadata.get("axes", {}))
    top = 0.915
    positions = []
    for axis in axes.values():
        try:
            positions.append(float(axis.get_position().y1))
        except Exception:
            continue
    if positions:
        top = max(positions)

    time_y = min(0.965, top + 0.006)
    title_size = 15.0
    title_color = None
    if title_artist is not None:
        try:
            title_size = float(title_artist.get_fontsize())
        except Exception:
            pass
        try:
            title_color = title_artist.get_color()
        except Exception:
            pass
    kwargs = {
        "ha": "center",
        "va": "bottom",
        "fontsize": max(8.0, title_size - 5.0),
        "zorder": 20,
    }
    if title_color is not None:
        kwargs["color"] = title_color
    return figure.text(0.5, time_y, "", **kwargs)


@dataclass
class ProjectionAnimationScene:
    figure: object
    scatters: list
    coordinate_pairs: list
    depth_keys: list
    colors: np.ndarray
    time_ms: np.ndarray
    base_rgba: np.ndarray
    title_artist: object | None
    base_title: str
    chi2_suffix: str
    cmap: str | None
    reverse_cmap: bool
    total_source_count: int
    animation_start_ms: float
    animation_end_ms: float
    preserve_depth_order: bool = True
    time_artist: object | None = None
    _offsets: list[np.ndarray] = field(default_factory=list, init=False, repr=False)
    _depth_orders: list[np.ndarray | None] = field(
        default_factory=list, init=False, repr=False
    )
    _rgba_buffer: np.ndarray = field(init=False, repr=False)
    _facecolor_buffer: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.colors = np.ascontiguousarray(self.colors, dtype=float)
        self.time_ms = np.ascontiguousarray(self.time_ms, dtype=float)
        self.base_rgba = np.ascontiguousarray(self.base_rgba, dtype=np.uint8)
        self.animation_start_ms = float(self.animation_start_ms)
        self.animation_end_ms = float(self.animation_end_ms)
        if (
            not np.isfinite(self.animation_start_ms)
            or not np.isfinite(self.animation_end_ms)
            or self.animation_end_ms < self.animation_start_ms
        ):
            raise DatasetError("Projection animation window limits are invalid")
        count = int(self.colors.size)
        if self.time_ms.shape != (count,) or self.base_rgba.shape != (count, 4):
            raise DatasetError("Projection animation arrays are inconsistent")
        self._rgba_buffer = np.empty_like(self.base_rgba)
        self._facecolor_buffer = np.empty((count, 4), dtype=np.float32)

        self._offsets = []
        for pair in self.coordinate_pairs:
            x_values = np.asarray(pair[0], dtype=float)
            y_values = np.asarray(pair[1], dtype=float)
            if x_values.shape != (count,) or y_values.shape != (count,):
                raise DatasetError("Projection animation coordinates are inconsistent")
            offsets = np.empty((count, 2), dtype=float)
            offsets[:, 0] = x_values
            offsets[:, 1] = y_values
            self._offsets.append(np.ascontiguousarray(offsets))

        self._depth_orders = []
        for index in range(len(self.scatters)):
            key = self.depth_keys[index] if index < len(self.depth_keys) else None
            if not self.preserve_depth_order or key is None:
                self._depth_orders.append(None)
                continue
            values = np.asarray(key, dtype=float)
            if values.shape != (count,):
                self._depth_orders.append(None)
                continue
            self._depth_orders.append(
                np.asarray(np.argsort(values, kind="stable"), dtype=np.int64)
            )

    @property
    def first_time_ms(self) -> float:
        """Beginning of the selected animation window in scene-relative ms."""

        return self.animation_start_ms

    @property
    def final_time_ms(self) -> float:
        """End of the selected animation window in scene-relative ms."""

        return self.animation_end_ms

    @property
    def first_source_time_ms(self) -> float:
        return float(self.time_ms[0])

    @property
    def final_source_time_ms(self) -> float:
        return float(self.time_ms[-1])

    @property
    def displayed_source_count(self) -> int:
        return int(self.colors.size)

    @property
    def dynamic_artists(self) -> tuple:
        artists = list(self.scatters)
        if self.title_artist is not None:
            artists.append(self.title_artist)
        if self.time_artist is not None:
            artists.append(self.time_artist)
        return tuple(artists)

    def _visible_bounds(
        self,
        current_time_ms: float,
        *,
        display_mode: DisplayMode,
        trail_ms: float,
    ) -> tuple[int, int]:
        current = float(current_time_ms)
        end = int(np.searchsorted(self.time_ms, current, side="right"))
        if display_mode == "trail":
            if trail_ms <= 0:
                raise ConfigurationError("Trail duration must be positive")
            start = int(
                np.searchsorted(self.time_ms, current - float(trail_ms), side="left")
            )
        else:
            start = 0
        return max(0, min(start, end)), max(0, min(end, self.time_ms.size))

    def update(
        self,
        current_time_ms: float,
        *,
        display_mode: DisplayMode,
        trail_ms: float,
        afterimage_ms: float,
        update_title: bool = True,
    ) -> int:
        if display_mode not in _DISPLAY_MODES:
            raise ConfigurationError(
                f"Unknown projection-animation display mode: {display_mode}"
            )
        start, end = self._visible_bounds(
            current_time_ms,
            display_mode=display_mode,
            trail_ms=trail_ms,
        )
        if end > start:
            rgba = frame_display_rgba(
                self.base_rgba[start:end],
                self.time_ms[start:end],
                float(current_time_ms),
                display_mode=display_mode,
                trail_ms=trail_ms,
                afterimage_ms=afterimage_ms,
                cmap=self.cmap,
                reverse_cmap=self.reverse_cmap,
                out=self._rgba_buffer[start:end],
            )
            facecolors = self._facecolor_buffer[start:end]
            np.multiply(
                rgba,
                np.float32(1.0 / 255.0),
                out=facecolors,
                casting="unsafe",
            )
            visible_count = int(np.count_nonzero(rgba[:, 3]))
        else:
            rgba = self._rgba_buffer[:0]
            facecolors = self._facecolor_buffer[:0]
            visible_count = 0

        for index, scatter in enumerate(self.scatters):
            order = self._depth_orders[index] if index < len(self._depth_orders) else None
            if end <= start:
                offsets = self._offsets[index][:0]
                panel_colors = facecolors
            elif order is None:
                offsets = self._offsets[index][start:end]
                panel_colors = facecolors
            else:
                selected = order[(order >= start) & (order < end)]
                offsets = self._offsets[index][selected]
                panel_colors = facecolors[selected - start]
            scatter.set_offsets(offsets)
            scatter.set_facecolors(panel_colors)

        if update_title and self.title_artist is not None:
            displayed = self.displayed_source_count
            total = max(displayed, int(self.total_source_count))
            if displayed < total:
                population = (
                    f"{visible_count:,} visible of {displayed:,} displayed; "
                    f"{total:,} sources in view"
                )
            else:
                population = f"{visible_count:,} visible of {total:,} sources in view"
            self.title_artist.set_text(
                f"{self.base_title} — {population}{self.chi2_suffix}"
            )
            if self.time_artist is not None:
                self.time_artist.set_text(
                    f"Source time: {float(current_time_ms):.3f} ms"
                )
        return visible_count

    def close(self) -> None:
        # Explicitly clear the large Matplotlib object graph so completed GUI and
        # headless workers do not retain source arrays at interpreter shutdown.
        try:
            self.figure.clear()
        except Exception:
            pass
        self.scatters.clear()
        self.coordinate_pairs.clear()
        self.depth_keys.clear()
        self._offsets.clear()
        self._depth_orders.clear()
        self._rgba_buffer = np.empty((0, 4), dtype=np.uint8)
        self._facecolor_buffer = np.empty((0, 4), dtype=np.float32)


def _subset_animation_arrays(
    indices: np.ndarray,
    coordinate_pairs: list,
    depth_keys: list,
    colors: np.ndarray,
    time_utc: np.ndarray,
) -> tuple[list, list, np.ndarray, np.ndarray]:
    take = np.asarray(indices, dtype=np.int64)
    coordinate_pairs = [
        (np.asarray(pair[0])[take], np.asarray(pair[1])[take])
        for pair in coordinate_pairs
    ]
    depth_keys = [
        None if key is None else np.asarray(key)[take]
        for key in depth_keys
    ]
    return coordinate_pairs, depth_keys, np.asarray(colors)[take], np.asarray(time_utc)[take]


def _animation_time_values(metadata: dict, coordinate_pairs: list, count: int) -> np.ndarray:
    scope = dict(metadata.get("selection_scopes", {})).get("filtered", {})
    values = np.asarray(scope.get("time", ()))
    if values.shape == (int(count),):
        return values.astype("datetime64[us]")

    time_numbers = np.asarray(coordinate_pairs[0][0], dtype=float)
    return np.asarray(
        [
            np.datetime64(value.replace(tzinfo=None), "us")
            for value in mdates.num2date(time_numbers)
        ]
    )


def _time_stratified_indices(time_utc: np.ndarray, limit: int) -> np.ndarray:
    count = int(np.asarray(time_utc).size)
    limit = int(limit)
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    order = np.argsort(np.asarray(time_utc).astype("datetime64[ns]").astype(np.int64), kind="stable")
    if limit <= 0 or count <= limit:
        return np.asarray(order, dtype=np.int64)
    positions = np.linspace(0, count - 1, limit, dtype=np.int64)
    return np.asarray(order[positions], dtype=np.int64)


def build_projection_animation_scene(
    project: LMAProject,
    *,
    width: int | None = None,
    height: int | None = None,
    custom_title: str | None = None,
    interactive: bool = False,
    point_limit: int | None = None,
    preserve_depth_order: bool | None = None,
) -> ProjectionAnimationScene:
    filters = combined_project_filter(project)
    plot = project.plot.validated()
    if point_limit is None:
        resolved_limit = int(plot.preview_point_limit) if interactive else 0
        # A disabled main-view cap should not accidentally send millions of
        # sources into an interactive animation on an ordinary machine.
        if interactive and resolved_limit <= 0:
            resolved_limit = 50_000
    else:
        resolved_limit = max(0, int(point_limit))
    figure = create_lma_figure(
        project,
        filters=filters,
        plot=replace(plot, preview_point_limit=resolved_limit),
    )
    metadata = getattr(figure, "_lmas_metadata", {})
    scatters = list(metadata.get("scatters", ()))
    coordinate_pairs = list(metadata.get("coordinate_pairs", ()))
    depth_keys = list(metadata.get("scatter_depth_keys", ()))
    colors = np.asarray(metadata.get("color_values", ()), dtype=float)
    if not scatters or len(scatters) != len(coordinate_pairs) or colors.size == 0:
        raise DatasetError("The LMAS figure does not expose an animatable projection layout")

    time_utc = _animation_time_values(metadata, coordinate_pairs, colors.size)
    source_ids = np.asarray(metadata.get("source_ids", ()), dtype=np.int64)

    # Projects may store exact source membership in addition to rectangular
    # time/spatial bounds. Projection animations use that same membership.
    if project.selected_source_ids is not None and source_ids.shape == colors.shape:
        membership = np.isin(
            source_ids, np.asarray(project.selected_source_ids, dtype=np.int64)
        )
        selected = np.flatnonzero(membership)
        coordinate_pairs, depth_keys, colors, time_utc = _subset_animation_arrays(
            selected, coordinate_pairs, depth_keys, colors, time_utc
        )
        if colors.size == 0:
            raise DatasetError("The saved project subset contains no matching sources")

    finite_time = ~np.isnat(np.asarray(time_utc).astype("datetime64[ns]"))
    if not np.all(finite_time):
        selected = np.flatnonzero(finite_time)
        coordinate_pairs, depth_keys, colors, time_utc = _subset_animation_arrays(
            selected, coordinate_pairs, depth_keys, colors, time_utc
        )
    if colors.size == 0:
        raise DatasetError("Projection animation has no finite source times")

    total_source_count = int(colors.size)
    sample_order = _time_stratified_indices(time_utc, resolved_limit)
    coordinate_pairs, depth_keys, colors, time_utc = _subset_animation_arrays(
        sample_order, coordinate_pairs, depth_keys, colors, time_utc
    )

    # All animation updates can now use two binary searches and contiguous time
    # slices.  Spatial painter order, when requested for saved products, is
    # calculated once below rather than re-sorted on every frame.
    time_order = np.argsort(
        np.asarray(time_utc).astype("datetime64[ns]").astype(np.int64),
        kind="stable",
    )
    coordinate_pairs, depth_keys, colors, time_utc = _subset_animation_arrays(
        time_order, coordinate_pairs, depth_keys, colors, time_utc
    )

    _apply_saved_view(figure, project)
    dpi = max(72, int(plot.dpi))
    figure.set_dpi(dpi)
    if width is not None and height is not None:
        if int(width) < 320 or int(height) < 240:
            raise ConfigurationError("Projection animation dimensions are too small")
        figure.set_size_inches(
            float(width) / dpi,
            float(height) / dpi,
            forward=True,
        )

    window_start_utc, window_end_utc = animation_window_bounds_utc(
        time_utc,
        start_time=filters.start_time,
        end_time=filters.end_time,
    )
    origin = np.datetime64(window_start_utc, "us")
    time_ms = np.asarray(
        (np.asarray(time_utc).astype("datetime64[us]") - origin)
        / np.timedelta64(1, "ms"),
        dtype=float,
    )
    animation_start_ms = 0.0
    animation_end_ms = float(
        (np.datetime64(window_end_utc, "us") - origin) / np.timedelta64(1, "ms")
    )

    if plot.color_by == "time":
        # Projection development is a self-contained view of the selected
        # flash. Color the animated population from its first source through its
        # final source, independent of a larger record's saved color limits.
        colors = np.asarray(time_ms / 1000.0, dtype=float)
        finite_colors = colors[np.isfinite(colors)]
        low = float(np.min(finite_colors))
        high = float(np.max(finite_colors))
        if high <= low:
            low -= 0.5
            high += 0.5
        animation_norm = mpl.colors.Normalize(vmin=low, vmax=high)
        for scatter in scatters:
            scatter.set_norm(animation_norm)
            scatter.set_clim(low, high)
        colorbar = metadata.get("colorbar")
        if colorbar is not None:
            colorbar.update_normal(scatters[0])
            colorbar.set_label(
                f"Seconds after {str(origin).replace('T', ' ')[:23]} UTC"
            )
    elif project.color_norm_limits is not None:
        low, high = project.color_norm_limits
        for scatter in scatters:
            scatter.set_clim(float(low), float(high))

    base_rgba = np.asarray(
        np.round(scatters[0].to_rgba(colors) * 255.0), dtype=np.uint8
    )
    for scatter in scatters:
        scatter.set_array(None)
    title_artist = metadata.get("title_artist")
    # Interactive source time is displayed in the viewer control row. Saved
    # animations retain an embedded source-time label in a dedicated compact
    # header that stays clear of the top axes on short displays.
    time_artist = (
        None
        if interactive
        else _install_animation_header(figure, metadata, title_artist)
    )
    base_title = str(custom_title or plot.title or project.data_source_stem)
    chi2_suffix = (
        f" (χ² < {project.filters.maximum_chi2:.2f})"
        if project.filters.maximum_chi2 is not None
        else ""
    )
    keep_depth = (not interactive) if preserve_depth_order is None else bool(preserve_depth_order)
    return ProjectionAnimationScene(
        figure=figure,
        scatters=scatters,
        coordinate_pairs=coordinate_pairs,
        depth_keys=depth_keys,
        colors=colors,
        time_ms=time_ms,
        base_rgba=base_rgba,
        title_artist=title_artist,
        time_artist=time_artist,
        base_title=base_title,
        chi2_suffix=chi2_suffix,
        cmap=plot.cmap if plot.color_by == "time" else None,
        reverse_cmap=plot.reverse_cmap,
        total_source_count=total_source_count,
        animation_start_ms=animation_start_ms,
        animation_end_ms=animation_end_ms,
        preserve_depth_order=keep_depth,
    )


def _frame_rgb(canvas: FigureCanvasAgg) -> np.ndarray:
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
    return np.ascontiguousarray(rgba[..., :3])


def animate_projection_project(
    project: LMAProject,
    *,
    output_path: str | Path,
    display_mode: DisplayMode = "cumulative",
    trail_ms: float = 30.0,
    afterimage_ms: float = 30.0,
    fps: int = 30,
    duration_s: float = 15.0,
    hold_end_s: float = 5.0,
    width: int = 1600,
    height: int = 900,
    video_quality: int = 8,
    custom_title: str | None = None,
) -> Path:
    if display_mode not in _DISPLAY_MODES:
        raise ConfigurationError(
            f"Unknown projection-animation display mode: {display_mode}"
        )
    if fps <= 0 or duration_s <= 0 or hold_end_s < 0:
        raise ConfigurationError(
            "Animation FPS/duration must be positive and hold cannot be negative"
        )
    if width < 320 or height < 240:
        raise ConfigurationError("Projection animation dimensions are too small")

    output = Path(output_path).expanduser().resolve()
    if output.suffix.lower() not in {".mp4", ".gif"}:
        raise ConfigurationError("Projection animation output must end in .mp4 or .gif")
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise ConfigurationError(
            "Projection animation requires imageio and imageio-ffmpeg"
        ) from exc

    scene: ProjectionAnimationScene | None = None
    canvas: FigureCanvasAgg | None = None
    writer = None
    writer_closed = False
    try:
        scene = build_projection_animation_scene(
            project, width=width, height=height, custom_title=custom_title
        )
        canvas = FigureCanvasAgg(scene.figure)
        frame_times = animation_frame_times(
            scene.first_time_ms,
            scene.final_time_ms,
            fps=fps,
            duration_s=duration_s,
            display_mode=display_mode,
            afterimage_ms=afterimage_ms,
        )
        hold_frames = max(0, int(round(hold_end_s * fps)))
        total_frames = len(frame_times) + hold_frames

        writer_kwargs = {"fps": int(fps)}
        if output.suffix.lower() == ".mp4":
            # -nostdin prevents ffmpeg's interactive command reader from
            # retaining the GUI terminal on Windows after encoding completes.
            writer_kwargs.update(
                {
                    "codec": "libx264",
                    "quality": int(video_quality),
                    "macro_block_size": None,
                    "ffmpeg_log_level": "error",
                    "input_params": ["-nostdin"],
                }
            )
        else:
            writer_kwargs.update({"loop": 0})

        progress = AnimationProgressReporter(
            total_frames, output, label="lma animate-projections"
        )
        progress.start()
        last_rgb = None
        completed = 0
        writer = imageio.get_writer(output, **writer_kwargs)
        for current in frame_times:
            scene.update(
                float(current),
                display_mode=display_mode,
                trail_ms=trail_ms,
                afterimage_ms=afterimage_ms,
            )
            last_rgb = _frame_rgb(canvas)
            writer.append_data(last_rgb)
            completed += 1
            progress.update(completed)
        if last_rgb is None:
            raise DatasetError("Projection animation produced no frames")
        for _ in range(hold_frames):
            writer.append_data(last_rgb)
            completed += 1
            progress.update(completed)
        progress.finalizing()
        writer.close()
        writer_closed = True
        progress.complete()
        return output
    finally:
        # imageio normally closes through the success path above. This second
        # guarded close handles exceptions without leaving ffmpeg or its pipe
        # alive, which was the likely source of the reported post-save hang.
        if writer is not None and not writer_closed:
            try:
                writer.close()
            except Exception:
                pass
        if scene is not None:
            scene.close()
        canvas = None
        scene = None
        writer = None
        gc.collect()


__all__ = [
    "DisplayMode",
    "ProjectionAnimationScene",
    "animate_projection_project",
    "build_projection_animation_scene",
    "combined_filter_spec",
    "combined_project_filter",
]
