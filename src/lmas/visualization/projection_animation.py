from __future__ import annotations

from dataclasses import replace

from dataclasses import dataclass
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
from .animation import AnimationProgressReporter, animation_frame_times
from .pyvista_3d import frame_display_rgba

DisplayMode = Literal["cumulative", "trail", "trail-afterimage"]
_DISPLAY_MODES = {"cumulative", "trail", "trail-afterimage"}


def combined_project_filter(project: LMAProject) -> FilterSpec:
    """Combine quality filtering with the non-destructive saved linked view."""
    quality = project.filters.validated()
    view = project.view_filters.validated()
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

    @property
    def first_time_ms(self) -> float:
        return float(np.nanmin(self.time_ms))

    @property
    def final_time_ms(self) -> float:
        return float(np.nanmax(self.time_ms))

    def update(
        self,
        current_time_ms: float,
        *,
        display_mode: DisplayMode,
        trail_ms: float,
        afterimage_ms: float,
    ) -> int:
        if display_mode not in _DISPLAY_MODES:
            raise ConfigurationError(
                f"Unknown projection-animation display mode: {display_mode}"
            )
        rgba = frame_display_rgba(
            self.base_rgba,
            self.time_ms,
            float(current_time_ms),
            display_mode=display_mode,
            trail_ms=trail_ms,
            afterimage_ms=afterimage_ms,
            cmap=self.cmap,
            reverse_cmap=self.reverse_cmap,
        )
        visible = rgba[:, 3] > 0
        base_selected = np.flatnonzero(visible)
        for index, (scatter, pair) in enumerate(
            zip(self.scatters, self.coordinate_pairs)
        ):
            selected = base_selected
            key = (
                np.asarray(self.depth_keys[index], dtype=float)
                if index < len(self.depth_keys)
                else None
            )
            if key is not None and key.shape == (self.colors.size,) and selected.size:
                selected = selected[np.argsort(key[selected], kind="stable")]
            if selected.size:
                offsets = np.column_stack(
                    (np.asarray(pair[0])[selected], np.asarray(pair[1])[selected])
                )
                facecolors = rgba[selected].astype(float) / 255.0
            else:
                offsets = np.empty((0, 2), dtype=float)
                facecolors = np.empty((0, 4), dtype=float)
            scatter.set_offsets(offsets)
            scatter.set_facecolors(facecolors)
        visible_count = int(np.count_nonzero(visible))
        if self.title_artist is not None:
            self.title_artist.set_text(
                f"{self.base_title} — {visible_count:,} visible of "
                f"{self.colors.size:,} sources in view{self.chi2_suffix}\n"
                f"Source time: {float(current_time_ms):.3f} ms"
            )
        return visible_count

    def close(self) -> None:
        # The animation worker is deliberately headless. Explicitly disconnect
        # and clear the large Matplotlib object graph so Qt/Matplotlib backend
        # state cannot keep a completed Windows worker alive at interpreter
        # shutdown.
        try:
            self.figure.clear()
        except Exception:
            pass
        self.scatters.clear()
        self.coordinate_pairs.clear()
        self.depth_keys.clear()


def build_projection_animation_scene(
    project: LMAProject,
    *,
    width: int | None = None,
    height: int | None = None,
    custom_title: str | None = None,
) -> ProjectionAnimationScene:
    filters = combined_project_filter(project)
    plot = project.plot.validated()
    figure = create_lma_figure(
        project, filters=filters, plot=replace(plot, preview_point_limit=0)
    )
    metadata = getattr(figure, "_lmas_metadata", {})
    scatters = list(metadata.get("scatters", ()))
    coordinate_pairs = list(metadata.get("coordinate_pairs", ()))
    depth_keys = list(metadata.get("scatter_depth_keys", ()))
    colors = np.asarray(metadata.get("color_values", ()), dtype=float)
    if not scatters or len(scatters) != len(coordinate_pairs) or colors.size == 0:
        raise DatasetError("The LMAS figure does not expose an animatable projection layout")

    # Projects may store exact source membership in addition to rectangular
    # time/spatial bounds. Projection animations use that same membership.
    if project.selected_source_ids is not None:
        source_ids = np.asarray(metadata.get("source_ids", ()), dtype=np.int64)
        if source_ids.shape == colors.shape:
            membership = np.isin(
                source_ids, np.asarray(project.selected_source_ids, dtype=np.int64)
            )
            coordinate_pairs = [
                (np.asarray(pair[0])[membership], np.asarray(pair[1])[membership])
                for pair in coordinate_pairs
            ]
            depth_keys = [
                None if key is None else np.asarray(key)[membership]
                for key in depth_keys
            ]
            colors = colors[membership]
            if colors.size == 0:
                raise DatasetError("The saved project subset contains no matching sources")

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

    time_numbers = np.asarray(coordinate_pairs[0][0], dtype=float)
    time_utc = np.asarray(
        [
            np.datetime64(value.replace(tzinfo=None), "us")
            for value in mdates.num2date(time_numbers)
        ]
    )
    origin = time_utc.min()
    time_ms = np.asarray(
        (time_utc - origin) / np.timedelta64(1, "ms"), dtype=float
    )
    if time_ms.size == 0 or not np.any(np.isfinite(time_ms)):
        raise DatasetError("Projection animation has no finite source times")

    if plot.color_by == "time":
        # Projection development is a self-contained view of the selected
        # flash. Color the entire animated source population from its first
        # source (0 s) through its final source, matching the established LMAS
        # cumulative-animation behavior. Saved GUI color limits may be relative
        # to a much earlier full-record origin; applying those limits to this
        # selected subset collapses every source to the purple end of the map.
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
    base_title = str(custom_title or plot.title or project.data_source_stem)
    chi2_suffix = (
        f" (χ² < {project.filters.maximum_chi2:.2f})"
        if project.filters.maximum_chi2 is not None
        else ""
    )
    return ProjectionAnimationScene(
        figure=figure,
        scatters=scatters,
        coordinate_pairs=coordinate_pairs,
        depth_keys=depth_keys,
        colors=colors,
        time_ms=time_ms,
        base_rgba=base_rgba,
        title_artist=title_artist,
        base_title=base_title,
        chi2_suffix=chi2_suffix,
        cmap=plot.cmap if plot.color_by == "time" else None,
        reverse_cmap=plot.reverse_cmap,
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
    "combined_project_filter",
]
