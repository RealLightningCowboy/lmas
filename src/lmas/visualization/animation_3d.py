from __future__ import annotations

from pathlib import Path

import numpy as np

from ..errors import ConfigurationError
from .animation import AnimationProgressReporter, animation_frame_times
from .pyvista_3d import (
    AnimationMode,
    DisplayMode,
    RenderProfile,
    _STATUS_FONT_SIZE,
    _add_point_cloud,
    _apply_camera,
    _camera_from_plotter,
    _configure_scene,
    _default_camera,
    _enable_anchored_axis_presentation,
    _enforce_integer_grid_actor,
    _parse_window_size,
    _require_pyvista,
    _reset_camera_clipping,
    _set_mesh_rgba,
    _synchronize_axis_presentation,
    frame_display_rgba,
    load_camera_settings,
    save_camera_settings,
)
from .snapshot import load_visualization_snapshot


def _open_animation_writer(plotter, output: Path, fps: int, quality: int = 7) -> None:
    suffix = output.suffix.lower()
    if suffix == ".gif":
        plotter.open_gif(str(output), fps=fps)
    elif suffix == ".mp4":
        try:
            plotter.open_movie(str(output), framerate=fps, quality=int(quality))
        except TypeError:
            plotter.open_movie(str(output), framerate=fps)
    else:
        raise ConfigurationError("Animation output must end in .mp4 or .gif")


def _silence_vtk_cleanup_logging() -> None:
    """Suppress duplicate VTK shutdown diagnostics after rendering is complete."""

    try:
        from vtkmodules.vtkCommonCore import vtkLogger, vtkObject

        try:
            vtkObject.GlobalWarningDisplayOff()
        except Exception:
            pass
        try:
            verbosity = getattr(vtkLogger, "VERBOSITY_OFF", None)
            if verbosity is not None:
                vtkLogger.SetStderrVerbosity(verbosity)
        except Exception:
            pass
    except Exception:
        pass


def _close_animation_writer(plotter) -> None:
    """Close PyVista's imageio writer exactly once."""

    writer = getattr(plotter, "mwriter", None)
    if writer is None:
        return
    try:
        writer.close()
    except Exception:
        pass
    try:
        plotter.mwriter = None
    except Exception:
        pass


def _finalize_animation_plotter(plotter) -> None:
    """Idempotently terminate VTK/PyVista without post-finalize renders."""

    if plotter is None or bool(getattr(plotter, "_lmas_finalized", False)):
        return
    try:
        setattr(plotter, "_lmas_finalized", True)
    except Exception:
        pass
    _silence_vtk_cleanup_logging()
    _close_animation_writer(plotter)

    iren = getattr(plotter, "iren", None)
    interactor = getattr(iren, "interactor", None) if iren is not None else None
    if interactor is None:
        interactor = iren
    if interactor is not None:
        for method in ("RemoveAllObservers", "Disable", "TerminateApp"):
            callback = getattr(interactor, method, None)
            if callable(callback):
                try:
                    callback()
                except Exception:
                    pass
    if iren is not None:
        close = getattr(iren, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    render_window = getattr(plotter, "render_window", None) or getattr(
        plotter, "ren_win", None
    )
    if render_window is not None:
        try:
            render_window.RemoveAllObservers()
        except Exception:
            pass
    try:
        plotter.clear()
    except Exception:
        pass
    closed_by_plotter = False
    try:
        plotter.close()
        closed_by_plotter = True
    except Exception:
        pass
    if not closed_by_plotter and render_window is not None:
        try:
            render_window.Finalize()
        except Exception:
            pass
    try:
        plotter.iren = None
    except Exception:
        pass
    try:
        plotter.render_window = None
    except Exception:
        pass


def animate_3d_snapshot(
    snapshot_path: str | Path,
    *,
    output_path: str | Path,
    mode: AnimationMode = "develop",
    display_mode: DisplayMode = "cumulative",
    trail_ms: float = 30.0,
    afterimage_ms: float = 30.0,
    point_size: float = 6.0,
    cmap: str = "turbo",
    reverse_cmap: bool = False,
    theme: str = "dark",
    render_profile: RenderProfile = "compatible",
    camera_path: str | Path | None = None,
    camera_output: str | Path | None = None,
    fps: int = 30,
    duration_s: float = 15.0,
    hold_end_s: float = 5.0,
    orbit_speed_deg_s: float = 14.0,
    video_quality: int = 7,
    show_grid_and_labels: bool = True,
    window_size: tuple[int, int] = (1400, 900),
) -> Path:
    """Render an off-screen 3D animation from an LMAS snapshot."""

    pv = _require_pyvista()
    _silence_vtk_cleanup_logging()
    if mode not in {"orbit", "develop", "develop-orbit"}:
        raise ConfigurationError(f"Unknown animation mode: {mode}")
    if fps <= 0 or duration_s <= 0 or hold_end_s < 0:
        raise ConfigurationError(
            "Animation FPS/duration must be positive and hold cannot be negative"
        )
    if point_size <= 0 or not np.isfinite(orbit_speed_deg_s):
        raise ConfigurationError("Point size must be positive and orbit speed finite")
    if not 0 <= int(video_quality) <= 10:
        raise ConfigurationError("Video quality must be between 0 and 10")

    snapshot = load_visualization_snapshot(snapshot_path)
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    window = _parse_window_size(window_size)
    plotter = None
    try:
        plotter = pv.Plotter(off_screen=True, window_size=window)
        mesh, base_rgba = _add_point_cloud(
            plotter,
            snapshot,
            cmap=cmap,
            reverse_cmap=reverse_cmap,
            point_size=point_size,
            render_profile=render_profile,
        )
        _configure_scene(
            plotter,
            snapshot,
            theme=theme,
            show_grid_and_labels=show_grid_and_labels,
            anchored_axis_presentation=mode in {"orbit", "develop-orbit"},
        )
        camera = (
            load_camera_settings(camera_path)
            if camera_path is not None
            else _default_camera(snapshot, window_size=window)
        )
        _apply_camera(plotter, camera)
        if camera_output is not None:
            save_camera_settings(camera_output, _camera_from_plotter(plotter))

        first_time, final_time = snapshot.time_limits
        frames_time = animation_frame_times(
            first_time,
            final_time,
            fps=fps,
            duration_s=duration_s,
            display_mode=display_mode if mode != "orbit" else "cumulative",
            afterimage_ms=afterimage_ms,
        )
        hold_frames = max(0, int(round(hold_end_s * fps)))
        total_frames = len(frames_time) + hold_frames
        progress = AnimationProgressReporter(total_frames, output)
        progress.start()
        _open_animation_writer(plotter, output, int(fps), int(video_quality))
        plotter.show(auto_close=False, interactive=False)
        _apply_camera(plotter, camera)
        _enforce_integer_grid_actor(plotter)
        _enable_anchored_axis_presentation(
            plotter, mode in {"orbit", "develop-orbit"}
        )

        orbit_step = float(orbit_speed_deg_s) / float(fps)
        completed_frames = 0
        for frame, timeline_time in enumerate(frames_time):
            if mode == "orbit":
                current = final_time
                active_display: DisplayMode = "full"
            else:
                current = float(timeline_time)
                active_display = display_mode
            rgba = frame_display_rgba(
                base_rgba,
                snapshot.time_ms,
                current,
                display_mode=active_display,
                trail_ms=trail_ms,
                afterimage_ms=afterimage_ms,
                cmap=cmap if snapshot.color_by == "time" else None,
                reverse_cmap=reverse_cmap,
            )
            _set_mesh_rgba(mesh, rgba)
            visible = int(np.count_nonzero(rgba[:, 3]))
            plotter.add_text(
                f"Source time: {current:.3f} ms   Visible: {visible:,}",
                position="lower_left",
                font_size=_STATUS_FONT_SIZE,
                name="time-status",
            )
            if mode in {"orbit", "develop-orbit"} and frame:
                plotter.camera.azimuth += orbit_step
                _reset_camera_clipping(plotter)
            _synchronize_axis_presentation(plotter)
            plotter.render()
            plotter.write_frame()
            completed_frames += 1
            progress.update(completed_frames)

        for _ in range(hold_frames):
            if mode in {"orbit", "develop-orbit"}:
                plotter.camera.azimuth += orbit_step
                _reset_camera_clipping(plotter)
                _synchronize_axis_presentation(plotter)
                plotter.render()
            plotter.write_frame()
            completed_frames += 1
            progress.update(completed_frames)
        progress.finalizing()
        _close_animation_writer(plotter)
        progress.complete()
        return output
    finally:
        _finalize_animation_plotter(plotter)


__all__ = ["animate_3d_snapshot"]
