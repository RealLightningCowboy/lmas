from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import time
from typing import Any, Literal

import matplotlib as mpl
import numpy as np

from ..errors import ConfigurationError, DependencyError
from ..plotting.common import resolved_cmap, theme_values
from ..source_selection import CHARGE_COLORS
from .animation import afterimage_completion_time, afterimage_grayscale_values
from .snapshot import VisualizationSnapshot, load_visualization_snapshot

DisplayMode = Literal["full", "cumulative", "trail", "trail-afterimage"]
AnimationMode = Literal["orbit", "develop", "develop-orbit"]
RenderProfile = Literal["compatible", "quality"]
InteractionMode = Literal["z-orbit", "full-3d"]


@dataclass
class PlaybackClock:
    first_time_ms: float
    final_time_ms: float
    duration_s: float
    current_time_ms: float
    playing: bool = False
    _last_monotonic_s: float | None = None

    def __post_init__(self) -> None:
        if self.final_time_ms < self.first_time_ms or self.duration_s <= 0:
            raise ConfigurationError("Invalid 3D playback interval or duration")
        self.current_time_ms = float(
            np.clip(self.current_time_ms, self.first_time_ms, self.final_time_ms)
        )

    @property
    def rate_ms_per_second(self) -> float:
        return (self.final_time_ms - self.first_time_ms) / self.duration_s

    def seek(self, value_ms: float, *, pause: bool = True) -> float:
        self.current_time_ms = float(np.clip(value_ms, self.first_time_ms, self.final_time_ms))
        if pause:
            self.pause()
        return self.current_time_ms

    def play(self, now_s: float | None = None) -> float:
        if self.current_time_ms >= self.final_time_ms - 1.0e-12:
            self.current_time_ms = self.first_time_ms
        self.playing = True
        self._last_monotonic_s = time.monotonic() if now_s is None else float(now_s)
        return self.current_time_ms

    def pause(self) -> float:
        self.playing = False
        self._last_monotonic_s = None
        return self.current_time_ms

    def toggle(self, now_s: float | None = None) -> float:
        return self.pause() if self.playing else self.play(now_s)

    def advance(self, now_s: float | None = None) -> tuple[float, bool]:
        if not self.playing:
            return self.current_time_ms, False
        now = time.monotonic() if now_s is None else float(now_s)
        if self._last_monotonic_s is None:
            self._last_monotonic_s = now
            return self.current_time_ms, False
        elapsed = max(0.0, now - self._last_monotonic_s)
        self._last_monotonic_s = now
        if elapsed <= 0:
            return self.current_time_ms, False
        next_time = self.current_time_ms + elapsed * self.rate_ms_per_second
        if next_time >= self.final_time_ms:
            self.current_time_ms = self.final_time_ms
            self.pause()
        else:
            self.current_time_ms = next_time
        return self.current_time_ms, True


@dataclass(frozen=True)
class CameraSettings:
    position: tuple[float, float, float]
    focal_point: tuple[float, float, float]
    view_up: tuple[float, float, float]
    parallel_projection: bool = False
    parallel_scale: float | None = None
    view_angle: float | None = None
    window_center: tuple[float, float] = (0.0, -0.14)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "position": list(self.position),
            "focal_point": list(self.focal_point),
            "view_up": list(self.view_up),
            "parallel_projection": self.parallel_projection,
            "window_center": list(self.window_center),
        }
        if self.parallel_scale is not None:
            payload["parallel_scale"] = self.parallel_scale
        if self.view_angle is not None:
            payload["view_angle"] = self.view_angle
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CameraSettings":
        def vector(name: str, length: int) -> tuple[float, ...]:
            value = payload.get(name)
            if not isinstance(value, (list, tuple)) or len(value) != length:
                raise ConfigurationError(f"Camera field {name!r} must contain {length} numbers")
            try:
                return tuple(float(item) for item in value)
            except (TypeError, ValueError) as exc:
                raise ConfigurationError(f"Camera field {name!r} is not numeric") from exc

        window_center = vector("window_center", 2) if "window_center" in payload else (0.0, -0.14)
        return cls(
            position=vector("position", 3),  # type: ignore[arg-type]
            focal_point=vector("focal_point", 3),  # type: ignore[arg-type]
            view_up=vector("view_up", 3),  # type: ignore[arg-type]
            parallel_projection=bool(payload.get("parallel_projection", False)),
            parallel_scale=None if payload.get("parallel_scale") is None else float(payload["parallel_scale"]),
            view_angle=None if payload.get("view_angle") is None else float(payload["view_angle"]),
            window_center=window_center,  # type: ignore[arg-type]
        )


def save_camera_settings(path: str | Path, settings: CameraSettings) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(settings.to_dict(), indent=2), encoding="utf-8")
    return destination


def load_camera_settings(path: str | Path) -> CameraSettings:
    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"Could not read camera file {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigurationError("Camera file must contain a JSON object")
    return CameraSettings.from_dict(payload)


def _require_pyvista():
    try:
        import pyvista as pv
    except Exception as exc:  # pragma: no cover - environment-specific
        raise DependencyError(
            "Interactive 3D visualization requires PyVista and VTK. Install the optional "
            "dependencies with `python -m pip install \"lmas[visualization]\"` "
            "or `mamba install -c conda-forge pyvista vtk imageio-ffmpeg ffmpeg`."
        ) from exc
    return pv


def _quiet_vtk_console() -> None:
    """Prevent repeated driver diagnostics from flooding detached terminals.

    A real Python/VTK exception still propagates normally. This only disables
    VTK's duplicate C++ warning/error stream, which can otherwise emit the same
    shader diagnostic hundreds of times after an OpenGL capability failure.
    """

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


def _finalize_interactive_plotter(plotter) -> None:
    """Close the interactive VTK window without post-close render attempts."""

    if plotter is None:
        return
    _quiet_vtk_console()
    render_window = getattr(plotter, "render_window", None) or getattr(
        plotter, "ren_win", None
    )
    try:
        plotter.close()
    except Exception:
        pass
    if render_window is not None:
        try:
            render_window.RemoveAllObservers()
        except Exception:
            pass
        try:
            render_window.Finalize()
        except Exception:
            pass


def _parse_window_size(window_size: tuple[int, int] | list[int]) -> tuple[int, int]:
    if len(window_size) != 2:
        raise ConfigurationError("Window size must contain width and height")
    width, height = int(window_size[0]), int(window_size[1])
    if width < 320 or height < 240:
        raise ConfigurationError("3D window size must be at least 320 x 240")
    return width, height


def _normalization(values: np.ndarray, *, logarithmic: bool):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if logarithmic:
        finite = finite[finite > 0]
    if finite.size == 0:
        raise ConfigurationError("No finite values are available for 3D coloring")
    low, high = float(np.min(finite)), float(np.max(finite))
    if high <= low:
        if logarithmic:
            low = max(low * 0.9, np.nextafter(0.0, 1.0))
            high = high * 1.1
        else:
            low -= 0.5
            high += 0.5
    return mpl.colors.LogNorm(low, high) if logarithmic else mpl.colors.Normalize(low, high)


def _base_rgba(snapshot: VisualizationSnapshot, cmap: str, *, reverse_cmap: bool = False) -> np.ndarray:
    if snapshot.color_by == "charge":
        values = np.asarray(snapshot.color_values, dtype=float)
        rgba = np.zeros((values.size, 4), dtype=np.uint8)
        for numeric, category in ((-1.0, "negative"), (0.0, "unassigned"), (1.0, "positive")):
            color = np.asarray(mpl.colors.to_rgba(CHARGE_COLORS[category]))
            rgba[np.isclose(values, numeric)] = np.asarray(np.round(color * 255.0), dtype=np.uint8)
        return rgba
    if snapshot.color_by == "group":
        values = np.asarray(snapshot.color_values, dtype=float)
        palette = tuple(snapshot.categorical_colors) or (CHARGE_COLORS["unassigned"],)
        rgba = np.zeros((values.size, 4), dtype=np.uint8)
        for numeric, color_text in enumerate(palette):
            color = np.asarray(mpl.colors.to_rgba(color_text))
            rgba[np.isclose(values, float(numeric))] = np.asarray(
                np.round(color * 255.0), dtype=np.uint8
            )
        return rgba
    try:
        color_map = mpl.colormaps[resolved_cmap(cmap, reverse=reverse_cmap)]
    except KeyError as exc:
        raise ConfigurationError(f"Unknown Matplotlib colormap: {cmap}") from exc
    norm = _normalization(snapshot.color_values, logarithmic=snapshot.logarithmic_color)
    return np.asarray(np.round(color_map(norm(snapshot.color_values)) * 255.0), dtype=np.uint8)


def visibility_alpha(
    time_ms: np.ndarray,
    current_ms: float,
    *,
    display_mode: DisplayMode,
    trail_ms: float,
    afterimage_ms: float,
    out: np.ndarray | None = None,
) -> np.ndarray:
    times = np.asarray(time_ms, dtype=float)
    if out is None:
        alpha = np.empty(times.size, dtype=np.uint8)
    else:
        alpha = np.asarray(out)
        if alpha.dtype != np.uint8 or alpha.shape != times.shape:
            raise ValueError("Alpha output buffer must be uint8 and match source times")
        if not alpha.flags.writeable:
            raise ValueError("Alpha output buffer must be writable")
    if display_mode == "full":
        alpha.fill(255)
        return alpha
    if display_mode == "cumulative":
        alpha.fill(0)
        alpha[times <= current_ms] = 255
        return alpha
    if display_mode == "trail":
        if trail_ms <= 0:
            raise ConfigurationError("Trail duration must be positive")
        age = current_ms - times
        visible = (age >= 0) & (age <= trail_ms)
        alpha.fill(0)
        if np.any(visible):
            fade = 0.18 + 0.82 * (1.0 - age[visible] / trail_ms)
            alpha[visible] = np.asarray(np.clip(np.round(255.0 * fade), 1, 255), dtype=np.uint8)
        return alpha
    if display_mode == "trail-afterimage":
        if afterimage_ms <= 0:
            raise ConfigurationError("Afterimage duration must be positive")
        age = current_ms - times
        completed = age >= 0
        recent = completed & (age <= afterimage_ms)
        alpha.fill(0)
        if np.any(recent):
            fade = 0.18 + 0.82 * (1.0 - age[recent] / afterimage_ms)
            alpha[recent] = np.asarray(np.clip(np.round(255.0 * fade), 1, 255), dtype=np.uint8)
        alpha[completed & ~recent] = 255
        return alpha
    raise ConfigurationError(f"Unknown 3D display mode: {display_mode}")


def frame_display_rgba(
    base_rgba: np.ndarray,
    time_ms: np.ndarray,
    current_ms: float,
    *,
    display_mode: DisplayMode,
    trail_ms: float,
    afterimage_ms: float,
    cmap: str | None = None,
    reverse_cmap: bool = False,
    out: np.ndarray | None = None,
) -> np.ndarray:
    times = np.asarray(time_ms, dtype=float)
    base = np.asarray(base_rgba, dtype=np.uint8)
    if base.ndim != 2 or base.shape[1:] != (4,) or times.shape != (base.shape[0],):
        raise ValueError("Base colors and source times must describe the same RGBA population")
    if out is None:
        rgba = np.array(base, copy=True)
    else:
        rgba = np.asarray(out)
        if rgba.dtype != np.uint8 or rgba.shape != base.shape:
            raise ValueError("RGBA output buffer must be uint8 and match the base-color array")
        if not rgba.flags.writeable:
            raise ValueError("RGBA output buffer must be writable")
        np.copyto(rgba, base)
    visibility_alpha(
        times,
        current_ms,
        display_mode=display_mode,
        trail_ms=trail_ms,
        afterimage_ms=afterimage_ms,
        out=rgba[:, 3],
    )
    if display_mode == "trail-afterimage":
        age = float(current_ms) - times
        recent = (age >= -1.0e-9) & (age < float(afterimage_ms) - 1.0e-9)
        if cmap is not None and np.any(recent):
            try:
                color_map = mpl.colormaps[resolved_cmap(cmap, reverse=reverse_cmap)]
            except KeyError as exc:
                raise ConfigurationError(f"Unknown Matplotlib colormap: {cmap}") from exc
            recent_times = times[recent]
            low, high = float(np.min(recent_times)), float(np.max(recent_times))
            if high <= low:
                pad = max(afterimage_ms * 0.5, 0.5)
                low, high = low - pad, high + pad
            dynamic = np.asarray(
                np.round(color_map(np.clip((recent_times - low) / (high - low), 0, 1)) * 255),
                dtype=np.uint8,
            )
            rgba[recent, :3] = dynamic[:, :3]
        shadow, gray = afterimage_grayscale_values(times, current_ms, afterimage_ms=afterimage_ms)
        gray_u8 = np.asarray(np.round(gray[shadow] * 255.0), dtype=np.uint8)
        rgba[shadow, :3] = gray_u8[:, None]
        rgba[shadow, 3] = 255
    return rgba


def _tight_bounds(bounds: tuple[float, float, float, float, float, float], padding_fraction: float = 0.04):
    x0, x1, y0, y1, z0, z1 = map(float, bounds)
    horizontal_span = max(x1 - x0, y1 - y0, 1.0e-6) * (1 + 2 * padding_fraction)
    xc, yc = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    zspan = max(z1 - z0, 1.0e-6)
    zpad = max(1.0e-6, zspan * padding_fraction)
    return (
        xc - horizontal_span / 2,
        xc + horizontal_span / 2,
        yc - horizontal_span / 2,
        yc + horizontal_span / 2,
        z0 - zpad,
        z1 + zpad,
    )


def _nice_integer_tick_step(span: float, *, maximum_intervals: int = 7) -> int:
    """Return a readable integer tick step for one 3D coordinate axis."""
    value = float(span)
    if not np.isfinite(value) or value <= 0:
        return 1
    required = max(1.0, value / max(1, int(maximum_intervals)))
    exponent = int(math.floor(math.log10(required)))
    candidates: list[int] = []
    for power in range(exponent - 1, exponent + 4):
        scale = 10.0 ** power
        for multiplier in (1.0, 2.0, 5.0, 10.0):
            candidate = max(1, int(math.ceil(multiplier * scale - 1.0e-12)))
            candidates.append(candidate)
    return min(candidate for candidate in sorted(set(candidates)) if candidate >= required)


def _integer_tick_axis(low: float, high: float, *, maximum_intervals: int = 7) -> tuple[float, float, int, int]:
    lo, hi = sorted((float(low), float(high)))
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ConfigurationError("3D bounds must be finite")
    step = _nice_integer_tick_step(hi - lo, maximum_intervals=maximum_intervals)
    maximum_labels = max(2, int(maximum_intervals) + 1)
    while True:
        snapped_low = float(math.floor(lo / step) * step)
        snapped_high = float(math.ceil(hi / step) * step)
        if snapped_high <= snapped_low:
            snapped_high = snapped_low + step
        count = int(round((snapped_high - snapped_low) / step)) + 1
        if count <= maximum_labels:
            return snapped_low, snapped_high, count, step
        step = _nice_integer_tick_step(float(step) + 1.0e-9, maximum_intervals=1)


def _integer_grid_spec(
    bounds: tuple[float, float, float, float, float, float],
) -> tuple[tuple[float, float, float, float, float, float], tuple[int, int, int], tuple[int, int, int]]:
    """Snap grid bounds to readable integer-spaced major ticks."""
    x0, x1, nx, sx = _integer_tick_axis(bounds[0], bounds[1])
    y0, y1, ny, sy = _integer_tick_axis(bounds[2], bounds[3])
    z0, z1, nz, sz = _integer_tick_axis(bounds[4], bounds[5])
    return (x0, x1, y0, y1, z0, z1), (nx, ny, nz), (sx, sy, sz)


def _default_camera(snapshot: VisualizationSnapshot, *, window_size=(1400, 900), margin=1.28, view_angle=30.0) -> CameraSettings:
    bounds = _tight_bounds(snapshot.bounds)
    center = np.array([
        0.5 * (bounds[0] + bounds[1]),
        0.5 * (bounds[2] + bounds[3]),
        0.5 * (bounds[4] + bounds[5]),
    ])
    spans = np.array([bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]])
    radius = max(0.5 * float(np.linalg.norm(spans)), 0.5)
    width, height = _parse_window_size(window_size)
    aspect = width / height
    half_vertical = math.radians(view_angle * 0.5)
    half_horizontal = math.atan(math.tan(half_vertical) * aspect)
    distance = margin * radius / max(math.sin(min(half_vertical, half_horizontal)), 1.0e-6)
    direction = np.array([1.45, -1.45, 0.85], dtype=float)
    direction /= np.linalg.norm(direction)
    position = center + direction * distance
    return CameraSettings(
        position=tuple(float(v) for v in position),
        focal_point=tuple(float(v) for v in center),
        view_up=(0.0, 0.0, 1.0),
        view_angle=float(view_angle),
    )


def _camera_from_plotter(plotter) -> CameraSettings:
    camera = plotter.camera
    parallel = bool(camera.GetParallelProjection())
    return CameraSettings(
        position=tuple(float(v) for v in camera.position),
        focal_point=tuple(float(v) for v in camera.focal_point),
        view_up=tuple(float(v) for v in camera.up),
        parallel_projection=parallel,
        parallel_scale=float(camera.parallel_scale) if parallel else None,
        view_angle=float(camera.view_angle),
        window_center=tuple(float(v) for v in camera.GetWindowCenter()),
    )


def _reset_camera_clipping(plotter) -> None:
    try:
        plotter.reset_camera_clipping_range()
    except Exception:
        renderer = getattr(plotter, "renderer", None)
        if renderer is not None:
            renderer.ResetCameraClippingRange()


def _apply_camera(plotter, settings: CameraSettings) -> None:
    if settings.parallel_projection:
        plotter.enable_parallel_projection()
        if settings.parallel_scale is not None:
            plotter.camera.parallel_scale = settings.parallel_scale
    else:
        plotter.disable_parallel_projection()
    if settings.view_angle is not None:
        plotter.camera.view_angle = float(settings.view_angle)
    plotter.camera_position = [list(settings.position), list(settings.focal_point), list(settings.view_up)]
    plotter.camera.SetWindowCenter(*settings.window_center)
    _reset_camera_clipping(plotter)


def _enable_z_orbit_style(plotter) -> None:
    try:
        from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera

        style = vtkInteractorStyleTrackballCamera()

        def azimuth_only(caller, _event: str) -> int:
            if caller.GetState() != 1:
                return 0
            interactor = caller.GetInteractor()
            renderer = caller.GetCurrentRenderer()
            if interactor is None or renderer is None:
                return 0
            x, _y = interactor.GetEventPosition()
            last_x, _last_y = interactor.GetLastEventPosition()
            width = max(1, int(interactor.GetRenderWindow().GetSize()[0]))
            camera = renderer.GetActiveCamera()
            camera.SetViewUp(0.0, 0.0, 1.0)
            camera.Azimuth(-180.0 * float(x - last_x) / float(width))
            camera.SetViewUp(0.0, 0.0, 1.0)
            camera.OrthogonalizeViewUp()
            renderer.ResetCameraClippingRange()
            interactor.Render()
            return 1

        style.AddObserver("MouseMoveEvent", azimuth_only, 1.0)
        plotter.iren.interactor.SetInteractorStyle(style)
        plotter._lmas_z_orbit_style = style
        plotter._lmas_z_orbit_callback = azimuth_only
    except Exception:
        plotter.enable_terrain_style(mouse_wheel_zooms=True, shift_pans=True)


def _set_interaction_mode(plotter, mode: InteractionMode) -> None:
    if mode == "z-orbit":
        _enable_z_orbit_style(plotter)
    elif mode == "full-3d":
        plotter.enable_trackball_style()
    else:
        raise ConfigurationError(f"Unknown 3D interaction mode: {mode}")


def _background(theme: str) -> str:
    return theme_values(theme)["axes"]


def _camera_horizontal_azimuth(settings: CameraSettings) -> float:
    """Return the camera's horizontal azimuth around its focal point."""

    position = np.asarray(settings.position, dtype=float)
    focal = np.asarray(settings.focal_point, dtype=float)
    delta = position - focal
    if not np.all(np.isfinite(delta)) or float(np.hypot(delta[0], delta[1])) <= 1.0e-12:
        return 0.0
    return float(np.degrees(np.arctan2(delta[1], delta[0])) % 360.0)


def axis_presentation_sector(settings: CameraSettings) -> int:
    """Return one of four stable horizontal sectors for cube-axis placement.

    Sector centers are the diagonal viewing directions (45, 135, 225, and
    315 degrees).  Boundaries therefore fall on the cardinal directions.  A
    rotating camera can move continuously inside one sector while the cube
    axes, grid planes, tick sides, and labels remain anchored to one common
    presentation state.
    """

    return int(math.floor(_camera_horizontal_azimuth(settings) / 90.0)) % 4


def anchored_axis_camera_settings(settings: CameraSettings) -> tuple[int, CameraSettings]:
    """Snap only horizontal camera direction to a stable sector center."""

    position = np.asarray(settings.position, dtype=float)
    focal = np.asarray(settings.focal_point, dtype=float)
    delta = position - focal
    horizontal = float(np.hypot(delta[0], delta[1]))
    sector = axis_presentation_sector(settings)
    center_deg = 45.0 + 90.0 * sector
    angle = math.radians(center_deg)
    anchored_delta = np.array(
        [horizontal * math.cos(angle), horizontal * math.sin(angle), float(delta[2])],
        dtype=float,
    )
    anchored = CameraSettings(
        position=tuple(float(value) for value in focal + anchored_delta),
        focal_point=settings.focal_point,
        view_up=settings.view_up,
        parallel_projection=settings.parallel_projection,
        parallel_scale=settings.parallel_scale,
        view_angle=settings.view_angle,
        window_center=settings.window_center,
    )
    return sector, anchored


_BASE_TICK_FONT_SIZE = 17
_CARDINAL_FONT_SIZE = 22
_EVENT_TITLE_FONT_SIZE = 14
_STATUS_FONT_SIZE = 14
_CONTROL_FONT_SIZE = 12

_LMAS_3D_ACTOR_NAMES = (
    "lmas-base-grid",
    "lmas-base-border",
    "lmas-base-x-labels",
    "lmas-base-y-labels",
    "lmas-base-cardinal-labels",
    # Legacy dev2 actor names retained only for cleanup during in-process refreshes.
    "lmas-base-x-title",
    "lmas-base-y-title",
)


def _remove_named_actor(plotter, name: str) -> None:
    """Remove one LMAS-owned actor without forcing a camera reset or render."""

    remover = getattr(plotter, "remove_actor", None)
    if not callable(remover):
        return
    for kwargs in (
        {"reset_camera": False, "render": False},
        {"reset_camera": False},
        {},
    ):
        try:
            remover(name, **kwargs)
            return
        except TypeError:
            continue
        except Exception:
            return


def _tick_values(low: float, high: float, count: int) -> np.ndarray:
    return np.linspace(float(low), float(high), max(2, int(count)), dtype=float)


def _paired_line_points(segments: list[tuple[tuple[float, float, float], tuple[float, float, float]]]) -> np.ndarray:
    if not segments:
        return np.empty((0, 3), dtype=float)
    return np.asarray([point for segment in segments for point in segment], dtype=float)


def _base_grid_line_points(
    bounds: tuple[float, float, float, float, float, float],
    counts: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return base-plane grid and border lines as disconnected point pairs."""

    x0, x1, y0, y1, z0, _z1 = (float(value) for value in bounds)
    x_values = _tick_values(x0, x1, counts[0])
    y_values = _tick_values(y0, y1, counts[1])
    grid_segments = [
        ((float(x), y0, z0), (float(x), y1, z0)) for x in x_values
    ] + [
        ((x0, float(y), z0), (x1, float(y), z0)) for y in y_values
    ]
    border_segments = [
        ((x0, y0, z0), (x1, y0, z0)),
        ((x1, y0, z0), (x1, y1, z0)),
        ((x1, y1, z0), (x0, y1, z0)),
        ((x0, y1, z0), (x0, y0, z0)),
    ]
    return _paired_line_points(grid_segments), _paired_line_points(border_segments)


def _sector_corner_signs(sector: int) -> tuple[int, int]:
    """Return the base corner nearest a camera in the selected azimuth sector."""

    return ((1, 1), (-1, 1), (-1, -1), (1, -1))[int(sector) % 4]


def _axis_overlay_layout(
    bounds: tuple[float, float, float, float, float, float],
    counts: tuple[int, int, int],
    sector: int,
) -> dict[str, Any]:
    """Build camera-side labels for the horizontal base plane only."""

    x0, x1, y0, y1, z0, _z1 = (float(value) for value in bounds)
    x_values = _tick_values(x0, x1, counts[0])
    y_values = _tick_values(y0, y1, counts[1])
    sx, sy = _sector_corner_signs(sector)
    x_span = max(x1 - x0, 1.0e-6)
    y_span = max(y1 - y0, 1.0e-6)
    horizontal_span = max(x_span, y_span)
    margin = max(0.045 * horizontal_span, 0.08)

    near_x = x1 if sx > 0 else x0
    near_y = y1 if sy > 0 else y0
    x_label_y = near_y + sy * margin
    y_label_x = near_x + sx * margin

    x_points = np.column_stack(
        (x_values, np.full_like(x_values, x_label_y), np.full_like(x_values, z0))
    )
    y_points = np.column_stack(
        (np.full_like(y_values, y_label_x), y_values, np.full_like(y_values, z0))
    )
    cardinal_margin = 2.45 * margin
    cardinal_points = np.asarray(
        [
            [0.5 * (x0 + x1), y1 + cardinal_margin, z0],  # North
            [0.5 * (x0 + x1), y0 - cardinal_margin, z0],  # South
            [x1 + cardinal_margin, 0.5 * (y0 + y1), z0],  # East
            [x0 - cardinal_margin, 0.5 * (y0 + y1), z0],  # West
        ],
        dtype=float,
    )
    return {
        "x_points": x_points,
        "x_labels": [f"{value:g}" for value in x_values],
        "y_points": y_points,
        "y_labels": [f"{value:g}" for value in y_values],
        "cardinal_points": cardinal_points,
        "cardinal_labels": ["N", "S", "E", "W"],
    }


def _add_lines_compat(
    plotter,
    points: np.ndarray,
    *,
    name: str,
    color: str,
    width: float,
    opacity: float = 1.0,
):
    if points.size == 0:
        return None
    kwargs = {
        "color": color,
        "width": float(width),
        "connected": False,
        "name": name,
        "opacity": float(opacity),
    }
    try:
        return plotter.add_lines(points, **kwargs)
    except TypeError:
        kwargs.pop("opacity", None)
        return plotter.add_lines(points, **kwargs)


def _add_point_labels_compat(
    plotter,
    points: np.ndarray,
    labels: list[str],
    *,
    name: str,
    color: str,
    font_size: int,
):
    kwargs: dict[str, Any] = {
        "name": name,
        "font_size": int(font_size),
        "text_color": color,
        "shape": None,
        "show_points": False,
        "always_visible": True,
        "render": False,
    }
    fallbacks = (
        (),
        ("render",),
        ("render", "shape"),
        ("render", "shape", "show_points"),
    )
    for removed in fallbacks:
        attempt = {key: value for key, value in kwargs.items() if key not in removed}
        try:
            return plotter.add_point_labels(points, labels, **attempt)
        except TypeError:
            continue
    return plotter.add_point_labels(points, labels, name=name, font_size=int(font_size))


def _add_base_grid(plotter) -> None:
    spec = getattr(plotter, "_lmas_grid_spec", None)
    theme = getattr(plotter, "_lmas_grid_theme", "dark")
    if spec is None:
        return
    bounds, counts = spec
    grid_points, border_points = _base_grid_line_points(bounds, counts)
    values = theme_values(theme)
    _add_lines_compat(
        plotter,
        grid_points,
        name="lmas-base-grid",
        color=values["grid"],
        width=1.0,
        opacity=0.58,
    )
    _add_lines_compat(
        plotter,
        border_points,
        name="lmas-base-border",
        color=values["text"],
        width=1.4,
        opacity=0.78,
    )


def _replace_base_axis_overlay(plotter, sector: int) -> None:
    spec = getattr(plotter, "_lmas_grid_spec", None)
    if spec is None:
        return
    bounds, counts = spec
    theme = getattr(plotter, "_lmas_grid_theme", "dark")
    values = theme_values(theme)
    layout = _axis_overlay_layout(bounds, counts, sector)
    for name in _LMAS_3D_ACTOR_NAMES[2:]:
        _remove_named_actor(plotter, name)

    _add_point_labels_compat(
        plotter,
        layout["x_points"],
        layout["x_labels"],
        name="lmas-base-x-labels",
        color=values["text"],
        font_size=_BASE_TICK_FONT_SIZE,
    )
    _add_point_labels_compat(
        plotter,
        layout["y_points"],
        layout["y_labels"],
        name="lmas-base-y-labels",
        color=values["text"],
        font_size=_BASE_TICK_FONT_SIZE,
    )
    _add_point_labels_compat(
        plotter,
        layout["cardinal_points"],
        layout["cardinal_labels"],
        name="lmas-base-cardinal-labels",
        color=values["text"],
        font_size=_CARDINAL_FONT_SIZE,
    )


def _enable_anchored_axis_presentation(plotter, enabled: bool) -> None:
    """Enable camera-side base labels for constrained-orbit presentation."""

    plotter._lmas_axis_anchor_enabled = getattr(plotter, "_lmas_grid_spec", None) is not None
    if plotter._lmas_axis_anchor_enabled:
        _synchronize_axis_presentation(plotter, force=True)


def _synchronize_axis_presentation(plotter, *, force: bool = False) -> bool:
    """Refresh only the horizontal base tick and cardinal labels."""

    if not bool(getattr(plotter, "_lmas_axis_anchor_enabled", False)):
        return False
    try:
        sector = axis_presentation_sector(_camera_from_plotter(plotter))
    except Exception:
        sector = 0
    current = getattr(plotter, "_lmas_axis_sector", None)
    if force or current != sector:
        _replace_base_axis_overlay(plotter, sector)
        plotter._lmas_axis_sector = sector
        return True
    return False


def _enforce_integer_grid_actor(plotter) -> None:
    """Compatibility shim for old call sites; refresh the custom overlays."""

    if getattr(plotter, "_lmas_grid_spec", None) is not None:
        _synchronize_axis_presentation(plotter, force=True)


def _configure_scene(
    plotter,
    snapshot: VisualizationSnapshot,
    *,
    theme: str,
    show_grid_and_labels: bool = True,
    anchored_axis_presentation: bool = False,
    total_source_count: int | None = None,
) -> None:
    background = _background(theme)
    plotter.set_background(background)
    text_color = theme_values(theme)["text"]
    try:
        plotter.theme.font.color = text_color
    except Exception:
        pass
    plotter._lmas_grid_actor = None  # legacy attribute: cube axes are no longer used
    plotter._lmas_grid_spec = None
    plotter._lmas_grid_theme = theme
    plotter._lmas_axis_anchor_enabled = False
    plotter._lmas_axis_sector = None
    if show_grid_and_labels:
        grid_bounds, label_counts, _steps = _integer_grid_spec(_tight_bounds(snapshot.bounds))
        plotter._lmas_grid_spec = (grid_bounds, label_counts)
        _add_base_grid(plotter)
        _enable_anchored_axis_presentation(plotter, anchored_axis_presentation)
    total = max(int(snapshot.source_count), int(total_source_count or snapshot.source_count))
    if snapshot.source_count < total:
        population = f"{snapshot.source_count:,} displayed of {total:,} sources"
    else:
        population = f"{total:,} sources"
    title = (
        f"{snapshot.title} — Interactive 3D Viewer ({population})\n"
        f"{snapshot.event_timestamp}\nAltitude: km MSL; no ground subtraction"
    )
    plotter.add_text(
        title,
        position="upper_left",
        font_size=_EVENT_TITLE_FONT_SIZE,
        name="event-title",
    )


def _point_render_options(render_profile: RenderProfile) -> dict[str, Any]:
    if render_profile == "compatible":
        return {
            "style": "points",
            "render_points_as_spheres": True,
            "emissive": False,
            "lighting": False,
        }
    if render_profile == "quality":
        return {
            "style": "points_gaussian",
            "render_points_as_spheres": True,
            "emissive": False,
            "lighting": False,
        }
    raise ConfigurationError(f"Unknown 3D render profile: {render_profile}")


def _add_point_cloud(plotter, snapshot: VisualizationSnapshot, *, cmap: str, reverse_cmap: bool, point_size: float, render_profile: RenderProfile):
    pv = _require_pyvista()
    mesh = pv.PolyData(snapshot.points_km)
    rgba = np.ascontiguousarray(
        _base_rgba(snapshot, cmap, reverse_cmap=reverse_cmap), dtype=np.uint8
    )
    # Keep the immutable base colors separate from the VTK-owned display array.
    # Otherwise an in-place frame update can silently corrupt the colors used to
    # construct every subsequent frame.
    mesh["source_rgba"] = np.array(rgba, copy=True)
    mesh["source_time_ms"] = snapshot.time_ms
    plotter.add_points(
        mesh,
        scalars="source_rgba",
        rgba=True,
        point_size=float(point_size),
        show_scalar_bar=False,
        copy_mesh=False,
        **_point_render_options(render_profile),
    )
    return mesh, rgba


def _set_mesh_rgba(mesh, rgba: np.ndarray) -> None:
    values = np.asarray(rgba, dtype=np.uint8)
    try:
        target = np.asarray(mesh["source_rgba"], dtype=np.uint8)
    except Exception:
        target = np.empty((0, 4), dtype=np.uint8)
    if target.shape == values.shape:
        np.copyto(target, values)
    else:
        mesh["source_rgba"] = np.ascontiguousarray(values)
    point_data = mesh.GetPointData()
    if point_data is not None:
        array = point_data.GetArray("source_rgba")
        if array is not None:
            array.Modified()
        point_data.Modified()
    mesh.Modified()


def _time_stratified_snapshot(
    snapshot: VisualizationSnapshot, point_limit: int
) -> VisualizationSnapshot:
    limit = int(point_limit)
    if limit < 0:
        raise ConfigurationError("Interactive point limit cannot be negative")
    count = int(snapshot.source_count)
    if limit <= 0 or count <= limit:
        return snapshot
    time_order = np.argsort(np.asarray(snapshot.time_ms, dtype=float), kind="stable")
    positions = time_order[np.linspace(0, count - 1, limit, dtype=np.int64)]
    return replace(
        snapshot,
        time_utc_ns=np.ascontiguousarray(snapshot.time_utc_ns[positions]),
        time_ms=np.ascontiguousarray(snapshot.time_ms[positions]),
        points_km=np.ascontiguousarray(snapshot.points_km[positions]),
        source_ids=np.ascontiguousarray(snapshot.source_ids[positions]),
        color_values=np.ascontiguousarray(snapshot.color_values[positions]),
    )


def _interactive_visibility_mode(display_mode: DisplayMode) -> DisplayMode:
    return "cumulative" if display_mode == "full" else display_mode


def view_3d_snapshot(
    snapshot_path: str | Path,
    *,
    display_mode: DisplayMode = "cumulative",
    trail_ms: float = 30.0,
    afterimage_ms: float = 30.0,
    point_size: float = 6.0,
    cmap: str = "turbo",
    reverse_cmap: bool = False,
    theme: str = "dark",
    render_profile: RenderProfile = "compatible",
    interaction_mode: InteractionMode = "z-orbit",
    camera_path: str | Path | None = None,
    camera_output: str | Path | None = None,
    playback_fps: float = 30.0,
    playback_duration_s: float = 15.0,
    point_limit: int = 50_000,
    start_playing: bool = False,
    show_grid_and_labels: bool = True,
    window_size: tuple[int, int] = (1400, 900),
) -> None:
    pv = _require_pyvista()
    _quiet_vtk_console()
    snapshot = load_visualization_snapshot(snapshot_path)
    total_source_count = int(snapshot.source_count)
    snapshot = _time_stratified_snapshot(snapshot, point_limit)
    if point_size <= 0 or playback_fps <= 0 or playback_duration_s <= 0:
        raise ConfigurationError("Point size, playback FPS, and duration must be positive")
    window = _parse_window_size(window_size)
    plotter = pv.Plotter(window_size=window)
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
        anchored_axis_presentation=interaction_mode == "z-orbit",
        total_source_count=total_source_count,
    )
    first_time, final_time = snapshot.time_limits
    visibility_mode = _interactive_visibility_mode(display_mode)
    playback_final = afterimage_completion_time(
        final_time,
        display_mode=visibility_mode,
        afterimage_ms=afterimage_ms,
    )
    source_span = max(final_time - first_time, 1.0e-12)
    playback_duration = playback_duration_s * (playback_final - first_time) / source_span
    # Always open at the beginning of the selected time window. Space/P
    # already toggles playback immediately; start_playing only controls whether
    # the clock begins advancing without a key press.
    initial_time = first_time
    playback = PlaybackClock(first_time, playback_final, playback_duration, initial_time)
    output_camera = (
        Path(camera_output).expanduser().resolve()
        if camera_output is not None
        else Path(snapshot_path).expanduser().resolve().with_suffix(".camera.json")
    )
    state: dict[str, Any] = {
        "programmatic_slider": False,
        "last_slider_render_s": 0.0,
        "last_time_status_s": 0.0,
    }
    rgba_buffer = np.empty_like(base_rgba)
    playback_status_actor = plotter.add_text(
        "",
        position=(18, 18),
        font_size=_STATUS_FONT_SIZE,
        name="playback-status",
    )
    time_status_actor = plotter.add_text(
        "",
        position=(max(18, window[0] - 430), 18),
        font_size=_STATUS_FONT_SIZE,
        name="time-status",
    )
    interaction_actor = plotter.add_text(
        "",
        position=(50, 153),
        font_size=_CONTROL_FONT_SIZE,
        name="interaction-mode",
    )

    def set_text(actor, text: str, *, fallback_name: str, position, font_size: int) -> None:
        try:
            actor.SetInput(str(text))
        except Exception:
            # Older PyVista/VTK combinations may return an annotation wrapper
            # instead of a vtkTextActor.  Retain a compatible fallback.
            plotter.add_text(
                str(text),
                position=position,
                font_size=font_size,
                name=fallback_name,
            )

    def mode_text() -> str:
        if visibility_mode == "trail":
            mode = f"trail: {trail_ms:g} ms"
        elif visibility_mode == "trail-afterimage":
            mode = f"trail + afterimage: {afterimage_ms:g} ms"
        else:
            mode = "cumulative"
        if snapshot.source_count < total_source_count:
            mode += f" | {snapshot.source_count:,} of {total_source_count:,} sources"
        return mode

    def update_status(*, render: bool = False) -> None:
        set_text(
            playback_status_actor,
            f"{'Playing' if playback.playing else 'Paused'} | {mode_text()}",
            fallback_name="playback-status",
            position=(18, 18),
            font_size=_STATUS_FONT_SIZE,
        )
        if render:
            plotter.render()

    def update_scene(value: float, *, render: bool = True) -> None:
        current = float(np.clip(value, first_time, playback_final))
        playback.current_time_ms = current
        rgba = frame_display_rgba(
            base_rgba,
            snapshot.time_ms,
            current,
            display_mode=visibility_mode,
            trail_ms=trail_ms,
            afterimage_ms=afterimage_ms,
            cmap=cmap if snapshot.color_by == "time" else None,
            reverse_cmap=reverse_cmap,
            out=rgba_buffer,
        )
        _set_mesh_rgba(mesh, rgba)
        visible = int(np.count_nonzero(rgba[:, 3]))
        now = time.monotonic()
        if render or not playback.playing or now - float(state["last_time_status_s"]) >= 0.10:
            state["last_time_status_s"] = now
            set_text(
                time_status_actor,
                f"Source time: {current:.3f} ms   Visible: {visible:,}",
                fallback_name="time-status",
                position=(max(18, window[0] - 430), 18),
                font_size=_STATUS_FONT_SIZE,
            )
        if render:
            plotter.render()

    slider = None

    def set_slider_value(value: float) -> None:
        if slider is None:
            return
        representation = slider.GetRepresentation()
        state["programmatic_slider"] = True
        try:
            representation.SetValue(float(value))
        finally:
            state["programmatic_slider"] = False

    def slider_callback(value: float) -> None:
        if state["programmatic_slider"]:
            # Playback updates the VTK slider and scene explicitly.  Rendering
            # from this callback as well doubles all per-frame animation work.
            return
        playback.seek(float(value), pause=True)
        update_status()
        now = time.monotonic()
        if now - float(state["last_slider_render_s"]) < (1.0 / 30.0):
            return
        state["last_slider_render_s"] = now
        update_scene(float(value))

    slider = plotter.add_slider_widget(
        slider_callback,
        rng=(first_time, playback_final),
        value=initial_time,
        title="Flash development — Source time (ms)",
        pointa=(0.20, 0.115),
        pointb=(0.80, 0.115),
        interaction_event="always",
        title_height=0.022,
        fmt="%0.3f",
        slider_width=0.018,
        tube_width=0.006,
    )

    def slider_end_callback(widget, _event=None) -> None:
        if state["programmatic_slider"]:
            return
        try:
            value = float(widget.GetRepresentation().GetValue())
        except Exception:
            return
        playback.seek(value, pause=True)
        state["last_slider_render_s"] = time.monotonic()
        update_status()
        update_scene(value)

    try:
        slider.AddObserver("EndInteractionEvent", slider_end_callback)
    except Exception:
        pass

    camera = load_camera_settings(camera_path) if camera_path is not None else _default_camera(snapshot, window_size=window)
    _apply_camera(plotter, camera)

    def set_z_orbit(enabled: bool) -> None:
        mode: InteractionMode = "z-orbit" if enabled else "full-3d"
        _set_interaction_mode(plotter, mode)
        _enable_anchored_axis_presentation(plotter, enabled)
        set_text(
            interaction_actor,
            "Camera: Z-axis orbit" if enabled else "Camera: full 3D orbit",
            fallback_name="interaction-mode",
            position=(50, 153),
            font_size=_CONTROL_FONT_SIZE,
        )
        _reset_camera_clipping(plotter)
        try:
            plotter.render()
        except Exception:
            pass

    plotter.add_checkbox_button_widget(
        set_z_orbit,
        value=interaction_mode == "z-orbit",
        position=(12, 145),
        size=26,
        border_size=2,
        color_on="royalblue",
        color_off="grey",
        background_color="white",
    )

    def toggle_playback() -> None:
        previous = playback.current_time_ms
        current = playback.toggle()
        if current != previous:
            set_slider_value(current)
            update_scene(current, render=False)
        update_status(render=True)

    def save_camera() -> None:
        saved = save_camera_settings(output_camera, _camera_from_plotter(plotter))
        print(f"[lmas view-3d] saved camera: {saved}")

    def reset_camera() -> None:
        _apply_camera(plotter, _default_camera(snapshot, window_size=window))
        plotter.render()

    plotter.add_key_event("space", toggle_playback)
    plotter.add_key_event("p", toggle_playback)
    plotter.add_key_event("s", save_camera)
    plotter.add_key_event("r", reset_camera)
    plotter.add_text(
        "Toggle: Z-axis/full 3D   Mouse: rotate / pan / zoom   "
        "Space/P: play-pause   S: save camera   R: reset camera",
        position="upper_right",
        font_size=_CONTROL_FONT_SIZE,
        name="controls",
    )

    update_scene(initial_time, render=False)
    if start_playing:
        playback.play()
    update_status(render=False)
    print(f"[lmas view-3d] camera save target: {output_camera}")
    if snapshot.source_count < total_source_count:
        print(
            f"[lmas view-3d] interactive preview: {snapshot.source_count:,} "
            f"of {total_source_count:,} sources"
        )
    plotter.show(auto_close=False, interactive_update=True)
    _enforce_integer_grid_actor(plotter)
    set_z_orbit(interaction_mode == "z-orbit")
    try:
        while not getattr(plotter, "_closed", False):
            plotter.update(stime=max(1, int(round(1000 / playback_fps))), force_redraw=False)
            axes_changed = _synchronize_axis_presentation(plotter)
            current, changed = playback.advance()
            if changed:
                set_slider_value(current)
                update_scene(current, render=False)
            if changed or axes_changed:
                plotter.render()
            time.sleep(min(0.005, 0.25 / playback_fps))
    finally:
        _finalize_interactive_plotter(plotter)



def animate_3d_snapshot(*args, **kwargs):
    """Compatibility wrapper for the dedicated off-screen animation module."""

    from .animation_3d import animate_3d_snapshot as _animate_3d_snapshot

    return _animate_3d_snapshot(*args, **kwargs)


__all__ = [
    "AnimationMode",
    "CameraSettings",
    "DisplayMode",
    "InteractionMode",
    "PlaybackClock",
    "RenderProfile",
    "anchored_axis_camera_settings",
    "animate_3d_snapshot",
    "axis_presentation_sector",
    "frame_display_rgba",
    "load_camera_settings",
    "save_camera_settings",
    "view_3d_snapshot",
    "visibility_alpha",
]
