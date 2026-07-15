from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import math
import sys
import time
from typing import Callable, Iterable, TextIO

import numpy as np

from ..errors import ConfigurationError

AFTERIMAGE_MODES = frozenset({"trail-afterimage"})
AFTERIMAGE_GRAY_MIN = 0.30
AFTERIMAGE_GRAY_MAX = 7.0 / 8.0


def is_afterimage_mode(display_mode: str) -> bool:
    return str(display_mode).strip().lower().replace("_", "-") in AFTERIMAGE_MODES


def afterimage_completion_time(final_source_time_ms: float, *, display_mode: str, afterimage_ms: float) -> float:
    final = float(final_source_time_ms)
    if not is_afterimage_mode(display_mode):
        return final
    width = float(afterimage_ms)
    if not np.isfinite(width) or width <= 0:
        raise ConfigurationError("Afterimage duration must be positive")
    return final + width


def afterimage_grayscale_values(
    time_ms: Iterable[float] | np.ndarray,
    current_time_ms: float,
    *,
    afterimage_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(time_ms, dtype=float)
    width = float(afterimage_ms)
    if not np.isfinite(width) or width <= 0:
        raise ConfigurationError("Afterimage duration must be positive")
    age = float(current_time_ms) - times
    shadow = age >= width - 1.0e-9
    values = np.zeros(times.shape, dtype=float)
    if not np.any(shadow):
        return shadow, values
    elapsed = np.maximum(age[shadow] - width, 0.0)
    span = float(np.max(elapsed))
    normalized = np.zeros_like(elapsed) if span <= 1.0e-12 else np.clip(elapsed / span, 0.0, 1.0)
    values[shadow] = AFTERIMAGE_GRAY_MAX - normalized * (AFTERIMAGE_GRAY_MAX - AFTERIMAGE_GRAY_MIN)
    return shadow, values


def animation_frame_times(
    start_ms: float,
    end_ms: float,
    *,
    fps: int,
    duration_s: float,
    display_mode: str = "cumulative",
    afterimage_ms: float = 30.0,
) -> np.ndarray:
    fps_value = int(fps)
    duration = float(duration_s)
    start = float(start_ms)
    end = float(end_ms)
    if fps_value <= 0 or duration <= 0 or not np.isfinite(duration):
        raise ConfigurationError("Animation FPS and duration must be positive")
    if not np.isfinite(start) or not np.isfinite(end) or end < start:
        raise ConfigurationError("Animation time limits must be finite and ordered")
    development_count = max(2, int(round(fps_value * duration)))
    development = np.linspace(start, end, development_count)
    if not is_afterimage_mode(display_mode):
        return development
    width = float(afterimage_ms)
    if width <= 0 or not np.isfinite(width):
        raise ConfigurationError("Afterimage duration must be positive")
    span = end - start
    if span <= 0:
        completion_count = max(1, int(round(fps_value * min(duration, 1.0))))
    else:
        simulated_step = span / max(1, development_count - 1)
        completion_count = max(1, int(np.ceil(width / simulated_step)))
    completion = np.linspace(end, end + width, completion_count + 1)[1:]
    return np.concatenate((development, completion))


def _format_elapsed(seconds: float) -> str:
    value = max(0, int(round(float(seconds))))
    hours, remainder = divmod(value, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


@dataclass
class AnimationProgressReporter:
    """Throttled terminal reporting for long 3D animation renders."""

    total_frames: int
    output_path: str | Path
    stream: TextIO = field(default_factory=lambda: sys.stdout)
    clock: Callable[[], float] = time.monotonic
    percent_step: float = 5.0
    min_interval_s: float = 1.0
    max_interval_s: float = 5.0
    label: str = "lma animate-3d"
    _start_s: float | None = field(default=None, init=False, repr=False)
    _last_report_s: float | None = field(default=None, init=False, repr=False)
    _next_percent: float = field(default=5.0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.total_frames = int(self.total_frames)
        self.output_path = Path(self.output_path).expanduser().resolve()
        self.percent_step = float(self.percent_step)
        self.min_interval_s = float(self.min_interval_s)
        self.max_interval_s = float(self.max_interval_s)
        if self.total_frames <= 0:
            raise ConfigurationError("Animation progress requires at least one frame")
        if self.percent_step <= 0 or self.min_interval_s < 0 or self.max_interval_s <= 0:
            raise ConfigurationError("Invalid animation progress reporting cadence")
        if self.max_interval_s < self.min_interval_s:
            self.max_interval_s = self.min_interval_s
        self._next_percent = self.percent_step

    def _print(self, message: str) -> None:
        print(message, file=self.stream, flush=True)

    def start(self) -> None:
        if self._start_s is not None:
            return
        now = float(self.clock())
        self._start_s = now
        self._last_report_s = now
        self._print(
            f"[{self.label}] Generating {self.total_frames:,} frames "
            f"→ {self.output_path}"
        )

    @property
    def elapsed_s(self) -> float:
        if self._start_s is None:
            return 0.0
        return max(0.0, float(self.clock()) - self._start_s)

    def update(self, completed_frames: int) -> bool:
        if self._start_s is None:
            self.start()
        assert self._start_s is not None
        assert self._last_report_s is not None
        completed = int(np.clip(int(completed_frames), 0, self.total_frames))
        now = float(self.clock())
        elapsed = max(0.0, now - self._start_s)
        since_last = max(0.0, now - self._last_report_s)
        percent = 100.0 * completed / self.total_frames
        finished = completed >= self.total_frames
        percent_due = percent + 1.0e-9 >= self._next_percent
        time_due = since_last >= self.max_interval_s
        if not finished:
            if since_last < self.min_interval_s and not time_due:
                return False
            if not (percent_due or time_due):
                return False
        eta = 0.0
        if completed > 0 and completed < self.total_frames and elapsed > 0:
            eta = elapsed * (self.total_frames - completed) / completed
        self._print(
            f"[{self.label}] Frame {completed:,}/{self.total_frames:,} "
            f"({percent:.0f}%) — elapsed {_format_elapsed(elapsed)} — "
            f"ETA {_format_elapsed(eta)}"
        )
        self._last_report_s = now
        self._next_percent = max(
            self._next_percent + self.percent_step,
            (math.floor(percent / self.percent_step) + 1.0) * self.percent_step,
        )
        return True

    def finalizing(self) -> None:
        suffix = self.output_path.suffix.lower().lstrip(".").upper() or "animation"
        self._print(f"[{self.label}] Finalizing {suffix}...")

    def complete(self) -> None:
        self._print(
            f"[{self.label}] Saved {self.output_path} — "
            f"total time {_format_elapsed(self.elapsed_s)}"
        )


__all__ = [
    "AFTERIMAGE_GRAY_MAX",
    "AFTERIMAGE_GRAY_MIN",
    "AFTERIMAGE_MODES",
    "AnimationProgressReporter",
    "afterimage_completion_time",
    "afterimage_grayscale_values",
    "animation_frame_times",
    "is_afterimage_mode",
]
