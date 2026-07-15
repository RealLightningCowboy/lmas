from __future__ import annotations

import math

import matplotlib.dates as mdates
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter

# Nice UTC-aligned intervals spanning microseconds through days.
_NICE_STEPS_S = np.array(
    [
        1e-6, 2e-6, 5e-6,
        1e-5, 2e-5, 5e-5,
        1e-4, 2e-4, 5e-4,
        1e-3, 2e-3, 5e-3,
        1e-2, 2e-2, 5e-2,
        0.1, 0.2, 0.5,
        1.0, 2.0, 5.0, 10.0, 15.0, 30.0,
        60.0, 120.0, 300.0, 600.0, 900.0, 1800.0,
        3600.0, 7200.0, 10800.0, 21600.0, 43200.0, 86400.0,
    ],
    dtype=float,
)


def _nice_step(span_s: float, target_intervals: int = 9) -> float:
    target = max(float(span_s) / max(int(target_intervals), 1), 1e-9)
    index = int(np.searchsorted(_NICE_STEPS_S, target, side="left"))
    if index >= _NICE_STEPS_S.size:
        magnitude = 10.0 ** math.floor(math.log10(target))
        for multiple in (1.0, 2.0, 5.0, 10.0):
            value = multiple * magnitude
            if value >= target:
                return value
        return 10.0 * magnitude
    return float(_NICE_STEPS_S[index])


class _UTCOffsetFormatter(FuncFormatter):
    """Adaptive UTC tick formatter with a ConciseDateFormatter-style anchor."""

    def __init__(self, function, offset_text: str) -> None:
        super().__init__(function)
        self._lmas_offset_text = str(offset_text)

    def get_offset(self) -> str:
        return self._lmas_offset_text


def _format_factory(step_s: float, span_s: float, low_value: float, high_value: float):
    """Return concise UTC labels plus a higher-order offset anchor.

    The visible tick labels intentionally omit date/hour/minute components that
    are already supplied by the right-aligned offset text below the axis.
    """

    low_stamp = mdates.num2date(float(low_value))
    high_stamp = mdates.num2date(float(high_value))

    if step_s >= 86400.0:
        return _UTCOffsetFormatter(
            lambda value, _pos: mdates.num2date(value).strftime("%Y-%m-%d"),
            "",
        )

    if step_s >= 3600.0 or span_s >= 6.0 * 3600.0:
        return _UTCOffsetFormatter(
            lambda value, _pos: mdates.num2date(value).strftime("%H:%M"),
            low_stamp.strftime("%Y %b %d"),
        )

    same_hour = (
        low_stamp.year,
        low_stamp.month,
        low_stamp.day,
        low_stamp.hour,
    ) == (
        high_stamp.year,
        high_stamp.month,
        high_stamp.day,
        high_stamp.hour,
    )
    same_minute = same_hour and low_stamp.minute == high_stamp.minute

    if step_s >= 1.0:
        if same_minute:
            # A sub-minute view needs only whole seconds; the offset carries
            # the date, hour, and minute.
            return _UTCOffsetFormatter(
                lambda value, _pos: mdates.num2date(value).strftime("%S"),
                low_stamp.strftime("%Y %b %d %H:%M"),
            )
        if same_hour:
            return _UTCOffsetFormatter(
                lambda value, _pos: mdates.num2date(value).strftime("%M:%S"),
                low_stamp.strftime("%Y %b %d %H"),
            )
        return _UTCOffsetFormatter(
            lambda value, _pos: mdates.num2date(value).strftime("%H:%M"),
            low_stamp.strftime("%Y %b %d"),
        )

    decimals = 3 if step_s >= 1e-3 else 6

    def formatter(value, _pos):
        stamp = mdates.num2date(value)
        seconds = stamp.second + stamp.microsecond / 1e6
        width = 2 + 1 + decimals
        if same_minute:
            return f"{seconds:0{width}.{decimals}f}"
        return f"{stamp:%M}:{seconds:0{width}.{decimals}f}"

    anchor = (
        low_stamp.strftime("%Y %b %d %H:%M")
        if same_minute
        else low_stamp.strftime("%Y %b %d %H")
    )
    return _UTCOffsetFormatter(formatter, anchor)


def configure_utc_time_axis(axis, limits: tuple[float, float] | None = None) -> float:
    """Apply scale-aware UTC major and minor ticks.

    The left visible boundary is always a major tick; a dense, human-readable
    interval is then repeated across the zoomed view.  The returned value is
    the selected major interval in seconds.
    """

    if limits is None:
        limits = axis.get_xlim()
    low, high = sorted(float(value) for value in limits)
    span_s = max((high - low) * 86400.0, 1e-9)
    step_s = _nice_step(span_s, target_intervals=9)
    low_s, high_s = low * 86400.0, high * 86400.0
    # The first visible tick is always an explicit major tick.  Subsequent
    # majors use one stable nice interval measured from that left-edge anchor.
    count = max(1, int(math.floor((high_s - low_s) / step_s)) + 1)
    major_s = low_s + np.arange(count, dtype=float) * step_s
    major = major_s / 86400.0
    if major.size < 2 and high > low:
        major = np.array([low, high], dtype=float)
        major_s = major * 86400.0

    # Three minor ticks between adjacent majors: four equal subdivisions from
    # the same left-edge anchor.
    minor_divisions = 4
    minor_step = step_s / minor_divisions
    minor_count = max(1, int(math.floor((high_s - low_s) / minor_step)) + 1)
    minor_s = low_s + np.arange(minor_count, dtype=float) * minor_step
    # Exclude points coincident with major ticks.
    if major_s.size:
        ratios = minor_s[:, None] - major_s[None, :]
        keep = np.all(np.abs(ratios) > max(minor_step, 1e-9) * 1e-6, axis=1)
        minor_s = minor_s[keep]

    axis.xaxis.set_major_locator(FixedLocator(major))
    axis.xaxis.set_minor_locator(FixedLocator(minor_s / 86400.0))
    axis.xaxis.set_major_formatter(_format_factory(step_s, span_s, low, high))
    return step_s


class _RelativeOffsetFormatter(FuncFormatter):
    """Relative-time labels with a fixed UTC origin shown as offset text."""

    def __init__(self, function, offset_text: str) -> None:
        super().__init__(function)
        self._lmas_offset_text = str(offset_text)

    def get_offset(self) -> str:
        return self._lmas_offset_text


def _relative_unit(window_span_s: float) -> tuple[float, str]:
    span = max(float(window_span_s), 0.0)
    if span < 0.02:
        return 1.0e6, "µs"
    if span < 2.0:
        return 1.0e3, "ms"
    if span < 120.0:
        return 1.0, "s"
    if span < 7200.0:
        return 1.0 / 60.0, "min"
    return 1.0 / 3600.0, "h"


def configure_relative_time_axis(
    axis,
    *,
    origin: float,
    window_span_s: float,
    limits: tuple[float, float] | None = None,
) -> tuple[float, str]:
    """Apply adaptive elapsed-time ticks anchored to a fixed window origin.

    Axis data remain Matplotlib UTC date numbers, so LMA, GLM, network, and
    saved-project timing stay exact.  Only labels and tick placement change.
    """

    if limits is None:
        limits = axis.get_xlim()
    low, high = sorted(float(value) for value in limits)
    visible_span_s = max((high - low) * 86400.0, 1e-9)
    step_s = _nice_step(visible_span_s, target_intervals=9)
    scale, unit = _relative_unit(window_span_s)
    origin_s = float(origin) * 86400.0
    low_s, high_s = low * 86400.0, high * 86400.0
    first = math.ceil((low_s - origin_s) / step_s - 1e-12)
    last = math.floor((high_s - origin_s) / step_s + 1e-12)
    if last >= first:
        major_s = origin_s + np.arange(first, last + 1, dtype=float) * step_s
    else:
        major_s = np.array([low_s, high_s], dtype=float)
    minor_step = step_s / 4.0
    first_minor = math.ceil((low_s - origin_s) / minor_step - 1e-12)
    last_minor = math.floor((high_s - origin_s) / minor_step + 1e-12)
    minor_s = (
        origin_s + np.arange(first_minor, last_minor + 1, dtype=float) * minor_step
        if last_minor >= first_minor else np.array([], dtype=float)
    )
    if major_s.size and minor_s.size:
        keep = np.all(
            np.abs(minor_s[:, None] - major_s[None, :]) > max(minor_step, 1e-9) * 1e-6,
            axis=1,
        )
        minor_s = minor_s[keep]

    def formatter(value, _position):
        elapsed = (float(value) - float(origin)) * 86400.0 * scale
        magnitude = abs(step_s * scale)
        if magnitude >= 10.0:
            decimals = 0
        elif magnitude >= 1.0:
            decimals = 1
        elif magnitude >= 0.1:
            decimals = 2
        else:
            decimals = 3
        text = f"{elapsed:.{decimals}f}"
        if text.startswith("-0") and abs(elapsed) < 0.5 * 10 ** (-decimals):
            text = text[1:]
        return text

    anchor = mdates.num2date(float(origin)).strftime("Window start: %Y-%m-%d %H:%M:%S.%f UTC")
    anchor = anchor.replace("000 UTC", " UTC")
    axis.xaxis.set_major_locator(FixedLocator(major_s / 86400.0))
    axis.xaxis.set_minor_locator(FixedLocator(minor_s / 86400.0))
    axis.xaxis.set_major_formatter(_RelativeOffsetFormatter(formatter, anchor))
    return step_s, unit


__all__ = ["configure_utc_time_axis", "configure_relative_time_axis"]
