"""Pure geometry helpers for responsive LMAS tool windows."""

from __future__ import annotations

from typing import NamedTuple


class Rect(NamedTuple):
    x: int
    y: int
    width: int
    height: int


def full_height_tool_geometry(
    main: Rect,
    current: Rect,
    available: Rect,
    *,
    gap: int = 8,
    minimum_width: int = 520,
    minimum_height: int = 420,
    work_area_margin: int = 12,
) -> Rect:
    """Fit a tool window to the main window's usable vertical extent.

    The tool is placed beside the main window when either side has room and is
    otherwise clamped into the monitor work area. Its current width is retained
    where possible. Returned geometry never exceeds the available work area.
    """

    margin = max(0, int(work_area_margin))
    usable_x = available.x + margin
    usable_y = available.y + margin
    usable_width = max(1, available.width - (2 * margin))
    usable_height = max(1, available.height - (2 * margin))
    available_right = usable_x + usable_width
    available_bottom = usable_y + usable_height
    main_bottom = main.y + max(0, main.height)

    top = max(usable_y, main.y)
    bottom = min(available_bottom, main_bottom)
    height = max(minimum_height, bottom - top)
    height = min(height, usable_height)
    if top + height > available_bottom:
        top = max(usable_y, available_bottom - height)

    width = max(minimum_width, current.width)
    width = min(width, usable_width)

    right_candidate = main.x + main.width + gap
    left_candidate = main.x - width - gap
    if right_candidate + width <= available_right:
        x = right_candidate
    elif left_candidate >= usable_x:
        x = left_candidate
    else:
        x = min(max(current.x, usable_x), max(usable_x, available_right - width))

    return Rect(int(x), int(top), int(width), int(height))


def clamp_frame_to_work_area(
    frame: Rect,
    available: Rect,
    *,
    margin: int = 16,
    minimum_width: int = 320,
    minimum_height: int = 320,
) -> Rect:
    """Clamp a decorated top-level window frame inside a screen work area.

    This second-stage clamp is intentionally separate from
    :func:`full_height_tool_geometry`.  Window-manager decoration sizes are not
    always final until after the Qt window is shown, especially on Windows.
    Applying this helper to the realized frame prevents the title bar from
    landing above the visible work area.
    """

    inset = max(0, int(margin))
    left = available.x + inset
    top = available.y + inset
    right = available.x + available.width - inset
    bottom = available.y + available.height - inset
    usable_width = max(1, right - left)
    usable_height = max(1, bottom - top)

    width = min(max(int(minimum_width), int(frame.width)), usable_width)
    height = min(max(int(minimum_height), int(frame.height)), usable_height)
    x = min(max(int(frame.x), left), max(left, right - width))
    y = min(max(int(frame.y), top), max(top, bottom - height))
    return Rect(x, y, width, height)


__all__ = ["Rect", "clamp_frame_to_work_area", "full_height_tool_geometry"]
