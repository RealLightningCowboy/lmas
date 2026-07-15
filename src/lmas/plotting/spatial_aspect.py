from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np


def coordinate_km_per_unit(name: str, reference_latitude: float) -> float | None:
    """Return physical kilometres represented by one plotted coordinate unit."""
    key = str(name)
    if key in {"east", "west", "north", "south", "altitude"}:
        return 1.0
    if key == "latitude":
        return 111.195
    if key == "longitude":
        return 111.195 * max(abs(float(np.cos(np.deg2rad(reference_latitude)))), 1.0e-6)
    return None


def axis_data_aspect(
    names: tuple[str, str], reference_latitude: float
) -> float | None:
    """Matplotlib data-aspect ratio for equal physical distance on x and y."""
    x_scale = coordinate_km_per_unit(names[0], reference_latitude)
    y_scale = coordinate_km_per_unit(names[1], reference_latitude)
    if x_scale is None or y_scale is None or x_scale <= 0 or y_scale <= 0:
        return None
    return float(y_scale / x_scale)


def symmetric_limits(
    limits: tuple[float, float],
    target_span: float,
    *,
    expand_only: bool = False,
    centre: float | None = None,
) -> tuple[float, float]:
    """Return an interval of ``target_span`` around the current centre.

    ``expand_only`` preserves the complete current interval.  This is used for
    the spatial panel that drove an interactive zoom.  Other linked panels may
    shrink as well as expand so their physical scale follows that driver rather
    than forcing a narrow Portrait selection back out to the full altitude
    envelope.
    """

    reverse = float(limits[0]) > float(limits[1])
    low, high = sorted((float(limits[0]), float(limits[1])))
    target = float(target_span)
    if expand_only:
        target = max(target, high - low)
    target = max(target, 1.0e-12)
    midpoint = 0.5 * (low + high) if centre is None else float(centre)
    if not np.isfinite(midpoint):
        midpoint = 0.5 * (low + high)
    result = (midpoint - 0.5 * target, midpoint + 0.5 * target)
    return (result[1], result[0]) if reverse else result


def expanded_symmetric_limits(
    limits: tuple[float, float], target_span: float
) -> tuple[float, float]:
    return symmetric_limits(limits, target_span, expand_only=True)


def enforce_true_spatial_scale(
    figure,
    axes: Sequence,
    coordinate_names: Sequence[Sequence[str]],
    *,
    axis_indices: Iterable[int],
    reference_latitude: float,
    anchors: Sequence[str] | None = None,
    driver_axis_index: int | None = None,
    coordinate_centres: Mapping[str, float] | None = None,
    emit: bool = False,
) -> tuple[float | None, ...]:
    """Enforce one common physical-distance scale across every spatial panel.

    LMAS keeps the designed axes rectangles fixed.  During initial rendering
    the largest current kilometres-per-inch requirement is used and shorter
    intervals are padded symmetrically.  During an interactive Portrait zoom,
    ``driver_axis_index`` makes the selected panel authoritative: its complete
    rectangle is preserved while the other spatial axes may shrink or expand
    around their current centres.  This prevents the shallow Portrait vertical
    panels from imposing the full altitude envelope on every plan-view zoom.
    """

    count = len(axes)
    selected = {int(index) for index in axis_indices if 0 <= int(index) < count}
    data_aspects: list[float | None] = []
    for index, names in enumerate(coordinate_names):
        pair = tuple(str(value) for value in names)
        data_aspects.append(
            axis_data_aspect(pair, reference_latitude) if len(pair) == 2 else None
        )
        if index < count:
            axes[index].set_aspect("auto")
            if anchors is not None and index < len(anchors):
                axes[index].set_anchor(anchors[index])

    if not selected:
        return tuple(data_aspects)

    fig_width, fig_height = (float(value) for value in figure.get_size_inches())
    requirements: list[float] = []
    dimensions: list[tuple[int, str, float, float, float]] = []

    for index in sorted(selected):
        if index >= len(coordinate_names):
            continue
        names = tuple(str(value) for value in coordinate_names[index])
        if len(names) != 2:
            continue
        axis = axes[index]
        position = axis.get_position().frozen()
        lengths = {
            "x": float(position.width) * fig_width,
            "y": float(position.height) * fig_height,
        }
        for dimension, name in zip(("x", "y"), names):
            km_per_unit = coordinate_km_per_unit(name, reference_latitude)
            physical_length = lengths[dimension]
            if km_per_unit is None or km_per_unit <= 0 or physical_length <= 0:
                continue
            limits = axis.get_xlim() if dimension == "x" else axis.get_ylim()
            span_units = abs(float(limits[1]) - float(limits[0]))
            if not np.isfinite(span_units) or span_units <= 0:
                continue
            span_km = span_units * km_per_unit
            requirement = span_km / physical_length
            requirements.append(requirement)
            dimensions.append(
                (index, dimension, km_per_unit, physical_length, requirement)
            )

    if not requirements:
        return tuple(data_aspects)

    driver = None if driver_axis_index is None else int(driver_axis_index)
    driver_requirements = [
        requirement
        for index, _dimension, _km_per_unit, _physical_length, requirement in dimensions
        if index == driver
    ]
    shared_km_per_inch = max(driver_requirements or requirements)
    for index, dimension, km_per_unit, physical_length, _requirement in dimensions:
        axis = axes[index]
        limits = axis.get_xlim() if dimension == "x" else axis.get_ylim()
        target_units = shared_km_per_inch * physical_length / km_per_unit
        names = tuple(str(value) for value in coordinate_names[index])
        coordinate_name = names[0 if dimension == "x" else 1]
        centre = (
            None
            if driver is None or index == driver or coordinate_centres is None
            else coordinate_centres.get(coordinate_name)
        )
        adjusted = symmetric_limits(
            limits,
            target_units,
            expand_only=(driver is None or index == driver),
            centre=centre,
        )
        if dimension == "x":
            axis.set_xlim(adjusted, emit=emit)
        else:
            axis.set_ylim(adjusted, emit=emit)

    return tuple(data_aspects)


__all__ = [
    "axis_data_aspect",
    "coordinate_km_per_unit",
    "enforce_true_spatial_scale",
    "expanded_symmetric_limits",
    "symmetric_limits",
]
