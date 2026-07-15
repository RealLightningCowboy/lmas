from __future__ import annotations

import numpy as np

from lmas.demo import synthetic_project
from lmas.interactions import LinkedViewController
from lmas.plotting import create_lma_figure


def _soft_startup_controller():
    project = synthetic_project(count=240)
    figure = create_lma_figure(project)
    controller = LinkedViewController(figure)
    assert controller.enabled

    index = 90
    limits = {}
    for name in ("time", "east", "north", "altitude"):
        values = np.asarray(controller._coordinate_values[name], dtype=float)
        span = max(float(np.ptp(values)), 1.0e-9)
        half_width = 0.025 * span
        centre = float(values[index])
        limits[name] = (centre - half_width, centre + half_width)

    assert controller.apply_interactive_limits(
        limits,
        initialize_all_matching_axes=True,
        soft_startup_view=True,
    )
    return controller, figure, controller._active_mask.size, limits


def _expand_axis_fraction(
    controller: LinkedViewController,
    axis,
    factor: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    x_limits = tuple(float(value) for value in axis.get_xlim())
    y_limits = tuple(float(value) for value in axis.get_ylim())
    x_centre = 0.5 * sum(x_limits)
    y_centre = 0.5 * sum(y_limits)
    x_span = abs(x_limits[1] - x_limits[0]) * factor
    y_span = abs(y_limits[1] - y_limits[0]) * factor
    requested_x = (x_centre - 0.5 * x_span, x_centre + 0.5 * x_span)
    requested_y = (y_centre - 0.5 * y_span, y_centre + 0.5 * y_span)
    axis.set_xlim(requested_x, emit=False)
    axis.set_ylim(requested_y, emit=False)
    controller._pending_explicit = True
    controller.update_now(axis, record_history=False)
    return requested_x, requested_y


def test_project_startup_keeps_exact_opening_but_uses_manual_navigation_constraints() -> None:
    controller, _figure, total, limits = _soft_startup_controller()
    opening_count = controller.visible_count

    assert 0 < opening_count < total
    assert controller._soft_startup_view is True
    assert set(controller._constraints) == {"time", "altitude"}
    assert controller._startup_display_constraints == limits

    exact_opening = controller._mask_for_constraints(limits)
    assert controller.visible_count == int(np.count_nonzero(exact_opening))


def test_first_project_scroll_expands_incrementally_without_home_jump() -> None:
    controller, figure, total, _limits = _soft_startup_controller()
    opening_count = controller.visible_count
    time_axis = figure._lmas_metadata["axis_order"][0]
    full_x, full_y = controller._initial_limits[0]

    requested_x, requested_y = _expand_axis_fraction(
        controller,
        time_axis,
        1.20,
    )

    assert controller._soft_startup_view is False
    assert controller._startup_display_constraints == {}
    assert set(controller._constraints) == {"time", "altitude"}
    assert opening_count < controller.visible_count < total
    assert np.allclose(time_axis.get_xlim(), requested_x)
    assert np.allclose(time_axis.get_ylim(), requested_y)
    assert abs(time_axis.get_xlim()[1] - time_axis.get_xlim()[0]) < 0.5 * abs(full_x[1] - full_x[0])
    assert not np.allclose(time_axis.get_ylim(), full_y)


def test_repeated_project_scrolls_grow_smoothly() -> None:
    controller, figure, total, _limits = _soft_startup_controller()
    time_axis = figure._lmas_metadata["axis_order"][0]

    counts = [controller.visible_count]
    x_spans = [abs(float(time_axis.get_xlim()[1] - time_axis.get_xlim()[0]))]
    for _ in range(3):
        requested_x, requested_y = _expand_axis_fraction(
            controller,
            time_axis,
            1.20,
        )
        counts.append(controller.visible_count)
        x_spans.append(abs(float(requested_x[1] - requested_x[0])))
        assert np.allclose(time_axis.get_xlim(), requested_x)
        assert np.allclose(time_axis.get_ylim(), requested_y)

    assert counts == sorted(counts)
    assert counts[-1] < total
    assert x_spans == sorted(x_spans)


def test_restored_startup_history_entry_remains_non_destructive() -> None:
    controller, figure, total, limits = _soft_startup_controller()
    opening_count = controller.visible_count
    state = controller.capture_view_state()
    assert state is not None
    assert state["soft_startup_view"] is True
    assert state["startup_display_constraints"] == limits

    controller.restore_full(record_history=False)
    assert controller.visible_count == total
    assert controller.restore_view_state(
        state,
        exact_membership=True,
        record_history=False,
    )
    assert controller.visible_count == opening_count
    assert controller._soft_startup_view is True
    assert set(controller._constraints) == {"time", "altitude"}

    time_axis = figure._lmas_metadata["axis_order"][0]
    _expand_axis_fraction(controller, time_axis, 1.20)
    assert opening_count < controller.visible_count < total
