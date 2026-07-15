from __future__ import annotations

from lmas.demo import demo_project, packaged_demo_project_path
from lmas.overlays.satellite import SatelliteOverlayManager


def test_packaged_hybrid_demo_loads_real_lma_and_dual_glm():
    project_path = packaged_demo_project_path()
    assert project_path.is_file()
    project = demo_project()
    assert project.event_count == 42_513
    assert len(project.source_files) == 1
    assert project.source_files[0].name == "LYLOUT_190430_144844_0060.dat.gz"
    assert project.view_filters.start_time == "2019-04-30T14:49:14.142212000"
    assert project.plot.show_legend is True

    manager = SatelliteOverlayManager()
    manager.restore_project_state(
        project.satellite_overlay_state,
        project_directory=project.project_path.parent,
    )
    assert manager.last_restore_errors == []
    records = manager.records
    assert len(records) == 2
    summary = {
        (record.observation.identity.spacecraft_name, record.observation.identity.position_name): (
            len(record.observation.identity.source_files),
            len(record.observation.events),
        )
        for record in records
    }
    assert summary[("GOES-16", "East")][0] == 4
    assert summary[("GOES-17", "West")][0] == 4
    assert summary[("GOES-16", "East")][1] > 20_000
    assert summary[("GOES-17", "West")][1] > 8_000


def test_packaged_hybrid_demo_glm_is_disabled_by_default():
    project = demo_project()
    state = project.satellite_overlay_state
    assert state["datasets"]
    assert all(not item["style"]["enabled"] for item in state["datasets"])


def test_demo_startup_view_zoom_out_reveals_surrounding_record() -> None:
    """A reopened demo scrolls outward gradually, exactly like a manual view."""

    import numpy as np

    from lmas.interactions import LinkedViewController
    from lmas.plotting import create_lma_figure

    project = demo_project()
    figure = create_lma_figure(project, plot=project.plot)
    controller = LinkedViewController(figure)
    assert controller.apply_interactive_limits(
        project.project_home_limits,
        initialize_all_matching_axes=True,
        soft_startup_view=True,
    )
    opening_count = controller.visible_count
    assert opening_count == 2_483
    assert set(controller._constraints) == {"time", "altitude"}

    time_axis = figure._lmas_metadata["axis_order"][0]
    full_x, _full_y = controller._initial_limits[0]
    opening_x = tuple(float(value) for value in time_axis.get_xlim())
    opening_y = tuple(float(value) for value in time_axis.get_ylim())
    centre = 0.5 * sum(opening_x)
    span = abs(opening_x[1] - opening_x[0]) * 1.20
    requested_x = (centre - 0.5 * span, centre + 0.5 * span)
    time_axis.set_xlim(requested_x, emit=False)
    time_axis.set_ylim(opening_y, emit=False)
    controller._pending_explicit = True
    controller.update_now(time_axis, record_history=False)

    assert controller.visible_count == 2_552
    assert opening_count < controller.visible_count < 10_000
    assert set(controller._constraints) == {"time", "altitude"}
    assert np.allclose(time_axis.get_xlim(), requested_x)
    assert np.allclose(time_axis.get_ylim(), opening_y)
    assert abs(time_axis.get_xlim()[1] - time_axis.get_xlim()[0]) < 0.5 * abs(full_x[1] - full_x[0])

