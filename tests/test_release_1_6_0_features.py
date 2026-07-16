from dataclasses import replace
from pathlib import Path

import matplotlib.dates as mdates
import numpy as np

from lmas.demo import synthetic_project
from lmas.model import PlotSpec
from lmas.plotting import create_lma_figure
from lmas.plotting.time_axis import configure_relative_time_axis
from lmas.plotting.spatial_aspect import coordinate_km_per_unit
from lmas.interactions import LinkedViewController
from lmas.plotting.map_underlay import cartography_backend


def test_map_forces_true_aspect_and_defaults_remain_free():
    default = PlotSpec().validated()
    assert default.show_map_underlay is False
    assert default.true_aspect is False
    mapped = replace(default, show_map_underlay=True, true_aspect=False).validated()
    assert mapped.show_map_underlay is True
    assert mapped.true_aspect is True


def test_relative_time_axis_keeps_fixed_origin_and_adaptive_units():
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots()
    origin = mdates.date2num(np.datetime64("2026-07-14T12:00:00", "us").astype(object))
    axis.set_xlim(origin + 0.100 / 86400.0, origin + 0.250 / 86400.0)
    _step, unit = configure_relative_time_axis(
        axis, origin=origin, window_span_s=0.5
    )
    assert unit == "ms"
    formatter = axis.xaxis.get_major_formatter()
    labels = [formatter(value, index) for index, value in enumerate(axis.get_xticks())]
    assert any(label.startswith("1") or label.startswith("2") for label in labels)
    assert "Window start:" in formatter.get_offset()
    plt.close(fig)


def test_true_aspect_metadata_local_and_geodetic():
    project = synthetic_project()
    for coordinate_system in ("local", "geodetic"):
        plot = replace(
            project.plot,
            coordinate_system=coordinate_system,
            true_aspect=True,
            relative_time_from_window_start=True,
        )
        figure = create_lma_figure(project, plot=plot)
        metadata = figure._lmas_metadata
        aspects = metadata["axis_data_aspects"]
        assert aspects[0] is None
        assert all(value is not None and value > 0 for value in aspects[1:])
        assert metadata["axes"]["time_altitude"].get_xlabel().startswith(
            "Time from window start"
        )


def test_overlay_browser_and_peak_current_contracts_are_present():
    root = Path(__file__).resolve().parents[1] / "src" / "lmas" / "gui"
    dialogs = (root / "data_dialogs.py").read_text(encoding="utf-8")
    satellite = (root / "satellite_overlay_window.py").read_text(encoding="utf-8")
    network = (root / "network_overlay_window.py").read_text(encoding="utf-8")
    main = (root / "main_window.py").read_text(encoding="utf-8")
    assert "DontUseNativeDialog" in dialogs
    assert "ShowDirsOnly" in dialogs
    assert "choose_directory_with_files_visible" in satellite
    assert "choose_directory_with_files_visible" in network
    assert "Peak current in the current linked time view" in network
    assert "_restore_and_focus_window" in main


def test_true_aspect_preserves_axes_boxes_and_pads_limits():
    import matplotlib.pyplot as plt

    project = synthetic_project()
    free = create_lma_figure(project, plot=replace(project.plot, true_aspect=False))
    true = create_lma_figure(project, plot=replace(project.plot, true_aspect=True))
    free_axes = free._lmas_metadata["axis_order"]
    true_axes = true._lmas_metadata["axis_order"]
    free.canvas.draw()
    true.canvas.draw()
    for index in (1, 2, 3):
        free_box = free_axes[index].get_position().bounds
        true_box = true_axes[index].get_position().bounds
        assert np.allclose(free_box, true_box, rtol=0.0, atol=1.0e-10)
        names = true._lmas_metadata["coordinate_names"][index]
        x_span = abs(np.diff(true_axes[index].get_xlim())[0])
        y_span = abs(np.diff(true_axes[index].get_ylim())[0])
        position = true_axes[index].get_position()
        figure_width, figure_height = true.get_size_inches()
        x_scale = coordinate_km_per_unit(names[0], project.reference_latitude)
        y_scale = coordinate_km_per_unit(names[1], project.reference_latitude)
        x_km_per_inch = x_span * x_scale / (position.width * figure_width)
        y_km_per_inch = y_span * y_scale / (position.height * figure_height)
        assert np.isclose(x_km_per_inch, y_km_per_inch, rtol=1.0e-9, atol=1.0e-9)
    plt.close(free)
    plt.close(true)


def test_true_aspect_control_can_turn_maps_off_instead_of_locking():
    controls = (
        Path(__file__).resolve().parents[1] / "src" / "lmas" / "gui" / "controls.py"
    ).read_text(encoding="utf-8")
    assert "self.true_aspect.setEnabled(True)" in controls
    assert "if not bool(checked) and self.show_map_underlay.isChecked()" in controls
    assert "self.show_map_underlay.setChecked(False)" in controls


def test_true_aspect_survives_preserved_runtime_view():
    import matplotlib.pyplot as plt

    project = synthetic_project()
    free = create_lma_figure(project, plot=replace(project.plot, true_aspect=False))
    free_controller = LinkedViewController(free)
    assert free_controller.apply_interactive_limits(
        {"east": (-15.0, 20.0), "north": (-10.0, 25.0), "altitude": (0.0, 20.0)},
        initialize_all_matching_axes=True,
    )
    state = free_controller.capture_view_state()

    true = create_lma_figure(project, plot=replace(project.plot, true_aspect=True))
    true_controller = LinkedViewController(true)
    assert true_controller.restore_view_state(state)
    true.canvas.draw()

    figure_width, figure_height = true.get_size_inches()
    scales = []
    for index in (1, 2, 3):
        axis = true._lmas_metadata["axis_order"][index]
        names = true._lmas_metadata["coordinate_names"][index]
        position = axis.get_position()
        x_scale = coordinate_km_per_unit(names[0], project.reference_latitude)
        y_scale = coordinate_km_per_unit(names[1], project.reference_latitude)
        scales.extend(
            [
                abs(np.diff(axis.get_xlim())[0]) * x_scale / (position.width * figure_width),
                abs(np.diff(axis.get_ylim())[0]) * y_scale / (position.height * figure_height),
            ]
        )
    assert np.allclose(scales, scales[0], rtol=1.0e-9, atol=1.0e-9)
    plt.close(free)
    plt.close(true)


def test_bundled_map_backend_draws_without_cartopy_or_basemap():
    import matplotlib.pyplot as plt

    assert cartography_backend() == "bundled"
    project = synthetic_project()
    figure = create_lma_figure(
        project,
        plot=replace(project.plot, true_aspect=True, show_map_underlay=True),
    )
    controller = LinkedViewController(figure)
    underlay = figure._lmas_metadata["map_underlay"]
    assert underlay is not None
    assert underlay.backend == "bundled"
    assert underlay.available
    assert "bundled" in figure._lmas_metadata["map_status"]
    assert sum(underlay.visible_counts.values()) > 0
    controller._refresh_map_underlay()
    plt.close(figure)


def test_project_home_translates_cardinal_viewpoints_before_true_aspect():
    """Project Home is viewpoint-independent and idempotent under True Aspect."""
    import matplotlib.pyplot as plt

    project = synthetic_project(count=360)
    project.plot = replace(
        project.plot,
        true_aspect=True,
        north_south_viewpoint="north",  # plots West rather than canonical East
        east_west_viewpoint="west",    # plots South rather than canonical North
    )
    figure = create_lma_figure(project, plot=project.plot)
    controller = LinkedViewController(figure)

    east = -np.asarray(controller._coordinate_values["west"], dtype=float)
    north = -np.asarray(controller._coordinate_values["south"], dtype=float)
    altitude = np.asarray(controller._coordinate_values["altitude"], dtype=float)
    time_values = np.asarray(controller._coordinate_values["time"], dtype=float)
    home = {
        "time": tuple(np.quantile(time_values, (0.20, 0.80))),
        "east": tuple(np.quantile(east, (0.20, 0.80))),
        "north": tuple(np.quantile(north, (0.20, 0.80))),
        "altitude": tuple(np.quantile(altitude, (0.05, 0.95))),
    }
    original_home = dict(home)

    assert controller.apply_interactive_limits(
        home,
        initialize_all_matching_axes=True,
        soft_startup_view=True,
    )
    first_limits = tuple(
        (tuple(axis.get_xlim()), tuple(axis.get_ylim()))
        for axis in figure._lmas_metadata["axis_order"]
    )

    # Canonical East/North bounds become sign-reversed West/South display
    # constraints without mutating the saved Project Home dictionary.
    assert home == original_home
    assert set(controller._startup_display_constraints) == {
        "time", "west", "south", "altitude"
    }
    expected_west_center = -0.5 * sum(home["east"])
    expected_south_center = -0.5 * sum(home["north"])
    plan = figure._lmas_metadata["axes"]["plan"]
    assert np.isclose(0.5 * sum(plan.get_xlim()), expected_west_center)
    assert np.isclose(0.5 * sum(plan.get_ylim()), expected_south_center)

    # Repeated Project Home restores must reproduce the same padded display,
    # not feed prior True-Aspect padding back into the canonical home limits.
    controller.restore_full(record_history=False)
    assert controller.apply_interactive_limits(
        home,
        initialize_all_matching_axes=True,
        soft_startup_view=True,
    )
    second_limits = tuple(
        (tuple(axis.get_xlim()), tuple(axis.get_ylim()))
        for axis in figure._lmas_metadata["axis_order"]
    )
    for first, second in zip(first_limits, second_limits):
        assert np.allclose(first[0], second[0], rtol=0.0, atol=1.0e-10)
        assert np.allclose(first[1], second[1], rtol=0.0, atol=1.0e-10)

    figure_width, figure_height = figure.get_size_inches()
    scales = []
    for index in (1, 2, 3):
        axis = figure._lmas_metadata["axis_order"][index]
        names = figure._lmas_metadata["coordinate_names"][index]
        position = axis.get_position()
        x_scale = coordinate_km_per_unit(names[0], project.reference_latitude)
        y_scale = coordinate_km_per_unit(names[1], project.reference_latitude)
        scales.extend(
            [
                abs(np.diff(axis.get_xlim())[0]) * x_scale
                / (position.width * figure_width),
                abs(np.diff(axis.get_ylim())[0]) * y_scale
                / (position.height * figure_height),
            ]
        )
    assert np.allclose(scales, scales[0], rtol=1.0e-9, atol=1.0e-9)
    plt.close(figure)


def test_portrait_legend_stays_inside_page_with_full_overlay_labels():
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    from lmas.plotting.figures import refresh_figure_legend

    project = synthetic_project(count=240)
    plot = replace(
        project.plot,
        layout="xlma",
        show_legend=True,
        show_colorbar=True,
    )
    figure = create_lma_figure(project, plot=plot, for_export=True)
    FigureCanvasAgg(figure)
    axes = tuple(figure._lmas_metadata["axis_order"])
    labels = (
        "GOES-16 (East) — GLM event footprints",
        "GOES-16 (East) — GLM group centroids",
        "GOES-17 (West) — GLM event footprints",
        "GOES-17 (West) — GLM group centroids",
        "ENTLN — negative CG",
        "ENTLN — positive CG",
        "ENTLN — IC",
    )
    for index, label in enumerate(labels):
        axes[index % len(axes)].plot([], [], marker="o", linestyle="None", label=label)
    legend = refresh_figure_legend(
        figure,
        axes,
        plot,
        extra_clearance_inches=0.52,
    )
    figure.canvas.draw()
    bbox = legend.get_window_extent(figure.canvas.get_renderer()).transformed(
        figure.transFigure.inverted()
    )
    assert bbox.y0 >= -1.0e-6
    assert bbox.x0 >= -1.0e-6
    assert bbox.x1 <= 1.0 + 1.0e-6
    # Keep the legend inside the dedicated caption gutter below the plan row.
    plan_bottom = figure._lmas_metadata["axes"]["plan"].get_position().y0
    assert bbox.y1 < plan_bottom
    plt.close(figure)


def test_portrait_fixed_canvas_separates_toe_source_colorbars_and_legend():
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import LogNorm

    from lmas.overlays.satellite.manager import SatelliteOverlayManager
    from lmas.overlays.satellite.rendering import SatelliteOverlayRenderer
    from lmas.plotting.figures import refresh_figure_legend

    project = synthetic_project(count=240)
    plot = replace(
        project.plot,
        layout="xlma",
        show_legend=True,
        show_colorbar=True,
    )
    figure = create_lma_figure(project, plot=plot, for_export=True)
    FigureCanvasAgg(figure)
    axes = tuple(figure._lmas_metadata["axis_order"])
    labels = (
        "GOES-16 (East) — GLM event footprints",
        "GOES-16 (East) — GLM group centroids",
        "GOES-17 (West) — GLM event footprints",
        "GOES-17 (West) — GLM group centroids",
        "ENTLN — negative CG",
        "ENTLN — positive CG",
        "ENTLN — IC",
    )
    for index, label in enumerate(labels):
        axes[index % len(axes)].plot([], [], marker="o", linestyle="None", label=label)

    renderer = SatelliteOverlayRenderer(
        SatelliteOverlayManager(), figure=figure, project=project
    )
    renderer._update_bottom_colorbars(((
        "GLM Total Optical Energy (fJ)",
        plt.get_cmap("viridis"),
        LogNorm(1.0, 100.0),
    ),))
    legend = refresh_figure_legend(figure, axes, plot, extra_clearance_inches=0.52)
    figure.canvas.draw()
    canvas_renderer = figure.canvas.get_renderer()
    figure_width, figure_height = (float(value) for value in figure.get_size_inches())

    legend_bbox = legend.get_window_extent(canvas_renderer).transformed(
        figure.transFigure.inverted()
    )
    toe_axis = renderer._bottom_colorbar_axes[0]
    toe_box = toe_axis.get_position()
    toe_label_bbox = toe_axis.yaxis.label.get_window_extent(canvas_renderer).transformed(
        figure.transFigure.inverted()
    )
    source_box = figure._lmas_metadata["colorbar"].ax.get_position()
    plan_box = figure._lmas_metadata["axes"]["plan"].get_position()

    assert np.allclose((figure_width, figure_height), (10.55, 11.0))
    assert np.isclose(plan_box.width * figure_width, 5.30, atol=1.0e-9)
    assert np.isclose(plan_box.height * figure_height, 5.30, atol=1.0e-9)
    assert renderer.colorbars[0].orientation == "vertical"
    assert toe_box.x1 < plan_box.x0
    assert source_box.x0 > max(axis.get_position().x1 for axis in axes)
    assert toe_label_bbox.x0 >= -1.0e-6
    assert legend_bbox.y1 < plan_box.y0
    assert legend_bbox.x0 >= -1.0e-6
    assert legend_bbox.x1 <= 1.0 + 1.0e-6
    plt.close(figure)

def test_portrait_without_glm_toe_keeps_fixed_canvas_and_caption_legend():
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    from lmas.plotting.figures import refresh_figure_legend

    project = synthetic_project(count=240)
    plot = replace(project.plot, layout="xlma", show_legend=True, show_colorbar=True)
    figure = create_lma_figure(project, plot=plot, for_export=True)
    FigureCanvasAgg(figure)
    axes = tuple(figure._lmas_metadata["axis_order"])
    axes[0].plot([], [], marker="v", linestyle="None", label="ENTLN — negative CG")
    axes[0].plot([], [], marker="x", linestyle="None", label="ENTLN — IC")
    legend = refresh_figure_legend(figure, axes, plot)
    figure.canvas.draw()

    assert np.allclose(figure.get_size_inches(), (10.55, 11.0))
    plan_box = figure._lmas_metadata["axes"]["plan"].get_position()
    legend_bbox = legend.get_window_extent(figure.canvas.get_renderer()).transformed(
        figure.transFigure.inverted()
    )
    assert legend_bbox.y0 >= -1.0e-6
    assert legend_bbox.y1 < plan_box.y0
    assert legend_bbox.x0 >= -1.0e-6
    assert legend_bbox.x1 <= 1.0 + 1.0e-6
    plt.close(figure)

def test_portrait_toe_toggle_preserves_fixed_canvas():
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import LogNorm

    from lmas.overlays.satellite.manager import SatelliteOverlayManager
    from lmas.overlays.satellite.rendering import SatelliteOverlayRenderer

    project = synthetic_project(count=240)
    plot = replace(project.plot, layout="xlma", show_legend=True, show_colorbar=True)
    figure = create_lma_figure(project, plot=plot, for_export=True)
    FigureCanvasAgg(figure)
    axes = tuple(figure._lmas_metadata["axis_order"])
    axes[0].plot([], [], marker="s", linestyle="None", label="GOES-16 (East) — GLM event footprints")
    axes[0].plot([], [], marker="o", linestyle="None", label="GOES-16 (East) — GLM group centroids")
    renderer = SatelliteOverlayRenderer(SatelliteOverlayManager(), figure=figure, project=project)
    initial_size = tuple(float(value) for value in figure.get_size_inches())

    renderer._update_bottom_colorbars(((
        "GLM Total Optical Energy (fJ)",
        plt.get_cmap("viridis"),
        LogNorm(1.0, 100.0),
    ),))
    renderer._refresh_figure_legend(True)
    assert np.allclose(figure.get_size_inches(), initial_size)
    assert renderer.colorbars[0].orientation == "vertical"

    renderer._update_bottom_colorbars(())
    renderer._refresh_figure_legend(False)
    figure.canvas.draw()
    assert np.allclose(figure.get_size_inches(), initial_size)
    assert np.allclose(figure._lmas_metadata["export_size_inches"], (10.55, 11.0))
    assert all(not axis.get_visible() for axis in renderer._bottom_colorbar_axes)
    plt.close(figure)

def test_portrait_true_aspect_uses_one_common_initial_spatial_scale():
    import matplotlib.pyplot as plt

    project = synthetic_project(count=600)
    plot = replace(project.plot, layout="xlma", true_aspect=True)
    figure = create_lma_figure(project, plot=plot)

    figure_width, figure_height = figure.get_size_inches()
    scales = []
    for index in (1, 2, 3):
        axis = figure._lmas_metadata["axis_order"][index]
        names = figure._lmas_metadata["coordinate_names"][index]
        position = axis.get_position()
        x_scale = coordinate_km_per_unit(names[0], project.reference_latitude)
        y_scale = coordinate_km_per_unit(names[1], project.reference_latitude)
        scales.extend(
            [
                abs(np.diff(axis.get_xlim())[0]) * x_scale
                / (position.width * figure_width),
                abs(np.diff(axis.get_ylim())[0]) * y_scale
                / (position.height * figure_height),
            ]
        )
    assert np.allclose(scales, scales[0], rtol=1.0e-9, atol=1.0e-9)
    plt.close(figure)

def test_portrait_overlay_toggles_do_not_resize_fixed_canvas():
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import LogNorm

    from lmas.overlays.satellite.manager import SatelliteOverlayManager
    from lmas.overlays.satellite.rendering import SatelliteOverlayRenderer

    project = synthetic_project(count=240)
    plot = replace(project.plot, layout="xlma", show_legend=True, show_colorbar=True)
    figure = create_lma_figure(project, plot=plot, for_export=False)
    FigureCanvasAgg(figure)
    renderer = SatelliteOverlayRenderer(
        SatelliteOverlayManager(), figure=figure, project=project
    )
    initial_live_size = tuple(float(value) for value in figure.get_size_inches())
    colorbar_spec = ((
        "GLM Total Optical Energy (fJ)",
        plt.get_cmap("viridis"),
        LogNorm(1.0, 100.0),
    ),)

    for _ in range(4):
        renderer._update_bottom_colorbars(colorbar_spec)
        renderer._refresh_figure_legend(True)
        assert np.allclose(figure.get_size_inches(), initial_live_size)
        assert np.allclose(figure._lmas_metadata["export_size_inches"], (10.55, 11.0))

        renderer._update_bottom_colorbars(())
        renderer._refresh_figure_legend(False)
        assert np.allclose(figure.get_size_inches(), initial_live_size)
        assert np.allclose(figure._lmas_metadata["export_size_inches"], (10.55, 11.0))

    plt.close(figure)

def test_landscape_glm_toe_bar_is_raised_and_matches_axis_typography():
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import LogNorm

    from lmas.overlays.satellite.manager import SatelliteOverlayManager
    from lmas.overlays.satellite.rendering import SatelliteOverlayRenderer

    project = synthetic_project(count=240)
    plot = replace(
        project.plot,
        layout="landscape",
        show_legend=True,
        show_colorbar=True,
    )
    figure = create_lma_figure(project, plot=plot, for_export=True)
    FigureCanvasAgg(figure)
    axes = tuple(figure._lmas_metadata["axis_order"])
    labels = (
        "GOES-16 (East) — GLM event footprints",
        "GOES-16 (East) — GLM group centroids",
        "GOES-17 (West) — GLM event footprints",
        "GOES-17 (West) — GLM group centroids",
        "ENTLN — negative CG",
        "ENTLN — positive CG",
        "ENTLN — IC",
    )
    for index, label in enumerate(labels):
        axes[index % len(axes)].plot([], [], marker="o", linestyle="None", label=label)

    renderer = SatelliteOverlayRenderer(
        SatelliteOverlayManager(), figure=figure, project=project
    )
    renderer._update_bottom_colorbars(((
        "GLM Total Optical Energy (fJ)",
        plt.get_cmap("viridis"),
        LogNorm(1.0, 100.0),
    ),))
    renderer._refresh_figure_legend(True)
    figure.canvas.draw()

    canvas_renderer = figure.canvas.get_renderer()
    figure_height = float(figure.get_figheight())
    bottom_normalized = min(axis.get_position().y0 for axis in axes)
    bottom_axis_inches = bottom_normalized * figure_height
    colorbar_axis = renderer._bottom_colorbar_axes[0]
    colorbar_box = colorbar_axis.get_position()
    colorbar_label_box = colorbar_axis.xaxis.label.get_window_extent(canvas_renderer).transformed(
        figure.transFigure.inverted()
    )
    legend = figure._lmas_metadata["legend"]
    legend_box = legend.get_window_extent(canvas_renderer).transformed(
        figure.transFigure.inverted()
    )

    expected_height = 0.014
    expected_y = max(0.055, bottom_normalized - 0.100) + 0.5 * expected_height
    assert np.isclose(colorbar_box.y0, expected_y)
    assert np.isclose(colorbar_box.height, expected_height)
    assert np.isclose(
        colorbar_axis.xaxis.label.get_fontsize(),
        axes[0].xaxis.label.get_fontsize(),
    )
    assert np.isclose(
        colorbar_axis.get_xticklabels()[0].get_fontsize(),
        axes[0].get_xticklabels()[0].get_fontsize(),
    )
    assert bottom_axis_inches - colorbar_label_box.y1 * figure_height < 0.60
    assert colorbar_box.y0 * figure_height - legend_box.y1 * figure_height >= 0.06
    assert legend.get_in_layout()
    plt.close(figure)

