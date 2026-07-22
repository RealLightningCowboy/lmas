from pathlib import Path

import numpy as np
import yaml

from lmas.demo import synthetic_project
from lmas.io.project import PROJECT_FORMAT, load_project, save_project
from lmas.source_selection import (
    CHARGE_COLORS,
    SourceSelectionGroup,
    SourceSelectionManager,
    clustered_hulls_geometry,
    concave_hull_geometry,
    convex_hull_geometry,
    projection_hull_geometry,
    reduce_points_for_hull,
)


def test_group_display_and_charge_assignment_are_undoable():
    manager = SourceSelectionManager()
    manager.apply([1, 2, 3], "replace")
    assert manager.set_display_style("concave_hull")
    assert manager.active_group.display_style == "concave_hull"
    assert manager.set_charge_category("positive")
    assert manager.active_group.charge_category == "positive"
    assert manager.active_group.color == CHARGE_COLORS["positive"]
    assert manager.undo()
    assert manager.active_group.charge_category == "unassigned"
    assert manager.active_group.display_style == "concave_hull"


def test_manager_round_trip_preserves_charge_provenance_and_style():
    manager = SourceSelectionManager()
    manager.apply([4, 8, 15, 16], "replace")
    manager.set_display_style("clustered_hulls")
    manager.set_charge_category("negative")
    payload = manager.to_dict()

    restored = SourceSelectionManager()
    restored.load_groups(payload["groups"], active_name=payload["active_group"])
    group = restored.active_group
    assert group.source_ids == frozenset({4, 8, 15, 16})
    assert group.display_style == "clustered_hulls"
    assert group.charge_category == "negative"
    assert group.color == CHARGE_COLORS["negative"]
    assert group.created_utc
    assert group.modified_utc
    assert group.created_with_lmas_version == "1.6.2"


def test_overlap_reporting_can_focus_on_assigned_groups():
    manager = SourceSelectionManager()
    manager.apply([1, 2, 3], "replace")
    manager.set_charge_category("positive")
    manager.new_group("Other", source_ids=[3, 4])
    assert manager.overlapping_source_ids("Selection 1") == frozenset({3})
    assert manager.overlapping_source_ids("Selection 1", assigned_only=True) == frozenset()
    manager.set_charge_category("negative")
    assert manager.overlapping_source_ids("Selection 1", assigned_only=True) == frozenset({3})


def _two_clouds():
    rng = np.random.default_rng(42)
    first = rng.normal(loc=(-1.0, -0.5), scale=(0.14, 0.09), size=(180, 2))
    second = rng.normal(loc=(1.0, 0.7), scale=(0.12, 0.10), size=(160, 2))
    return np.vstack((first, second))


def test_three_hull_modes_produce_projection_geometry():
    points = _two_clouds()
    convex = convex_hull_geometry(points)
    concave = concave_hull_geometry(points)
    clustered = clustered_hulls_geometry(points)
    assert len(convex.faces) == 1
    assert convex.boundaries
    assert concave.faces and concave.boundaries
    assert clustered.faces and len(clustered.boundaries) >= 2
    assert projection_hull_geometry(points, "convex_hull").method == "convex"
    assert projection_hull_geometry(points, "concave_hull").method == "concave"
    assert projection_hull_geometry(points, "clustered_hulls").method == "clustered"


def test_hull_reduction_is_deterministic_and_bounded():
    rng = np.random.default_rng(5)
    points = rng.normal(size=(25000, 2))
    first = reduce_points_for_hull(points, max_points=900)
    second = reduce_points_for_hull(points, max_points=900)
    assert first.shape[0] <= 900
    np.testing.assert_allclose(first, second)


def test_project_round_trip_preserves_source_selection_and_charge_state(tmp_path: Path):
    project = synthetic_project(count=32)
    manager = SourceSelectionManager()
    manager.apply([1, 2, 7], "replace")
    manager.set_charge_category("positive")
    manager.set_display_style("convex_hull")
    project.source_selection_state = {
        **manager.to_dict(),
        "category_visibility": {
            "unassigned": False,
            "positive": True,
            "negative": True,
        },
        "selection_scope": "filtered",
        "member_display_scope": "all",
    }
    destination = save_project(project, tmp_path / "charge-test.lmas-project.yaml")
    payload = yaml.safe_load(destination.read_text(encoding="utf-8"))
    assert payload["format"] == PROJECT_FORMAT
    assert payload["analysis"]["source_selection"]["groups"][0]["charge_category"] == "positive"

    loaded = load_project(destination)
    state = loaded.source_selection_state
    assert state["active_group"] == "Selection 1"
    assert state["groups"][0]["source_ids"] == [1, 2, 7]
    assert state["groups"][0]["display_style"] == "convex_hull"
    assert state["groups"][0]["charge_category"] == "positive"
    assert state["category_visibility"]["unassigned"] is False
    assert state["member_display_scope"] == "all"


def test_old_group_payloads_migrate_to_safe_defaults():
    group = SourceSelectionGroup.from_dict({"name": "Legacy", "source_ids": [2, 3]})
    assert group.display_style == "recolor"
    assert group.charge_category == "unassigned"
    assert group.color == CHARGE_COLORS["unassigned"]


def test_charge_color_values_preserve_conflicts_and_show_latest_assignment():
    from lmas.source_selection import charge_values_for_source_ids

    ids = np.array([1, 2, 3, 4, 5])
    state = {
        "groups": [
            {"name": "Positive", "source_ids": [1, 2, 3], "charge_category": "positive"},
            {"name": "Negative", "source_ids": [3, 4], "charge_category": "negative"},
        ]
    }
    values, conflicts = charge_values_for_source_ids(ids, state)
    np.testing.assert_array_equal(values, np.array([1.0, 1.0, -1.0, -1.0, 0.0]))
    np.testing.assert_array_equal(conflicts, np.array([False, False, True, False, False]))


def test_charge_plot_mode_and_selection_scope_metadata():
    from matplotlib.colors import BoundaryNorm

    from lmas.model import PlotSpec
    from lmas.plotting import create_lma_figure

    project = synthetic_project(count=80)
    source_ids = project.dataset["event_source_index"].values.astype(int)
    project.source_selection_state = {
        "groups": [
            {"name": "Positive", "source_ids": source_ids[:8].tolist(), "charge_category": "positive"},
            {"name": "Negative", "source_ids": source_ids[8:16].tolist(), "charge_category": "negative"},
        ]
    }
    plot = PlotSpec(color_by="charge", log_color_scale=True)
    assert plot.validated().log_color_scale is False
    figure = create_lma_figure(project, plot=plot)
    metadata = figure._lmas_metadata
    assert metadata["categorical_color"] is True
    assert metadata["remap_colormap"] is False
    assert isinstance(metadata["norm"], BoundaryNorm)
    assert set(np.unique(metadata["color_values"])).issubset({-1.0, 0.0, 1.0})
    assert set(metadata["selection_scopes"]) == {"filtered", "all"}
    assert len(metadata["selection_scopes"]["all"]["source_ids"]) == project.event_count
    assert "charge" in project.available_color_fields


def test_unassigned_charge_maps_to_neutral_gray():
    from matplotlib.colors import to_rgba

    from lmas.plotting.figures import _charge_cmap_and_norm

    cmap, norm = _charge_cmap_and_norm()
    np.testing.assert_allclose(cmap(norm([0.0]))[0], to_rgba(CHARGE_COLORS["unassigned"]))


def test_charge_redraw_does_not_restore_continuous_norm():
    from matplotlib.colors import BoundaryNorm

    from lmas.interactions import LinkedViewController
    from lmas.model import PlotSpec
    from lmas.plotting import create_lma_figure

    project = synthetic_project(count=80)
    time_figure = create_lma_figure(project, plot=PlotSpec(color_by="time"))
    time_controller = LinkedViewController(time_figure)
    state = time_controller.capture_view_state()
    assert state is not None

    charge_figure = create_lma_figure(project, plot=PlotSpec(color_by="charge"))
    charge_controller = LinkedViewController(charge_figure)
    assert charge_controller.restore_view_state(state, exact_membership=False)
    for scatter in charge_figure._lmas_metadata["scatters"]:
        assert isinstance(scatter.norm, BoundaryNorm)
