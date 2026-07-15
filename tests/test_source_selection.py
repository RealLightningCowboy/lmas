import numpy as np

from lmas.source_selection import SourceSelectionManager, source_ids_inside_polygon


def test_new_groups_default_to_recolor():
    manager = SourceSelectionManager()
    assert manager.active_group.display_style == "recolor"
    second = manager.new_group("Second")
    assert second.display_style == "recolor"


def test_selection_operations_and_undo():
    manager = SourceSelectionManager()
    assert manager.apply([1, 2, 3], "replace")
    assert manager.active_group.source_ids == frozenset({1, 2, 3})
    assert manager.apply([3, 4], "add")
    assert manager.active_group.source_ids == frozenset({1, 2, 3, 4})
    assert manager.apply([2, 9], "subtract")
    assert manager.active_group.source_ids == frozenset({1, 3, 4})
    assert manager.apply([1, 4, 7], "intersect")
    assert manager.active_group.source_ids == frozenset({1, 4})
    assert manager.undo()
    assert manager.active_group.source_ids == frozenset({1, 3, 4})


def test_locked_group_cannot_be_edited():
    manager = SourceSelectionManager()
    manager.apply([10], "replace")
    assert manager.set_locked(True)
    assert not manager.apply([20], "replace")
    assert manager.active_group.source_ids == frozenset({10})


def test_named_groups_preserve_source_identity():
    manager = SourceSelectionManager()
    manager.apply([2, 5], "replace")
    second = manager.new_group("Leader", source_ids=[7, 8])
    assert second.name == "Leader"
    assert manager.set_active("Selection 1")
    assert manager.active_group.source_ids == frozenset({2, 5})
    renamed = manager.rename_group("Charge region")
    assert renamed == "Charge region"
    assert manager.active_name == "Charge region"


def test_polygon_selection_uses_full_arrays():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    ids = np.array([100, 101, 102, 103])
    selected = source_ids_inside_polygon(
        x,
        y,
        ids,
        [(-0.5, -0.5), (2.5, -0.5), (2.5, 2.5), (-0.5, 2.5)],
    )
    assert selected.tolist() == [100, 101, 102]


def test_invert_uses_current_filtered_universe_but_keeps_identity():
    manager = SourceSelectionManager()
    manager.apply([1, 9], "replace")
    assert manager.invert([1, 2, 3])
    assert manager.active_group.source_ids == frozenset({2, 3})


def test_linked_limit_mask_uses_all_named_dimensions():
    from lmas.source_selection import source_mask_in_linked_limits

    ids = np.array([10, 11, 12, 13])
    time = np.array([0.0, 1.0, 2.0, 3.0])
    altitude = np.array([5.0, 5.5, 6.0, 6.5])
    east = np.array([-1.0, 0.0, 1.0, 2.0])
    north = np.array([0.0, 0.5, 1.0, 1.5])
    names = (("time", "altitude"), ("north", "altitude"), ("east", "altitude"), ("east", "north"))
    pairs = ((time, altitude), (north, altitude), (east, altitude), (east, north))
    mask = source_mask_in_linked_limits(
        ids,
        names,
        pairs,
        {"time": (0.5, 2.5), "altitude": (5.0, 6.1), "east": (-0.5, 1.5), "north": (0.0, 1.1)},
    )
    assert ids[mask].tolist() == [11, 12]


def test_recolor_overlay_yields_to_non_charge_color_modes():
    from lmas.source_selection import effective_group_display_style

    assert effective_group_display_style("recolor", color_by="charge") == "outline"
    for color_by in ("time", "altitude", "power", "stations", "chi2", None):
        assert effective_group_display_style("recolor", color_by=color_by) == "outline"

    assert effective_group_display_style("halo", color_by="time") == "halo"
    assert effective_group_display_style("convex_hull", color_by="power") == "convex_hull"


def test_charge_overlays_are_opt_in_outside_charge_coloring():
    from lmas.source_selection import charge_group_overlay_visible

    for category in ("positive", "negative"):
        assert charge_group_overlay_visible(
            category,
            color_by="charge",
            show_with_other_color_modes=False,
        )
        assert not charge_group_overlay_visible(
            category,
            color_by="time",
            show_with_other_color_modes=False,
        )
        assert charge_group_overlay_visible(
            category,
            color_by="time",
            show_with_other_color_modes=True,
        )

    # Unassigned groups may be ordinary Source Selection groups and remain
    # visible independently of the Charge Analysis overlay preference.
    assert charge_group_overlay_visible(
        "unassigned",
        color_by="altitude",
        show_with_other_color_modes=False,
    )
