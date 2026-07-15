from lmas.gui.window_geometry import (
    Rect,
    clamp_frame_to_work_area,
    full_height_tool_geometry,
)


def test_tool_window_uses_right_side_and_leaves_work_area_margin():
    geometry = full_height_tool_geometry(
        Rect(100, 50, 1200, 900),
        Rect(200, 200, 610, 700),
        Rect(0, 0, 2200, 1040),
    )
    assert geometry.y == 50
    assert geometry.height == 900
    assert geometry.x == 1308
    assert geometry.width == 610
    assert geometry.y >= 12
    assert geometry.y + geometry.height <= 1028


def test_tool_window_clamps_inside_inset_work_area():
    geometry = full_height_tool_geometry(
        Rect(0, -40, 1500, 1200),
        Rect(1600, 100, 900, 800),
        Rect(0, 0, 1920, 1040),
    )
    assert geometry.y == 12
    assert geometry.height == 1016
    assert geometry.width == 900
    assert 12 <= geometry.x <= 1008
    assert geometry.y + geometry.height <= 1028


def test_realized_frame_clamp_keeps_title_bar_and_bottom_visible():
    geometry = clamp_frame_to_work_area(
        Rect(1300, -14, 620, 1084),
        Rect(0, 0, 1920, 1040),
        margin=18,
    )
    assert geometry.y == 18
    assert geometry.height == 1004
    assert geometry.y + geometry.height == 1022
    assert geometry.x + geometry.width <= 1902
