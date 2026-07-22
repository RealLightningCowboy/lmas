from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from lmas.visualization.projection_animation import (
    ProjectionAnimationScene,
    _install_animation_header,
)


ROOT = Path(__file__).resolve().parents[1]
PLOT_WIDGET = ROOT / "src" / "lmas" / "gui" / "plot_widget.py"


class _Scatter:
    def set_offsets(self, _values) -> None:
        pass

    def set_facecolors(self, _values) -> None:
        pass


class _Text:
    def __init__(self) -> None:
        self.text = ""

    def set_text(self, value: str) -> None:
        self.text = str(value)


def _scene(*, title_artist, time_artist) -> ProjectionAnimationScene:
    return ProjectionAnimationScene(
        figure=object(),
        scatters=[_Scatter()],
        coordinate_pairs=[(np.asarray([1.0]), np.asarray([2.0]))],
        depth_keys=[None],
        colors=np.asarray([0.0]),
        time_ms=np.asarray([10.0]),
        base_rgba=np.asarray([[255, 0, 0, 255]], dtype=np.uint8),
        title_artist=title_artist,
        time_artist=time_artist,
        base_title="test flash",
        chi2_suffix="",
        cmap=None,
        reverse_cmap=False,
        total_source_count=1,
        animation_start_ms=0.0,
        animation_end_ms=20.0,
        preserve_depth_order=False,
    )


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    return next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == name
    )


def _method(node: ast.ClassDef, name: str) -> ast.FunctionDef:
    return next(
        child for child in node.body
        if isinstance(child, ast.FunctionDef) and child.name == name
    )


def test_projection_time_is_not_a_second_title_line() -> None:
    title = _Text()
    scene = _scene(title_artist=title, time_artist=None)
    scene.update(
        10.0,
        display_mode="cumulative",
        trail_ms=30.0,
        afterimage_ms=30.0,
    )
    assert "\n" not in title.text
    assert "Source time" not in title.text


def test_saved_projection_time_has_its_own_header_artist() -> None:
    title = _Text()
    source_time = _Text()
    scene = _scene(title_artist=title, time_artist=source_time)
    scene.update(
        10.0,
        display_mode="cumulative",
        trail_ms=30.0,
        afterimage_ms=30.0,
    )
    assert "\n" not in title.text
    assert source_time.text == "Source time: 10.000 ms"


def test_saved_projection_header_clears_top_axes() -> None:
    figure = Figure(figsize=(16.0, 9.0), dpi=100)
    axis = figure.add_axes([0.065, 0.690, 0.850, 0.2325])
    title = figure.suptitle("", y=0.975, fontsize=15)
    source_time = _install_animation_header(
        figure, {"axes": {"time_altitude": axis}}, title
    )
    title.set_text("Test flash — 12,345 visible of 50,000 sources in view")
    source_time.set_text("Source time: 123.456 ms")

    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    renderer = canvas.get_renderer()
    assert not source_time.get_window_extent(renderer).overlaps(
        axis.get_window_extent(renderer)
    )
    assert not title.get_window_extent(renderer).overlaps(
        source_time.get_window_extent(renderer)
    )


def test_replacement_figure_replays_canvas_resize_path() -> None:
    source = PLOT_WIDGET.read_text(encoding="utf-8")
    tree = ast.parse(source)

    canvas_method = _method(
        _class(tree, "LMASFigureCanvas"), "sync_figure_size_to_widget"
    )
    canvas_segment = ast.get_source_segment(source, canvas_method)
    assert canvas_segment is not None
    assert "QResizeEvent(size, size)" in canvas_segment
    assert "self.resizeEvent" in canvas_segment

    set_figure = _method(_class(tree, "FigureHost"), "set_figure")
    host_segment = ast.get_source_segment(source, set_figure)
    assert host_segment is not None
    aspect_position = host_segment.index(
        "self._canvas_holder.set_aspect_ratio(aspect_ratio)"
    )
    sync_position = host_segment.index(
        "self._canvas.sync_figure_size_to_widget()"
    )
    controller_position = host_segment.index(
        "self._linked_view = LinkedViewController("
    )
    assert aspect_position < sync_position < controller_position
