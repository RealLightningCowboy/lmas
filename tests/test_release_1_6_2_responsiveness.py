from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from lmas.model import FilterSpec, LMAProject, PlotSpec
from lmas.visualization.animation import animation_window_bounds_utc
from lmas.visualization.projection_animation import ProjectionAnimationScene
from lmas.visualization.snapshot import build_visualization_snapshot


class _Scatter:
    def __init__(self) -> None:
        self.offsets = None
        self.facecolors = None

    def set_offsets(self, values) -> None:
        self.offsets = np.asarray(values)

    def set_facecolors(self, values) -> None:
        self.facecolors = np.asarray(values)


def _project() -> LMAProject:
    times = np.asarray(
        [
            np.datetime64("2026-07-06T21:18:35.010", "ns"),
            np.datetime64("2026-07-06T21:18:35.020", "ns"),
        ]
    )
    dataset = xr.Dataset(
        {
            "event_time": ("number_of_events", times),
            "event_latitude": ("number_of_events", np.asarray([33.0, 33.001])),
            "event_longitude": ("number_of_events", np.asarray([-107.0, -107.001])),
            "event_altitude": ("number_of_events", np.asarray([5000.0, 5100.0])),
            "event_power": ("number_of_events", np.asarray([1.0, 2.0])),
        }
    )
    dataset["event_altitude"].attrs["units"] = "m"
    return LMAProject(dataset=dataset, name="1.6.3 timing test")


def test_explicit_animation_window_precedes_first_source() -> None:
    times = np.asarray(
        [
            np.datetime64("2026-07-06T21:18:35.010", "ns"),
            np.datetime64("2026-07-06T21:18:35.020", "ns"),
        ]
    )
    start, end = animation_window_bounds_utc(
        times,
        start_time="2026-07-06T21:18:35.000",
        end_time="2026-07-06T21:18:35.030",
    )
    assert start == np.datetime64("2026-07-06T21:18:35.000", "ns")
    assert end == np.datetime64("2026-07-06T21:18:35.030", "ns")


def test_projection_scene_can_open_before_first_source() -> None:
    scatter = _Scatter()
    scene = ProjectionAnimationScene(
        figure=object(),
        scatters=[scatter],
        coordinate_pairs=[(np.asarray([1.0, 2.0]), np.asarray([3.0, 4.0]))],
        depth_keys=[None],
        colors=np.asarray([0.0, 1.0]),
        time_ms=np.asarray([10.0, 20.0]),
        base_rgba=np.asarray([[255, 0, 0, 255], [0, 0, 255, 255]], dtype=np.uint8),
        title_artist=None,
        base_title="test",
        chi2_suffix="",
        cmap=None,
        reverse_cmap=False,
        total_source_count=2,
        animation_start_ms=0.0,
        animation_end_ms=30.0,
        preserve_depth_order=False,
    )
    assert scene.first_time_ms == 0.0
    assert scene.first_source_time_ms == 10.0
    assert scene.update(
        0.0,
        display_mode="cumulative",
        trail_ms=30.0,
        afterimage_ms=30.0,
    ) == 0
    assert scatter.offsets.shape == (0, 2)
    assert scene.update(
        10.0,
        display_mode="cumulative",
        trail_ms=30.0,
        afterimage_ms=30.0,
    ) == 1
    assert scatter.offsets.shape == (1, 2)


def test_3d_snapshot_round_trip_preserves_window(tmp_path: Path) -> None:
    destination = tmp_path / "window.lmas3d.npz"
    snapshot = build_visualization_snapshot(
        _project(),
        filters=FilterSpec(
            start_time="2026-07-06T21:18:35.000",
            end_time="2026-07-06T21:18:35.030",
            minimum_stations=None,
            maximum_chi2=None,
        ),
        plot=PlotSpec(color_by="time"),
        output_path=destination,
    )
    assert snapshot.time_limits == (0.0, 30.0)
    assert np.allclose(snapshot.time_ms, np.asarray([10.0, 20.0]))
    assert snapshot.source_time_limits == (10.0, 20.0)


def test_shortcut_contracts_are_present() -> None:
    package = Path(__file__).parents[1] / "src" / "lmas"
    projection = (package / "gui" / "projection_animation_viewer.py").read_text(
        encoding="utf-8"
    )
    pyvista = (package / "visualization" / "pyvista_3d.py").read_text(
        encoding="utf-8"
    )
    assert 'QShortcut(QKeySequence("Space"), self)' in projection
    assert "Qt.ShortcutContext.WindowShortcut" in projection
    assert 'plotter.add_key_event("space", toggle_playback)' in pyvista
    assert 'plotter.add_key_event("p", toggle_playback)' in pyvista
