from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lmas.help_docs import NETWORK_OVERLAYS, read_help_document
from lmas.demo import synthetic_project
from lmas.io.project import load_project, save_project
from lmas.model import PlotSpec
from lmas.overlays.network import (
    NetworkCSVOptions,
    NetworkOverlayManager,
    NetworkOverlayRenderer,
    network_dataframe,
    network_dataset,
    read_network_csv,
    write_generic_network_example,
)


def test_entln_oriented_split_date_clock_and_units(tmp_path: Path):
    path = tmp_path / "lxarchive_pulse_reference.csv"
    pd.DataFrame(
        {
            "date": ["2019-04-30", "2019-04-30"],
            "time": ["14:49:14.100", "14:49:14.200"],
            "lat": [35.5, 35.51],
            "lon": [-97.5, -97.49],
            "type": ["-CG", "IC"],
            "peakCurrent": [-25000.0, np.nan],
            "numberSensors": [18, 12],
            "majorAxis": [500.0, 700.0],
            "minorAxis": [200.0, 300.0],
            "bearing": [0.0, 90.0],
            "icHeight": [np.nan, 8500.0],
        }
    ).to_csv(path, index=False)

    observation = read_network_csv(
        path,
        options=NetworkCSVOptions(
            peak_current_unit="A", ellipse_unit="m", altitude_unit="m"
        ),
    )
    assert observation.identity.provider_id == "entln"
    assert observation.events.time_ns.astype("datetime64[ns]").astype(str).tolist() == [
        "2019-04-30T14:49:14.100000000",
        "2019-04-30T14:49:14.200000000",
    ]
    assert observation.events.event_type.tolist() == ["CG", "IC"]
    assert observation.events.polarity.tolist() == [-1, 0]
    assert observation.events.peak_current_ka[0] == -25.0
    assert np.allclose(observation.events.ellipse_major_km, [0.5, 0.7])
    # Bearing is clockwise from north; normalized render angle is CCW from east.
    assert np.allclose(observation.events.ellipse_angle_deg, [90.0, 0.0])
    assert observation.events.sensor_count.tolist() == [18, 12]

    explicit_generic = read_network_csv(path, options=NetworkCSVOptions(provider="generic"))
    assert explicit_generic.identity.provider_id == "generic"


def test_entln_numeric_type_codes_are_provider_scoped_and_preserved(tmp_path: Path):
    path = tmp_path / "lxarchive_pulse_numeric.csv"
    pd.DataFrame(
        {
            "timestamp": ["2019-04-30T14:49:14.100Z", "2019-04-30T14:49:14.200Z"],
            "lat": [35.5, 35.51],
            "lon": [-97.5, -97.49],
            "type": [0, 1],
            "peakCurrent": [-25.0, 8.0],
        }
    ).to_csv(path, index=False)

    observation = read_network_csv(path)
    assert observation.identity.provider_id == "entln"
    assert observation.events.event_type.tolist() == ["CG", "IC"]
    assert observation.events.original_event_type.tolist() == ["0", "1"]
    assert observation.identity.schema["event_type_decoding"] == "ENTLN numeric: 0 = CG, 1 = IC"
    frame = network_dataframe(observation)
    assert frame["original_event_type"].tolist() == ["0", "1"]

    generic = read_network_csv(path, options=NetworkCSVOptions(provider="generic"))
    assert generic.events.event_type.tolist() == ["0", "1"]


def test_generic_example_roundtrip_selection_and_exports(tmp_path: Path):
    path = write_generic_network_example(tmp_path / "network.csv")
    observation = read_network_csv(path)
    assert len(observation.events) == 7
    assert set(observation.events.event_type) == {"CG", "IC"}
    assert np.count_nonzero(np.isfinite(observation.events.ellipse_major_km)) == 7

    center = int(np.datetime64("2019-04-30T14:49:14.265", "ns").astype(np.int64))
    selected = observation.select(
        time_range_ns=(center - 100_000_000, center + 100_000_000),
        minimum_sensor_count=13,
    ).event_indices
    assert selected.size == 4

    frame = network_dataframe(observation, selected)
    dataset = network_dataset(observation, selected)
    assert len(frame) == 4
    assert dataset.sizes["number_of_network_events"] == 4
    assert dataset.attrs["provider"] == "generic"


def _network_figure():
    fig = plt.figure(figsize=(8, 5))
    time_axis = fig.add_axes([0.10, 0.72, 0.82, 0.20])
    plan = fig.add_axes([0.20, 0.12, 0.60, 0.48])
    start = np.datetime64("2019-04-30T14:49:13.9", "ns").astype("datetime64[us]").astype(object)
    end = np.datetime64("2019-04-30T14:49:14.7", "ns").astype("datetime64[us]").astype(object)
    time_axis.set_xlim(mdates.date2num(start), mdates.date2num(end))
    time_axis.set_ylim(0.0, 20.0)
    plan.set_xlim(-97.70, -97.40)
    plan.set_ylim(35.38, 35.60)
    plot = PlotSpec(show_legend=True).validated()
    fig._lmas_metadata = {
        "axes": {"time_altitude": time_axis, "plan": plan},
        "axis_order": (time_axis, plan),
        "coordinate_names": (("time", "altitude"), ("longitude", "latitude")),
        "plot_spec": plot,
    }
    fig._lmas_theme = {"axes": "white"}
    return fig, plot


def test_renderer_uses_retained_categories_ellipses_and_time_rail(tmp_path: Path):
    path = write_generic_network_example(tmp_path / "network.csv")
    manager = NetworkOverlayManager()
    record = manager.add_csv_paths(path)
    record.style = replace(record.style, show_uncertainty=True).validated()
    fig, plot = _network_figure()
    project = SimpleNamespace(reference_longitude=-97.5, reference_latitude=35.5, plot=plot)
    renderer = NetworkOverlayRenderer(manager)
    renderer.bind(fig, project)
    first_artist_ids = tuple(id(artist) for artist in renderer.artists)
    assert renderer.summaries[0].visible_events == 7
    assert renderer.summaries[0].visible_ellipses == 7
    assert getattr(fig, "_lmas_metadata")["legend"] is not None
    time_artists = [artist for artist in renderer.artists if artist in renderer._bundles[record.key].time_categories.values()]
    visible_time = [artist for artist in time_artists if artist.get_visible()]
    assert visible_time
    assert all(np.min(artist.get_offsets()[:, 1]) > fig._lmas_metadata["axes"]["time_altitude"].get_position().y1 for artist in visible_time)
    assert all(not artist.get_clip_on() for artist in visible_time)

    record.style = replace(record.style, show_ic=False).validated()
    renderer.refresh()
    assert renderer.summaries[0].visible_events == 3
    assert renderer.summaries[0].visible_ellipses == 3
    # Existing artists are updated rather than rebuilt on each linked refresh.
    assert tuple(id(artist) for artist in renderer.artists) == first_artist_ids
    plt.close(fig)


def test_manager_project_state_and_missing_file_tolerance(tmp_path: Path):
    data = write_generic_network_example(tmp_path / "data" / "network.csv")
    manager = NetworkOverlayManager()
    record = manager.add_csv_paths(data, options=NetworkCSVOptions(provider="generic"))
    record.style = replace(record.style, marker_size=55.0, show_uncertainty=True).validated()
    state = manager.project_state()

    restored = NetworkOverlayManager()
    restored.restore_project_state(state)
    assert len(restored.records) == 1
    assert restored.records[0].style.marker_size == 55.0
    assert restored.records[0].style.show_uncertainty is True

    data.unlink()
    missing = NetworkOverlayManager()
    missing.restore_project_state(state)
    assert not missing.records
    assert missing.last_restore_errors



def test_network_project_paths_are_portable(tmp_path: Path):
    data = write_generic_network_example(tmp_path / "workspace" / "data" / "network.csv")
    manager = NetworkOverlayManager()
    manager.add_csv_paths(data)
    project = synthetic_project(count=40)
    project.network_overlay_state = manager.project_state()
    project_path = save_project(project, tmp_path / "workspace" / "projects" / "case.lmas-project.yaml")
    text = project_path.read_text(encoding="utf-8")
    assert "../data/network.csv" in text.replace("\\", "/")
    loaded = load_project(project_path)
    restored = NetworkOverlayManager()
    restored.restore_project_state(loaded.network_overlay_state, project_directory=project_path.parent)
    assert len(restored.records) == 1
    assert len(restored.records[0].observation.events) == 7

def test_network_help_document_is_packaged():
    text = read_help_document(NETWORK_OVERLAYS)
    assert "Network Overlays" in text
    assert "No artificial altitude" in text
