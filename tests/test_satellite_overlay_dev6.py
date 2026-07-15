from __future__ import annotations

from dataclasses import replace

from lmas.overlays.satellite import GLMOverlayStyle, SatelliteOverlayManager


def test_dev6_group_marker_defaults_and_color_roundtrip():
    style = GLMOverlayStyle().validated()
    assert style.group_marker_size == 15.0
    assert style.group_marker_color == "auto"
    custom = GLMOverlayStyle.from_dict({"group_marker_color": "#12ab34"})
    assert custom.group_marker_color == "#12ab34"
    invalid = GLMOverlayStyle.from_dict({"group_marker_color": "not-a-color"})
    assert invalid.group_marker_color == "auto"


def test_dev6_reader_backend_persists_in_project_state():
    manager = SatelliteOverlayManager()
    manager.glm_backend = "glmtools"
    state = manager.project_state()
    assert state["glm_backend"] == "glmtools"
    restored = SatelliteOverlayManager()
    restored.restore_project_state(state)
    assert restored.glm_backend == "glmtools"


def test_dev6_global_layer_state_updates_every_loaded_record(monkeypatch):
    manager = SatelliteOverlayManager()
    class Record:
        def __init__(self):
            self.style = GLMOverlayStyle()
    manager._records = {"a": Record(), "b": Record()}
    manager.set_global_layer_state(
        show_event_footprints=False,
        show_group_centroids=False,
        show_group_time_rail=False,
    )
    for record in manager._records.values():
        assert not record.style.show_event_footprints
        assert not record.style.show_group_centroids
        assert not record.style.show_group_time_rail


def test_dev6_migrates_legacy_default_marker_size_but_preserves_v2_explicit_45():
    assert GLMOverlayStyle.from_dict({"group_marker_size": 45.0}).group_marker_size == 15.0
    payload = GLMOverlayStyle(group_marker_size=45.0).to_dict()
    assert payload["format"] == "lmas-glm-overlay-style-v2"
    assert GLMOverlayStyle.from_dict(payload).group_marker_size == 45.0
