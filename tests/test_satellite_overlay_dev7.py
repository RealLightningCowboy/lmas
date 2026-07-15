from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from lmas.overlays.satellite import GLMOverlayStyle, SatelliteOverlayManager
from lmas.overlays.satellite.glm.model import (
    GLMDatasetIdentity,
    GLMProjectionMetadata,
)
from lmas.overlays.satellite.rendering import SatelliteOverlayRenderer, _ArtistBundle


def _identity(platform: str = "G16", role: str = "east") -> GLMDatasetIdentity:
    return GLMDatasetIdentity(
        instrument_family="GLM",
        platform_id=platform,
        operational_role=role,
        operational_role_source="test",
        product_level="L2_LCFA",
        observation_start_ns=0,
        observation_end_ns=1,
        projection=GLMProjectionMetadata(),
        source_files=(),
    )


def test_dev7_spacecraft_position_and_legend_labels_are_explicit():
    identity = _identity("G16", "east")
    assert identity.spacecraft_name == "GOES-16"
    assert identity.position_name == "East"
    assert identity.legend_prefix == "GOES-16 (East)"
    west = _identity("GOES-17", "west")
    assert west.spacecraft_name == "GOES-17"
    assert west.legend_prefix == "GOES-17 (West)"


def test_dev7_footprint_render_padding_defaults_to_one_third_and_roundtrips():
    style = GLMOverlayStyle().validated()
    assert np.isclose(style.footprint_render_padding_fraction, 1.0 / 3.0)
    restored = GLMOverlayStyle.from_dict(style.to_dict())
    assert np.isclose(restored.footprint_render_padding_fraction, 1.0 / 3.0)
    assert GLMOverlayStyle.from_dict({"footprint_render_padding_fraction": 9}).footprint_render_padding_fraction == 1.0
    assert GLMOverlayStyle.from_dict({"footprint_render_padding_fraction": -2}).footprint_render_padding_fraction == 0.0


def test_dev7_padded_event_center_selection_extends_beyond_axes():
    figure = Figure()
    axis = figure.add_subplot(111)
    axis.set_xlim(0.0, 10.0)
    axis.set_ylim(0.0, 10.0)
    renderer = SatelliteOverlayRenderer(SatelliteOverlayManager())
    lon = np.array([-3.0, -3.5, 5.0, 13.0, 13.5])
    lat = np.full(lon.shape, 5.0)
    _, _, exact = renderer._point_coordinates(
        lon, lat, x_name="longitude", y_name="latitude", plan=axis
    )
    _, _, padded = renderer._point_coordinates(
        lon,
        lat,
        x_name="longitude",
        y_name="latitude",
        plan=axis,
        padding_fraction=1.0 / 3.0,
    )
    assert exact.tolist() == [False, False, True, False, False]
    assert padded.tolist() == [True, False, True, True, False]


def test_dev7_highest_energy_star_uses_spacecraft_group_color():
    figure = Figure()
    axis = figure.add_subplot(111)
    renderer = SatelliteOverlayRenderer(SatelliteOverlayManager())
    bundle = _ArtistBundle(axis)
    renderer._update_maximum(
        bundle,
        axis,
        np.array([[1.0, 2.0]]),
        style=GLMOverlayStyle(show_maximum_group=True),
        group_color="crimson",
    )
    face = bundle.maximum.get_facecolors()[0]
    # Matplotlib's crimson is approximately RGBA (0.86, 0.08, 0.24, 1).
    assert face[0] > 0.8 and face[1] < 0.2 and face[2] < 0.4


def test_what_lmas_can_do_uses_markdown_bullets():
    text = Path("WHAT_LMAS_CAN_DO.md").read_text(encoding="utf-8")
    assert "- Open" in text
    assert "- Render" in text
    assert "- Use Precision Mode" in text
