from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import matplotlib.dates as mdates
from matplotlib.figure import Figure
import numpy as np

from lmas.overlays.satellite import (
    GLMOverlayStyle,
    SatelliteOverlayManager,
    SatelliteOverlayRenderer,
    group_glm_paths_by_platform,
)
from lmas.overlays.satellite.colormaps import SATELLITE_COLORMAP_NAMES, satellite_colormap


def test_satellite_colormap_catalog_requires_no_seaborn_runtime():
    assert {"mako", "crest", "inferno", "magma", "plasma", "viridis", "cividis", "turbo"} <= set(SATELLITE_COLORMAP_NAMES)
    for name in SATELLITE_COLORMAP_NAMES:
        assert satellite_colormap(name).N >= 256


def test_glm_path_grouping_keeps_spacecraft_independent(tmp_path):
    values = [
        tmp_path / "OR_GLM-L2-LCFA_G16_a.nc",
        tmp_path / "OR_GLM-L2-LCFA_G17_b.nc",
        tmp_path / "OR_GLM-L2-LCFA_G16_c.nc",
    ]
    grouped = group_glm_paths_by_platform(values)
    assert tuple(grouped) == ("G16", "G17")
    assert len(grouped["G16"]) == 2
    assert len(grouped["G17"]) == 1


def test_overlay_style_roundtrip_preserves_research_defaults():
    style = GLMOverlayStyle.from_dict(GLMOverlayStyle().to_dict())
    assert style.colormap == "mako"
    assert style.show_group_centroids
    assert style.show_event_footprints
    assert style.group_marker_size == 15.0
    assert style.maximum_group_color == "springgreen"
    assert not style.show_maximum_group
    assert style.zorder < 1.0
    assert style.maximum_interactive_events == 1500


def test_dev3_fixed_zorder_migrates_behind_lma_sources():
    style = GLMOverlayStyle.from_dict({"zorder": 90.0, "show_maximum_group": True})
    assert style.zorder == 0.5
    # Explicit saved layer visibility remains authoritative.
    assert style.show_maximum_group
