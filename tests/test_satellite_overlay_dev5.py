from __future__ import annotations

import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator, LogLocator
from matplotlib.dates import ConciseDateFormatter

from lmas.overlays.satellite import GLMOverlayStyle, configure_group_energy_time_axis


def test_dev5_defaults_use_separate_layer_order_and_time_rail():
    style = GLMOverlayStyle().validated()
    assert style.show_colorbar
    assert style.show_group_time_rail
    assert not style.show_time_rail_labels
    assert style.footprint_zorder < style.group_zorder < 1.0
    assert not style.show_maximum_group


def test_dev4_single_zorder_migrates_without_changing_old_visual_order():
    style = GLMOverlayStyle.from_dict({"zorder": 0.75})
    assert style.footprint_zorder == 0.75
    assert style.group_zorder == 0.85
    payload = style.to_dict()
    assert "zorder" not in payload
    assert payload["footprint_zorder"] == 0.75
    assert payload["group_zorder"] == 0.85


def test_group_energy_axis_uses_concise_time_and_minor_ticks():
    figure = Figure()
    axis = figure.add_subplot(111)
    axis.set_yscale("log")
    start = mdates.datestr2num("2019-04-30 14:49:13.265")
    axis.set_xlim(start, start + 2.0 / 86400.0)
    configure_group_energy_time_axis(axis)
    assert isinstance(axis.xaxis.get_major_formatter(), ConciseDateFormatter)
    assert isinstance(axis.xaxis.get_minor_locator(), AutoMinorLocator)
    assert isinstance(axis.yaxis.get_minor_locator(), LogLocator)
