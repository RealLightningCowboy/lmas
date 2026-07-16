from lmas import __version__
from lmas.model import PlotSpec


def test_release_version():
    assert __version__ == "1.6.1"


def test_public_plot_defaults():
    plot = PlotSpec().validated()
    assert plot.coordinate_system == "local"
    assert plot.preview_point_limit == 12000
    assert plot.three_d_trail_ms == 30.0
    assert plot.three_d_afterimage_ms == 30.0
