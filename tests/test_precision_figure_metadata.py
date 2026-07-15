from __future__ import annotations

from lmas.demo import synthetic_project
from lmas.model import PlotSpec
from lmas.plotting import create_lma_figure


def test_precision_metadata_is_complete_in_both_layouts() -> None:
    for layout in ("intfs", "xlma"):
        project = synthetic_project()
        plot = PlotSpec(layout=layout, show_histogram=True)
        figure = create_lma_figure(project, plot=plot)
        metadata = figure._lmas_metadata
        values = metadata["precision_source_values"]
        count = metadata["source_count"]
        assert count > 0
        assert len(metadata["axis_order"]) == 4
        assert len(metadata["coordinate_pairs"]) == 4
        assert len(metadata["coordinate_names"]) == 4
        assert metadata["theme"] == plot.theme
        assert metadata["selection_dataset_key"] == (id(project.dataset), project.event_count)
        for key in (
            "time", "time_num", "latitude", "longitude", "altitude_km",
            "east_km", "north_km", "source_id", "power", "chi2", "stations",
        ):
            assert len(values[key]) == count
