import matplotlib.dates as mdates
from datetime import datetime, timezone

from lmas.figure_export import default_custom_title
from lmas.source_selection import SourceSelectionGroup


def test_default_title_floors_view_start_second():
    low = mdates.date2num(datetime(2019, 4, 30, 14, 49, 14, 142212, tzinfo=timezone.utc))
    high = mdates.date2num(datetime(2019, 4, 30, 14, 49, 15, 79513, tzinfo=timezone.utc))
    title = default_custom_title(
        "Old label — 2,483 visible of 2,982 sources in view (χ² < 1.00)",
        (low, high),
    )
    assert title == "2019-04-30 14:49:14 UTC — 2,483 visible of 2,982 sources in view (χ² < 1.00)"
