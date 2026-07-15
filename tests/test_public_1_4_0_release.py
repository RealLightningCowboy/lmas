from pathlib import Path
import matplotlib.dates as mdates
from datetime import datetime, timezone

from lmas.figure_export import default_custom_title
from lmas.source_selection import SourceSelectionGroup


def test_private_leader_speed_modules_are_not_packaged():
    package = Path(__file__).parents[1] / "src" / "lmas"
    assert not (package / "leader_speed.py").exists()
    assert not (package / "leader_steps.py").exists()
    assert not (package / "gui" / "leader_speed_window.py").exists()
    assert not (package / "docs" / "CURVED_LEADER_FRONT_METHOD.md").exists()


def test_legacy_leader_domain_migrates_to_custom_selection():
    group = SourceSelectionGroup.from_dict({
        "name": "Imported group",
        "domain": "leader",
        "subtype": "leader_track",
        "source_ids": [1, 2, 3],
    })
    assert group.domain == "custom"
    assert group.source_ids == frozenset({1, 2, 3})


def test_default_title_floors_view_start_second():
    low = mdates.date2num(datetime(2019, 4, 30, 14, 49, 14, 142212, tzinfo=timezone.utc))
    high = mdates.date2num(datetime(2019, 4, 30, 14, 49, 15, 79513, tzinfo=timezone.utc))
    title = default_custom_title(
        "Old label — 2,483 visible of 2,982 sources in view (χ² < 1.00)",
        (low, high),
    )
    assert title == "2019-04-30 14:49:14 UTC — 2,483 visible of 2,982 sources in view (χ² < 1.00)"
