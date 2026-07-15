from lmas.source_selection import charge_region_label


def test_charge_region_label_defaults_to_leader_polarity():
    assert charge_region_label({}) == "Leader polarity"
    assert charge_region_label(None) == "Leader polarity"


def test_charge_region_label_supports_broader_region_wording():
    state = {"charge_region_label": "charge_region_polarity"}
    assert charge_region_label(state) == "Charge region polarity"


def test_unknown_charge_region_label_migrates_to_default():
    assert charge_region_label({"charge_region_label": "old-value"}) == "Leader polarity"
