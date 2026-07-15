from pathlib import Path

from lmas.model import PlotSpec


def test_source_power_cannot_use_second_log_normalization():
    plot = PlotSpec(color_by="power", log_color_scale=True).validated()
    assert plot.color_by == "power"
    assert plot.log_color_scale is False


def test_charge_dialog_uses_neutral_assignment_wording():
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "lmas"
        / "gui"
        / "selection_window.py"
    ).read_text(encoding="utf-8")
    assert 'QGroupBox("Polarity Assignment")' in source
    assert 'addRow("Polarity", self.charge_category)' in source
    assert 'addRow("Active group", self.charge_active_label)' in source
    assert "Active charge region polarity" not in source


def test_source_power_log_checkbox_is_disabled_by_contract():
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "lmas"
        / "gui"
        / "controls.py"
    ).read_text(encoding="utf-8")
    assert 'is_power = mode == "power"' in source
    assert 'dBW is already logarithmic' in source
