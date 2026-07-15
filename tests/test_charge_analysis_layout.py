from pathlib import Path


def test_charge_group_list_has_protected_responsive_space():
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "lmas"
        / "gui"
        / "selection_window.py"
    ).read_text(encoding="utf-8")
    assert "self.charge_group_list.setMinimumHeight(180)" in source
    assert "QSplitter(Qt.Orientation.Vertical)" in source
    assert "self._charge_splitter.setChildrenCollapsible(False)" in source
    assert "details_scroll = QScrollArea()" in source
    assert "details_scroll.setWidgetResizable(True)" in source
