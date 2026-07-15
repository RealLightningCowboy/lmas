from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "src" / "lmas" / "gui"


def test_plain_lasso_defaults_to_add_and_replace_remains_available():
    source = (ROOT / "selection_window.py").read_text(encoding="utf-8")
    add_pos = source.index('self.operation.addItem("Add", "add")')
    replace_pos = source.index('self.operation.addItem("Replace", "replace")')
    assert add_pos < replace_pos
    assert 'return str(self.operation.currentData() or "add")' in source
    assert "Default tool action" in source
    assert "Point edit" in source


def test_charge_category_change_rearms_lasso_after_delayed_redraw():
    source = (ROOT / "selection_window.py").read_text(encoding="utf-8")
    assert "self._selector_rearm_timer" in source
    assert "self._recover_selection_interaction" in source
    assert "QTimer.singleShot(0, self._rearm_current_selectors)" in source
    assert "self._selector_rearm_timer.start()" in source


def test_charge_geometry_accounts_for_window_frame_and_margin():
    source = (ROOT / "main_window.py").read_text(encoding="utf-8")
    assert "frame_extra_height" in source
    assert "frame_top" in source
    assert "work_area_margin=18" in source
    assert "fitted.y + frame_top" in source
    assert "self._clamp_charge_window" in source
    assert "QTimer.singleShot(120" in source


def test_charge_overlay_preference_is_visible_and_defaults_off():
    source = (ROOT / "selection_window.py").read_text(encoding="utf-8")
    assert "Show charge overlays with other Color by modes" in source
    assert "self._show_charge_overlays_with_other_color_modes = False" in source
    assert 'payload.get("show_charge_overlays_with_other_color_modes", False)' in source
