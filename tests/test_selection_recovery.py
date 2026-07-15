from __future__ import annotations

from pathlib import Path

from lmas.source_selection import SourceSelectionManager


GUI_ROOT = Path(__file__).resolve().parents[1] / "src" / "lmas" / "gui"


def test_shift_override_subtracts_from_active_group() -> None:
    source = (GUI_ROOT / "selection_window.py").read_text(encoding="utf-8")
    shift = 'if modifiers & Qt.KeyboardModifier.ShiftModifier:\n            return "subtract"'
    assert shift in source
    assert "Shift temporarily removes sources from the active group" in source

    manager = SourceSelectionManager()
    manager.apply([1, 2, 3], "replace")
    other = manager.new_group("Other", source_ids=[10, 11])
    assert manager.active_name == other.name
    manager.set_active("Selection 1")
    assert manager.apply([2], "subtract")
    assert manager.active_group is not None
    assert manager.active_group.source_ids == frozenset({1, 3})
    assert next(group for group in manager.groups if group.name == "Other").source_ids == frozenset({10, 11})


def test_explicit_group_and_tool_actions_resume_selection() -> None:
    source = (GUI_ROOT / "selection_window.py").read_text(encoding="utf-8")
    assert "def _resume_selection_interaction" in source
    assert "def _selection_canvas_press" in source
    assert 'self._event_connections["recovery_press"]' in source
    # Recovery is connected before selectors so the first press can participate
    # in the same lasso gesture rather than requiring a second attempt.
    assert source.index('self._event_connections["recovery_press"]') < source.index(
        "selector = LassoSelector("
    )
    assert source.count("self._resume_selection_interaction()") >= 5
    assert "Selection paused. Choose Lasso, Point edit, or a source group to resume." in source


