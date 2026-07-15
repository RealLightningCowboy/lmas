from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parents[1]
PACKAGE = ROOT / "src" / "lmas"


def test_source_distributions_exposes_required_sets_and_theme_application() -> None:
    text = (PACKAGE / "gui" / "source_distributions_window.py").read_text(encoding="utf-8")
    assert 'self.source_set.addItem("Full dataset", "full")' in text
    assert 'self.source_set.addItem("Selected subset (current view)", "subset")' in text
    assert 'self.source_set.addItem("Active source group", "active_group")' in text
    assert "apply_figure_theme" in text
    assert "theme_values" in text


def test_active_tab_scopes_transient_group_overlays() -> None:
    text = (PACKAGE / "gui" / "selection_window.py").read_text(encoding="utf-8")
    assert "if group.domain != self._current_domain():" in text
    assert 'state["active_domain"] = self._current_domain()' in text
    assert "self._remove_overlays()" in text


def test_public_release_omits_private_leader_workspace() -> None:
    main = (PACKAGE / "gui" / "main_window.py").read_text(encoding="utf-8")
    assert "Leader Speed Analysis" not in main
    assert "Leader Analysis" not in (PACKAGE / "gui" / "selection_window.py").read_text(encoding="utf-8")
