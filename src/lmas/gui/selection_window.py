"""Non-modal linked Source Selection and Charge Analysis workspace."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.widgets import LassoSelector
from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..source_selection import (
    CHARGE_COLORS,
    CHARGE_REGION_LABELS,
    DEFAULT_CHARGE_REGION_LABEL,
    SourceSelectionManager,
    charge_group_overlay_visible,
    effective_group_display_style,
    projection_hull_geometry,
    refresh_charge_source_colors,
    source_ids_inside_polygon,
    source_mask_in_linked_limits,
)
from .icon import application_icon


_DISPLAY_STYLE_ITEMS = (
    ("Halo", "halo"),
    ("Recolor", "recolor"),
    ("Outline", "outline"),
    ("Convex Hull", "convex_hull"),
    ("Concave Hull", "concave_hull"),
    ("Clustered Hulls", "clustered_hulls"),
    ("Hidden", "hidden"),
)
_CHARGE_ITEMS = (
    ("Unassigned", "unassigned"),
    ("Positive", "positive"),
    ("Negative", "negative"),
)
_MEMBER_DISPLAY_ITEMS = (
    ("Passing current filters", "filtered"),
    ("All group members", "all"),
    ("Filtered-out members only", "filtered_out"),
)


class SourceSelectionWindow(QMainWindow):
    """Shared linked selection workspace with a dedicated Charge Analysis tab."""

    charge_default_requested = Signal()
    selection_state_changed = Signal()
    polarity_export_requested = Signal(str, str)
    polarity_import_requested = Signal()

    def __init__(self, figure_host, parent=None) -> None:
        # Major analysis workspaces are independent top-level windows.  Giving
        # them the main window as a Qt parent creates an owned-window relationship
        # on Windows: the workspace stays above LMAS and receives no independent
        # taskbar entry.  MainWindow already retains a strong Python reference.
        super().__init__(None, Qt.WindowType.Window)
        self._owner_window = parent
        self.setAttribute(Qt.WidgetAttribute.WA_QuitOnClose, False)
        self.figure_host = figure_host
        self.setWindowTitle("Source Selection — LMAS")
        self.setWindowIcon(application_icon())
        self.resize(720, 800)

        self.manager = SourceSelectionManager()
        self._figure = None
        self._metadata: dict[str, Any] = {}
        self._lassos: list[LassoSelector] = []
        self._event_connections: dict[str, int] = {}
        self._axis_limit_connections: list[tuple[Any, str, int]] = []
        self._overlay_artists: list[Any] = []
        self._shortcuts: list[QShortcut] = []
        self._updating_widgets = False
        self._selection_active = False
        self._dataset_key = None
        self._charge_default_requested_for_dataset = False
        self._hull_cache: dict[tuple[Any, ...], Any] = {}
        self._member_mask_cache: dict[tuple[Any, ...], np.ndarray] = {}
        self._charge_category_visibility = {
            "unassigned": True,
            "positive": True,
            "negative": True,
        }
        self._show_charge_overlays_with_other_color_modes = False
        self._active_by_domain: dict[str, str | None] = {
            "custom": None, "charge": None
        }

        self._overlay_timer = QTimer(self)
        self._overlay_timer.setSingleShot(True)
        self._overlay_timer.setInterval(35)
        self._overlay_timer.timeout.connect(self._draw_overlays_now)

        # Charge-category changes can trigger a delayed full figure redraw when
        # Charge coloring is active. Re-arm the selectors after that redraw so
        # the shared lasso remains immediately usable on the replacement canvas.
        self._selector_rearm_timer = QTimer(self)
        self._selector_rearm_timer.setSingleShot(True)
        self._selector_rearm_timer.setInterval(300)
        self._selector_rearm_timer.timeout.connect(self._recover_selection_interaction)

        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(7, 7, 7, 7)
        layout.setSpacing(6)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_selection_tab(), "Custom Selection")
        self.tabs.addTab(self._build_charge_tab(), "Charge Analysis")
        self.tabs.currentChanged.connect(self._tab_changed)
        layout.addWidget(self.tabs, 1)

        self.status = QLabel("Open LMA data to begin selection.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        hint = QLabel(
            "Window keys: L lasso · E point edit · Ctrl+Z undo · Delete clear · Esc hide"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("font-size: 9pt;")
        layout.addWidget(hint)

        self.setCentralWidget(central)
        self._install_shortcuts()
        self._refresh_group_widgets()
        self.bind_current_figure()

    def _build_selection_tab(self) -> QWidget:
        tab = QWidget()
        outer = QVBoxLayout(tab)
        outer.setContentsMargins(6, 8, 6, 6)
        outer.setSpacing(8)

        intro = QLabel(
            "<b>Custom Selection</b> creates stable source-ID groups from any "
            "linked projection. By default, lasso and point edits use sources "
            "passing the active filters and lying inside the complete linked view."
        )
        intro.setWordWrap(True)
        outer.addWidget(intro)

        mode_box = QGroupBox("Selection tool")
        mode_layout = QVBoxLayout(mode_box)
        mode_row = QHBoxLayout()
        self.lasso_mode = QRadioButton("Lasso")
        self.point_mode = QRadioButton("Point edit")
        self.lasso_mode.setChecked(True)
        self.lasso_mode.toggled.connect(self._tool_changed)
        self.point_mode.toggled.connect(self._tool_changed)
        mode_row.addWidget(self.lasso_mode)
        mode_row.addWidget(self.point_mode)
        mode_row.addStretch(1)
        mode_row.addWidget(QLabel("Default tool action"))
        self.operation = QComboBox()
        self.operation.addItem("Add", "add")
        self.operation.addItem("Replace", "replace")
        self.operation.addItem("Remove", "subtract")
        self.operation.addItem("Intersect", "intersect")
        self.operation.currentIndexChanged.connect(self._operation_changed)
        mode_row.addWidget(self.operation)
        mode_layout.addLayout(mode_row)
        scope_row = QHBoxLayout()
        scope_row.addWidget(QLabel("Selection scope"))
        self.selection_scope = QComboBox()
        self.selection_scope.addItem("Filtered sources", "filtered")
        self.selection_scope.addItem("All loaded sources", "all")
        self.selection_scope.setToolTip(
            "Filtered sources uses the active quality filters and complete linked-view limits. "
            "All loaded sources ignores those filters for selection only."
        )
        self.selection_scope.currentIndexChanged.connect(
            lambda _index: self._selection_scope_changed(self.selection_scope)
        )
        scope_row.addWidget(self.selection_scope)
        scope_row.addStretch(1)
        mode_layout.addLayout(scope_row)
        member_row = QHBoxLayout()
        member_row.addWidget(QLabel("Display members"))
        self.member_display = QComboBox()
        for label, value in _MEMBER_DISPLAY_ITEMS:
            self.member_display.addItem(label, value)
        self.member_display.setToolTip(
            "Passing current filters is the normal scientific view. All group members "
            "adds dim hollow markers for members outside the current filtered linked view. "
            "Filtered-out members only isolates sources rejected by the active quality filters."
        )
        self.member_display.currentIndexChanged.connect(
            lambda _index: self._member_display_changed(self.member_display)
        )
        member_row.addWidget(self.member_display)
        member_row.addStretch(1)
        mode_layout.addLayout(member_row)
        gesture = QLabel(
            "The selected default action applies to both lasso and point edit. "
            "Shift temporarily removes sources from the active group, Alt also removes, "
            "and Ctrl intersects. Point edit applies the same set operation to the nearest source."
        )
        gesture.setWordWrap(True)
        mode_layout.addWidget(gesture)
        outer.addWidget(mode_box)

        groups_box = QGroupBox("Custom source groups")
        groups_layout = QVBoxLayout(groups_box)
        self.group_list = QListWidget()
        self.group_list.currentTextChanged.connect(self._group_selected)
        self.group_list.itemDoubleClicked.connect(lambda _item: self.rename_group())
        groups_layout.addWidget(self.group_list)
        group_buttons = QHBoxLayout()
        for label, callback in (
            ("+ New Group", self.new_group),
            ("Rename", self.rename_group),
            ("Delete", self.delete_group),
            ("Color", self.choose_color),
        ):
            button = QPushButton(label)
            button.clicked.connect(callback)
            group_buttons.addWidget(button)
        groups_layout.addLayout(group_buttons)
        option_row = QHBoxLayout()
        self.visible = QCheckBox("Visible")
        self.locked = QCheckBox("Locked")
        self.visible.toggled.connect(self._visibility_changed)
        self.locked.toggled.connect(self._locked_changed)
        option_row.addWidget(self.visible)
        option_row.addWidget(self.locked)
        option_row.addStretch(1)
        option_row.addWidget(QLabel("Display"))
        self.display_style = QComboBox()
        for label, value in _DISPLAY_STYLE_ITEMS:
            self.display_style.addItem(label, value)
        self.display_style.currentIndexChanged.connect(self._display_style_changed)
        option_row.addWidget(self.display_style)
        groups_layout.addLayout(option_row)
        outer.addWidget(groups_box)

        summary_box = QGroupBox("Active selection")
        summary_form = QFormLayout(summary_box)
        self.count_label = QLabel("0")
        self.filtered_label = QLabel("0")
        self.available_label = QLabel("0")
        self.filtered_out_label = QLabel("0")
        self.time_label = QLabel("—")
        self.altitude_label = QLabel("—")
        self.bounds_label = QLabel("—")
        for widget in (
            self.count_label,
            self.filtered_label,
            self.available_label,
            self.filtered_out_label,
            self.time_label,
            self.altitude_label,
            self.bounds_label,
        ):
            widget.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            widget.setWordWrap(True)
        summary_form.addRow("Sources in group", self.count_label)
        summary_form.addRow("Passing current filters", self.filtered_label)
        summary_form.addRow("Visible in current view", self.available_label)
        summary_form.addRow("Filtered-out members", self.filtered_out_label)
        summary_form.addRow("Time range", self.time_label)
        summary_form.addRow("Altitude range", self.altitude_label)
        summary_form.addRow("Projection bounds", self.bounds_label)
        outer.addWidget(summary_box)

        action_row = QHBoxLayout()
        for label, callback in (
            ("Clear", self.clear_active),
            ("Invert selection scope", self.invert_active),
            ("Undo", self.undo_last_action),
        ):
            button = QPushButton(label)
            button.clicked.connect(callback)
            action_row.addWidget(button)
        outer.addLayout(action_row)
        outer.addStretch(1)
        return tab

    def _build_charge_tab(self) -> QWidget:
        tab = QWidget()
        outer = QVBoxLayout(tab)
        outer.setContentsMargins(6, 8, 6, 6)
        outer.setSpacing(8)

        intro = QLabel(
            "Use the lasso to select LMA source groups and assign Positive, "
            "Negative, or Unassigned polarity."
        )
        intro.setWordWrap(True)
        outer.addWidget(intro)

        tool_box = QGroupBox("Selection tool")
        tool_row = QHBoxLayout(tool_box)
        self.charge_lasso_mode = QRadioButton("Lasso")
        self.charge_point_mode = QRadioButton("Point edit")
        self.charge_lasso_mode.setChecked(True)
        self.charge_lasso_mode.toggled.connect(
            lambda checked: checked and self.lasso_mode.setChecked(True)
        )
        self.charge_point_mode.toggled.connect(
            lambda checked: checked and self.point_mode.setChecked(True)
        )
        tool_row.addWidget(self.charge_lasso_mode)
        tool_row.addWidget(self.charge_point_mode)
        tool_row.addStretch(1)
        tool_row.addWidget(QLabel("Default tool action"))
        self.charge_operation = QComboBox()
        for label, value in (("Add", "add"), ("Replace", "replace"), ("Remove", "subtract"), ("Intersect", "intersect")):
            self.charge_operation.addItem(label, value)
        self.charge_operation.currentIndexChanged.connect(self._charge_operation_changed)
        tool_row.addWidget(self.charge_operation)
        outer.addWidget(tool_box)

        group_box = QGroupBox("Charge-analysis groups")
        group_layout = QVBoxLayout(group_box)
        self.charge_group_list = QListWidget()
        self.charge_group_list.setMinimumHeight(180)
        self.charge_group_list.setUniformItemSizes(True)
        self.charge_group_list.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.charge_group_list.currentTextChanged.connect(self._charge_group_selected)
        self.charge_group_list.itemDoubleClicked.connect(lambda _item: self.rename_group())
        group_layout.addWidget(self.charge_group_list, 1)
        create_row = QHBoxLayout()
        for label, category in (
            ("+ Positive", "positive"),
            ("+ Negative", "negative"),
            ("+ Unassigned", "unassigned"),
        ):
            button = QPushButton(label)
            button.clicked.connect(
                lambda _checked=False, value=category: self.new_charge_group(value)
            )
            create_row.addWidget(button)
        group_layout.addLayout(create_row)
        edit_row = QHBoxLayout()
        for label, callback in (
            ("Rename", self.rename_group),
            ("Delete", self.delete_group),
            ("Color", self.choose_color),
            ("Reset color", self.reset_charge_category_color),
        ):
            button = QPushButton(label)
            button.clicked.connect(callback)
            edit_row.addWidget(button)
        group_layout.addLayout(edit_row)
        group_box.setMinimumHeight(245)
        group_box.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding
        )

        details = QWidget()
        details_layout = QVBoxLayout(details)
        details_layout.setContentsMargins(0, 0, 0, 0)
        details_layout.setSpacing(8)

        assign_box = QGroupBox("Polarity Assignment")
        assign_form = QFormLayout(assign_box)
        self.charge_category = QComboBox()
        for label, value in _CHARGE_ITEMS:
            self.charge_category.addItem(label, value)
        self.charge_category.currentIndexChanged.connect(self._charge_category_changed)
        assign_form.addRow("Polarity", self.charge_category)
        self.charge_region_label = QComboBox()
        for value, label in CHARGE_REGION_LABELS.items():
            self.charge_region_label.addItem(label, value)
        default_label_index = self.charge_region_label.findData(
            DEFAULT_CHARGE_REGION_LABEL
        )
        self.charge_region_label.setCurrentIndex(max(0, default_label_index))
        self.charge_region_label.setToolTip(
            "Controls the label on Charge-colored figures. Leader polarity is the "
            "default; choose Charge region polarity only when that interpretation "
            "is intended."
        )
        self.charge_region_label.currentIndexChanged.connect(
            self._charge_region_label_changed
        )
        assign_form.addRow("Figure label", self.charge_region_label)
        self.charge_style = QComboBox()
        for label, value in _DISPLAY_STYLE_ITEMS:
            self.charge_style.addItem(label, value)
        self.charge_style.currentIndexChanged.connect(self._charge_style_changed)
        assign_form.addRow("Display style", self.charge_style)
        self.charge_selection_scope = QComboBox()
        self.charge_selection_scope.addItem("Filtered sources", "filtered")
        self.charge_selection_scope.addItem("All loaded sources", "all")
        self.charge_selection_scope.currentIndexChanged.connect(
            lambda _index: self._selection_scope_changed(self.charge_selection_scope)
        )
        assign_form.addRow("Selection scope", self.charge_selection_scope)
        self.charge_member_display = QComboBox()
        for label, value in _MEMBER_DISPLAY_ITEMS:
            self.charge_member_display.addItem(label, value)
        self.charge_member_display.currentIndexChanged.connect(
            lambda _index: self._member_display_changed(self.charge_member_display)
        )
        assign_form.addRow("Show members", self.charge_member_display)
        details_layout.addWidget(assign_box)

        visibility_box = QGroupBox("Category visibility")
        visibility_row = QHBoxLayout(visibility_box)
        self.show_unassigned = QCheckBox("Unassigned")
        self.show_positive = QCheckBox("Positive")
        self.show_negative = QCheckBox("Negative")
        for category, widget in (
            ("unassigned", self.show_unassigned),
            ("positive", self.show_positive),
            ("negative", self.show_negative),
        ):
            widget.setChecked(True)
            widget.toggled.connect(
                lambda checked, value=category: self._category_visibility_changed(
                    value, checked
                )
            )
            visibility_row.addWidget(widget)
        visibility_row.addStretch(1)
        details_layout.addWidget(visibility_box)

        self.show_charge_overlays_other_modes = QCheckBox(
            "Show charge overlays with other Color by modes"
        )
        self.show_charge_overlays_other_modes.setChecked(False)
        self.show_charge_overlays_other_modes.setToolTip(
            "When off, Positive and Negative group halos/outlines are hidden while "
            "the main figure is colored by time, altitude, power, stations, or χ². "
            "Charge coloring itself is unchanged."
        )
        self.show_charge_overlays_other_modes.toggled.connect(
            self._charge_overlay_preference_changed
        )
        details_layout.addWidget(self.show_charge_overlays_other_modes)

        product_box = QGroupBox("Polarity products")
        product_layout = QVBoxLayout(product_box)
        product_form = QFormLayout()
        self.polarity_export_scope = QComboBox()
        for label, value in (
            ("All loaded sources", "all"),
            ("Passing saved filters/view", "filtered"),
            ("Assigned sources only", "assigned"),
            ("Active group only", "active_group"),
        ):
            self.polarity_export_scope.addItem(label, value)
        self.polarity_export_scope.setToolTip(
            "All loaded sources is the authoritative round-trip product. Scoped exports "
            "are useful analysis subsets and require explicit partial-import permission."
        )
        product_form.addRow("Export scope", self.polarity_export_scope)
        product_layout.addLayout(product_form)
        product_buttons = QHBoxLayout()
        export_csv = QPushButton("Export CSV")
        export_csv.clicked.connect(
            lambda: self.polarity_export_requested.emit(
                "csv", str(self.polarity_export_scope.currentData() or "all")
            )
        )
        export_netcdf = QPushButton("Export NetCDF")
        export_netcdf.clicked.connect(
            lambda: self.polarity_export_requested.emit(
                "netcdf", str(self.polarity_export_scope.currentData() or "all")
            )
        )
        import_netcdf = QPushButton("Import NetCDF")
        import_netcdf.clicked.connect(self.polarity_import_requested.emit)
        for button in (export_csv, export_netcdf, import_netcdf):
            product_buttons.addWidget(button)
        product_layout.addLayout(product_buttons)
        product_note = QLabel(
            "NetCDF is the complete LMAS polarity product and preserves the original "
            "LMA dataset, named groups, sparse membership, conflicts, and provenance."
        )
        product_note.setWordWrap(True)
        product_layout.addWidget(product_note)
        details_layout.addWidget(product_box)

        summary_box = QGroupBox("Assignment summary")
        summary_form = QFormLayout(summary_box)
        self.charge_active_label = QLabel("Unassigned")
        self.charge_counts_label = QLabel("—")
        self.charge_overlap_label = QLabel("—")
        self.charge_provenance_label = QLabel("—")
        for widget in (
            self.charge_active_label,
            self.charge_counts_label,
            self.charge_overlap_label,
            self.charge_provenance_label,
        ):
            widget.setWordWrap(True)
            widget.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        summary_form.addRow("Active group", self.charge_active_label)
        summary_form.addRow("Assigned sources", self.charge_counts_label)
        summary_form.addRow("Overlap check", self.charge_overlap_label)
        summary_form.addRow("Group provenance", self.charge_provenance_label)
        details_layout.addWidget(summary_box)

        explanation = QLabel(
            "Positive groups use red, Negative groups use blue, and Unassigned uses "
            "neutral gray by default. Custom group colors remain available from "
            "Source Selection."
        )
        explanation.setWordWrap(True)
        details_layout.addWidget(explanation)
        details_layout.addStretch(1)

        details_scroll = QScrollArea()
        details_scroll.setWidgetResizable(True)
        details_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        details_scroll.setMinimumHeight(250)
        details_scroll.setWidget(details)

        self._charge_splitter = QSplitter(Qt.Orientation.Vertical)
        self._charge_splitter.setChildrenCollapsible(False)
        self._charge_splitter.addWidget(group_box)
        self._charge_splitter.addWidget(details_scroll)
        self._charge_splitter.setStretchFactor(0, 1)
        self._charge_splitter.setStretchFactor(1, 2)
        self._charge_splitter.setSizes([270, 430])
        outer.addWidget(self._charge_splitter, 1)
        return tab


    def _shortcut(self, sequence: str, callback: Callable[[], None]) -> None:
        shortcut = QShortcut(QKeySequence(sequence), self)
        shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        shortcut.setAutoRepeat(False)
        shortcut.activated.connect(callback)
        self._shortcuts.append(shortcut)

    def _install_shortcuts(self) -> None:
        self._shortcut("L", lambda: self.lasso_mode.setChecked(True))
        self._shortcut("E", lambda: self.point_mode.setChecked(True))
        self._shortcut("Ctrl+Z", self.undo_last_action)
        self._shortcut("Delete", self.clear_active)
        self._shortcut("Escape", self.hide_selection_mode)

    def select_tab(self, name: str) -> None:
        key = str(name).casefold()
        index = 1 if key.startswith("charge") else 0
        self.tabs.setCurrentIndex(index)

    @staticmethod
    def _domain_index(domain: str) -> int:
        value = str(domain or "custom").strip().lower()
        return {"custom": 0, "charge": 1}.get(value, 0)

    def _current_domain(self) -> str:
        return ("custom", "charge")[max(0, min(1, self.tabs.currentIndex()))]

    def _activate_domain_group(self, domain: str) -> None:
        current = self.manager.active_group
        if current is not None and current.domain == domain:
            self._active_by_domain[domain] = current.name
            return
        requested = self._active_by_domain.get(domain)
        candidates = self.manager.groups_for_domain(domain)
        if not candidates:
            if domain == "charge":
                created = self.manager.new_group(
                    "Unassigned 1",
                    charge_category="unassigned",
                    color=CHARGE_COLORS["unassigned"],
                    display_style="recolor",
                    domain="charge",
                    subtype="polarity_group",
                    record_history=False,
                )
            else:
                created = self.manager.new_group(
                    "Selection 1", domain="custom", subtype="generic_selection",
                    record_history=False,
                )
            candidates = (created,)
        target = next((g for g in candidates if g.name == requested), candidates[0] if candidates else None)
        if target is not None:
            self.manager.set_active(target.name)
            self._active_by_domain[domain] = target.name

    def _tab_changed(self, _index: int) -> None:
        domain = self._current_domain()
        self._activate_domain_group(domain)
        if domain == "charge":
            self._request_charge_default_once()
        if self._selection_active:
            label = {"custom": "Custom Selection", "charge": "Charge Analysis"}[domain]
            self.status.setText(f"{label} active; lasso and point edits modify the active {domain} group.")
        self._refresh_group_widgets()
        self._refresh_base_charge_colors(draw=False)
        self._draw_overlays()
        self._refresh_charge_summary()

    def _request_charge_default_once(self) -> None:
        if self._charge_default_requested_for_dataset:
            return
        self._charge_default_requested_for_dataset = True
        self.charge_default_requested.emit()

    def _selection_scope_changed(self, source: QComboBox) -> None:
        if self._updating_widgets:
            return
        value = str(source.currentData() or "filtered")
        self._updating_widgets = True
        try:
            for combo in (self.selection_scope, self.charge_selection_scope):
                index = combo.findData(value)
                combo.setCurrentIndex(max(0, index))
        finally:
            self._updating_widgets = False
        label = str(source.currentText() or "Filtered sources")
        self.status.setText(f"Selection scope changed to {label}.")
        self._refresh_summary()
        self.selection_state_changed.emit()

    def _member_display_changed(self, source: QComboBox) -> None:
        if self._updating_widgets:
            return
        value = str(source.currentData() or "filtered")
        self._updating_widgets = True
        try:
            for combo in (self.member_display, self.charge_member_display):
                index = combo.findData(value)
                combo.setCurrentIndex(max(0, index))
        finally:
            self._updating_widgets = False
        label = str(source.currentText() or "Passing current filters")
        self.status.setText(f"Group-member display changed to {label}.")
        self._draw_overlays()
        self._refresh_summary()
        self.selection_state_changed.emit()

    def _view_limits_changed(self, _axis) -> None:
        self._draw_overlays()

    def _scope_payload(self, scope: str | None = None) -> dict[str, Any]:
        value = str(scope or self.selection_scope.currentData() or "filtered")
        scopes = self._metadata.get("selection_scopes") or {}
        payload = scopes.get(value) if isinstance(scopes, Mapping) else None
        if isinstance(payload, Mapping):
            return dict(payload)
        return {
            "source_ids": self._metadata.get("source_ids", ()),
            "coordinate_pairs": self._metadata.get("coordinate_pairs", ()),
            "time": self._metadata.get("precision_source_values", {}).get("time", ()),
            "altitude_km": self._metadata.get("precision_source_values", {}).get("altitude_km", ()),
        }

    def _current_view_mask(self, payload: Mapping[str, Any]) -> np.ndarray:
        ids = np.asarray(payload.get("source_ids", ()), dtype=np.int64)
        mask = np.ones(ids.shape, dtype=bool)
        if ids.size == 0:
            return mask
        axes = tuple(self._metadata.get("axis_order", ()))
        pairs = tuple(payload.get("coordinate_pairs", ()))
        names = tuple(self._metadata.get("coordinate_names", ()))
        limits_by_name: dict[str, tuple[float, float]] = {}
        for axis, pair_names in zip(axes, names):
            if len(pair_names) != 2:
                continue
            limits_by_name.setdefault(str(pair_names[0]), tuple(sorted(axis.get_xlim())))
            limits_by_name.setdefault(str(pair_names[1]), tuple(sorted(axis.get_ylim())))
        return source_mask_in_linked_limits(
            ids, names, pairs, limits_by_name
        )

    def _selection_candidates(
        self, operation: str | None = None
    ) -> tuple[np.ndarray, tuple[Any, ...], np.ndarray]:
        """Return editable source candidates for the current tool operation.

        Normal filtered selection remains limited to sources passing the active
        filters and linked-view bounds.  Removal-style edits additionally include
        every existing member of the active group from the all-loaded payload.
        This makes an assigned source removable even when it is currently visible
        only through the group overlay because it no longer passes a filter.
        """

        scope = str(self.selection_scope.currentData() or "filtered")
        payload = self._scope_payload(scope)
        ids = np.asarray(payload.get("source_ids", ()), dtype=np.int64)
        pairs = tuple(payload.get("coordinate_pairs", ()))
        mask = np.ones(ids.shape, dtype=bool)
        if scope != "filtered":
            return ids, pairs, mask

        mask = self._current_view_mask(payload)
        removal_operation = str(operation or "").lower() in {
            "subtract",
            "intersect",
            "toggle",
        }
        group = self.manager.active_group
        if not removal_operation or group is None or not group.source_ids:
            return ids, pairs, mask

        all_payload = self._scope_payload("all")
        all_ids = np.asarray(all_payload.get("source_ids", ()), dtype=np.int64)
        all_pairs = tuple(all_payload.get("coordinate_pairs", ()))
        if all_ids.size == 0 or len(all_pairs) != len(pairs):
            return ids, pairs, mask
        if any(
            np.asarray(pair[0]).shape != all_ids.shape
            or np.asarray(pair[1]).shape != all_ids.shape
            for pair in all_pairs
        ):
            return ids, pairs, mask

        filtered_available = ids[np.asarray(mask, dtype=bool)]
        active_members = np.fromiter(group.source_ids, dtype=np.int64)
        editable = np.isin(
            all_ids,
            np.concatenate((filtered_available, active_members)),
            assume_unique=False,
        )
        return all_ids, all_pairs, editable

    def project_state(self) -> dict[str, Any]:
        state = self.manager.to_dict()
        state["category_visibility"] = dict(self._charge_category_visibility)
        state["selection_scope"] = str(self.selection_scope.currentData() or "filtered")
        state["member_display_scope"] = str(self.member_display.currentData() or "filtered")
        state["charge_region_label"] = str(
            self.charge_region_label.currentData() or DEFAULT_CHARGE_REGION_LABEL
        )
        state["show_charge_overlays_with_other_color_modes"] = bool(
            self._show_charge_overlays_with_other_color_modes
        )
        state["active_groups_by_domain"] = dict(self._active_by_domain)
        # Viewer overlays are editing aids scoped to the active workspace tab.
        # Scientific base colors such as Color by Charge remain independent.
        state["active_domain"] = self._current_domain()
        return state

    def restore_project_state(self, state: Mapping[str, Any] | None) -> None:
        payload = dict(state or {})
        self.manager.load_groups(
            payload.get("groups") or (), active_name=payload.get("active_group")
        )
        scope = str(payload.get("selection_scope") or "filtered")
        member_scope = str(payload.get("member_display_scope") or "filtered")
        show_charge_overlays = bool(
            payload.get("show_charge_overlays_with_other_color_modes", False)
        )
        saved_active = dict(payload.get("active_groups_by_domain") or {})
        for domain in self._active_by_domain:
            value = saved_active.get(domain)
            self._active_by_domain[domain] = str(value) if value else None
        active_domain = str(payload.get("active_domain") or "").strip().lower()
        if active_domain not in {"custom", "charge"}:
            active_group = self.manager.active_group
            active_domain = active_group.domain if active_group is not None else "custom"
        self._updating_widgets = True
        try:
            previous = self.tabs.blockSignals(True)
            try:
                self.tabs.setCurrentIndex(self._domain_index(active_domain))
            finally:
                self.tabs.blockSignals(previous)
            for combo in (self.selection_scope, self.charge_selection_scope):
                index = combo.findData(scope)
                combo.setCurrentIndex(max(0, index))
            for combo in (self.member_display, self.charge_member_display):
                index = combo.findData(member_scope)
                combo.setCurrentIndex(max(0, index))
            label_mode = str(
                payload.get("charge_region_label") or DEFAULT_CHARGE_REGION_LABEL
            )
            label_index = self.charge_region_label.findData(label_mode)
            self.charge_region_label.setCurrentIndex(max(0, label_index))
            self.show_charge_overlays_other_modes.setChecked(show_charge_overlays)
        finally:
            self._updating_widgets = False
        self._show_charge_overlays_with_other_color_modes = show_charge_overlays
        # Make the manager's active group agree with the restored tab before
        # group widgets and overlays are rebuilt.
        self._activate_domain_group(active_domain)
        visibility = payload.get("category_visibility") or {}
        for category in self._charge_category_visibility:
            if category in visibility:
                self._charge_category_visibility[category] = bool(visibility[category])
        self._sync_category_visibility_widgets()
        self._hull_cache.clear()
        self._member_mask_cache.clear()
        self._refresh_group_widgets()
        self._refresh_base_charge_colors(draw=False)
        self._draw_overlays()
        self._refresh_summary()
        self._refresh_charge_summary()

    def bind_current_figure(self) -> None:
        self.bind_figure(self.figure_host.figure)

    def bind_figure(self, figure) -> None:
        if figure is self._figure and figure is not None:
            self._metadata = getattr(figure, "_lmas_metadata", {})
            self._refresh_base_charge_colors(draw=False)
            self._draw_overlays()
            self._refresh_summary()
            self._refresh_charge_summary()
            return
        self._disconnect_figure()
        self._figure = figure
        self._metadata = getattr(figure, "_lmas_metadata", {}) if figure is not None else {}
        dataset_key = self._metadata.get("selection_dataset_key")
        if self._dataset_key is not None and dataset_key != self._dataset_key:
            self.manager = SourceSelectionManager()
            self._charge_default_requested_for_dataset = False
            self._hull_cache.clear()
            self._member_mask_cache.clear()
            self._refresh_group_widgets()
            self.status.setText("Source groups were reset for the newly opened dataset.")
        self._dataset_key = dataset_key
        if figure is None or not self._metadata.get("linked_view"):
            self.status.setText("Open LMA data to begin selection.")
            self._refresh_summary()
            self._refresh_charge_summary()
            return
        axes = tuple(self._metadata.get("axis_order", ()))
        # Register the recovery callback before the Matplotlib selectors. If a
        # non-modal workspace left selection paused, the first new press can
        # reactivate the selector in time for that same lasso/click gesture.
        self._event_connections["recovery_press"] = figure.canvas.mpl_connect(
            "button_press_event", self._selection_canvas_press
        )
        for index, axis in enumerate(axes):
            selector = LassoSelector(
                axis,
                lambda vertices, axis_index=index: self._lasso_complete(
                    axis_index, vertices
                ),
                useblit=True,
                props={"color": "#ff4fd8", "linewidth": 1.1, "alpha": 0.9},
            )
            selector.set_active(False)
            self._lassos.append(selector)
            for signal_name in ("xlim_changed", "ylim_changed"):
                callback_id = axis.callbacks.connect(signal_name, self._view_limits_changed)
                self._axis_limit_connections.append((axis, signal_name, callback_id))
        self._event_connections["press"] = figure.canvas.mpl_connect(
            "button_press_event", self._point_click
        )
        self._set_selector_activity()
        QTimer.singleShot(0, self._rearm_current_selectors)
        self._refresh_base_charge_colors(draw=False)
        self._draw_overlays()
        self._refresh_summary()
        self._refresh_charge_summary()
        self.status.setText(
            "Selection ready. Draw a lasso in any scientific panel or use Point edit."
        )

    def _disconnect_figure(self) -> None:
        for selector in self._lassos:
            try:
                selector.disconnect_events()
            except Exception:
                pass
        self._lassos = []
        for axis, _signal_name, callback_id in self._axis_limit_connections:
            try:
                axis.callbacks.disconnect(callback_id)
            except Exception:
                pass
        self._axis_limit_connections = []
        if self._figure is not None:
            for connection in self._event_connections.values():
                try:
                    self._figure.canvas.mpl_disconnect(connection)
                except Exception:
                    pass
        self._event_connections = {}
        self._remove_overlays()
        self._hull_cache.clear()
        self._member_mask_cache.clear()

    def _toolbar_busy(self) -> bool:
        toolbar = getattr(self.figure_host, "_toolbar", None)
        mode = getattr(toolbar, "mode", "") if toolbar is not None else ""
        text = str(getattr(mode, "name", mode)).casefold()
        return "zoom" in text or "pan" in text

    def _selection_canvas_press(self, event) -> None:
        """Recover a paused selection session before the actual edit callback.

        This deliberately does not steal clicks from Precision Mode or an active
        pan/zoom tool. It only repairs the silent state where the selection window
        is visible, an edit tool is selected, but the shared canvas flag was left
        inactive by another non-modal workflow.
        """

        if (
            self._selection_active
            or not self.isVisible()
            or self.figure_host.precision_mode_active
            or self._toolbar_busy()
            or getattr(event, "button", None) != 1
            or event.inaxes not in tuple(self._metadata.get("axis_order", ()))
        ):
            return
        self._resume_selection_interaction()

    def activate_for_selection(self) -> None:
        self.select_tab("selection")
        self._activate_workspace()

    def activate_for_charge_analysis(self) -> None:
        self.select_tab("charge")
        self._request_charge_default_once()
        self._activate_workspace()


    def _activate_workspace(self) -> None:
        self.bind_current_figure()
        self._selection_active = True
        self.show()
        self.raise_()
        self.activateWindow()
        self.figure_host.activate_selection_mode(crosshair_cursor=self.lasso_mode.isChecked())
        self._set_selector_activity()

    def suspend_selection(self) -> None:
        """Stop figure interaction while retaining groups and visible overlays."""

        self._selection_active = False
        self._set_selector_activity()
        if self.isVisible():
            self.status.setText(
                "Selection paused. Choose Lasso, Point edit, or a source group to resume."
            )

    def _resume_selection_interaction(self) -> bool:
        """Reclaim the main plotting canvas for source-group editing.

        Other workspaces and the navigation toolbar can pause selection while the
        non-modal Source Selection window remains visible.  Any explicit editing
        action in this window must therefore restore the shared interaction state
        instead of leaving both lasso and point edit silently inert.
        """

        if self.figure_host.figure is None or not self.isVisible():
            return False
        self.bind_current_figure()
        self._selection_active = True
        if not self.figure_host.activate_selection_mode(crosshair_cursor=self.lasso_mode.isChecked()):
            self._selection_active = False
            return False
        self._rearm_current_selectors()
        return True

    def hide_selection_mode(self) -> None:
        """Hide interaction and remove transient Source Selection overlays."""

        self._selection_active = False
        self._set_selector_activity()
        self.figure_host.deactivate_selection_mode()
        self._overlay_timer.stop()
        self._refresh_base_charge_colors(draw=False)
        self._remove_overlays()
        self.hide()
        figure = self._figure
        canvas = getattr(figure, "canvas", None) if figure is not None else None
        if canvas is not None:
            canvas.draw_idle()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        event.ignore()
        self.hide_selection_mode()

    def _tool_changed(self) -> None:
        if self._updating_widgets:
            return
        self._updating_widgets = True
        try:
            self.charge_lasso_mode.setChecked(self.lasso_mode.isChecked())
            self.charge_point_mode.setChecked(self.point_mode.isChecked())
        finally:
            self._updating_widgets = False
        if self.isVisible():
            self._resume_selection_interaction()
        else:
            self._set_selector_activity()
        if self._selection_active:
            tool = "lasso" if self.lasso_mode.isChecked() else "point edit"
            self.status.setText(f"{tool.title()} active on all scientific panels.")

    def _set_selector_activity(self) -> None:
        active = bool(
            self._selection_active
            and self.figure_host.selection_mode_active
            and self.isVisible()
            and self.lasso_mode.isChecked()
            and not self._toolbar_busy()
        )
        for selector in self._lassos:
            selector.set_active(active)

    def _rearm_current_selectors(self) -> None:
        """Re-arm selectors after a canvas redraw or figure replacement."""

        if not self._selection_active or not self.isVisible() or self._figure is None:
            return
        self.figure_host.activate_selection_mode(crosshair_cursor=self.lasso_mode.isChecked())
        for selector in self._lassos:
            selector.set_active(False)
        self._set_selector_activity()
        canvas = getattr(self._figure, "canvas", None)
        if canvas is not None:
            try:
                canvas.setFocus(Qt.FocusReason.ShortcutFocusReason)
            except Exception:
                pass
            canvas.draw_idle()

    def _recover_selection_interaction(self) -> None:
        """Recover lasso interaction after delayed Charge-color redraws."""

        if not self._selection_active or not self.isVisible():
            return
        self.bind_current_figure()
        self._rearm_current_selectors()

    @staticmethod
    def _keyboard_modifiers() -> Qt.KeyboardModifier:
        return QApplication.keyboardModifiers()

    def _resolved_operation(self, *, point: bool = False) -> str:
        del point  # Lasso and point edit deliberately share one operation policy.
        modifiers = self._keyboard_modifiers()
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            return "intersect"
        if modifiers & Qt.KeyboardModifier.ShiftModifier:
            return "subtract"
        if modifiers & Qt.KeyboardModifier.AltModifier:
            return "subtract"
        return str(self.operation.currentData() or "add")

    def _lasso_complete(self, axis_index: int, vertices) -> None:
        """Apply one lasso edit and always return the selector to an idle state.

        Matplotlib clears a LassoSelector only *after* its callback returns.  An
        exception in this callback therefore leaves the polygon hanging on-screen
        and blocks subsequent interaction.  Keep every failure inside this GUI
        boundary and explicitly re-arm the selectors on the next event-loop turn.
        """

        try:
            if (
                not self._selection_active
                or not self.figure_host.selection_mode_active
                or not self.lasso_mode.isChecked()
            ):
                return
            group = self.manager.active_group
            if group is None or group.locked:
                self.status.setText("The active source group is locked.")
                return
            operation = self._resolved_operation()
            ids, pairs, scope_mask = self._selection_candidates(operation)
            if axis_index < 0 or axis_index >= len(pairs) or ids.size == 0:
                return
            x_values = np.asarray(pairs[axis_index][0], dtype=float)
            y_values = np.asarray(pairs[axis_index][1], dtype=float)
            if x_values.shape != ids.shape or y_values.shape != ids.shape:
                return
            candidates = (
                np.asarray(scope_mask, dtype=bool)
                & np.isfinite(x_values)
                & np.isfinite(y_values)
            )
            selected = source_ids_inside_polygon(
                x_values[candidates], y_values[candidates], ids[candidates], vertices
            )
            changed = (
                self.manager.apply_charge_assignment(selected, operation)
                if group.domain == "charge"
                else self.manager.apply(selected, operation)
            )
            scope_label = str(
                self.selection_scope.currentText() or "Filtered sources"
            )
            operation_label = "Remove" if operation == "subtract" else operation.title()
            self._after_state_change(
                f"{operation_label} lasso matched {selected.size:,} "
                f"{scope_label.lower()}.",
                changed=changed,
            )
        except Exception as exc:
            self.status.setText(
                f"Selection edit could not be completed: {exc}. The tool was reset."
            )
        finally:
            QTimer.singleShot(0, self._rearm_current_selectors)

    def _point_click(self, event) -> None:
        if (
            not self._selection_active
            or not self.figure_host.selection_mode_active
            or not self.point_mode.isChecked()
            or self._toolbar_busy()
            or event.inaxes not in tuple(self._metadata.get("axis_order", ()))
            or event.xdata is None
            or event.ydata is None
            or getattr(event, "button", None) != 1
        ):
            return
        group = self.manager.active_group
        if group is None or group.locked:
            self.status.setText("The active source group is locked.")
            return
        axes = tuple(self._metadata.get("axis_order", ()))
        operation = self._resolved_operation(point=True)
        ids, pairs, scope_mask = self._selection_candidates(operation)
        index = axes.index(event.inaxes)
        if index >= len(pairs) or ids.size == 0:
            return
        x_values = np.asarray(pairs[index][0], dtype=float)
        y_values = np.asarray(pairs[index][1], dtype=float)
        if x_values.shape != ids.shape or y_values.shape != ids.shape:
            return
        finite = np.asarray(scope_mask, dtype=bool) & np.isfinite(x_values) & np.isfinite(y_values)
        if not np.any(finite):
            return
        points = np.column_stack((x_values[finite], y_values[finite]))
        display = event.inaxes.transData.transform(points)
        distance = np.hypot(display[:, 0] - float(event.x), display[:, 1] - float(event.y))
        nearest = int(np.argmin(distance))
        if float(distance[nearest]) > 12.0:
            self.status.setText("No source was close enough to the click.")
            return
        source_id = int(ids[finite][nearest])
        changed = (
            self.manager.apply_charge_assignment([source_id], operation)
            if group.domain == "charge"
            else self.manager.apply([source_id], operation)
        )
        operation_label = "remove" if operation == "subtract" else operation
        self._after_state_change(
            f"Source {source_id} {operation_label} operation applied.", changed=changed
        )

    def _source_ids(self) -> np.ndarray:
        payload = self._scope_payload("filtered")
        return np.asarray(payload.get("source_ids", ()), dtype=np.int64)

    def _refresh_base_charge_colors(self, *, draw: bool = False) -> bool:
        """Keep Color by Charge independent of the active workspace tab."""

        return refresh_charge_source_colors(
            self._figure,
            self.project_state(),
            draw=draw,
        )

    def _remove_overlays(self) -> None:
        for artist in self._overlay_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._overlay_artists = []

    def _draw_overlays(self) -> None:
        self._overlay_timer.start()

    def _group_should_render(self, group) -> bool:
        # Halo/recolor/outline/hull artists are tab-local editing aids. Base
        # scientific color modes are handled separately and remain persistent.
        if group.domain != self._current_domain():
            return False
        return bool(
            group.visible
            and group.display_style != "hidden"
            and self._charge_category_visibility.get(group.charge_category, True)
            and charge_group_overlay_visible(
                group.charge_category,
                color_by=self._metadata.get("color_by"),
                show_with_other_color_modes=(
                    self._show_charge_overlays_with_other_color_modes
                ),
            )
        )

    def _source_center_kwargs(self, axis_index: int, mask: np.ndarray) -> dict[str, Any]:
        values = np.asarray(self._metadata.get("color_values", ()), dtype=float)
        scatters = tuple(self._metadata.get("scatters", ()))
        kwargs: dict[str, Any] = {}
        if values.shape == mask.shape:
            kwargs["c"] = values[mask]
            kwargs["norm"] = self._metadata.get("norm")
            if axis_index < len(scatters):
                kwargs["cmap"] = scatters[axis_index].get_cmap()
        return kwargs

    def _membership_mask(
        self,
        source_ids: np.ndarray,
        group,
        *,
        scope: str,
    ) -> np.ndarray:
        ids = np.asarray(source_ids, dtype=np.int64)
        key = (
            self._dataset_key,
            str(scope),
            group.name,
            hash(group.source_ids),
            ids.size,
        )
        cached = self._member_mask_cache.get(key)
        if cached is None or cached.shape != ids.shape:
            requested = np.fromiter(group.source_ids, dtype=np.int64)
            cached = np.isin(ids, requested, assume_unique=False)
            self._member_mask_cache[key] = cached
            if len(self._member_mask_cache) > 64:
                oldest = next(iter(self._member_mask_cache))
                self._member_mask_cache.pop(oldest, None)
        return cached

    def _hull_geometry(self, group, axis_index: int, points: np.ndarray):
        key = (
            self._dataset_key,
            group.name,
            hash(group.source_ids),
            group.display_style,
            axis_index,
            points.shape[0],
            hash(np.ascontiguousarray(points, dtype=float).tobytes()),
        )
        geometry = self._hull_cache.get(key)
        if geometry is None:
            geometry = projection_hull_geometry(points, group.display_style)
            self._hull_cache[key] = geometry
            if len(self._hull_cache) > 96:
                oldest = next(iter(self._hull_cache))
                self._hull_cache.pop(oldest, None)
        return geometry

    @staticmethod
    def _bounded_display_mask(mask: np.ndarray, *, max_points: int) -> np.ndarray:
        """Deterministically thin overlay artists without changing membership."""

        values = np.asarray(mask, dtype=bool)
        indices = np.flatnonzero(values)
        if indices.size <= max_points:
            return values
        positions = np.linspace(0, indices.size - 1, int(max_points), dtype=np.int64)
        reduced = np.zeros(values.shape, dtype=bool)
        reduced[indices[positions]] = True
        return reduced

    def _draw_group_subset(
        self,
        group,
        axes: tuple[Any, ...],
        pairs: tuple[Any, ...],
        mask: np.ndarray,
        scatters: tuple[Any, ...],
        *,
        active: bool,
    ) -> None:
        if not np.any(mask):
            return
        point_mask = self._bounded_display_mask(mask, max_points=50_000)
        color_by_group = str(self._metadata.get("color_by") or "").strip().lower() == "group"
        for axis_index, (axis, pair) in enumerate(zip(axes, pairs)):
            x = np.asarray(pair[0], dtype=float)[point_mask]
            y = np.asarray(pair[1], dtype=float)[point_mask]
            valid = np.isfinite(x) & np.isfinite(y)
            if not np.any(valid):
                continue
            x = x[valid]
            y = y[valid]
            base_z = (
                float(scatters[axis_index].get_zorder())
                if axis_index < len(scatters)
                else 1.0
            )
            style = effective_group_display_style(
                group.display_style,
                color_by=self._metadata.get("color_by"),
            )
            # ``Color by → Source group`` is the base source-color mode, not a
            # selection-overlay style.  Fill the selected sources with their
            # custom group color immediately, even before MainWindow's delayed
            # full redraw replaces the figure.  The group's Halo/Outline/Hull
            # choice remains an optional supplemental overlay.
            if color_by_group:
                fill = axis.scatter(
                    x,
                    y,
                    s=10 if active else 7.5,
                    facecolors=group.color,
                    edgecolors="none",
                    alpha=0.98 if active else 0.88,
                    zorder=base_z + 0.20,
                    clip_on=True,
                )
                self._overlay_artists.append(fill)
                if style == "recolor":
                    continue
            if style == "halo":
                halo = axis.scatter(
                    x,
                    y,
                    s=24 if active else 19,
                    facecolors=group.color,
                    edgecolors="none",
                    alpha=0.34 if active else 0.24,
                    zorder=base_z - 0.05,
                    clip_on=True,
                )
                center_mask = point_mask.copy()
                center_mask[point_mask] = valid
                centers = axis.scatter(
                    x,
                    y,
                    s=6.5 if active else 5.0,
                    edgecolors="none",
                    alpha=1.0,
                    zorder=base_z + 0.15,
                    clip_on=True,
                    **self._source_center_kwargs(axis_index, center_mask),
                )
                self._overlay_artists.extend((halo, centers))
            elif style == "recolor":
                artist = axis.scatter(
                    x,
                    y,
                    s=11 if active else 8,
                    facecolors=group.color,
                    edgecolors="none",
                    alpha=0.95 if active else 0.82,
                    zorder=base_z + 0.2,
                    clip_on=True,
                )
                self._overlay_artists.append(artist)
            elif style == "outline":
                artist = axis.scatter(
                    x,
                    y,
                    s=20 if active else 15,
                    facecolors="none",
                    edgecolors=group.color,
                    linewidths=1.0 if active else 0.75,
                    alpha=0.95 if active else 0.75,
                    zorder=base_z + 0.25,
                    clip_on=True,
                )
                self._overlay_artists.append(artist)
            elif style in {"convex_hull", "concave_hull", "clustered_hulls"}:
                points = np.column_stack((x, y))
                geometry = self._hull_geometry(group, axis_index, points)
                if geometry.faces:
                    collection = PolyCollection(
                        geometry.faces,
                        facecolors=group.color,
                        edgecolors="none",
                        alpha=0.18 if active else 0.12,
                        zorder=base_z - 0.08,
                        clip_on=True,
                    )
                    axis.add_collection(collection)
                    self._overlay_artists.append(collection)
                for boundary in geometry.boundaries:
                    if boundary.shape[0] < 2:
                        continue
                    line = axis.plot(
                        boundary[:, 0],
                        boundary[:, 1],
                        color=group.color,
                        linewidth=1.35 if active else 0.95,
                        alpha=0.95 if active else 0.72,
                        zorder=base_z + 0.18,
                        clip_on=True,
                    )[0]
                    self._overlay_artists.append(line)

    def _draw_ghost_subset(
        self,
        group,
        axes: tuple[Any, ...],
        pairs: tuple[Any, ...],
        mask: np.ndarray,
        scatters: tuple[Any, ...],
        *,
        active: bool,
    ) -> None:
        """Draw members outside the normal filtered linked view distinctly."""

        ghost_mask = self._bounded_display_mask(mask, max_points=25_000)
        if not np.any(ghost_mask):
            return
        for axis_index, (axis, pair) in enumerate(zip(axes, pairs)):
            x = np.asarray(pair[0], dtype=float)[ghost_mask]
            y = np.asarray(pair[1], dtype=float)[ghost_mask]
            valid = np.isfinite(x) & np.isfinite(y)
            if not np.any(valid):
                continue
            base_z = (
                float(scatters[axis_index].get_zorder())
                if axis_index < len(scatters)
                else 1.0
            )
            artist = axis.scatter(
                x[valid],
                y[valid],
                s=14 if active else 11,
                facecolors="none",
                edgecolors=group.color,
                linewidths=0.65 if active else 0.5,
                alpha=0.42 if active else 0.30,
                zorder=base_z + 0.1,
                clip_on=True,
            )
            self._overlay_artists.append(artist)

    def _draw_overlays_now(self) -> None:
        self._remove_overlays()
        if self._figure is None or not self.isVisible():
            return
        axes = tuple(self._metadata.get("axis_order", ()))
        filtered_payload = self._scope_payload("filtered")
        filtered_pairs = tuple(filtered_payload.get("coordinate_pairs", ()))
        filtered_ids = np.asarray(filtered_payload.get("source_ids", ()), dtype=np.int64)
        all_payload = self._scope_payload("all")
        all_pairs = tuple(all_payload.get("coordinate_pairs", ()))
        all_ids = np.asarray(all_payload.get("source_ids", ()), dtype=np.int64)
        if filtered_ids.size == 0 and all_ids.size == 0:
            self._refresh_summary()
            return
        in_view = self._current_view_mask(filtered_payload)
        scatters = tuple(self._metadata.get("scatters", ()))
        display_mode = str(self.member_display.currentData() or "filtered")
        groups = list(self.manager.groups)
        # Draw the active group last so it has the same deterministic overlap
        # precedence as the categorical Source-group color mapping.
        groups.sort(key=lambda group: group.name == self.manager.active_name)
        for group in groups:
            if not self._group_should_render(group) or not group.source_ids:
                continue
            active = group.name == self.manager.active_name
            normal_mask = (
                self._membership_mask(filtered_ids, group, scope="filtered") & in_view
                if filtered_ids.size
                else np.zeros(filtered_ids.shape, dtype=bool)
            )
            if display_mode != "filtered_out":
                self._draw_group_subset(
                    group,
                    axes,
                    filtered_pairs,
                    normal_mask,
                    scatters,
                    active=active,
                )
            if display_mode not in {"all", "filtered_out"} or all_ids.size == 0:
                continue
            all_selected = self._membership_mask(all_ids, group, scope="all")
            if display_mode == "filtered_out":
                ghost_mask = all_selected & ~np.isin(
                    all_ids, filtered_ids, assume_unique=False
                )
            else:
                normally_drawn_ids = filtered_ids[normal_mask]
                ghost_mask = all_selected & ~np.isin(
                    all_ids, normally_drawn_ids, assume_unique=False
                )
            self._draw_ghost_subset(
                group,
                axes,
                all_pairs,
                ghost_mask,
                scatters,
                active=active,
            )
        self._figure.canvas.draw_idle()
        self._refresh_summary()

    def _style_index(self, combo: QComboBox, value: str) -> int:
        index = combo.findData(value)
        return max(0, index)

    def _refresh_group_widgets(self) -> None:
        self._updating_widgets = True
        try:
            lists = {
                "custom": self.group_list,
                "charge": self.charge_group_list,
            }
            for widget in lists.values():
                widget.clear()
            for domain, widget in lists.items():
                groups = self.manager.groups_for_domain(domain)
                requested = self._active_by_domain.get(domain)
                selected_row = -1
                for row, group in enumerate(groups):
                    item = QListWidgetItem(group.name)
                    item.setForeground(QBrush(QColor(group.color)))
                    flags = [
                        group.subtype.replace("_", " "),
                        group.display_style.replace("_", " "),
                        f"{len(group.source_ids):,} sources",
                    ]
                    if domain == "charge":
                        flags.insert(0, group.charge_category.title())
                    if not group.visible:
                        flags.append("hidden")
                    if group.locked:
                        flags.append("locked")
                    item.setToolTip(", ".join(flags))
                    widget.addItem(item)
                    if group.name == requested or (requested is None and group.name == self.manager.active_name):
                        selected_row = row
                if selected_row >= 0:
                    widget.setCurrentRow(selected_row)

            group = self.manager.active_group
            self.visible.setChecked(bool(group and group.visible))
            self.locked.setChecked(bool(group and group.locked))
            if group is not None:
                self.display_style.setCurrentIndex(
                    self._style_index(self.display_style, group.display_style)
                )
                self.charge_style.setCurrentIndex(
                    self._style_index(self.charge_style, group.display_style)
                )
                category_index = self.charge_category.findData(group.charge_category)
                self.charge_category.setCurrentIndex(max(0, category_index))
            operation_index = self.charge_operation.findData(self.operation.currentData())
            self.charge_operation.setCurrentIndex(max(0, operation_index))
            self.charge_lasso_mode.setChecked(self.lasso_mode.isChecked())
            self.charge_point_mode.setChecked(self.point_mode.isChecked())
        finally:
            self._updating_widgets = False
        self._refresh_charge_summary()

    def _select_group_name(self, name: str, domain: str) -> None:
        if self._updating_widgets or not name:
            return
        group = next((item for item in self.manager.groups_for_domain(domain) if item.name == name), None)
        if group is None:
            return
        self.manager.set_active(group.name)
        self._active_by_domain[domain] = group.name
        if self.isVisible():
            self._resume_selection_interaction()
        self._refresh_group_widgets()
        self._draw_overlays()
        self._refresh_summary()
        self._refresh_charge_summary()

    def _group_selected(self, name: str) -> None:
        self._select_group_name(name, "custom")

    def _charge_group_selected(self, name: str) -> None:
        self._select_group_name(name, "charge")


    def new_group(self) -> None:
        group = self.manager.new_group(domain="custom", subtype="generic_selection")
        self._resume_selection_interaction()
        self._after_state_change(
            f"Created source group {group.name}; draw a lasso or use Point edit.",
            changed=True,
        )

    def new_charge_group(self, category: str) -> None:
        value = str(category or "unassigned").strip().lower()
        base = value.title()
        number = 1 + sum(
            1 for group in self.manager.groups if group.charge_category == value
        )
        group = self.manager.new_group(
            f"{base} {number}",
            charge_category=value,
            color=CHARGE_COLORS.get(value, CHARGE_COLORS["unassigned"]),
            display_style="recolor",
            domain="charge",
            subtype="polarity_group",
        )
        self._active_by_domain["charge"] = group.name
        self._resume_selection_interaction()
        self._after_state_change(
            f"Created {group.name}; draw a lasso or use Point edit.", changed=True
        )








    def _operation_changed(self, _index: int) -> None:
        if self._updating_widgets:
            return
        self._updating_widgets = True
        try:
            index = self.charge_operation.findData(self.operation.currentData())
            self.charge_operation.setCurrentIndex(max(0, index))
        finally:
            self._updating_widgets = False

    def _charge_operation_changed(self, _index: int) -> None:
        if self._updating_widgets:
            return
        index = self.operation.findData(self.charge_operation.currentData())
        if index >= 0:
            self.operation.setCurrentIndex(index)

    def rename_group(self) -> None:
        group = self.manager.active_group
        if group is None:
            return
        name, accepted = QInputDialog.getText(
            self, "Rename source group", "Group name:", text=group.name
        )
        if not accepted:
            return
        renamed = self.manager.rename_group(name)
        self._after_state_change(
            f"Renamed active group to {renamed}." if renamed else "Group name was unchanged.",
            changed=renamed is not None,
        )

    def delete_group(self) -> None:
        group = self.manager.active_group
        if group is None:
            return
        changed = self.manager.delete_group()
        self._after_state_change(f"Deleted source group {group.name}.", changed=changed)

    def choose_color(self) -> None:
        group = self.manager.active_group
        if group is None:
            return
        color = QColorDialog.getColor(QColor(group.color), self, "Source-group color")
        if not color.isValid():
            return
        changed = self.manager.set_color(color.name())
        self._after_state_change("Source-group color updated.", changed=changed)

    def _visibility_changed(self, checked: bool) -> None:
        if self._updating_widgets:
            return
        changed = self.manager.set_visible(checked)
        self._after_state_change("Source-group visibility updated.", changed=changed)

    def _locked_changed(self, checked: bool) -> None:
        if self._updating_widgets:
            return
        changed = self.manager.set_locked(checked)
        self._after_state_change("Source-group lock updated.", changed=changed)

    def _display_style_changed(self, _index: int) -> None:
        if self._updating_widgets:
            return
        changed = self.manager.set_display_style(str(self.display_style.currentData()))
        self._after_state_change("Source-group display style updated.", changed=changed)

    def _charge_style_changed(self, _index: int) -> None:
        if self._updating_widgets:
            return
        changed = self.manager.set_display_style(str(self.charge_style.currentData()))
        self._after_state_change("Charge-group display style updated.", changed=changed)

    def _charge_category_changed(self, _index: int) -> None:
        if self._updating_widgets:
            return
        category = str(self.charge_category.currentData() or "unassigned")
        changed = self.manager.set_charge_category(category, apply_default_color=True)
        self._after_state_change(
            f"Active group changed to {category.title()}.", changed=changed
        )
        if changed and self._selection_active:
            # MainWindow may replace the Charge-colored figure on a short timer.
            # Re-arm once immediately and once after that delayed replacement.
            QTimer.singleShot(0, self._rearm_current_selectors)
            self._selector_rearm_timer.start()

    def reset_charge_category_color(self) -> None:
        group = self.manager.active_group
        if group is None:
            return
        changed = self.manager.set_color(CHARGE_COLORS[group.charge_category])
        self._after_state_change("Category color restored.", changed=changed)

    def _charge_region_label_changed(self, _index: int) -> None:
        if self._updating_widgets:
            return
        self.selection_state_changed.emit()
        self.status.setText(
            f"Charge-colored figures will use the label “{self.charge_region_label.currentText()}”."
        )

    def _category_visibility_changed(self, category: str, checked: bool) -> None:
        if self._updating_widgets:
            return
        self._charge_category_visibility[str(category)] = bool(checked)
        self.status.setText(f"{str(category).title()} category visibility updated.")
        self._draw_overlays()

    def _charge_overlay_preference_changed(self, checked: bool) -> None:
        if self._updating_widgets:
            return
        self._show_charge_overlays_with_other_color_modes = bool(checked)
        self.status.setText(
            "Charge overlays will be shown with other Color by modes."
            if checked
            else "Charge overlays are hidden in non-Charge Color by modes."
        )
        self._draw_overlays()
        self.selection_state_changed.emit()

    def _sync_category_visibility_widgets(self) -> None:
        self._updating_widgets = True
        try:
            self.show_unassigned.setChecked(self._charge_category_visibility["unassigned"])
            self.show_positive.setChecked(self._charge_category_visibility["positive"])
            self.show_negative.setChecked(self._charge_category_visibility["negative"])
        finally:
            self._updating_widgets = False


    def clear_active(self) -> None:
        changed = self.manager.clear()
        self._after_state_change("Active source group cleared.", changed=changed)

    def invert_active(self) -> None:
        ids, _pairs, scope_mask = self._selection_candidates()
        universe = ids[np.asarray(scope_mask, dtype=bool)]
        changed = self.manager.invert(universe)
        scope_label = str(self.selection_scope.currentText() or "Filtered sources")
        self._after_state_change(
            f"Active group inverted against {scope_label.lower()}.", changed=changed
        )

    def undo_last_action(self) -> bool:
        changed = self.manager.undo()
        self._after_state_change(
            "Last source-selection or charge action undone."
            if changed
            else "No source-selection action is available to undo.",
            changed=changed,
        )
        return changed

    def _after_state_change(self, message: str, *, changed: bool) -> None:
        self.status.setText(message)
        if changed:
            active = self.manager.active_group
            if active is not None:
                self._active_by_domain[active.domain] = active.name
            self._hull_cache.clear()
            self._member_mask_cache.clear()
            self._refresh_group_widgets()
            self._refresh_base_charge_colors(draw=False)
            self._draw_overlays()
            self._refresh_summary()
            self._refresh_charge_summary()
            self.selection_state_changed.emit()

    @staticmethod
    def _utc(value: np.datetime64) -> str:
        if np.isnat(value):
            return "—"
        return (
            np.datetime_as_string(value.astype("datetime64[us]"), unit="us").replace("T", " ")
            + " UTC"
        )

    def _refresh_summary(self) -> None:
        group = self.manager.active_group
        payload = self._scope_payload("filtered")
        ids = np.asarray(payload.get("source_ids", ()), dtype=np.int64)
        if group is None:
            self.count_label.setText("0")
            self.filtered_label.setText("0")
            self.available_label.setText("0")
            self.filtered_out_label.setText("0")
            self.time_label.setText("—")
            self.altitude_label.setText("—")
            self.bounds_label.setText("—")
            return

        requested = np.fromiter(group.source_ids, dtype=np.int64)
        filtered_members = np.isin(ids, requested, assume_unique=False)
        view_members = filtered_members & self._current_view_mask(payload)
        total = len(group.source_ids)
        filtered_count = int(np.count_nonzero(filtered_members))
        view_count = int(np.count_nonzero(view_members))
        self.count_label.setText(f"{total:,}")
        self.filtered_label.setText(
            f"{filtered_count:,}"
            + (f" ({total - filtered_count:,} outside current filters)" if total > filtered_count else "")
        )
        self.available_label.setText(
            f"{view_count:,}"
            + (f" ({filtered_count - view_count:,} passing sources outside view)" if filtered_count > view_count else "")
        )
        self.filtered_out_label.setText(f"{max(0, total - filtered_count):,}")
        if not view_count:
            self.time_label.setText("—")
            self.altitude_label.setText("—")
            self.bounds_label.setText("—")
            return

        times = np.asarray(payload.get("time", ())).astype("datetime64[ns]")
        if times.shape == ids.shape:
            chosen = times[view_members]
            chosen = chosen[~np.isnat(chosen)]
            self.time_label.setText(
                f"{self._utc(chosen.min())} → {self._utc(chosen.max())}"
                if chosen.size
                else "—"
            )
        else:
            self.time_label.setText("—")

        altitude = np.asarray(payload.get("altitude_km", ()), dtype=float)
        if altitude.shape == ids.shape:
            chosen_altitude = altitude[view_members]
            chosen_altitude = chosen_altitude[np.isfinite(chosen_altitude)]
            self.altitude_label.setText(
                f"{chosen_altitude.min():.3f}–{chosen_altitude.max():.3f} km MSL"
                if chosen_altitude.size
                else "—"
            )
        else:
            self.altitude_label.setText("—")

        coordinate_names = tuple(self._metadata.get("coordinate_names", ()))
        coordinate_pairs = tuple(payload.get("coordinate_pairs", ()))
        seen: dict[str, str] = {}
        for names, pair in zip(coordinate_names, coordinate_pairs):
            for name, raw in zip(names, pair):
                key = str(name)
                if key in seen:
                    continue
                array = np.asarray(raw, dtype=float)
                if array.shape != ids.shape:
                    continue
                selected = array[view_members]
                selected = selected[np.isfinite(selected)]
                if selected.size:
                    if key == "time":
                        span_ms = (selected.max() - selected.min()) * 86_400_000.0
                        seen[key] = f"Δt {span_ms:.3f} ms"
                    else:
                        seen[key] = f"{key} {selected.min():.3f}–{selected.max():.3f}"
        self.bounds_label.setText(" · ".join(seen.values()) or "—")

    def _refresh_charge_summary(self) -> None:
        group = self.manager.active_group
        if group is None:
            self.charge_active_label.setText("—")
            self.charge_counts_label.setText("—")
            self.charge_overlap_label.setText("—")
            self.charge_provenance_label.setText("—")
            return
        self.charge_active_label.setText(group.charge_category.title())
        counts = self.manager.category_counts()
        self.charge_counts_label.setText(
            f"Unassigned {counts['unassigned']:,} · Positive {counts['positive']:,} · "
            f"Negative {counts['negative']:,}"
        )
        overlap = self.manager.overlapping_source_ids(group.name)
        assigned_overlap = self.manager.overlapping_source_ids(
            group.name, assigned_only=True
        )
        if overlap:
            detail = f"{len(overlap):,} source IDs overlap another group"
            if assigned_overlap:
                detail += f"; {len(assigned_overlap):,} overlap an assigned group"
            self.charge_overlap_label.setText(detail)
            self.charge_overlap_label.setStyleSheet("font-weight: 600;")
        else:
            self.charge_overlap_label.setText("No overlap detected")
            self.charge_overlap_label.setStyleSheet("")
        self.charge_provenance_label.setText(
            f"Created {group.created_utc} with LMAS {group.created_with_lmas_version}; "
            f"modified {group.modified_utc}"
        )


__all__ = ["SourceSelectionWindow"]
