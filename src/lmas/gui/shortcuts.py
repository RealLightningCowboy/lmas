from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PySide6.QtCore import QEvent, QObject, QTimer, Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QKeySequenceEdit,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
)


@dataclass(frozen=True)
class ShortcutDefinition:
    action_id: str
    label: str
    category: str
    default: str


DEFAULT_SHORTCUTS: tuple[ShortcutDefinition, ...] = (
    ShortcutDefinition("open_data", "Open LMA data", "File", "Ctrl+O"),
    ShortcutDefinition("save_project", "Save project", "File", "Ctrl+S"),
    ShortcutDefinition("save_figure", "Save figure", "File", "Ctrl+Shift+S"),
    ShortcutDefinition("quit", "Quit LMAS", "File", "Ctrl+Q"),
    ShortcutDefinition("full_view", "Restore full linked view", "Navigation", "Home"),
    ShortcutDefinition("history_back", "Previous scientific view", "Navigation", "Alt+Left"),
    ShortcutDefinition("history_forward", "Next scientific view", "Navigation", "Alt+Right"),
    ShortcutDefinition("rectangle_zoom", "Rectangle zoom (zoom to box)", "Navigation", "W"),
    ShortcutDefinition("pan_drag", "Pan (click and drag)", "Navigation", "D"),
    ShortcutDefinition("precision_mode", "Open or focus Precision Mode", "Precision Mode", "P"),
    ShortcutDefinition("source_selection", "Open or focus Source Selection", "Source Selection", "L"),
    ShortcutDefinition("precision_cursor_a", "Select Precision cursor A", "Precision Mode", "Ctrl+1"),
    ShortcutDefinition("precision_cursor_b", "Select Precision cursor B", "Precision Mode", "Ctrl+2"),
    ShortcutDefinition("precision_previous", "Previous source by time", "Precision Mode", "Ctrl+Left"),
    ShortcutDefinition("precision_next", "Next source by time", "Precision Mode", "Ctrl+Right"),
    ShortcutDefinition("precision_previous_10", "Previous 10 sources by time", "Precision Mode", "Ctrl+Shift+Left"),
    ShortcutDefinition("precision_next_10", "Next 10 sources by time", "Precision Mode", "Ctrl+Shift+Right"),
    ShortcutDefinition("precision_swap", "Swap Precision cursors A/B", "Precision Mode", "Ctrl+Shift+X"),
    ShortcutDefinition("precision_undo", "Undo active analysis action", "Analysis", "Ctrl+Z"),
    ShortcutDefinition("precision_clear", "Clear active Precision cursor", "Precision Mode", "Delete"),
    ShortcutDefinition("precision_clear_all", "Clear both Precision cursors", "Precision Mode", "Shift+Delete"),
    ShortcutDefinition("precision_copy", "Copy Precision measurements", "Precision Mode", "Ctrl+C"),
    ShortcutDefinition("toggle_grid", "Toggle grid", "Plot", "G"),
    ShortcutDefinition("toggle_colorbar", "Toggle colorbar", "Plot", "C"),
    ShortcutDefinition("toggle_stations", "Toggle station markers", "Plot", "S"),
    ShortcutDefinition("toggle_station_labels", "Toggle station labels", "Plot", "Shift+S"),
    ShortcutDefinition("toggle_auto_fit", "Toggle linked auto-fit", "Plot", "A"),
    ShortcutDefinition("toggle_remap", "Toggle colormap remapping", "Plot", "M"),
    ShortcutDefinition("layout_landscape", "Landscape layout", "Plot", "1"),
    ShortcutDefinition("layout_portrait", "Portrait layout", "Plot", "2"),
    ShortcutDefinition("open_3d", "Open interactive 3D viewer", "Plot", "3"),
    ShortcutDefinition("fullscreen", "Toggle full screen", "Window", "F11"),
    ShortcutDefinition("keybind_help", "Open keybind reference", "Help", "F1"),
)

VIEWER_SHORTCUTS: tuple[tuple[str, str, str], ...] = (
    ("Projection/3D viewer", "Play or pause", "Space"),
    ("Projection viewer", "Previous / next frame", "Left / Right"),
    ("Projection viewer", "Larger time step", "Shift+Left / Shift+Right"),
    ("Projection viewer", "Decrease / increase playback speed", "[ / ]"),
    ("Projection viewer", "Restart playback", "R"),
    ("Hovered projection", "Zoom both dimensions around cursor", "Mouse wheel"),
    ("Hovered projection", "Zoom horizontal dimension only", "Shift+wheel"),
    ("Hovered projection", "Zoom vertical dimension only", "Ctrl+wheel"),
    ("Precision Mode window", "Select cursor A / B", "A / B"),
    ("Precision Mode window", "Previous / next source by time", "Left / Right"),
    ("Precision Mode window", "Step 10 sources by time", "Shift+Left / Shift+Right"),
    ("Precision Mode window", "Swap cursors", "X"),
    ("Precision Mode window", "Undo last cursor action", "Ctrl+Z"),
    ("Precision Mode window", "Hide Precision Mode", "Esc"),
    ("Source Selection window", "Lasso / point-edit tool", "L / E"),
    ("Source Selection window", "Undo last selection action", "Ctrl+Z"),
    ("Source Selection window", "Clear active group", "Delete"),
    ("Source Selection window", "Hide Source Selection", "Esc"),
)


def _portable(sequence: str | QKeySequence) -> str:
    value = sequence if isinstance(sequence, QKeySequence) else QKeySequence(str(sequence))
    return value.toString(QKeySequence.SequenceFormat.PortableText)


class ShortcutManager(QObject):
    """Persistent, focus-safe LMAS keyboard shortcut dispatcher.

    Qt ``QShortcut`` objects provide reliable delivery while the main window
    or any embedded plotting-canvas child has focus.  The application event
    filter only manages the focus guard; it no longer tries to reconstruct and
    compare raw key events.
    """

    def __init__(self, main_window, settings) -> None:
        super().__init__(main_window)
        self.main_window = main_window
        self.settings = settings
        self._callbacks: dict[str, Callable[[], None]] = {}
        self._bindings = self._load_bindings()
        self._shortcuts: dict[str, QShortcut] = {}
        for definition in DEFAULT_SHORTCUTS:
            shortcut = QShortcut(
                QKeySequence(self._bindings.get(definition.action_id, "")),
                self.main_window,
            )
            shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
            shortcut.setAutoRepeat(False)
            shortcut.activated.connect(
                lambda action_id=definition.action_id: self._dispatch(action_id)
            )
            self._shortcuts[definition.action_id] = shortcut
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
        self._sync_enabled()

    @property
    def definitions(self) -> tuple[ShortcutDefinition, ...]:
        return DEFAULT_SHORTCUTS

    def _load_bindings(self) -> dict[str, str]:
        bindings: dict[str, str] = {}
        for definition in DEFAULT_SHORTCUTS:
            stored = self.settings.value(
                f"shortcuts/{definition.action_id}", definition.default
            )
            bindings[definition.action_id] = _portable(str(stored or ""))
        return bindings

    def register(self, action_id: str, callback: Callable[[], None]) -> None:
        self._callbacks[str(action_id)] = callback

    def bindings(self) -> dict[str, str]:
        return dict(self._bindings)

    def set_bindings(self, bindings: dict[str, str]) -> None:
        known = {definition.action_id for definition in DEFAULT_SHORTCUTS}
        for action_id in known:
            value = _portable(bindings.get(action_id, self._bindings.get(action_id, "")))
            self._bindings[action_id] = value
            self.settings.setValue(f"shortcuts/{action_id}", value)
            shortcut = self._shortcuts.get(action_id)
            if shortcut is not None:
                shortcut.setKey(QKeySequence(value))
        self.settings.sync()
        self._sync_enabled()

    def restore_defaults(self) -> None:
        self.set_bindings(
            {definition.action_id: definition.default for definition in DEFAULT_SHORTCUTS}
        )

    @staticmethod
    def conflicts(bindings: dict[str, str]) -> dict[str, list[str]]:
        reverse: dict[str, list[str]] = {}
        for action_id, sequence in bindings.items():
            normalized = _portable(sequence)
            if normalized:
                reverse.setdefault(normalized, []).append(action_id)
        return {sequence: ids for sequence, ids in reverse.items() if len(ids) > 1}

    @staticmethod
    def _text_entry_has_priority(focus) -> bool:
        return isinstance(
            focus,
            (QLineEdit, QAbstractSpinBox, QComboBox, QTextEdit, QPlainTextEdit, QKeySequenceEdit),
        )

    def _main_window_is_active(self) -> bool:
        app = QApplication.instance()
        return app is not None and app.activeWindow() is self.main_window

    def _shortcuts_allowed(self) -> bool:
        app = QApplication.instance()
        if app is None or not self._main_window_is_active():
            return False
        return not self._text_entry_has_priority(app.focusWidget())

    def _sync_enabled(self) -> None:
        enabled = self._shortcuts_allowed()
        for shortcut in self._shortcuts.values():
            shortcut.setEnabled(enabled)

    def _dispatch(self, action_id: str) -> None:
        if not self._shortcuts_allowed():
            return
        callback = self._callbacks.get(action_id)
        if callback is not None:
            callback()

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt API
        if event.type() in {
            QEvent.Type.FocusIn,
            QEvent.Type.FocusOut,
            QEvent.Type.WindowActivate,
            QEvent.Type.WindowDeactivate,
            QEvent.Type.Show,
            QEvent.Type.Hide,
        }:
            # Focus/activation state is finalized after the current Qt event.
            QTimer.singleShot(0, self._sync_enabled)
        return False


class ShortcutSettingsDialog(QDialog):
    def __init__(self, manager: ShortcutManager, parent=None) -> None:
        super().__init__(parent)
        self.manager = manager
        self.setWindowTitle("LMAS Keyboard Shortcuts")
        self.resize(680, 620)
        outer = QVBoxLayout(self)
        self.table = QTableWidget(len(manager.definitions), 3)
        self.table.setHorizontalHeaderLabels(["Category", "Action", "Shortcut"])
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self._editors: dict[str, QKeySequenceEdit] = {}
        current = manager.bindings()
        for row, definition in enumerate(manager.definitions):
            category = QTableWidgetItem(definition.category)
            action = QTableWidgetItem(definition.label)
            category.setFlags(category.flags() & ~Qt.ItemFlag.ItemIsEditable)
            action.setFlags(action.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 0, category)
            self.table.setItem(row, 1, action)
            editor = QKeySequenceEdit(QKeySequence(current.get(definition.action_id, "")))
            editor.setClearButtonEnabled(True)
            self.table.setCellWidget(row, 2, editor)
            self._editors[definition.action_id] = editor
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setStretchLastSection(True)
        outer.addWidget(self.table, 1)

        button_row = QHBoxLayout()
        restore = QPushButton("Restore Defaults")
        restore.clicked.connect(self._restore_defaults)
        button_row.addWidget(restore)
        button_row.addStretch(1)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        button_row.addWidget(buttons)
        outer.addLayout(button_row)

    def _restore_defaults(self) -> None:
        defaults = {definition.action_id: definition.default for definition in DEFAULT_SHORTCUTS}
        for action_id, editor in self._editors.items():
            editor.setKeySequence(QKeySequence(defaults[action_id]))

    def _values(self) -> dict[str, str]:
        return {
            action_id: _portable(editor.keySequence())
            for action_id, editor in self._editors.items()
        }

    def _accept(self) -> None:
        values = self._values()
        conflicts = ShortcutManager.conflicts(values)
        if conflicts:
            lines = [f"{sequence}: {', '.join(ids)}" for sequence, ids in conflicts.items()]
            QMessageBox.warning(
                self,
                "Conflicting shortcuts",
                "Each shortcut can control only one action. Resolve these conflicts:\n\n"
                + "\n".join(lines),
            )
            return
        self.manager.set_bindings(values)
        self.accept()


class ShortcutReferenceDialog(QDialog):
    def __init__(self, manager: ShortcutManager, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("LMAS Keybinds")
        self.resize(720, 650)
        outer = QVBoxLayout(self)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search actions or keybinds…")
        outer.addWidget(self.search)
        rows = len(manager.definitions) + len(VIEWER_SHORTCUTS)
        self.table = QTableWidget(rows, 3)
        self.table.setHorizontalHeaderLabels(["Context", "Action", "Keybind"])
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        bindings = manager.bindings()
        row = 0
        for definition in manager.definitions:
            for column, text in enumerate(
                (definition.category, definition.label, bindings.get(definition.action_id, ""))
            ):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(row, column, item)
            row += 1
        for context, label, binding in VIEWER_SHORTCUTS:
            for column, text in enumerate((context, label, binding)):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(row, column, item)
            row += 1
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setStretchLastSection(True)
        outer.addWidget(self.table, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.close)
        buttons.accepted.connect(self.close)
        outer.addWidget(buttons)
        self.search.textChanged.connect(self._filter)

    def _filter(self, text: str) -> None:
        needle = str(text).strip().lower()
        for row in range(self.table.rowCount()):
            haystack = " ".join(
                self.table.item(row, column).text()
                for column in range(self.table.columnCount())
                if self.table.item(row, column) is not None
            ).lower()
            self.table.setRowHidden(row, bool(needle and needle not in haystack))


__all__ = [
    "DEFAULT_SHORTCUTS",
    "ShortcutDefinition",
    "ShortcutManager",
    "ShortcutReferenceDialog",
    "ShortcutSettingsDialog",
]
