"""Non-modal Precision Mode ("scope mode") measurement window."""

from __future__ import annotations

from collections.abc import Callable
import math
from typing import Any

import matplotlib.dates as mdates
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..precision import (
    PrecisionSource,
    canonical_coordinate_name,
    format_apparent_speed,
    from_canonical_coordinate,
    precision_coordinate_difference,
    to_canonical_coordinate,
    utc_text,
)
from .icon import application_icon


_SOURCE_FIELDS: tuple[tuple[str, str], ...] = (
    ("utc", "UTC"),
    ("source_id", "Source ID"),
    ("local", "Local E / N"),
    ("latlon", "Latitude / longitude"),
    ("altitude", "Altitude"),
    ("power", "Power"),
    ("chi2", "Reduced χ²"),
    ("stations", "Stations"),
)

_DIFFERENCE_FIELDS: tuple[tuple[str, str], ...] = (
    ("dt", "Δt (B − A)"),
    ("displacement", "ΔE / ΔN / ΔZ"),
    ("horizontal", "Horizontal distance"),
    ("distance_3d", "3D distance"),
    ("bearing", "Bearing A → B"),
    ("speed_horizontal", "Apparent horizontal speed"),
    ("speed_3d", "Apparent 3D speed"),
)

_AXIS_SLOT_ORDER = ("time", "x", "y", "altitude")


class PrecisionModeWindow(QMainWindow):
    """Linked point and axis-cursor measurement window."""

    def __init__(self, figure_host, parent=None) -> None:
        # Major analysis workspaces are independent top-level windows.  Giving
        # them the main window as a Qt parent creates an owned-window relationship
        # on Windows: the workspace stays above LMAS and receives no independent
        # taskbar entry.  MainWindow already retains a strong Python reference.
        super().__init__(None, Qt.WindowType.Window)
        self._owner_window = parent
        self.setAttribute(Qt.WidgetAttribute.WA_QuitOnClose, False)
        self.figure_host = figure_host
        self.setWindowTitle("Precision Mode — LMAS")
        self.setWindowIcon(application_icon())
        self.resize(700, 860)

        self._figure = None
        self._metadata: dict[str, Any] = {}
        self._event_connections: dict[str, int] = {}
        self._markers: dict[str, list[tuple[Any, Any]]] = {"A": [], "B": []}
        self._cursor_states: dict[str, dict[str, Any]] = {
            "A": {"index": None, "coordinates": {}},
            "B": {"index": None, "coordinates": {}},
        }
        self._axis_lines: dict[str, dict[int, list[tuple[Any, str, str]]]] = {}
        self._axis_values: dict[str, list[float | None]] = {}
        self._slot_dimensions: dict[str, str | None] = {
            slot: None for slot in _AXIS_SLOT_ORDER
        }
        self._active_cursor = "A"
        self._display_enabled = False
        self._shortcuts: list[QShortcut] = []
        self._history: list[dict[str, Any]] = []
        self._drag_state: dict[str, Any] | None = None
        self._restoring_state = False

        central = QWidget(self)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        intro = QLabel(
            "<b>Precision measurement mode / scope mode</b> combines linked "
            "source inspection with free measurement cursors and "
            "oscilloscope-style axis lines."
        )
        intro.setWordWrap(True)
        outer.addWidget(intro)

        control_row = QHBoxLayout()
        control_row.addWidget(QLabel("Active cursor"))
        self.cursor_a = QRadioButton("A")
        self.cursor_b = QRadioButton("B")
        self.cursor_a.setChecked(True)
        self.cursor_a.toggled.connect(
            lambda checked: self.select_cursor("A") if checked else None
        )
        self.cursor_b.toggled.connect(
            lambda checked: self.select_cursor("B") if checked else None
        )
        control_row.addWidget(self.cursor_a)
        control_row.addWidget(self.cursor_b)
        control_row.addSpacing(16)
        control_row.addWidget(QLabel("Snap"))
        self.snap_mode = QComboBox()
        self.snap_mode.addItem("Off — Free", "free")
        self.snap_mode.addItem("Nearest visible source", "visible")
        self.snap_mode.addItem("Nearest full filtered source", "filtered")
        self.snap_mode.setCurrentIndex(1)
        self.snap_mode.setToolTip(
            "Free preserves the coordinates placed on each panel. Visible snaps to "
            "the exact linked subset in the current view. Full filtered uses every "
            "source that passes the quality filters."
        )
        control_row.addWidget(self.snap_mode, 1)
        outer.addLayout(control_row)

        option_row = QHBoxLayout()
        self.show_cursor_labels = QCheckBox("Show cursor labels")
        self.show_cursor_labels.setChecked(False)
        self.show_cursor_labels.setToolTip(
            "Show or hide the A/B text labels. Crosshair markers remain visible."
        )
        self.show_cursor_labels.toggled.connect(self._update_markers)
        option_row.addWidget(self.show_cursor_labels)
        option_row.addStretch(1)
        option_row.addWidget(QLabel("Click: active cursor · Shift+click: cursor B"))
        outer.addLayout(option_row)

        source_grid = QGridLayout()
        self._source_labels: dict[str, dict[str, QLabel]] = {}
        for column, cursor_name in enumerate(("A", "B")):
            box = QGroupBox(f"Cursor {cursor_name}")
            form = QFormLayout(box)
            labels: dict[str, QLabel] = {}
            for key, label in _SOURCE_FIELDS:
                value = QLabel("—")
                value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
                value.setMinimumWidth(190)
                form.addRow(label, value)
                labels[key] = value
            self._source_labels[cursor_name] = labels
            source_grid.addWidget(box, 0, column)
        outer.addLayout(source_grid)

        difference_box = QGroupBox("A → B measurements")
        difference_form = QFormLayout(difference_box)
        self._difference_labels: dict[str, QLabel] = {}
        for key, label in _DIFFERENCE_FIELDS:
            value = QLabel("—")
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            difference_form.addRow(label, value)
            self._difference_labels[key] = value
        note = QLabel("Speeds are apparent cursor-derived values, not fitted propagation estimates.")
        note.setWordWrap(True)
        note.setStyleSheet("font-style: italic;")
        difference_form.addRow(note)
        outer.addWidget(difference_box)

        self.axis_box = QGroupBox("Axis cursors")
        axis_layout = QGridLayout(self.axis_box)
        axis_layout.addWidget(QLabel("Pair"), 0, 0)
        axis_layout.addWidget(QLabel("Line 1"), 0, 1)
        axis_layout.addWidget(QLabel("Line 2"), 0, 2)
        axis_layout.addWidget(QLabel("Difference"), 0, 3)
        self._axis_rows: dict[str, dict[str, Any]] = {}
        for row, slot in enumerate(_AXIS_SLOT_ORDER, start=1):
            enabled = QCheckBox(slot.title())
            first = QLineEdit()
            second = QLineEdit()
            delta = QLabel("—")
            for line_edit in (first, second):
                line_edit.setPlaceholderText("—")
                line_edit.setMinimumWidth(150)
                line_edit.setClearButtonEnabled(True)
                line_edit.setToolTip(
                    "Enter a coordinate directly. Time accepts UTC or milliseconds "
                    "from the first source according to the Time entry control."
                )
            delta.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            enabled.toggled.connect(
                lambda checked, slot_name=slot: self._axis_enabled_changed(
                    slot_name, checked
                )
            )
            first.editingFinished.connect(
                lambda slot_name=slot: self._commit_axis_entry(slot_name, 0)
            )
            second.editingFinished.connect(
                lambda slot_name=slot: self._commit_axis_entry(slot_name, 1)
            )
            axis_layout.addWidget(enabled, row, 0)
            axis_layout.addWidget(first, row, 1)
            axis_layout.addWidget(second, row, 2)
            axis_layout.addWidget(delta, row, 3)
            self._axis_rows[slot] = {
                "enabled": enabled,
                "first": first,
                "second": second,
                "delta": delta,
            }

        time_row = len(_AXIS_SLOT_ORDER) + 1
        axis_layout.addWidget(QLabel("Time entry"), time_row, 0)
        self.time_entry_mode = QComboBox()
        self.time_entry_mode.addItem("UTC", "utc")
        self.time_entry_mode.addItem("Milliseconds from first source", "relative_ms")
        self.time_entry_mode.currentIndexChanged.connect(self._refresh_axis_readouts)
        axis_layout.addWidget(self.time_entry_mode, time_row, 1, 1, 3)

        action_row = time_row + 1
        set_from_ab = QPushButton("Set lines from A/B")
        set_from_ab.clicked.connect(self.set_axis_lines_from_ab)
        apply_ab = QPushButton("Apply A–B time interval")
        apply_ab.clicked.connect(self.apply_ab_time_interval)
        clear_axis = QPushButton("Clear axis cursors")
        clear_axis.clicked.connect(self.clear_axis_cursors)
        axis_layout.addWidget(set_from_ab, action_row, 0)
        axis_layout.addWidget(apply_ab, action_row, 1, 1, 2)
        axis_layout.addWidget(clear_axis, action_row, 3)
        axis_hint = QLabel(
            "Ctrl+click sets line 1 for both dimensions on a panel; "
            "Ctrl+Shift+click sets line 2. Drag any visible free cursor or axis "
            "line to refine it. Lines are linked across compatible panels."
        )
        axis_hint.setWordWrap(True)
        axis_layout.addWidget(axis_hint, action_row + 1, 0, 1, 4)
        outer.addWidget(self.axis_box)

        button_row = QHBoxLayout()
        swap = QPushButton("Swap A/B")
        swap.clicked.connect(self.swap_cursors)
        clear = QPushButton("Clear active")
        clear.clicked.connect(self.clear_active)
        clear_all = QPushButton("Clear both")
        clear_all.clicked.connect(self.clear_all)
        copy = QPushButton("Copy measurements")
        copy.clicked.connect(self.copy_measurements)
        button_row.addWidget(swap)
        button_row.addWidget(clear)
        button_row.addWidget(clear_all)
        button_row.addStretch(1)
        button_row.addWidget(copy)
        outer.addLayout(button_row)

        self.status = QLabel(
            "Click a scientific panel to place cursor A. Shift+click places cursor B."
        )
        self.status.setWordWrap(True)
        outer.addWidget(self.status)

        shortcut_hint = QLabel(
            "Window keys: A/B select cursor · ←/→ step by time · Shift+←/→ step 10 · "
            "X swap · Ctrl+Z undo cursor action · Delete clear · Shift+Delete clear both · "
            "Ctrl+C copy · Esc hide"
        )
        shortcut_hint.setWordWrap(True)
        shortcut_hint.setStyleSheet("font-size: 9pt;")
        outer.addWidget(shortcut_hint)

        self.setCentralWidget(central)
        self._install_shortcuts()
        self.bind_current_figure()

    @staticmethod
    def _text_entry_has_priority(widget) -> bool:
        return isinstance(
            widget,
            (QLineEdit, QAbstractSpinBox, QComboBox, QTextEdit, QPlainTextEdit),
        )

    def _shortcut(
        self, sequence: str, callback: Callable[[], None], *, allow_text: bool = False
    ) -> None:
        shortcut = QShortcut(QKeySequence(sequence), self)
        shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        shortcut.setAutoRepeat(False)

        def dispatch() -> None:
            focus = QApplication.focusWidget()
            if allow_text or not self._text_entry_has_priority(focus):
                callback()

        shortcut.activated.connect(dispatch)
        self._shortcuts.append(shortcut)

    def _install_shortcuts(self) -> None:
        self._shortcut("A", lambda: self.select_cursor("A"))
        self._shortcut("B", lambda: self.select_cursor("B"))
        self._shortcut("Left", lambda: self.step_active(-1))
        self._shortcut("Right", lambda: self.step_active(1))
        self._shortcut("Shift+Left", lambda: self.step_active(-10))
        self._shortcut("Shift+Right", lambda: self.step_active(10))
        self._shortcut("X", self.swap_cursors)
        self._shortcut("Ctrl+Z", self.undo_last_action)
        self._shortcut("Delete", self.clear_active)
        self._shortcut("Shift+Delete", self.clear_all)
        self._shortcut("Ctrl+C", self.copy_measurements)
        self._shortcut("Escape", self.hide_precision_mode, allow_text=True)

    def bind_current_figure(self) -> None:
        self.bind_figure(self.figure_host.figure)

    def bind_figure(self, figure) -> None:
        if figure is self._figure and figure is not None:
            self._metadata = getattr(figure, "_lmas_metadata", {})
            self._configure_axis_rows()
            self._update_markers()
            self._update_axis_lines()
            self._refresh_readouts()
            return

        previous_states = {
            name: {
                "index": state.get("index"),
                "coordinates": dict(state.get("coordinates", {})),
                "source_id": self._state_source_id(state),
            }
            for name, state in self._cursor_states.items()
        }
        previous_axis_values = {
            key: list(values) for key, values in self._axis_values.items()
        }

        if self._figure is not None:
            for connection in self._event_connections.values():
                try:
                    self._figure.canvas.mpl_disconnect(connection)
                except Exception:
                    pass
        self._event_connections = {}
        self._drag_state = None
        self._history.clear()
        self._remove_artists()
        self._figure = figure
        self._metadata = getattr(figure, "_lmas_metadata", {}) if figure is not None else {}
        self._cursor_states = {
            "A": {"index": None, "coordinates": {}},
            "B": {"index": None, "coordinates": {}},
        }

        if figure is None or not self._metadata.get("linked_view"):
            self._configure_axis_rows()
            self.status.setText("Open LMA data to use Precision Mode.")
            self._refresh_readouts()
            return
        values = self._metadata.get("precision_source_values", {})
        count = np.asarray(values.get("time", ())).size
        if count <= 0:
            self._configure_axis_rows()
            self.status.setText("The current figure has no Precision Mode source data.")
            self._refresh_readouts()
            return

        self._event_connections = {
            "press": figure.canvas.mpl_connect(
                "button_press_event", self._on_figure_press
            ),
            "motion": figure.canvas.mpl_connect(
                "motion_notify_event", self._on_figure_motion
            ),
            "release": figure.canvas.mpl_connect(
                "button_release_event", self._on_figure_release
            ),
        }
        ids = np.asarray(values.get("source_id", ()), dtype=np.int64)
        for name, state in previous_states.items():
            source_id = state.get("source_id")
            if source_id is not None:
                matches = np.flatnonzero(ids == int(source_id))
                if matches.size:
                    self._set_source_state(name, int(matches[0]), redraw=False)
            elif state.get("coordinates"):
                self._cursor_states[name] = {
                    "index": None,
                    "coordinates": dict(state["coordinates"]),
                }

        self._configure_axis_rows()
        available = set(self._available_dimensions())
        self._axis_values = {
            dimension: list(previous_axis_values.get(dimension, [None, None]))
            for dimension in available
        }
        self._create_artists()
        self._update_markers()
        self._update_axis_lines()
        self._refresh_readouts()
        self.status.setText(
            "Precision Mode ready. Click to place the active cursor; Shift+click places B."
        )

    def _state_source_id(self, state: dict[str, Any]) -> int | None:
        index = state.get("index")
        if index is None:
            return None
        ids = np.asarray(
            self._metadata.get("precision_source_values", {}).get("source_id", ()),
            dtype=np.int64,
        )
        idx = int(index)
        return int(ids[idx]) if 0 <= idx < ids.size else None

    def _remove_artists(self) -> None:
        for marker_pairs in self._markers.values():
            for marker, label in marker_pairs:
                for artist in (marker, label):
                    try:
                        artist.remove()
                    except Exception:
                        pass
        for dimensions in self._axis_lines.values():
            for line_sets in dimensions.values():
                for artist, _, _ in line_sets:
                    try:
                        artist.remove()
                    except Exception:
                        pass
        self._markers = {"A": [], "B": []}
        self._axis_lines = {}

    def _theme_styles(self) -> dict[str, str]:
        theme = str(self._metadata.get("theme", "dark")).casefold()
        if theme == "light":
            return {"A": "#087830", "B": "#9A5500"}
        return {"A": "#39FF14", "B": "#FFD166"}

    def _create_artists(self) -> None:
        axes = tuple(self._metadata.get("axis_order", ()))
        styles = self._theme_styles()
        for cursor_name in ("A", "B"):
            color = styles[cursor_name]
            pairs: list[tuple[Any, Any]] = []
            for axis in axes:
                marker, = axis.plot(
                    [],
                    [],
                    linestyle="None",
                    marker="+",
                    markersize=13,
                    markeredgewidth=1.1,
                    color=color,
                    zorder=250,
                    clip_on=True,
                    picker=7,
                )
                label = axis.annotate(
                    cursor_name,
                    (0.0, 0.0),
                    xytext=(5, 5),
                    textcoords="offset points",
                    color=color,
                    fontsize=9,
                    fontweight="bold",
                    zorder=251,
                    visible=False,
                )
                pairs.append((marker, label))
            self._markers[cursor_name] = pairs

        coordinate_names = tuple(self._metadata.get("coordinate_names", ()))
        for axis, names in zip(axes, coordinate_names):
            for orientation, display_name in zip(("x", "y"), names):
                dimension, _ = canonical_coordinate_name(display_name)
                self._axis_lines.setdefault(dimension, {0: [], 1: []})
                for line_number, color_key in ((0, "A"), (1, "B")):
                    kwargs = {
                        "color": styles[color_key],
                        "linewidth": 0.9,
                        "linestyle": (0, (5, 3)),
                        "alpha": 0.92,
                        "zorder": 245,
                        "visible": False,
                        "picker": 6,
                    }
                    artist = axis.axvline(0.0, **kwargs) if orientation == "x" else axis.axhline(0.0, **kwargs)
                    self._axis_lines[dimension][line_number].append(
                        (artist, str(display_name), orientation)
                    )

    def _toolbar_busy(self) -> bool:
        toolbar = getattr(self.figure_host, "_toolbar", None)
        mode = getattr(toolbar, "mode", "") if toolbar is not None else ""
        value = str(getattr(mode, "name", mode)).casefold()
        return "zoom" in value or "pan" in value

    @staticmethod
    def _event_modifiers(event) -> tuple[bool, bool]:
        key = str(getattr(event, "key", "") or "").casefold()
        shift = "shift" in key
        control = "control" in key or "ctrl" in key
        gui_event = getattr(event, "guiEvent", None)
        if gui_event is not None and hasattr(gui_event, "modifiers"):
            try:
                modifiers = gui_event.modifiers()
                shift = shift or bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
                control = control or bool(modifiers & Qt.KeyboardModifier.ControlModifier)
            except Exception:
                pass
        return shift, control

    def _capture_precision_state(self) -> dict[str, Any]:
        return {
            "cursors": {
                name: {
                    "source_id": self._state_source_id(state),
                    "coordinates": dict(state.get("coordinates", {})),
                }
                for name, state in self._cursor_states.items()
            },
            "axis_values": {
                dimension: list(values)
                for dimension, values in self._axis_values.items()
            },
            "axis_enabled": {
                slot: bool(row["enabled"].isChecked())
                for slot, row in self._axis_rows.items()
            },
        }

    def _push_history(self) -> None:
        if self._restoring_state:
            return
        snapshot = self._capture_precision_state()
        if not self._history or self._history[-1] != snapshot:
            self._history.append(snapshot)
            if len(self._history) > 100:
                del self._history[0]

    def _restore_precision_state(self, snapshot: dict[str, Any]) -> None:
        self._restoring_state = True
        try:
            ids = np.asarray(
                self._metadata.get("precision_source_values", {}).get("source_id", ()),
                dtype=np.int64,
            )
            restored: dict[str, dict[str, Any]] = {}
            for name in ("A", "B"):
                saved = dict(snapshot.get("cursors", {}).get(name, {}))
                source_id = saved.get("source_id")
                coordinates = dict(saved.get("coordinates", {}))
                index = None
                if source_id is not None:
                    matches = np.flatnonzero(ids == int(source_id))
                    if matches.size:
                        index = int(matches[0])
                        coordinates = self._source_coordinates(index)
                restored[name] = {"index": index, "coordinates": coordinates}
            self._cursor_states = restored
            self._axis_values = {
                str(dimension): list(values)
                for dimension, values in dict(snapshot.get("axis_values", {})).items()
            }
            enabled = dict(snapshot.get("axis_enabled", {}))
            for slot, row in self._axis_rows.items():
                checkbox: QCheckBox = row["enabled"]
                checkbox.blockSignals(True)
                checkbox.setChecked(bool(enabled.get(slot, False)) and checkbox.isEnabled())
                checkbox.blockSignals(False)
        finally:
            self._restoring_state = False
        self._update_markers()
        self._update_axis_lines()
        self._refresh_readouts()

    def undo_last_action(self) -> None:
        if not self._history:
            self.status.setText("No Precision Mode cursor action is available to undo.")
            return
        self._restore_precision_state(self._history.pop())
        self.status.setText("Last Precision Mode cursor action undone.")

    def _axis_index_for_event(self, event) -> int | None:
        if event.inaxes is None:
            return None
        axes = tuple(self._metadata.get("axis_order", ()))
        try:
            return axes.index(event.inaxes)
        except ValueError:
            return None

    def _hit_test_drag(self, event, axis_index: int) -> dict[str, Any] | None:
        if event.x is None or event.y is None:
            return None
        coordinate_names = tuple(self._metadata.get("coordinate_names", ()))
        axes = tuple(self._metadata.get("axis_order", ()))
        if axis_index >= len(coordinate_names) or axis_index >= len(axes):
            return None
        axis = axes[axis_index]
        candidates: list[tuple[float, dict[str, Any]]] = []

        point_order = (self._active_cursor, "B" if self._active_cursor == "A" else "A")
        for cursor_name in point_order:
            state = self._cursor_states[cursor_name]
            if state.get("index") is not None:
                continue
            coordinates = state.get("coordinates", {})
            display_values: list[float] = []
            for display_name in coordinate_names[axis_index]:
                dimension, _ = canonical_coordinate_name(display_name)
                value = coordinates.get(dimension)
                if value is None or not math.isfinite(float(value)):
                    display_values = []
                    break
                display_values.append(from_canonical_coordinate(display_name, float(value)))
            if len(display_values) == 2:
                px, py = axis.transData.transform(display_values)
                distance = math.hypot(float(event.x) - px, float(event.y) - py)
                if distance <= 8.0:
                    candidates.append((distance, {
                        "kind": "point",
                        "cursor": cursor_name,
                        "axis_index": axis_index,
                    }))

        for orientation, display_name in zip(("x", "y"), coordinate_names[axis_index]):
            dimension, _ = canonical_coordinate_name(display_name)
            if not self._dimension_enabled(dimension):
                continue
            values = self._axis_values.get(dimension, [None, None])
            for line_number, value in enumerate(values[:2]):
                if value is None or not math.isfinite(float(value)):
                    continue
                display_value = from_canonical_coordinate(display_name, float(value))
                if orientation == "x":
                    pixel = axis.transData.transform((display_value, axis.get_ylim()[0]))[0]
                    distance = abs(float(event.x) - pixel)
                else:
                    pixel = axis.transData.transform((axis.get_xlim()[0], display_value))[1]
                    distance = abs(float(event.y) - pixel)
                if distance <= 6.0:
                    candidates.append((distance, {
                        "kind": "axis",
                        "dimension": dimension,
                        "display_name": str(display_name),
                        "orientation": orientation,
                        "line_number": line_number,
                        "axis_index": axis_index,
                    }))
        return min(candidates, key=lambda item: item[0])[1] if candidates else None

    def _on_figure_press(self, event) -> None:
        if not self.isVisible() or event.button != 1 or event.inaxes is None:
            return
        if not self.figure_host.precision_mode_active or self._toolbar_busy():
            return
        axis_index = self._axis_index_for_event(event)
        if axis_index is None:
            return
        shift, control = self._event_modifiers(event)
        if not control:
            target = self._hit_test_drag(event, axis_index)
            if target is not None:
                target.update({
                    "start_x": float(event.x),
                    "start_y": float(event.y),
                    "moved": False,
                })
                self._drag_state = target
                return
        self._apply_figure_click(event, axis_index, shift=shift, control=control)

    def _apply_figure_click(
        self, event, axis_index: int, *, shift: bool, control: bool
    ) -> None:
        if control:
            if event.xdata is None or event.ydata is None:
                return
            self._place_axis_cursor(
                axis_index, 1 if shift else 0, event.xdata, event.ydata
            )
            return
        cursor_name = "B" if shift else self._active_cursor
        if self.snap_mode.currentData() == "free":
            if event.xdata is None or event.ydata is None:
                return
            self._place_free_cursor(cursor_name, axis_index, event.xdata, event.ydata)
            return
        if event.x is None or event.y is None:
            return
        index = self._nearest_source_index(axis_index, float(event.x), float(event.y))
        if index is None:
            self.status.setText("No eligible source is available for that click.")
            return
        self.set_cursor_index(cursor_name, index)
        self.status.setText(
            f"Cursor {cursor_name} snapped to source {self._source(index).source_id}."
        )

    def _on_figure_motion(self, event) -> None:
        drag = self._drag_state
        if drag is None or event.x is None or event.y is None:
            return
        if self._axis_index_for_event(event) != drag.get("axis_index"):
            return
        if event.xdata is None or event.ydata is None:
            return
        if not drag.get("moved"):
            distance = math.hypot(
                float(event.x) - float(drag["start_x"]),
                float(event.y) - float(drag["start_y"]),
            )
            if distance < 3.0:
                return
            self._push_history()
            drag["moved"] = True

        if drag["kind"] == "point":
            cursor_name = str(drag["cursor"])
            coordinate_names = tuple(self._metadata.get("coordinate_names", ()))
            coordinates = dict(self._cursor_states[cursor_name].get("coordinates", {}))
            for display_name, value in zip(
                coordinate_names[int(drag["axis_index"])],
                (event.xdata, event.ydata),
            ):
                dimension, canonical_value = to_canonical_coordinate(
                    display_name, float(value)
                )
                coordinates[dimension] = canonical_value
            self._cursor_states[cursor_name] = {
                "index": None,
                "coordinates": coordinates,
            }
            self._update_markers()
            self._refresh_readouts()
        else:
            raw_value = event.xdata if drag["orientation"] == "x" else event.ydata
            _, canonical_value = to_canonical_coordinate(
                drag["display_name"], float(raw_value)
            )
            self._axis_values.setdefault(
                str(drag["dimension"]), [None, None]
            )[int(drag["line_number"])] = canonical_value
            self._update_axis_lines()
            self._refresh_axis_readouts()

    def _on_figure_release(self, event) -> None:
        drag = self._drag_state
        self._drag_state = None
        if drag is None or not drag.get("moved"):
            return
        if drag["kind"] == "point":
            self.status.setText(f"Cursor {drag['cursor']} moved freely.")
        else:
            label = self._dimension_label(str(drag["dimension"]))
            self.status.setText(
                f"{label} axis line {int(drag['line_number']) + 1} moved."
            )

    def _place_free_cursor(
        self, cursor_name: str, axis_index: int, x_value: float, y_value: float
    ) -> None:
        coordinate_names = tuple(self._metadata.get("coordinate_names", ()))
        if axis_index >= len(coordinate_names):
            return
        self._push_history()
        state = self._cursor_states[cursor_name]
        coordinates = dict(state.get("coordinates", {}))
        for display_name, value in zip(coordinate_names[axis_index], (x_value, y_value)):
            dimension, canonical_value = to_canonical_coordinate(display_name, float(value))
            coordinates[dimension] = canonical_value
        self._cursor_states[cursor_name] = {
            "index": None,
            "coordinates": coordinates,
        }
        self._update_markers()
        self._refresh_readouts()
        self.status.setText(
            f"Cursor {cursor_name} placed freely on the selected panel."
        )

    def _place_axis_cursor(
        self, axis_index: int, line_number: int, x_value: float, y_value: float
    ) -> None:
        coordinate_names = tuple(self._metadata.get("coordinate_names", ()))
        if axis_index >= len(coordinate_names):
            return
        self._push_history()
        changed: list[str] = []
        for display_name, value in zip(coordinate_names[axis_index], (x_value, y_value)):
            dimension, canonical_value = to_canonical_coordinate(display_name, float(value))
            self._axis_values.setdefault(dimension, [None, None])[line_number] = canonical_value
            slot = self._slot_for_dimension(dimension)
            if slot is not None:
                checkbox = self._axis_rows[slot]["enabled"]
                checkbox.blockSignals(True)
                checkbox.setChecked(True)
                checkbox.blockSignals(False)
            changed.append(self._dimension_label(dimension))
        self._update_axis_lines()
        self._refresh_axis_readouts()
        self.status.setText(
            f"Axis line {line_number + 1} placed for " + " and ".join(changed) + "."
        )

    def _candidate_indices(self) -> np.ndarray:
        values = self._metadata.get("precision_source_values", {})
        count = np.asarray(values.get("time", ())).size
        if count <= 0:
            return np.array([], dtype=np.int64)
        if self.snap_mode.currentData() == "visible":
            mask = np.asarray(self._metadata.get("visible_mask", ()), dtype=bool)
            if mask.size == count:
                return np.flatnonzero(mask)
        return np.arange(count, dtype=np.int64)

    def _nearest_source_index(
        self, axis_index: int, click_x: float, click_y: float
    ) -> int | None:
        coordinate_pairs = tuple(self._metadata.get("coordinate_pairs", ()))
        axes = tuple(self._metadata.get("axis_order", ()))
        if axis_index >= len(coordinate_pairs) or axis_index >= len(axes):
            return None
        candidates = self._candidate_indices()
        if not candidates.size:
            return None
        x_values = np.asarray(coordinate_pairs[axis_index][0], dtype=float)
        y_values = np.asarray(coordinate_pairs[axis_index][1], dtype=float)
        valid = (
            (candidates < x_values.size)
            & (candidates < y_values.size)
            & np.isfinite(x_values[candidates])
            & np.isfinite(y_values[candidates])
        )
        candidates = candidates[valid]
        if not candidates.size:
            return None
        points = np.column_stack((x_values[candidates], y_values[candidates]))
        display = axes[axis_index].transData.transform(points)
        distance2 = (display[:, 0] - click_x) ** 2 + (display[:, 1] - click_y) ** 2
        finite = np.isfinite(distance2)
        if not np.any(finite):
            return None
        local = np.flatnonzero(finite)[int(np.argmin(distance2[finite]))]
        return int(candidates[local])

    def select_cursor(self, cursor_name: str) -> None:
        name = "B" if str(cursor_name).upper() == "B" else "A"
        self._active_cursor = name
        if self.isVisible():
            self.figure_host.activate_precision_mode()
        self.cursor_a.blockSignals(True)
        self.cursor_b.blockSignals(True)
        self.cursor_a.setChecked(name == "A")
        self.cursor_b.setChecked(name == "B")
        self.cursor_a.blockSignals(False)
        self.cursor_b.blockSignals(False)
        self.status.setText(
            f"Cursor {name} is active. Click to place it; Shift+click always places B."
        )

    def _source_coordinates(self, index: int) -> dict[str, float]:
        values = self._metadata.get("precision_source_values", {})
        idx = int(index)
        mapping = {
            "time": "time_num",
            "altitude": "altitude_km",
            "east": "east_km",
            "north": "north_km",
            "longitude": "longitude",
            "latitude": "latitude",
        }
        result: dict[str, float] = {}
        for dimension, key in mapping.items():
            array = np.asarray(values.get(key, ()), dtype=float)
            if 0 <= idx < array.size and math.isfinite(float(array[idx])):
                result[dimension] = float(array[idx])
        return result

    def _set_source_state(self, cursor_name: str, index: int, *, redraw: bool = True) -> None:
        self._cursor_states[cursor_name] = {
            "index": int(index),
            "coordinates": self._source_coordinates(index),
        }
        if redraw:
            self._update_markers()
            self._refresh_readouts()

    def set_cursor_index(
        self, cursor_name: str, index: int | None, *, record_history: bool = True
    ) -> None:
        name = "B" if str(cursor_name).upper() == "B" else "A"
        values = self._metadata.get("precision_source_values", {})
        count = np.asarray(values.get("time", ())).size
        current = self._cursor_states[name].get("index")
        if index is None:
            if current is None and not self._cursor_states[name].get("coordinates"):
                return
            if record_history:
                self._push_history()
            self._cursor_states[name] = {"index": None, "coordinates": {}}
        else:
            idx = int(index)
            if idx < 0 or idx >= count or current == idx:
                return
            if record_history:
                self._push_history()
            self._set_source_state(name, idx, redraw=False)
        self._update_markers()
        self._refresh_readouts()

    def step_active(self, amount: int) -> None:
        candidates = self._candidate_indices()
        if not candidates.size:
            self.status.setText("No eligible sources are available to step through.")
            return
        values = self._metadata.get("precision_source_values", {})
        times = np.asarray(values.get("time", ())).astype("datetime64[ns]")
        order = candidates[np.argsort(times[candidates], kind="stable")]
        current = self._cursor_states[self._active_cursor].get("index")
        if current is None:
            position = 0 if amount >= 0 else len(order) - 1
        else:
            matches = np.flatnonzero(order == int(current))
            position = int(matches[0]) if matches.size else 0
            position = min(max(position + int(amount), 0), len(order) - 1)
        self.set_cursor_index(self._active_cursor, int(order[position]))
        self.status.setText(
            f"Cursor {self._active_cursor}: source {self._source(int(order[position])).source_id}."
        )

    def swap_cursors(self) -> None:
        self._push_history()
        self._cursor_states["A"], self._cursor_states["B"] = (
            self._cursor_states["B"],
            self._cursor_states["A"],
        )
        self._update_markers()
        self._refresh_readouts()
        self.status.setText("Cursor A and B were swapped.")

    def clear_active(self) -> None:
        self.set_cursor_index(self._active_cursor, None)
        self.status.setText(f"Cursor {self._active_cursor} cleared.")

    def clear_all(self) -> None:
        if not any(state.get("coordinates") for state in self._cursor_states.values()):
            return
        self._push_history()
        self._cursor_states = {
            "A": {"index": None, "coordinates": {}},
            "B": {"index": None, "coordinates": {}},
        }
        self._update_markers()
        self._refresh_readouts()
        self.status.setText("Both Precision Mode cursors were cleared.")

    def clear_axis_cursors(self) -> None:
        if not any(
            value is not None
            for values in self._axis_values.values()
            for value in values
        ):
            return
        self._push_history()
        for values in self._axis_values.values():
            values[:] = [None, None]
        self._update_axis_lines()
        self._refresh_axis_readouts()
        self.status.setText("All axis cursors were cleared.")

    def _source(self, index: int) -> PrecisionSource:
        return PrecisionSource.from_values(
            self._metadata.get("precision_source_values", {}), int(index)
        )

    @staticmethod
    def _number(value: float | None, unit: str = "", digits: int = 3) -> str:
        if value is None or not math.isfinite(float(value)):
            return "—"
        suffix = f" {unit}" if unit else ""
        return f"{float(value):.{digits}f}{suffix}"

    @staticmethod
    def _time_text_from_num(value: float | None) -> str:
        if value is None or not math.isfinite(float(value)):
            return "—"
        try:
            dt = mdates.num2date(float(value)).replace(tzinfo=None)
            return utc_text(np.datetime64(dt, "ns"))
        except Exception:
            return "—"

    def _source_texts(self, source: PrecisionSource) -> dict[str, str]:
        return {
            "utc": utc_text(source.time),
            "source_id": str(source.source_id),
            "local": f"{source.east_km:.3f} / {source.north_km:.3f} km",
            "latlon": f"{source.latitude:.6f} / {source.longitude:.6f}°",
            "altitude": f"{source.altitude_km:.3f} km MSL",
            "power": self._number(source.power_dbw, "dBW", 2),
            "chi2": self._number(source.chi2, "", 3),
            "stations": "—" if source.stations is None else str(source.stations),
        }

    def _free_texts(self, coordinates: dict[str, float]) -> dict[str, str]:
        east, north = coordinates.get("east"), coordinates.get("north")
        latitude, longitude = coordinates.get("latitude"), coordinates.get("longitude")
        altitude = coordinates.get("altitude")
        return {
            "utc": self._time_text_from_num(coordinates.get("time")),
            "source_id": "Free",
            "local": (
                f"{east:.3f} / {north:.3f} km"
                if east is not None and north is not None
                else "—"
            ),
            "latlon": (
                f"{latitude:.6f} / {longitude:.6f}°"
                if latitude is not None and longitude is not None
                else "—"
            ),
            "altitude": (
                f"{altitude:.3f} km MSL" if altitude is not None else "—"
            ),
            "power": "—",
            "chi2": "—",
            "stations": "—",
        }

    @staticmethod
    def _optional_signed(value: float | None, digits: int = 3) -> str:
        if value is None or not math.isfinite(float(value)):
            return "—"
        return f"{float(value):+.{digits}f}"

    def _difference_texts(self) -> dict[str, str]:
        a = self._cursor_states["A"]["coordinates"]
        b = self._cursor_states["B"]["coordinates"]
        if not a or not b:
            return {key: "—" for key, _ in _DIFFERENCE_FIELDS}
        result = precision_coordinate_difference(a, b)
        displacement = (
            f"{self._optional_signed(result.delta_east_km)} / "
            f"{self._optional_signed(result.delta_north_km)} / "
            f"{self._optional_signed(result.delta_altitude_km)} km"
        )
        return {
            "dt": (
                "—"
                if result.delta_time_ms is None
                else f"{result.delta_time_ms:+.6f} ms"
            ),
            "displacement": displacement,
            "horizontal": self._number(result.horizontal_distance_km, "km", 3),
            "distance_3d": self._number(result.distance_3d_km, "km", 3),
            "bearing": (
                "—"
                if result.bearing_deg is None
                else f"{result.bearing_deg:.2f}° clockwise from north"
            ),
            "speed_horizontal": self._speed_text(
                result.apparent_horizontal_speed_km_s
            ),
            "speed_3d": self._speed_text(result.apparent_3d_speed_km_s),
        }

    def _refresh_readouts(self) -> None:
        for name in ("A", "B"):
            state = self._cursor_states[name]
            index = state.get("index")
            texts = (
                self._source_texts(self._source(int(index)))
                if index is not None
                else self._free_texts(state.get("coordinates", {}))
            )
            for key, _ in _SOURCE_FIELDS:
                self._source_labels[name][key].setText(texts.get(key, "—"))
        difference = self._difference_texts()
        for key, label in self._difference_labels.items():
            label.setText(difference[key])
        self._refresh_axis_readouts()

    @staticmethod
    def _speed_text(value_km_s: float | None) -> str:
        return format_apparent_speed(value_km_s)

    def _update_markers(self, *_args) -> None:
        coordinate_names = tuple(self._metadata.get("coordinate_names", ()))
        labels_enabled = self.show_cursor_labels.isChecked()
        for cursor_name in ("A", "B"):
            coordinates = self._cursor_states[cursor_name].get("coordinates", {})
            marker_pairs = self._markers.get(cursor_name, ())
            for axis_index, (marker, label) in enumerate(marker_pairs):
                if (
                    not self._display_enabled
                    or axis_index >= len(coordinate_names)
                    or not coordinates
                ):
                    marker.set_data([], [])
                    label.set_visible(False)
                    continue
                display_values: list[float] = []
                for display_name in coordinate_names[axis_index]:
                    dimension, _ = canonical_coordinate_name(display_name)
                    value = coordinates.get(dimension)
                    if value is None or not math.isfinite(float(value)):
                        display_values = []
                        break
                    display_values.append(
                        from_canonical_coordinate(display_name, float(value))
                    )
                if len(display_values) != 2:
                    marker.set_data([], [])
                    label.set_visible(False)
                    continue
                x_value, y_value = display_values
                marker.set_data([x_value], [y_value])
                label.xy = (x_value, y_value)
                label.set_visible(bool(labels_enabled))
        if self._figure is not None:
            self._figure.canvas.draw_idle()

    def _available_dimensions(self) -> tuple[str, ...]:
        found: list[str] = []
        for names in self._metadata.get("coordinate_names", ()):
            for display_name in names:
                dimension, _ = canonical_coordinate_name(display_name)
                if dimension not in found:
                    found.append(dimension)
        return tuple(found)

    def _configure_axis_rows(self) -> None:
        available = set(self._available_dimensions())
        horizontal = [
            dim
            for dim in ("east", "longitude", "north", "latitude")
            if dim in available
        ]
        self._slot_dimensions = {
            "time": "time" if "time" in available else None,
            "x": horizontal[0] if horizontal else None,
            "y": horizontal[1] if len(horizontal) > 1 else None,
            "altitude": "altitude" if "altitude" in available else None,
        }
        for slot, row in self._axis_rows.items():
            dimension = self._slot_dimensions.get(slot)
            checkbox: QCheckBox = row["enabled"]
            checkbox.blockSignals(True)
            checkbox.setEnabled(dimension is not None)
            checkbox.setText(self._dimension_label(dimension) if dimension else slot.title())
            for key in ("first", "second"):
                row[key].setEnabled(dimension is not None)
            if dimension is None:
                checkbox.setChecked(False)
            checkbox.blockSignals(False)

    def _slot_for_dimension(self, dimension: str) -> str | None:
        for slot, value in self._slot_dimensions.items():
            if value == dimension:
                return slot
        return None

    @staticmethod
    def _dimension_label(dimension: str | None) -> str:
        labels = {
            "time": "Time",
            "east": "East–west",
            "north": "North–south",
            "longitude": "Longitude",
            "latitude": "Latitude",
            "altitude": "Altitude",
        }
        return labels.get(str(dimension), str(dimension).title())

    def _axis_enabled_changed(self, slot: str, _checked: bool) -> None:
        self._update_axis_lines()
        self._refresh_axis_readouts()

    def _dimension_enabled(self, dimension: str) -> bool:
        slot = self._slot_for_dimension(dimension)
        return bool(slot is not None and self._axis_rows[slot]["enabled"].isChecked())

    def _update_axis_lines(self) -> None:
        for dimension, line_sets in self._axis_lines.items():
            values = self._axis_values.get(dimension, [None, None])
            enabled = self._display_enabled and self._dimension_enabled(dimension)
            for line_number in (0, 1):
                value = values[line_number] if line_number < len(values) else None
                visible = enabled and value is not None and math.isfinite(float(value))
                for artist, display_name, orientation in line_sets[line_number]:
                    if visible:
                        display_value = from_canonical_coordinate(
                            display_name, float(value)
                        )
                        if orientation == "x":
                            artist.set_xdata([display_value, display_value])
                        else:
                            artist.set_ydata([display_value, display_value])
                    artist.set_visible(bool(visible))
        if self._figure is not None:
            self._figure.canvas.draw_idle()

    def _time_origin_num(self) -> float | None:
        values = np.asarray(
            self._metadata.get("precision_source_values", {}).get("time_num", ()),
            dtype=float,
        )
        finite = values[np.isfinite(values)]
        return float(np.min(finite)) if finite.size else None

    def _format_axis_entry_value(
        self, dimension: str, value: float | None
    ) -> str:
        if value is None or not math.isfinite(float(value)):
            return ""
        if dimension == "time":
            if self.time_entry_mode.currentData() == "relative_ms":
                origin = self._time_origin_num()
                if origin is None:
                    return ""
                return f"{(float(value) - origin) * 86_400_000.0:.6f}"
            return self._time_text_from_num(value).replace(" UTC", "")
        if dimension in {"longitude", "latitude"}:
            return f"{float(value):.6f}"
        return f"{float(value):.3f}"

    def _parse_axis_entry(self, dimension: str, text: str) -> float | None:
        cleaned = str(text).strip()
        if not cleaned or cleaned == "—":
            return None
        if dimension == "time":
            if self.time_entry_mode.currentData() == "relative_ms":
                origin = self._time_origin_num()
                if origin is None:
                    raise ValueError("No source time is available as a relative-time origin")
                numeric = cleaned.casefold().replace("milliseconds", "").replace("ms", "").strip()
                value_ms = float(numeric)
                if not math.isfinite(value_ms):
                    raise ValueError("Relative time must be finite")
                return origin + value_ms / 86_400_000.0

            utc_value = cleaned.strip()
            if utc_value.upper().endswith("UTC"):
                utc_value = utc_value[:-3].strip()
            if utc_value.endswith(("Z", "z")):
                utc_value = utc_value[:-1].strip()
            if "T" not in utc_value and " " not in utc_value:
                origin = self._time_origin_num()
                if origin is None:
                    raise ValueError("A full UTC date is required")
                anchor = mdates.num2date(origin).replace(tzinfo=None).date().isoformat()
                utc_value = f"{anchor}T{utc_value}"
            else:
                utc_value = utc_value.replace(" ", "T", 1)
            timestamp = np.datetime64(utc_value, "ns")
            if np.isnat(timestamp):
                raise ValueError("UTC time is not valid")
            return float(
                mdates.date2num(timestamp.astype("datetime64[us]").astype(object))
            )

        numeric = (
            cleaned.casefold()
            .replace("kilometres", "")
            .replace("kilometers", "")
            .replace("km", "")
            .replace("degrees", "")
            .replace("degree", "")
            .replace("deg", "")
            .replace("°", "")
            .strip()
        )
        value = float(numeric)
        if not math.isfinite(value):
            raise ValueError("Coordinate must be finite")
        return value

    def _commit_axis_entry(self, slot: str, line_number: int) -> None:
        dimension = self._slot_dimensions.get(slot)
        if dimension is None or self._restoring_state:
            return
        field: QLineEdit = self._axis_rows[slot][
            "first" if line_number == 0 else "second"
        ]
        old_value = self._axis_values.get(dimension, [None, None])[line_number]
        try:
            new_value = self._parse_axis_entry(dimension, field.text())
        except (TypeError, ValueError, OverflowError) as exc:
            self.status.setText(
                f"Could not set {self._dimension_label(dimension)} line "
                f"{line_number + 1}: {exc}."
            )
            self._refresh_axis_readouts()
            return
        if new_value == old_value or (
            new_value is not None
            and old_value is not None
            and math.isclose(float(new_value), float(old_value), rel_tol=0.0, abs_tol=1e-12)
        ):
            self._refresh_axis_readouts()
            return
        self._push_history()
        self._axis_values.setdefault(dimension, [None, None])[line_number] = new_value
        checkbox: QCheckBox = self._axis_rows[slot]["enabled"]
        checkbox.blockSignals(True)
        checkbox.setChecked(new_value is not None or checkbox.isChecked())
        checkbox.blockSignals(False)
        self._update_axis_lines()
        self._refresh_axis_readouts()
        action = "cleared" if new_value is None else "set"
        self.status.setText(
            f"{self._dimension_label(dimension)} axis line {line_number + 1} {action}."
        )

    def set_axis_lines_from_ab(self) -> None:
        a = self._cursor_states["A"].get("coordinates", {})
        b = self._cursor_states["B"].get("coordinates", {})
        changed: list[str] = []
        pending: list[tuple[str, float, float, str]] = []
        for slot in _AXIS_SLOT_ORDER:
            dimension = self._slot_dimensions.get(slot)
            if dimension is None:
                continue
            first, second = a.get(dimension), b.get(dimension)
            if first is None or second is None:
                continue
            if not (math.isfinite(float(first)) and math.isfinite(float(second))):
                continue
            pending.append((dimension, float(first), float(second), slot))
        if not pending:
            self.status.setText(
                "A and B do not share coordinates that can populate the current axis pairs."
            )
            return
        self._push_history()
        for dimension, first, second, slot in pending:
            self._axis_values[dimension] = [first, second]
            checkbox: QCheckBox = self._axis_rows[slot]["enabled"]
            checkbox.blockSignals(True)
            checkbox.setChecked(True)
            checkbox.blockSignals(False)
            changed.append(self._dimension_label(dimension))
        self._update_axis_lines()
        self._refresh_axis_readouts()
        self.status.setText("Axis lines set from A/B for " + ", ".join(changed) + ".")

    def apply_ab_time_interval(self) -> None:
        a_time = self._cursor_states["A"].get("coordinates", {}).get("time")
        b_time = self._cursor_states["B"].get("coordinates", {}).get("time")
        if a_time is None or b_time is None:
            self.status.setText("Both A and B need time coordinates before applying an interval.")
            return
        low, high = sorted((float(a_time), float(b_time)))
        if not (math.isfinite(low) and math.isfinite(high)) or high <= low:
            self.status.setText("The A–B time interval must have a nonzero finite duration.")
            return
        applied = self.figure_host.set_interactive_limits(
            {"time": (low, high)}, initialize_all_matching_axes=True
        )
        if applied:
            self.status.setText(
                f"Applied A–B time interval ({(high - low) * 86_400_000.0:.6f} ms)."
            )
        else:
            self.status.setText(
                "The A–B time interval could not be applied to the current linked view."
            )

    def _format_axis_value(self, dimension: str, value: float | None) -> str:
        if value is None or not math.isfinite(float(value)):
            return "—"
        if dimension == "time":
            return self._time_text_from_num(value).replace(" UTC", "")
        if dimension in {"longitude", "latitude"}:
            return f"{float(value):.6f}°"
        return f"{float(value):.3f} km"

    def _format_axis_delta(
        self, dimension: str, first: float | None, second: float | None
    ) -> str:
        if first is None or second is None:
            return "—"
        delta = float(second) - float(first)
        if dimension == "time":
            return f"{delta * 86_400_000.0:+.6f} ms"
        if dimension in {"longitude", "latitude"}:
            return f"{delta:+.6f}°"
        return f"{delta:+.3f} km"

    def _refresh_axis_readouts(self, *_args) -> None:
        for slot, row in self._axis_rows.items():
            dimension = self._slot_dimensions.get(slot)
            values = self._axis_values.get(dimension, [None, None]) if dimension else [None, None]
            first, second = values[0], values[1]
            for key, value in (("first", first), ("second", second)):
                field: QLineEdit = row[key]
                if not field.hasFocus():
                    field.setText(
                        self._format_axis_entry_value(dimension, value)
                        if dimension
                        else ""
                    )
            row["delta"].setText(
                self._format_axis_delta(dimension, first, second)
                if dimension
                else "—"
            )

    def measurement_text(self) -> str:
        lines = ["LMAS Precision Mode measurements"]
        for name in ("A", "B"):
            state = self._cursor_states[name]
            index = state.get("index")
            lines.append("")
            lines.append(f"Cursor {name}")
            if index is not None:
                texts = self._source_texts(self._source(int(index)))
            elif state.get("coordinates"):
                texts = self._free_texts(state["coordinates"])
            else:
                lines.append("  not set")
                continue
            for key, label in _SOURCE_FIELDS:
                lines.append(f"  {label}: {texts[key]}")
        difference = self._difference_texts()
        lines.extend(["", "A -> B"])
        for key, label in _DIFFERENCE_FIELDS:
            lines.append(f"  {label}: {difference[key]}")
        enabled_axis_rows = []
        for slot in _AXIS_SLOT_ORDER:
            dimension = self._slot_dimensions.get(slot)
            if dimension and self._dimension_enabled(dimension):
                values = self._axis_values.get(dimension, [None, None])
                enabled_axis_rows.append(
                    (
                        self._dimension_label(dimension),
                        self._format_axis_value(dimension, values[0]),
                        self._format_axis_value(dimension, values[1]),
                        self._format_axis_delta(dimension, values[0], values[1]),
                    )
                )
        if enabled_axis_rows:
            lines.extend(["", "Axis cursors"])
            for label, first, second, delta in enabled_axis_rows:
                lines.append(f"  {label}: line 1={first}; line 2={second}; difference={delta}")
        return "\n".join(lines)

    def copy_measurements(self) -> None:
        QApplication.clipboard().setText(self.measurement_text())
        self.status.setText("Precision Mode measurements copied to the clipboard.")

    def activate_for_placement(self) -> None:
        self.figure_host.activate_precision_mode()
        self._display_enabled = True
        self.show()
        self._update_markers()
        self._update_axis_lines()
        self.raise_()
        self.activateWindow()
        self.status.setText(
            f"Cursor {self._active_cursor} is active. Click to place it; Shift+click places B."
        )

    def hide_precision_mode(self) -> None:
        self.figure_host.deactivate_precision_mode()
        self._display_enabled = False
        self._update_markers()
        self._update_axis_lines()
        self.hide()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        event.ignore()
        self.hide_precision_mode()


__all__ = ["PrecisionModeWindow"]
