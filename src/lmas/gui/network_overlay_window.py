from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable

import matplotlib.dates as mdates
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QSettings, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..overlays.network import (
    NetworkCSVOptions,
    NetworkOverlayManager,
    NetworkOverlayStyle,
    export_network_csv,
    export_network_netcdf,
)
from ..plotting.common import apply_figure_theme, theme_values
from ..plotting.time_axis import configure_utc_time_axis
from .data_dialogs import choose_directory_with_files_visible, choose_existing_files
from .icon import application_icon


class NetworkOverlayWindow(QMainWindow):
    """Compact workspace for ground lightning-location-network overlays."""

    overlays_changed = Signal()

    def __init__(
        self,
        manager: NetworkOverlayManager,
        figure_host,
        project_getter: Callable[[], object | None],
        parent=None,
    ) -> None:
        super().__init__(None)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.setWindowFlag(Qt.WindowType.Tool, False)
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setWindowIcon(application_icon())
        self.setWindowTitle("Network Overlays")
        self.resize(940, 860)
        self.setMinimumSize(820, 680)
        self.manager = manager
        self.figure_host = figure_host
        self.project_getter = project_getter
        self._settings = QSettings("LMAS", "NetworkOverlays")
        self._updating = False
        self._item_keys: dict[int, str] = {}
        self._restored_geometry = False
        self.peak_figure = Figure(figsize=(5.4, 2.6), dpi=100, constrained_layout=True)
        self.peak_canvas = FigureCanvasQTAgg(self.peak_figure)
        self.peak_canvas.setMinimumHeight(220)
        self._colors = {
            "positive_color": "#ef5350",
            "negative_color": "#2196f3",
            "intracloud_color": "#ffca28",
            "unknown_color": "#b0bec5",
            "ellipse_color": "auto",
        }

        self.setCentralWidget(self._build_page())
        geometry = self._settings.value("geometry_dev1")
        if geometry is not None:
            self._restored_geometry = bool(self.restoreGeometry(geometry))
        main_sizes = self._settings.value("main_splitter_dev1")
        if isinstance(main_sizes, (list, tuple)) and main_sizes:
            self.main_splitter.setSizes([int(value) for value in main_sizes])
        left_sizes = self._settings.value("left_splitter_dev1")
        if isinstance(left_sizes, (list, tuple)) and left_sizes:
            self.left_splitter.setSizes([int(value) for value in left_sizes])
        self._refresh_tree()
        self.refresh_diagnostics(force=True)

    def _build_page(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(7)

        intro = QLabel(
            "Overlay reports from ground-based lightning-location networks. "
            "LMAS auto-detects common ENTLN-style and generic CSV columns while "
            "keeping each loaded network dataset independent."
        )
        intro.setWordWrap(True)
        outer.addWidget(intro)

        buttons = QHBoxLayout()
        add_files = QPushButton("Add CSV files")
        add_files.clicked.connect(self._add_files)
        add_directory = QPushButton("Add directory")
        add_directory.clicked.connect(self._add_directory)
        remove = QPushButton("Remove selected")
        remove.clicked.connect(self._remove_selected)
        clear = QPushButton("Clear")
        clear.clicked.connect(self._clear)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self._emit_changed)
        for button in (add_files, add_directory, remove, clear, refresh):
            buttons.addWidget(button)
        buttons.addStretch(1)
        outer.addLayout(buttons)

        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("File or directory"))
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Paste or type a network CSV file or directory path")
        self.path_edit.returnPressed.connect(self._load_entered_path)
        load_path = QPushButton("Load")
        load_path.clicked.connect(self._load_entered_path)
        path_row.addWidget(self.path_edit, 1)
        path_row.addWidget(load_path)
        outer.addLayout(path_row)

        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.left_splitter = QSplitter(Qt.Orientation.Vertical)

        data_panel = QWidget()
        data_layout = QVBoxLayout(data_panel)
        data_layout.setContentsMargins(0, 0, 0, 0)
        self.dataset_tree = QTreeWidget()
        self.dataset_tree.setHeaderLabels(
            ("On", "Network", "Product", "Coverage UTC", "Events", "CG", "IC", "Ellipses")
        )
        self.dataset_tree.setRootIsDecorated(False)
        self.dataset_tree.setAlternatingRowColors(True)
        self.dataset_tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.dataset_tree.itemSelectionChanged.connect(self._selected_dataset_changed)
        self.dataset_tree.itemChanged.connect(self._dataset_item_changed)
        data_layout.addWidget(self.dataset_tree, 1)

        import_row = QHBoxLayout()
        import_row.addWidget(QLabel("New-file provider"))
        self.provider_combo = QComboBox()
        self.provider_combo.addItem("Auto detect", "auto")
        self.provider_combo.addItem("ENTLN / Earth Networks", "entln")
        self.provider_combo.addItem("NLDN", "nldn")
        self.provider_combo.addItem("GLD360", "gld360")
        self.provider_combo.addItem("Generic CSV", "generic")
        import_row.addWidget(self.provider_combo, 1)
        data_layout.addLayout(import_row)
        self.left_splitter.addWidget(data_panel)

        peak_group = QGroupBox("Peak current in the current linked time view")
        peak_layout = QVBoxLayout(peak_group)
        peak_layout.setContentsMargins(6, 8, 6, 5)
        peak_layout.addWidget(self.peak_canvas, 1)
        self.left_splitter.addWidget(peak_group)

        diagnostics = QGroupBox("Current linked view")
        diagnostics_layout = QVBoxLayout(diagnostics)
        self.diagnostics_text = QTextEdit()
        self.diagnostics_text.setReadOnly(True)
        self.diagnostics_text.setMinimumHeight(180)
        diagnostics_layout.addWidget(self.diagnostics_text)
        export_row = QHBoxLayout()
        export_csv = QPushButton("Export selected CSV")
        export_csv.clicked.connect(self._export_selected_csv)
        export_nc = QPushButton("Export selected NetCDF")
        export_nc.clicked.connect(self._export_selected_netcdf)
        export_row.addWidget(export_csv)
        export_row.addWidget(export_nc)
        export_row.addStretch(1)
        diagnostics_layout.addLayout(export_row)
        self.left_splitter.addWidget(diagnostics)
        self.left_splitter.setStretchFactor(0, 3)
        self.left_splitter.setStretchFactor(1, 2)
        self.left_splitter.setStretchFactor(2, 2)
        self.left_splitter.setSizes([360, 260, 220])
        self.main_splitter.addWidget(self.left_splitter)

        controls = QWidget()
        right = QVBoxLayout(controls)
        right.setContentsMargins(4, 0, 4, 0)
        right.setSpacing(7)

        layers = QGroupBox("Layers — all enabled datasets")
        layer_layout = QVBoxLayout(layers)
        self.show_events = QCheckBox("Network events")
        self.show_uncertainty = QCheckBox("Location uncertainty ellipses")
        self.show_time_rail = QCheckBox("Network event time rail")
        self.show_legend = QCheckBox("Legend entries")
        for widget in (self.show_events, self.show_uncertainty, self.show_time_rail, self.show_legend):
            widget.toggled.connect(self._global_layers_changed)
            layer_layout.addWidget(widget)
        right.addWidget(layers)

        filters = QGroupBox("Selected dataset filters")
        filter_layout = QVBoxLayout(filters)
        type_row = QHBoxLayout()
        self.show_cg = QCheckBox("CG")
        self.show_ic = QCheckBox("IC")
        self.show_other = QCheckBox("Other")
        for widget in (self.show_cg, self.show_ic, self.show_other):
            widget.toggled.connect(self._selected_style_changed)
            type_row.addWidget(widget)
        type_row.addStretch(1)
        filter_layout.addLayout(type_row)
        polarity_row = QHBoxLayout()
        self.show_negative = QCheckBox("Negative")
        self.show_positive = QCheckBox("Positive")
        self.show_unknown_polarity = QCheckBox("Unknown")
        for widget in (self.show_negative, self.show_positive, self.show_unknown_polarity):
            widget.toggled.connect(self._selected_style_changed)
            polarity_row.addWidget(widget)
        polarity_row.addStretch(1)
        filter_layout.addLayout(polarity_row)
        form = QFormLayout()
        self.minimum_current = QDoubleSpinBox()
        self.minimum_current.setRange(-1.0, 10000.0)
        self.minimum_current.setSpecialValueText("Any")
        self.minimum_current.setSuffix(" kA")
        self.minimum_current.setDecimals(1)
        self.minimum_current.valueChanged.connect(self._selected_style_changed)
        self.minimum_sensors = QSpinBox()
        self.minimum_sensors.setRange(-1, 10000)
        self.minimum_sensors.setSpecialValueText("Any")
        self.minimum_sensors.valueChanged.connect(self._selected_style_changed)
        self.follow_spatial = QCheckBox("Limit events and time rail to the current map view")
        self.follow_spatial.toggled.connect(self._selected_style_changed)
        form.addRow("Minimum |peak current|", self.minimum_current)
        form.addRow("Minimum sensors", self.minimum_sensors)
        form.addRow(self.follow_spatial)
        filter_layout.addLayout(form)
        right.addWidget(filters)

        appearance = QGroupBox("Selected dataset appearance")
        appearance_form = QFormLayout(appearance)
        self.marker_size = QDoubleSpinBox()
        self.marker_size.setRange(1.0, 500.0)
        self.marker_size.setSingleStep(2.0)
        self.marker_size.valueChanged.connect(self._selected_style_changed)
        self.scale_current = QCheckBox("Scale markers by |peak current|")
        self.scale_current.toggled.connect(self._selected_style_changed)
        self.opacity = QDoubleSpinBox()
        self.opacity.setRange(0.0, 1.0)
        self.opacity.setSingleStep(0.05)
        self.opacity.setDecimals(2)
        self.opacity.valueChanged.connect(self._selected_style_changed)
        appearance_form.addRow("Marker size", self.marker_size)
        appearance_form.addRow(self.scale_current)
        appearance_form.addRow("Marker opacity", self.opacity)

        for key, label in (
            ("negative_color", "Negative CG"),
            ("positive_color", "Positive CG"),
            ("intracloud_color", "IC"),
            ("unknown_color", "Unknown/other"),
            ("ellipse_color", "Uncertainty ellipse"),
        ):
            button = QPushButton()
            button.setProperty("lmasColorKey", key)
            button.clicked.connect(lambda _checked=False, k=key: self._choose_color(k))
            setattr(self, f"{key}_button", button)
            appearance_form.addRow(label, button)

        self.ellipse_opacity = QDoubleSpinBox()
        self.ellipse_opacity.setRange(0.0, 1.0)
        self.ellipse_opacity.setSingleStep(0.05)
        self.ellipse_opacity.setDecimals(2)
        self.ellipse_opacity.valueChanged.connect(self._selected_style_changed)
        self.event_zorder = QDoubleSpinBox()
        self.event_zorder.setRange(-20.0, 100.0)
        self.event_zorder.setDecimals(2)
        self.event_zorder.setSingleStep(0.25)
        self.event_zorder.valueChanged.connect(self._selected_style_changed)
        self.ellipse_zorder = QDoubleSpinBox()
        self.ellipse_zorder.setRange(-20.0, 100.0)
        self.ellipse_zorder.setDecimals(2)
        self.ellipse_zorder.setSingleStep(0.25)
        self.ellipse_zorder.valueChanged.connect(self._selected_style_changed)
        self.time_rail_size = QDoubleSpinBox()
        self.time_rail_size.setRange(1.0, 500.0)
        self.time_rail_size.valueChanged.connect(self._selected_style_changed)
        self.maximum_events = QSpinBox()
        self.maximum_events.setRange(0, 1_000_000)
        self.maximum_events.setSpecialValueText("Unlimited")
        self.maximum_events.setSingleStep(500)
        self.maximum_events.valueChanged.connect(self._selected_style_changed)
        self.maximum_ellipses = QSpinBox()
        self.maximum_ellipses.setRange(0, 100_000)
        self.maximum_ellipses.setSpecialValueText("Unlimited")
        self.maximum_ellipses.setSingleStep(100)
        self.maximum_ellipses.valueChanged.connect(self._selected_style_changed)
        appearance_form.addRow("Ellipse opacity", self.ellipse_opacity)
        appearance_form.addRow("Event z-order", self.event_zorder)
        appearance_form.addRow("Ellipse z-order", self.ellipse_zorder)
        appearance_form.addRow("Time-rail marker size", self.time_rail_size)
        appearance_form.addRow("Interactive event limit", self.maximum_events)
        appearance_form.addRow("Interactive ellipse limit", self.maximum_ellipses)
        right.addWidget(appearance)

        details = QGroupBox("Selected dataset")
        details_layout = QVBoxLayout(details)
        self.dataset_details = QTextEdit()
        self.dataset_details.setReadOnly(True)
        self.dataset_details.setMinimumHeight(170)
        details_layout.addWidget(self.dataset_details)
        right.addWidget(details)
        right.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(controls)
        self.main_splitter.addWidget(scroll)
        self.main_splitter.setStretchFactor(0, 3)
        self.main_splitter.setStretchFactor(1, 2)
        self.main_splitter.setSizes([570, 370])
        outer.addWidget(self.main_splitter, 1)
        return page

    def _new_options(self) -> NetworkCSVOptions:
        provider = str(self.provider_combo.currentData() or "auto")
        return NetworkCSVOptions(provider=provider)

    def _browser_start(self, fallback: Path | str | None) -> Path:
        typed = Path(self.path_edit.text().strip().strip('"').strip("'")).expanduser() if self.path_edit.text().strip() else None
        if typed is not None and typed.exists():
            return typed if typed.is_dir() else typed.parent
        remembered = self._settings.value("last_data_directory")
        if remembered:
            candidate = Path(str(remembered)).expanduser()
            if candidate.exists():
                return candidate
        return Path(fallback or Path.home()).expanduser()

    def _add_files(self) -> None:
        project = self.project_getter()
        start = project.output_directory if project is not None else Path.home()
        selected = choose_existing_files(
            self,
            "Add ground-network CSV files",
            self._browser_start(start),
            ("CSV and text tables (*.csv *.txt)", "All files (*)"),
        )
        if selected:
            self._settings.setValue("last_data_directory", str(selected[0].parent))
            self.path_edit.setText(str(selected[0]) if len(selected) == 1 else str(selected[0].parent))
            self._load_paths(selected)

    def _add_directory(self) -> None:
        project = self.project_getter()
        start = project.output_directory if project is not None else Path.home()
        selected = choose_directory_with_files_visible(
            self,
            "Add ground-network CSV directory",
            self._browser_start(start),
            ("CSV and text tables (*.csv *.txt)", "All files (*)"),
        )
        if selected is None:
            return
        self._settings.setValue("last_data_directory", str(selected))
        self.path_edit.setText(str(selected))
        paths = self._csv_paths_from_entry(selected)
        if not paths:
            QMessageBox.information(self, "Network Overlays", "No CSV files were found.")
            return
        self._load_paths(paths)

    @staticmethod
    def _csv_paths_from_entry(path: Path) -> list[Path]:
        path = path.expanduser()
        if path.is_file():
            return [path.resolve()]
        if path.is_dir():
            return sorted(item.resolve() for item in path.glob("*.csv") if item.is_file())
        return []

    def _load_entered_path(self) -> None:
        raw = self.path_edit.text().strip().strip('"')
        if not raw:
            QMessageBox.information(self, "Network Overlays", "Enter a CSV file or directory path.")
            return
        path = Path(raw).expanduser()
        paths = self._csv_paths_from_entry(path)
        if not paths:
            QMessageBox.critical(
                self, "Could not load network data",
                f"The path does not identify a CSV file or a directory containing CSV files:\n{path}",
            )
            return
        self._load_paths(paths)

    def _load_paths(self, paths: list[Path]) -> None:
        try:
            record = self.manager.add_csv_paths(paths, options=self._new_options())
        except Exception as exc:
            QMessageBox.critical(self, "Could not load network data", str(exc))
            return
        self._refresh_tree(select_key=record.key)
        self._emit_changed()

    def _remove_selected(self) -> None:
        key = self._selected_key()
        if key is None:
            return
        self.manager.remove(key)
        self._refresh_tree()
        self._emit_changed()

    def _clear(self) -> None:
        self.manager.clear()
        self._refresh_tree()
        self._emit_changed()

    def _refresh_tree(self, *, select_key: str | None = None) -> None:
        self._updating = True
        try:
            self.dataset_tree.clear()
            self._item_keys.clear()
            selected = None
            for record in self.manager.records:
                events = record.observation.events
                cg = int(np.count_nonzero(np.char.upper(events.event_type.astype("U16")) == "CG"))
                ic = int(np.count_nonzero(np.char.upper(events.event_type.astype("U16")) == "IC"))
                ellipses = int(np.count_nonzero(
                    np.isfinite(events.ellipse_major_km) & np.isfinite(events.ellipse_minor_km)
                    & (events.ellipse_major_km > 0) & (events.ellipse_minor_km > 0)
                ))
                identity = record.observation.identity
                coverage = f"{str(identity.observation_start)[11:23]}–{str(identity.observation_end)[11:23]}"
                item = QTreeWidgetItem((
                    "", record.display_name, identity.product_name, coverage,
                    f"{len(events):,}", f"{cg:,}", f"{ic:,}", f"{ellipses:,}",
                ))
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(0, Qt.CheckState.Checked if record.style.enabled else Qt.CheckState.Unchecked)
                self.dataset_tree.addTopLevelItem(item)
                self._item_keys[id(item)] = record.key
                if record.key == select_key:
                    selected = item
            for column in range(self.dataset_tree.columnCount()):
                self.dataset_tree.resizeColumnToContents(column)
            if selected is None and self.dataset_tree.topLevelItemCount():
                selected = self.dataset_tree.topLevelItem(0)
            if selected is not None:
                self.dataset_tree.setCurrentItem(selected)
        finally:
            self._updating = False
        self._selected_dataset_changed()

    def _selected_key(self) -> str | None:
        items = self.dataset_tree.selectedItems()
        return None if not items else self._item_keys.get(id(items[0]))

    def _selected_dataset_changed(self) -> None:
        key = self._selected_key()
        self._updating = True
        try:
            global_style = self.manager.global_layer_state()
            has_data = self.manager.has_data
            for widget in (self.show_events, self.show_uncertainty, self.show_time_rail, self.show_legend):
                widget.setEnabled(has_data)
            self.show_events.setChecked(global_style.show_events)
            self.show_uncertainty.setChecked(global_style.show_uncertainty)
            self.show_time_rail.setChecked(global_style.show_time_rail)
            self.show_legend.setChecked(global_style.show_legend)
            selected_widgets = (
                self.show_cg, self.show_ic, self.show_other, self.show_negative,
                self.show_positive, self.show_unknown_polarity, self.minimum_current,
                self.minimum_sensors, self.follow_spatial, self.marker_size,
                self.scale_current, self.opacity, self.ellipse_opacity, self.event_zorder,
                self.ellipse_zorder, self.time_rail_size, self.maximum_events,
                self.maximum_ellipses,
            )
            for widget in selected_widgets:
                widget.setEnabled(key is not None)
            for color_key in self._colors:
                getattr(self, f"{color_key}_button").setEnabled(key is not None)
            if key is None:
                self.dataset_details.clear()
                self._refresh_peak_current_plot()
                return
            record = self.manager.record(key)
            if record.observation.identity.source_files:
                source_paths = [item.path for item in record.observation.identity.source_files]
                self.path_edit.setText(
                    str(source_paths[0]) if len(source_paths) == 1 else str(source_paths[0].parent)
                )
            style = record.style
            self.show_cg.setChecked(style.show_cg)
            self.show_ic.setChecked(style.show_ic)
            self.show_other.setChecked(style.show_other_types)
            self.show_negative.setChecked(style.show_negative)
            self.show_positive.setChecked(style.show_positive)
            self.show_unknown_polarity.setChecked(style.show_unknown_polarity)
            self.minimum_current.setValue(-1.0 if style.minimum_absolute_peak_current_ka is None else style.minimum_absolute_peak_current_ka)
            self.minimum_sensors.setValue(-1 if style.minimum_sensor_count is None else style.minimum_sensor_count)
            self.follow_spatial.setChecked(style.follow_spatial_view)
            self.marker_size.setValue(style.marker_size)
            self.scale_current.setChecked(style.scale_by_peak_current)
            self.opacity.setValue(style.marker_alpha)
            self.ellipse_opacity.setValue(style.ellipse_alpha)
            self.event_zorder.setValue(style.event_zorder)
            self.ellipse_zorder.setValue(style.ellipse_zorder)
            self.time_rail_size.setValue(style.time_rail_marker_size)
            self.maximum_events.setValue(style.maximum_interactive_events)
            self.maximum_ellipses.setValue(style.maximum_interactive_ellipses)
            for color_key in self._colors:
                self._colors[color_key] = str(getattr(style, color_key))
                self._update_color_button(color_key)
            identity = record.observation.identity
            schema = "\n".join(f"  {name}: {column}" for name, column in sorted(identity.schema.items()))
            files = "\n".join(f"  {item.path}" for item in identity.source_files)
            self.dataset_details.setPlainText(
                f"Network: {identity.display_name}\n"
                f"Provider: {identity.provider_id}\n"
                f"Product: {identity.product_name}\n"
                f"Events: {len(record.observation.events):,}\n"
                f"Coverage: {identity.observation_start} to {identity.observation_end}\n\n"
                f"Resolved columns:\n{schema or '  None'}\n\nSource files:\n{files}"
            )
        finally:
            self._updating = False
        self.refresh_diagnostics(force=True)

    def _dataset_item_changed(self, item, column: int) -> None:
        if self._updating or column != 0:
            return
        key = self._item_keys.get(id(item))
        if key is None:
            return
        record = self.manager.record(key)
        record.style = replace(record.style, enabled=item.checkState(0) == Qt.CheckState.Checked).validated()
        self._emit_changed()

    def _global_layers_changed(self) -> None:
        if self._updating:
            return
        self.manager.set_global_layer_state(
            show_events=self.show_events.isChecked(),
            show_uncertainty=self.show_uncertainty.isChecked(),
            show_time_rail=self.show_time_rail.isChecked(),
            show_legend=self.show_legend.isChecked(),
        )
        self._emit_changed()

    def _selected_style_changed(self) -> None:
        if self._updating:
            return
        key = self._selected_key()
        if key is None:
            return
        old = self.manager.record(key).style
        self.manager.record(key).style = NetworkOverlayStyle(
            enabled=old.enabled,
            show_events=old.show_events,
            show_uncertainty=old.show_uncertainty,
            show_time_rail=old.show_time_rail,
            show_legend=old.show_legend,
            follow_spatial_view=self.follow_spatial.isChecked(),
            marker_size=self.marker_size.value(),
            scale_by_peak_current=self.scale_current.isChecked(),
            marker_alpha=self.opacity.value(),
            marker_edge_width=old.marker_edge_width,
            positive_color=self._colors["positive_color"],
            negative_color=self._colors["negative_color"],
            intracloud_color=self._colors["intracloud_color"],
            unknown_color=self._colors["unknown_color"],
            ellipse_color=self._colors["ellipse_color"],
            ellipse_alpha=self.ellipse_opacity.value(),
            ellipse_line_width=old.ellipse_line_width,
            event_zorder=self.event_zorder.value(),
            ellipse_zorder=self.ellipse_zorder.value(),
            time_rail_zorder=old.time_rail_zorder,
            time_rail_marker_size=self.time_rail_size.value(),
            minimum_absolute_peak_current_ka=(None if self.minimum_current.value() < 0 else self.minimum_current.value()),
            minimum_sensor_count=(None if self.minimum_sensors.value() < 0 else self.minimum_sensors.value()),
            show_positive=self.show_positive.isChecked(),
            show_negative=self.show_negative.isChecked(),
            show_unknown_polarity=self.show_unknown_polarity.isChecked(),
            show_cg=self.show_cg.isChecked(),
            show_ic=self.show_ic.isChecked(),
            show_other_types=self.show_other.isChecked(),
            maximum_interactive_events=self.maximum_events.value(),
            maximum_interactive_ellipses=self.maximum_ellipses.value(),
        ).validated()
        self._emit_changed()

    def _choose_color(self, key: str) -> None:
        initial = self._colors[key]
        if initial.lower() == "auto":
            initial = self._colors["unknown_color"]
        color = QColorDialog.getColor(QColor(initial), self, "Network overlay color")
        if not color.isValid():
            return
        self._colors[key] = color.name(QColor.NameFormat.HexRgb)
        self._update_color_button(key)
        self._selected_style_changed()

    def _update_color_button(self, key: str) -> None:
        button = getattr(self, f"{key}_button")
        value = self._colors[key]
        if value.lower() == "auto":
            button.setText("Automatic")
            button.setStyleSheet("")
            return
        color = QColor(value)
        text = "black" if color.lightnessF() > 0.6 else "white"
        button.setText(value)
        button.setStyleSheet(f"background-color: {value}; color: {text};")

    def _visible_indices(self, record):
        """Return the exact scientific event set in the current linked view."""
        figure = getattr(self.figure_host, "figure", None)
        renderer = self.manager.renderer
        if renderer is None or figure is None:
            return np.arange(len(record.observation.events), dtype=np.int64)
        metadata = getattr(figure, "_lmas_metadata", {}) or {}
        axes = metadata.get("axes") or {}
        plan = axes.get("plan")
        time_axis = axes.get("time_altitude")
        time_range = renderer._time_range_ns(time_axis)
        style = record.style.validated()
        indices = record.observation.select(
            time_range_ns=time_range,
            minimum_absolute_peak_current_ka=style.minimum_absolute_peak_current_ka,
            minimum_sensor_count=style.minimum_sensor_count,
        ).event_indices
        if not indices.size:
            return indices
        plan_names = renderer._plan_coordinate_names(metadata, plan) if plan is not None else None
        if plan is not None and plan_names is not None:
            x, y = renderer._point_coordinates(
                record.observation.events.longitude_deg[indices],
                record.observation.events.latitude_deg[indices],
                x_name=plan_names[0], y_name=plan_names[1],
            )
            keep = np.isfinite(x) & np.isfinite(y)
            if style.follow_spatial_view:
                xlim = sorted(plan.get_xlim())
                ylim = sorted(plan.get_ylim())
                keep &= (x >= xlim[0]) & (x <= xlim[1]) & (y >= ylim[0]) & (y <= ylim[1])
            indices = indices[keep]
        categories = renderer._categories(record, indices)
        return indices[categories != "HIDDEN"]

    def _export_selected_csv(self) -> None:
        key = self._selected_key()
        if key is None:
            return
        record = self.manager.record(key)
        project = self.project_getter()
        start = project.output_directory if project is not None else Path.home()
        path, _ = QFileDialog.getSaveFileName(
            self, "Export normalized network CSV", str((start or Path.home()) / f"{record.display_name}_network.csv"),
            "CSV files (*.csv)",
        )
        if not path:
            return
        try:
            export_network_csv(record.observation, path, self._visible_indices(record))
        except Exception as exc:
            QMessageBox.critical(self, "Could not export network CSV", str(exc))

    def _export_selected_netcdf(self) -> None:
        key = self._selected_key()
        if key is None:
            return
        record = self.manager.record(key)
        project = self.project_getter()
        start = project.output_directory if project is not None else Path.home()
        path, _ = QFileDialog.getSaveFileName(
            self, "Export normalized network NetCDF", str((start or Path.home()) / f"{record.display_name}_network.nc"),
            "NetCDF files (*.nc)",
        )
        if not path:
            return
        try:
            export_network_netcdf(record.observation, path, self._visible_indices(record))
        except Exception as exc:
            QMessageBox.critical(self, "Could not export network NetCDF", str(exc))

    def _emit_changed(self) -> None:
        self.overlays_changed.emit()
        self.refresh_diagnostics(force=True)

    def bind_current_figure(self) -> None:
        self.refresh_diagnostics(force=True)

    def refresh_diagnostics(self, *, force: bool = False) -> None:
        lines: list[str] = []
        summaries = {item.dataset_key: item for item in getattr(self.manager.renderer, "summaries", ())}
        for record in self.manager.records:
            if not record.style.enabled:
                continue
            summary = summaries.get(record.key)
            events = record.observation.events
            current = events.peak_current_ka
            finite_current = current[np.isfinite(current)]
            current_text = "not available"
            if finite_current.size:
                current_text = f"{np.min(finite_current):+.1f} to {np.max(finite_current):+.1f} kA"
            if summary is None:
                visible_text = "awaiting linked refresh"
            else:
                visible_text = (
                    f"{summary.visible_events:,} visible; {summary.rendered_events:,} rendered"
                    + (" (interactive cap)" if summary.truncated else "")
                )
            lines.extend((
                record.display_name,
                f"  Loaded: {len(events):,} events",
                f"  Current view: {visible_text}",
                f"  Uncertainty ellipses: {0 if summary is None else summary.visible_ellipses:,}",
                f"  Peak current: {current_text}",
                "",
            ))
        if not lines:
            lines = ["No enabled ground-network datasets are loaded."]
        self.diagnostics_text.setPlainText("\n".join(lines).rstrip())
        self._refresh_peak_current_plot()

    def _refresh_peak_current_plot(self) -> None:
        self.peak_figure.clear()
        axis = self.peak_figure.add_subplot(111)
        metadata = getattr(getattr(self.figure_host, "figure", None), "_lmas_metadata", {}) or {}
        theme_name = str(metadata.get("theme", "dark"))
        colors = theme_values(theme_name)
        key = self._selected_key()
        message = None
        if key is None:
            message = "Select a network dataset"
        else:
            record = self.manager.record(key)
            indices = self._visible_indices(record)
            events = record.observation.events
            current = np.asarray(events.peak_current_ka[indices], dtype=float)
            valid = np.isfinite(current) & (np.asarray(events.time_ns[indices], dtype=np.int64) != np.iinfo(np.int64).min)
            indices = indices[valid]
            current = current[valid]
            if not indices.size:
                message = "No peak-current observations in the current linked view"
            else:
                times = np.asarray(events.time_ns[indices], dtype="datetime64[ns]")
                x = mdates.date2num(times.astype("datetime64[us]").astype(object))
                event_type = np.char.upper(np.asarray(events.event_type[indices]).astype("U24"))
                polarity = np.asarray(events.polarity[indices], dtype=np.int8)
                categories = (
                    (event_type == "CG") & (polarity < 0),
                    (event_type == "CG") & (polarity > 0),
                    event_type == "IC",
                    ~(((event_type == "CG") & (polarity != 0)) | (event_type == "IC")),
                )
                styles = (
                    (self._colors["negative_color"], "v", "Negative CG"),
                    (self._colors["positive_color"], "^", "Positive CG"),
                    (self._colors["intracloud_color"], "D", "IC"),
                    (self._colors["unknown_color"], "x", "Other/unknown"),
                )
                for keep, (color, marker, label) in zip(categories, styles, strict=False):
                    if np.any(keep):
                        axis.scatter(x[keep], current[keep], s=22, marker=marker,
                                     color=color, alpha=0.85, linewidths=0.65,
                                     label=label, zorder=3)
                axis.axhline(0.0, color=colors["text"], linewidth=0.75, alpha=0.55, zorder=1)
                axis.set_xlim(tuple(sorted((float(np.min(x)), float(np.max(x))))))
                configure_utc_time_axis(axis)
                axis.legend(loc="best", fontsize=7, framealpha=0.65, ncols=2)
        if message is not None:
            axis.text(0.5, 0.5, message, transform=axis.transAxes,
                      ha="center", va="center", color=colors["text"], fontsize=9)
            axis.set_xticks([])
            axis.set_yticks([])
        axis.set_ylabel("Peak current (kA)")
        axis.set_xlabel("Time (UTC)")
        apply_figure_theme(self.peak_figure, (axis,), theme_name, show_grid=True)
        self.peak_canvas.draw_idle()

    def position_next_to(self, main_window) -> None:
        if self._restored_geometry:
            return
        screen = main_window.screen()
        if screen is None:
            return
        main_frame = main_window.frameGeometry()
        current = screen.availableGeometry()
        regions = [item.availableGeometry() for item in QApplication.screens()]
        width = min(980, max(880, current.width() // 2))
        height = min(current.height() - 24, max(760, main_frame.height()))
        preferred_x = main_frame.right() + 10
        candidates = [region for region in regions if region.left() > main_frame.right() or preferred_x + width <= region.right() + 1]
        target = min(candidates, key=lambda region: abs(region.left() - preferred_x)) if candidates else current
        x = max(target.left(), preferred_x)
        if x + width > target.right() + 1:
            x = max(target.left(), main_frame.left() - width - 10)
        y = min(max(target.top(), main_frame.top()), target.bottom() - height + 1)
        self.resize(width, height)
        self.move(x, y)

    def closeEvent(self, event) -> None:
        self._settings.setValue("geometry_dev1", self.saveGeometry())
        self._settings.setValue("main_splitter_dev1", self.main_splitter.sizes())
        self._settings.setValue("left_splitter_dev1", self.left_splitter.sizes())
        self.hide()
        event.ignore()


__all__ = ["NetworkOverlayWindow"]
