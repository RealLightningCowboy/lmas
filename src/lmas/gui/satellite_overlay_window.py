from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib.dates as mdates
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.collections import PolyCollection
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
    QTabWidget,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..overlays.satellite import (
    GLMOverlayStyle, SatelliteOverlayManager, configure_group_energy_time_axis,
)
from ..overlays.satellite.colormaps import SATELLITE_COLORMAP_NAMES
from ..plotting.common import apply_figure_theme, theme_values
from .data_dialogs import choose_directory_with_files_visible, choose_existing_files
from .icon import application_icon

_ROLE_COLORS = {"east": "chartreuse", "west": "crimson"}


class SatelliteOverlayWindow(QMainWindow):
    """First user-facing satellite-lightning overlay workspace."""

    overlays_changed = Signal()

    def __init__(
        self,
        manager: SatelliteOverlayManager,
        figure_host,
        project_getter: Callable[[], object | None],
        parent=None,
    ) -> None:
        # Deliberately independent from the LMAS main window. A parented Qt
        # top-level stays transiently stacked above its parent on Windows; this
        # workspace should receive a normal taskbar entry and may move behind
        # the main scientific window like the other analysis workspaces.
        super().__init__(None)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.setWindowFlag(Qt.WindowType.Tool, False)
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setWindowIcon(application_icon())
        self.setWindowTitle("Satellite Overlays")
        self.resize(940, 960)
        self.setMinimumSize(820, 720)
        self.manager = manager
        self.figure_host = figure_host
        self.project_getter = project_getter
        self._updating_controls = False
        self._item_keys: dict[int, str] = {}
        self._diagnostic_signature = None
        self._settings = QSettings("LMAS", "SatelliteOverlays")
        self._group_color_value = "auto"
        self._restored_geometry = False

        tabs = QTabWidget(self)
        tabs.addTab(self._build_glm_tab(), "GLM")
        self.setCentralWidget(tabs)
        geometry = self._settings.value("geometry_rc1")
        if geometry is not None:
            self._restored_geometry = bool(self.restoreGeometry(geometry))
        splitter_sizes = self._settings.value("main_splitter_rc1")
        if isinstance(splitter_sizes, (list, tuple)) and splitter_sizes:
            self.main_splitter.setSizes([int(value) for value in splitter_sizes])
        left_sizes = self._settings.value("left_splitter_rc1")
        if isinstance(left_sizes, (list, tuple)) and left_sizes:
            self.left_splitter.setSizes([int(value) for value in left_sizes])
        self._refresh_dataset_tree()
        self.refresh_diagnostics()

    def _build_glm_tab(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(7)
        intro = QLabel(
            "Load GOES-R GLM L2 LCFA files. LMAS keeps each spacecraft as an "
            "independent dataset and identifies its historical East/West role."
        )
        intro.setWordWrap(True)
        outer.addWidget(intro)

        data_buttons = QHBoxLayout()
        add_files = QPushButton("Add files")
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
            data_buttons.addWidget(button)
        data_buttons.addStretch(1)
        outer.addLayout(data_buttons)

        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("File or directory"))
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Paste or type a GLM LCFA NetCDF file or directory path")
        self.path_edit.returnPressed.connect(self._load_entered_path)
        load_path = QPushButton("Load")
        load_path.clicked.connect(self._load_entered_path)
        path_row.addWidget(self.path_edit, 1)
        path_row.addWidget(load_path)
        outer.addLayout(path_row)

        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter = main_splitter

        left_splitter = QSplitter(Qt.Orientation.Vertical)
        self.left_splitter = left_splitter
        data_panel = QWidget()
        data_layout = QVBoxLayout(data_panel)
        data_layout.setContentsMargins(0, 0, 0, 0)
        reader_row = QHBoxLayout()
        reader_row.addWidget(QLabel("GLM reader"))
        self.backend_combo = QComboBox()
        self.backend_combo.addItem("Auto (prefer LMAS native)", "auto")
        self.backend_combo.addItem("LMAS native", "native")
        self.backend_combo.addItem("glmtools", "glmtools")
        backend_index = self.backend_combo.findData(self.manager.glm_backend)
        self.backend_combo.setCurrentIndex(max(0, backend_index))
        self.backend_combo.currentIndexChanged.connect(self._reader_backend_changed)
        reader_row.addWidget(self.backend_combo, 1)
        reload_button = QPushButton("Reload")
        reload_button.setToolTip("Reload all currently listed GLM files with the selected reader")
        reload_button.clicked.connect(self._reload_all_with_backend)
        reader_row.addWidget(reload_button)
        data_layout.addLayout(reader_row)

        self.dataset_tree = QTreeWidget()
        self.dataset_tree.setHeaderLabels(
            ("On", "Spacecraft", "Position", "Coverage UTC", "Events", "Groups", "Flashes")
        )
        self.dataset_tree.setRootIsDecorated(False)
        self.dataset_tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.dataset_tree.itemSelectionChanged.connect(self._selected_dataset_changed)
        self.dataset_tree.itemChanged.connect(self._dataset_item_changed)
        data_layout.addWidget(self.dataset_tree, 1)
        self.shared_scale = QCheckBox("Share event-energy scale across enabled GLM datasets")
        self.shared_scale.setChecked(self.manager.shared_energy_scale)
        self.shared_scale.toggled.connect(self._shared_scale_changed)
        data_layout.addWidget(self.shared_scale)
        left_splitter.addWidget(data_panel)

        diagnostic_group = QGroupBox("GLM group energy in the current LMAS time view")
        diagnostic_layout = QVBoxLayout(diagnostic_group)
        self.energy_figure = Figure(figsize=(7.0, 4.2), dpi=100, constrained_layout=True)
        self.energy_canvas = FigureCanvasQTAgg(self.energy_figure)
        self.energy_canvas.setMinimumHeight(260)
        diagnostic_layout.addWidget(self.energy_canvas)
        left_splitter.addWidget(diagnostic_group)
        left_splitter.setStretchFactor(0, 1)
        left_splitter.setStretchFactor(1, 3)
        left_splitter.setSizes([230, 560])
        main_splitter.addWidget(left_splitter)

        control_page = QWidget()
        right_layout = QVBoxLayout(control_page)
        right_layout.setContentsMargins(4, 0, 4, 0)

        layers = QGroupBox("GLM layers — all enabled spacecraft")
        layer_form = QFormLayout(layers)
        self.show_footprints = QCheckBox("Event footprints")
        self.show_groups = QCheckBox("GLM group centroids")
        self.show_flashes = QCheckBox("GLM flash centroids")
        self.show_maximum = QCheckBox("Emphasize highest-energy visible GLM group")
        self.show_colorbar = QCheckBox("Bottom GLM total optical energy colorbar")
        self.show_time_rail = QCheckBox("GLM group time rail")
        self.show_time_labels = QCheckBox("Label East/West time-rail tracks")
        for widget in (
            self.show_footprints, self.show_groups, self.show_flashes,
            self.show_maximum, self.show_colorbar, self.show_time_rail,
            self.show_time_labels,
        ):
            widget.toggled.connect(self._global_layer_controls_changed)
            layer_form.addRow(widget)
        right_layout.addWidget(layers)

        appearance = QGroupBox("Selected spacecraft appearance")
        form = QFormLayout(appearance)
        self.colormap = QComboBox()
        self.colormap.addItems(SATELLITE_COLORMAP_NAMES)
        self.colormap.currentTextChanged.connect(self._selected_style_controls_changed)

        self.opacity = QDoubleSpinBox()
        self.opacity.setRange(0.0, 1.0)
        self.opacity.setSingleStep(0.05)
        self.opacity.setDecimals(2)
        self.opacity.valueChanged.connect(self._selected_style_controls_changed)

        self.footprint_padding = QDoubleSpinBox()
        self.footprint_padding.setRange(0.0, 100.0)
        self.footprint_padding.setSingleStep(5.0)
        self.footprint_padding.setDecimals(0)
        self.footprint_padding.setSuffix(" %")
        self.footprint_padding.setToolTip(
            "Expand the event-center selection beyond the visible axes before "
            "polygon clipping. This prevents partial edge holes while zoomed in."
        )
        self.footprint_padding.valueChanged.connect(self._selected_style_controls_changed)

        self.marker_size = QDoubleSpinBox()
        self.marker_size.setRange(1.0, 500.0)
        self.marker_size.setSingleStep(2.0)
        self.marker_size.valueChanged.connect(self._selected_style_controls_changed)

        color_widget = QWidget()
        color_layout = QHBoxLayout(color_widget)
        color_layout.setContentsMargins(0, 0, 0, 0)
        self.group_color_button = QPushButton("Automatic East/West")
        self.group_color_button.clicked.connect(self._choose_group_marker_color)
        reset_color = QPushButton("Auto")
        reset_color.clicked.connect(self._reset_group_marker_color)
        color_layout.addWidget(self.group_color_button, 1)
        color_layout.addWidget(reset_color)

        self.footprint_zorder = QDoubleSpinBox()
        self.footprint_zorder.setRange(-20.0, 100.0)
        self.footprint_zorder.setDecimals(2)
        self.footprint_zorder.setSingleStep(0.25)
        self.footprint_zorder.setToolTip(
            "LMA source collections are near z-order 1. Values below 1 place "
            "GLM event footprints behind LMA."
        )
        self.footprint_zorder.valueChanged.connect(self._selected_style_controls_changed)

        self.group_zorder = QDoubleSpinBox()
        self.group_zorder.setRange(-20.0, 100.0)
        self.group_zorder.setDecimals(2)
        self.group_zorder.setSingleStep(0.25)
        self.group_zorder.valueChanged.connect(self._selected_style_controls_changed)

        self.time_rail_marker_size = QDoubleSpinBox()
        self.time_rail_marker_size.setRange(1.0, 500.0)
        self.time_rail_marker_size.setSingleStep(2.0)
        self.time_rail_marker_size.valueChanged.connect(self._selected_style_controls_changed)

        self.time_rail_zorder = QDoubleSpinBox()
        self.time_rail_zorder.setRange(-20.0, 100.0)
        self.time_rail_zorder.setDecimals(2)
        self.time_rail_zorder.setSingleStep(0.25)
        self.time_rail_zorder.valueChanged.connect(self._selected_style_controls_changed)

        self.maximum_events = QSpinBox()
        self.maximum_events.setRange(0, 1_000_000)
        self.maximum_events.setSingleStep(500)
        self.maximum_events.setSpecialValueText("Unlimited")
        self.maximum_events.valueChanged.connect(self._selected_style_controls_changed)

        form.addRow("Event colormap", self.colormap)
        form.addRow("Footprint opacity", self.opacity)
        form.addRow("Footprint render padding", self.footprint_padding)
        form.addRow("GLM group marker size", self.marker_size)
        form.addRow("GLM group marker color", color_widget)
        form.addRow("Event-footprint z-order", self.footprint_zorder)
        form.addRow("GLM group-centroid z-order", self.group_zorder)
        form.addRow("Time-rail marker size", self.time_rail_marker_size)
        form.addRow("Time-rail z-order", self.time_rail_zorder)
        form.addRow("Interactive event limit", self.maximum_events)
        right_layout.addWidget(appearance)

        details_group = QGroupBox("Selected dataset")
        details_layout = QVBoxLayout(details_group)
        self.dataset_details = QTextEdit()
        self.dataset_details.setReadOnly(True)
        self.dataset_details.setMinimumHeight(190)
        details_layout.addWidget(self.dataset_details)
        right_layout.addWidget(details_group)
        right_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(control_page)
        main_splitter.addWidget(scroll)
        main_splitter.setStretchFactor(0, 3)
        main_splitter.setStretchFactor(1, 2)
        main_splitter.setSizes([560, 380])
        outer.addWidget(main_splitter, 1)
        return page

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
            "Add GLM L2 LCFA files",
            self._browser_start(start),
            ("GLM NetCDF files (*.nc *.netcdf)", "All files (*)"),
        )
        if selected:
            self._settings.setValue("last_data_directory", str(selected[0].parent))
            self.path_edit.setText(str(selected[0]) if len(selected) == 1 else str(selected[0].parent))
            self._load_paths(selected)

    def _add_directory(self) -> None:
        project = self.project_getter()
        start = project.output_directory if project is not None else Path.home()
        directory = choose_directory_with_files_visible(
            self,
            "Add GLM directory",
            self._browser_start(start),
            ("GLM NetCDF files (*.nc *.netcdf)", "All files (*)"),
        )
        if directory is None:
            return
        self._settings.setValue("last_data_directory", str(directory))
        self.path_edit.setText(str(directory))
        paths = self._glm_paths_from_entry(directory)
        if not paths:
            QMessageBox.information(self, "Satellite Overlays", "No GLM LCFA NetCDF files were found.")
            return
        self._load_paths(paths)

    @staticmethod
    def _glm_paths_from_entry(path: Path) -> list[Path]:
        if path.is_file():
            return [path] if path.suffix.lower() in {".nc", ".netcdf"} else []
        if not path.is_dir():
            return []
        patterns = (
            "*GLM-L2-LCFA*_G??_*.nc",
            "*GLM-L2-LCFA*.nc",
            "*GLM-L2-LCFA*.netcdf",
        )
        found: dict[Path, None] = {}
        for pattern in patterns:
            for candidate in sorted(path.glob(pattern)):
                if candidate.is_file():
                    found[candidate] = None
        return list(found)

    def _load_entered_path(self) -> None:
        raw = self.path_edit.text().strip().strip('"').strip("'")
        if not raw:
            QMessageBox.information(self, "Satellite Overlays", "Enter a GLM file or directory path.")
            return
        path = Path(raw).expanduser()
        paths = self._glm_paths_from_entry(path)
        if not paths:
            QMessageBox.critical(
                self,
                "Could not load GLM data",
                f"The path does not identify a GLM LCFA NetCDF file or a directory containing GLM files:\n{path}",
            )
            return
        self._load_paths(paths)

    def _load_paths(self, paths: list[Path]) -> None:
        try:
            records = self.manager.add_glm_paths(paths)
        except Exception as exc:
            QMessageBox.critical(self, "Could not load GLM data", str(exc))
            return
        self._refresh_dataset_tree(select_key=records[-1].key if records else None)
        self._emit_changed()

    def _remove_selected(self) -> None:
        key = self._selected_key()
        if key is None:
            return
        self.manager.remove(key)
        self._refresh_dataset_tree()
        self._emit_changed()

    def _clear(self) -> None:
        self.manager.clear()
        self._refresh_dataset_tree()
        self._emit_changed()

    def _refresh_dataset_tree(self, *, select_key: str | None = None) -> None:
        self._updating_controls = True
        try:
            self.dataset_tree.clear()
            self._item_keys.clear()
            selected_item = None
            for record in self.manager.records:
                identity = record.observation.identity
                coverage = (
                    f"{str(identity.observation_start)[11:23]}–"
                    f"{str(identity.observation_end)[11:23]}"
                )
                item = QTreeWidgetItem(
                    (
                        "",
                        identity.spacecraft_name,
                        identity.position_name,
                        coverage,
                        f"{len(record.observation.events):,}",
                        f"{len(record.observation.groups):,}",
                        f"{len(record.observation.flashes):,}",
                    )
                )
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(
                    0,
                    Qt.CheckState.Checked if record.style.enabled else Qt.CheckState.Unchecked,
                )
                self.dataset_tree.addTopLevelItem(item)
                self._item_keys[id(item)] = record.key
                if record.key == select_key:
                    selected_item = item
            for column in range(self.dataset_tree.columnCount()):
                self.dataset_tree.resizeColumnToContents(column)
            if selected_item is None and self.dataset_tree.topLevelItemCount():
                selected_item = self.dataset_tree.topLevelItem(0)
            if selected_item is not None:
                self.dataset_tree.setCurrentItem(selected_item)
        finally:
            self._updating_controls = False
        self._selected_dataset_changed()

    def _selected_key(self) -> str | None:
        items = self.dataset_tree.selectedItems()
        if not items:
            return None
        return self._item_keys.get(id(items[0]))

    def _selected_dataset_changed(self) -> None:
        key = self._selected_key()
        self._updating_controls = True
        try:
            has_data = self.manager.has_data
            for widget in (
                self.show_footprints, self.show_groups, self.show_flashes,
                self.show_maximum, self.show_colorbar, self.show_time_rail,
                self.show_time_labels,
            ):
                widget.setEnabled(has_data)
            global_style = self.manager.global_layer_state()
            self.show_footprints.setChecked(global_style.show_event_footprints)
            self.show_groups.setChecked(global_style.show_group_centroids)
            self.show_flashes.setChecked(global_style.show_flash_centroids)
            self.show_maximum.setChecked(global_style.show_maximum_group)
            self.show_colorbar.setChecked(global_style.show_colorbar)
            self.show_time_rail.setChecked(global_style.show_group_time_rail)
            self.show_time_labels.setChecked(global_style.show_time_rail_labels)

            selected_widgets = (
                self.colormap, self.opacity, self.footprint_padding, self.marker_size,
                self.group_color_button, self.footprint_zorder, self.group_zorder,
                self.time_rail_marker_size, self.time_rail_zorder, self.maximum_events,
            )
            enabled = key is not None
            for widget in selected_widgets:
                widget.setEnabled(enabled)
            if not enabled:
                self.dataset_details.clear()
                self._group_color_value = "auto"
                self._update_group_color_button()
                return
            record = self.manager.record(key)
            style = record.style.validated()
            self.colormap.setCurrentText(style.colormap)
            self.opacity.setValue(style.footprint_alpha)
            self.footprint_padding.setValue(style.footprint_render_padding_fraction * 100.0)
            self.marker_size.setValue(style.group_marker_size)
            self._group_color_value = style.group_marker_color
            self._update_group_color_button()
            self.footprint_zorder.setValue(style.footprint_zorder)
            self.group_zorder.setValue(style.group_zorder)
            self.time_rail_marker_size.setValue(style.time_rail_marker_size)
            self.time_rail_zorder.setValue(style.time_rail_zorder)
            self.maximum_events.setValue(style.maximum_interactive_events)
            identity = record.observation.identity
            if identity.source_files:
                source_paths = [item.path for item in identity.source_files]
                self.path_edit.setText(
                    str(source_paths[0]) if len(source_paths) == 1 else str(source_paths[0].parent)
                )
            projection = identity.projection
            backend = str(identity.attributes.get("lmas_reader_backend", "unknown"))
            self.dataset_details.setPlainText(
                "\n".join(
                    (
                        identity.display_name,
                        f"Reader backend: {backend}",
                        f"Platform: {identity.platform_id}",
                        f"Operational role: {identity.operational_role} ({identity.operational_role_source})",
                        f"Product: {identity.product_level}",
                        f"Coverage: {identity.observation_start} to {identity.observation_end}",
                        f"Projection FOV longitude: {projection.field_of_view_lon_deg}°",
                        f"Nominal subpoint longitude: {projection.nominal_subpoint_lon_deg}°",
                        f"Source files: {len(identity.source_files)}",
                    )
                )
            )
        finally:
            self._updating_controls = False

    def _dataset_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._updating_controls or column != 0:
            return
        key = self._item_keys.get(id(item))
        if key is None:
            return
        self.manager.record(key).style.enabled = item.checkState(0) == Qt.CheckState.Checked
        self._emit_changed()

    def _shared_scale_changed(self, checked: bool) -> None:
        if self._updating_controls:
            return
        self.manager.shared_energy_scale = bool(checked)
        self._emit_changed()

    def _global_layer_controls_changed(self, *_args) -> None:
        if self._updating_controls:
            return
        self.manager.set_global_layer_state(
            show_event_footprints=self.show_footprints.isChecked(),
            show_group_centroids=self.show_groups.isChecked(),
            show_flash_centroids=self.show_flashes.isChecked(),
            show_maximum_group=self.show_maximum.isChecked(),
            show_colorbar=self.show_colorbar.isChecked(),
            show_group_time_rail=self.show_time_rail.isChecked(),
            show_time_rail_labels=self.show_time_labels.isChecked(),
        )
        self._emit_changed()

    def _selected_style_controls_changed(self, *_args) -> None:
        if self._updating_controls:
            return
        key = self._selected_key()
        if key is None:
            return
        old = self.manager.record(key).style
        self.manager.record(key).style = GLMOverlayStyle(
            enabled=old.enabled,
            show_event_footprints=old.show_event_footprints,
            show_group_centroids=old.show_group_centroids,
            show_flash_centroids=old.show_flash_centroids,
            show_maximum_group=old.show_maximum_group,
            show_colorbar=old.show_colorbar,
            show_group_time_rail=old.show_group_time_rail,
            show_time_rail_labels=old.show_time_rail_labels,
            colormap=self.colormap.currentText(),
            logarithmic_energy=True,
            footprint_alpha=self.opacity.value(),
            footprint_edge_width=old.footprint_edge_width,
            group_marker_size=self.marker_size.value(),
            group_marker_color=self._group_color_value,
            group_edge_width=old.group_edge_width,
            maximum_group_size=old.maximum_group_size,
            maximum_group_color=old.maximum_group_color,
            footprint_zorder=self.footprint_zorder.value(),
            group_zorder=self.group_zorder.value(),
            time_rail_marker_size=self.time_rail_marker_size.value(),
            time_rail_zorder=self.time_rail_zorder.value(),
            maximum_interactive_events=self.maximum_events.value(),
            footprint_render_padding_fraction=self.footprint_padding.value() / 100.0,
        ).validated()
        self._emit_changed()

    def _reader_backend_changed(self, _index: int) -> None:
        if self._updating_controls:
            return
        backend = str(self.backend_combo.currentData() or "auto")
        self.manager.glm_backend = backend

    def _reload_all_with_backend(self) -> None:
        paths = [path for record in self.manager.records for path in record.source_paths]
        if not paths:
            return
        styles_by_platform = {
            record.observation.identity.platform_id: record.style
            for record in self.manager.records
        }
        try:
            replacement = SatelliteOverlayManager()
            replacement.glm_backend = self.manager.glm_backend
            replacement.shared_energy_scale = self.manager.shared_energy_scale
            records = replacement.add_glm_paths(paths)
            for record in records:
                style = styles_by_platform.get(record.observation.identity.platform_id)
                if style is not None:
                    record.style = style
        except Exception as exc:
            QMessageBox.critical(self, "Could not reload GLM data", str(exc))
            return
        self.manager.clear()
        self.manager._records.update(replacement._records)
        self._refresh_dataset_tree(select_key=records[0].key if records else None)
        self._emit_changed()

    def _choose_group_marker_color(self) -> None:
        key = self._selected_key()
        if key is None:
            return
        role = self.manager.record(key).observation.identity.operational_role.lower()
        initial = _ROLE_COLORS.get(role, "deepskyblue")
        if self._group_color_value.lower() != "auto":
            initial = self._group_color_value
        color = QColorDialog.getColor(QColor(initial), self, "GLM group marker color")
        if not color.isValid():
            return
        self._group_color_value = color.name(QColor.NameFormat.HexRgb)
        self._update_group_color_button()
        self._selected_style_controls_changed()

    def _reset_group_marker_color(self) -> None:
        if self._selected_key() is None:
            return
        self._group_color_value = "auto"
        self._update_group_color_button()
        self._selected_style_controls_changed()

    def _update_group_color_button(self) -> None:
        value = str(self._group_color_value or "auto")
        if value.lower() == "auto":
            self.group_color_button.setText("Automatic East/West")
            self.group_color_button.setStyleSheet("")
            return
        color = QColor(value)
        text = "black" if color.lightnessF() > 0.6 else "white"
        self.group_color_button.setText(value)
        self.group_color_button.setStyleSheet(
            f"background-color: {value}; color: {text};"
        )

    def _emit_changed(self) -> None:
        self.refresh_diagnostics()
        self.overlays_changed.emit()

    def bind_current_figure(self) -> None:
        self.refresh_diagnostics()

    def refresh_diagnostics(self, *, force: bool = False) -> None:
        project = self.project_getter()
        theme_name = getattr(getattr(project, "plot", None), "theme", "space")
        limits = None
        figure = getattr(self.figure_host, "figure", None)
        metadata = getattr(figure, "_lmas_metadata", {}) if figure is not None else {}
        time_axis = (metadata.get("axes") or {}).get("time_altitude") if isinstance(metadata, dict) else None
        if time_axis is not None:
            limits = tuple(float(value) for value in sorted(time_axis.get_xlim()))
        signature = (
            theme_name,
            None if limits is None else tuple(round(value, 12) for value in limits),
            tuple(
                (record.key, bool(record.style.enabled), len(record.observation.groups))
                for record in self.manager.records
            ),
        )
        if not force and signature == self._diagnostic_signature:
            return
        self._diagnostic_signature = signature

        self.energy_figure.clear()
        axis = self.energy_figure.add_subplot(111)
        series = []
        for record in self.manager.records:
            if not record.style.enabled:
                continue
            groups = record.observation.groups
            times_num = mdates.date2num(
                groups.time_ns.astype("datetime64[ns]").astype("datetime64[us]").astype(object)
            )
            energy_fj = np.asarray(groups.energy_j, dtype=float) * 1.0e15
            keep = np.isfinite(times_num) & np.isfinite(energy_fj) & (energy_fj > 0)
            if limits is not None:
                keep &= (times_num >= limits[0]) & (times_num <= limits[1])
            if np.any(keep):
                series.append((record, times_num[keep], energy_fj[keep]))

        if series:
            all_energy = np.concatenate([item[2] for item in series])
            baseline = max(float(np.min(all_energy)) * 0.50, np.finfo(float).tiny)
            high = float(np.max(all_energy))
            width_days = 1.65 / 86_400_000.0
            half_width = width_days / 2.0
            for record, times_num, energy_fj in series:
                vertices = np.empty((times_num.size, 4, 2), dtype=float)
                vertices[:, 0, 0] = times_num - half_width
                vertices[:, 1, 0] = times_num - half_width
                vertices[:, 2, 0] = times_num + half_width
                vertices[:, 3, 0] = times_num + half_width
                vertices[:, 0, 1] = baseline
                vertices[:, 1, 1] = energy_fj
                vertices[:, 2, 1] = energy_fj
                vertices[:, 3, 1] = baseline
                color = _ROLE_COLORS.get(
                    record.observation.identity.operational_role.lower(), "deepskyblue"
                )
                collection = PolyCollection(
                    vertices,
                    facecolors=color,
                    edgecolors="none",
                    linewidths=0.0,
                    alpha=0.88,
                    label=record.display_name,
                    rasterized=times_num.size > 5000,
                )
                axis.add_collection(collection, autolim=False)
            axis.set_ylim(baseline, high * 1.25)
            axis.legend(loc="upper right")
        else:
            axis.text(
                0.5, 0.5,
                "No enabled GLM groups in the current time view",
                ha="center", va="center", transform=axis.transAxes,
            )

        axis.set_yscale("log")
        axis.set_ylabel("GLM group energy (fJ)")
        axis.set_xlabel("Time (UTC)")
        configure_group_energy_time_axis(axis)
        if limits is not None:
            axis.set_xlim(limits)
        apply_figure_theme(self.energy_figure, (axis,), theme_name, show_grid=True)
        self.energy_canvas.draw_idle()

    def position_next_to(self, main_window) -> None:
        """Place the first Satellite Overlays opening to the main window's right.

        Prefer a monitor physically to the right of the LMAS main window.  If
        none exists, use free space on the current monitor and finally fall
        back to the left while keeping the complete workspace on-screen.
        """
        if self._restored_geometry:
            return
        main_screen = main_window.screen()
        if main_screen is None:
            return
        main_frame = main_window.frameGeometry()
        screens = QApplication.screens()
        available_regions = [screen.availableGeometry() for screen in screens]
        current = main_screen.availableGeometry()
        width = min(980, max(900, current.width() // 2))
        height = min(current.height() - 24, max(840, main_frame.height()))

        preferred_x = main_frame.right() + 10
        preferred_y = main_frame.top()
        right_regions = [
            region for region in available_regions
            if region.left() > main_frame.right()
            or (region == current and preferred_x + width <= region.right() + 1)
        ]
        if right_regions:
            target = min(right_regions, key=lambda region: abs(region.left() - preferred_x))
            x = max(target.left(), preferred_x)
            if x + width > target.right() + 1:
                x = max(target.left(), target.right() - width + 1)
        else:
            target = current
            x = preferred_x
            if x + width > target.right() + 1:
                x = max(target.left(), main_frame.left() - width - 10)
        y = min(max(target.top(), preferred_y), target.bottom() - height + 1)
        self.resize(width, height)
        self.move(x, y)

    def closeEvent(self, event) -> None:
        # Preserve loaded overlays and controls while behaving like an
        # independent workspace. Closing hides rather than destroys it.
        self._settings.setValue("geometry_rc1", self.saveGeometry())
        self._settings.setValue("main_splitter_rc1", self.main_splitter.sizes())
        self._settings.setValue("left_splitter_rc1", self.left_splitter.sizes())
        self.hide()
        event.ignore()


__all__ = ["SatelliteOverlayWindow"]
