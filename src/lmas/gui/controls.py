from __future__ import annotations

from contextlib import contextmanager

import matplotlib.dates as mdates
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..coordinates import (
    altitude_km,
    event_local_coordinates,
    station_center_latlon,
    station_center_local_km,
)
from ..model import FilterSpec, LMAProject, PlotSpec
from .numeric_editors import DeferredDoubleSpinBox, DeferredSpinBox


class InteractiveDoubleRange(QWidget):
    changed = Signal()

    def __init__(self, label: str, *, decimals: int = 3) -> None:
        super().__init__()
        self.label = QLabel(label)
        self.minimum = DeferredDoubleSpinBox()
        self.maximum = DeferredDoubleSpinBox()
        for editor in (self.minimum, self.maximum):
            editor.setDecimals(decimals)
            editor.setRange(-1.0e9, 1.0e9)
            editor.setMinimumWidth(72)
            editor.setMaximumWidth(86)
            editor.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            editor.valueChanged.connect(self.changed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.label)

        range_row = QHBoxLayout()
        range_row.setContentsMargins(0, 0, 0, 0)
        range_row.setSpacing(4)
        range_row.addWidget(QLabel("Min"))
        range_row.addWidget(self.minimum)
        range_row.addSpacing(5)
        range_row.addWidget(QLabel("Max"))
        range_row.addWidget(self.maximum)
        range_row.addStretch(1)
        layout.addLayout(range_row)

    def set_label(self, label: str) -> None:
        self.label.setText(label)

    def values(self) -> tuple[float, float]:
        return float(self.minimum.value()), float(self.maximum.value())

    def set_values(self, minimum: float, maximum: float) -> None:
        for editor, value in ((self.minimum, minimum), (self.maximum, maximum)):
            blocked = editor.blockSignals(True)
            try:
                editor.setValue(float(value))
            finally:
                editor.blockSignals(blocked)


class InteractiveTimeRange(QWidget):
    changed = Signal()

    def __init__(self, label: str = "Time (UTC)") -> None:
        super().__init__()
        self.label = QLabel(label)
        self.minimum = QLineEdit()
        self.maximum = QLineEdit()
        for editor in (self.minimum, self.maximum):
            editor.setPlaceholderText("YYYY-MM-DDTHH:MM:SS.sss")
            editor.setMinimumWidth(0)
            editor.setMaximumWidth(184)
            editor.setFixedWidth(184)
            editor.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            editor.editingFinished.connect(self.changed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        layout.addWidget(self.label)

        for caption, editor in (("Start", self.minimum), ("End", self.maximum)):
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(5)
            row_label = QLabel(caption)
            row_label.setFixedWidth(34)
            row.addWidget(row_label)
            row.addWidget(editor)
            row.addStretch(1)
            layout.addLayout(row)

    @staticmethod
    def _format(value: float) -> str:
        stamp = mdates.num2date(float(value)).replace(tzinfo=None)
        return stamp.strftime("%Y-%m-%dT%H:%M:%S.%f").rstrip("0").rstrip(".")

    @staticmethod
    def _parse(value: str) -> float:
        parsed = np.datetime64(value.strip(), "ns")
        if np.isnat(parsed):
            raise ValueError(f"Invalid UTC time: {value!r}")
        return float(mdates.date2num(parsed.astype("datetime64[us]").astype(object)))

    def values(self) -> tuple[float, float]:
        return self._parse(self.minimum.text()), self._parse(self.maximum.text())

    def set_values(self, minimum: float, maximum: float) -> None:
        for editor, value in ((self.minimum, minimum), (self.maximum, maximum)):
            blocked = editor.blockSignals(True)
            try:
                editor.setText(self._format(float(value)))
            finally:
                editor.blockSignals(blocked)


class ControlPanel(QWidget):
    """LMAS analysis controls.

    Source-quality filters are intentionally separate from interactive plot
    limits.  The latter always mirror the current linked view and therefore do
    not need enable checkboxes or duplicate time/spatial filter controls.
    """

    redraw_requested = Signal(bool)
    reset_requested = Signal()
    save_requested = Signal()
    detach_requested = Signal()
    interactive_projection_animation_requested = Signal()
    save_projection_animation_requested = Signal()
    interactive_3d_requested = Signal()
    save_animation_requested = Signal()
    linked_behavior_changed = Signal(bool, bool)
    interactive_limits_changed = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._project: LMAProject | None = None
        self._base_filters = FilterSpec()
        self._change_block_depth = 0
        self._power_available = False
        self._minimum_power_auto = True
        self._maximum_power_auto = True
        self._three_d_hold_end_s = 5.0
        self._three_d_orbit_speed_deg_s = 14.0
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        self.summary = QLabel("Open solved LMA data or the built-in demonstration.")
        self.summary.setWordWrap(True)
        outer.addWidget(self.summary)
        self.view_count = QLabel("")
        self.view_count.setWordWrap(True)
        outer.addWidget(self.view_count)

        display_group = QGroupBox("Display")
        display_form = QFormLayout(display_group)
        display_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)
        display_form.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        display_form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        display_form.setHorizontalSpacing(7)
        display_form.setVerticalSpacing(5)
        self.layout_combo = QComboBox()
        self.layout_combo.addItem("Landscape", "intfs")
        self.layout_combo.addItem("Portrait", "xlma")
        self.show_histogram = QCheckBox("Hist")
        self.show_histogram.setChecked(False)
        layout_row = QWidget()
        layout_row_layout = QHBoxLayout(layout_row)
        layout_row_layout.setContentsMargins(0, 0, 0, 0)
        layout_row_layout.setSpacing(7)
        layout_row_layout.addWidget(self.layout_combo)
        layout_row_layout.addWidget(self.show_histogram)
        layout_row_layout.addStretch(1)
        self.coordinate_system = QComboBox()
        self.coordinate_system.addItem("Local (km)", "local")
        self.coordinate_system.addItem("Geodetic (lat/lon)", "geodetic")
        self.text_size_preset = QComboBox()
        self.text_size_preset.addItem("Normal", "normal")
        self.text_size_preset.addItem("Publication", "publication")
        self.text_size_preset.addItem("Poster", "poster")
        self.color_combo = QComboBox()
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(
            ["turbo", "jet", "viridis", "plasma", "inferno", "magma", "cividis", "gray"]
        )
        self.log_cmap = QCheckBox("Log")
        cmap_row = QWidget()
        cmap_layout = QHBoxLayout(cmap_row)
        cmap_layout.setContentsMargins(0, 0, 0, 0)
        cmap_layout.setSpacing(5)
        cmap_layout.addWidget(self.cmap_combo)
        cmap_layout.addWidget(self.log_cmap)
        cmap_layout.addStretch(1)
        self.reverse_cmap = QCheckBox("Reverse colormap")
        self.theme_combo = QComboBox()
        self.theme_combo.addItem("Light", "light")
        self.theme_combo.addItem("Space", "space")
        self.theme_combo.addItem("Dark", "dark")
        self.theme_combo.setCurrentIndex(2)
        self.point_size = DeferredDoubleSpinBox()
        self.point_size.setRange(0.0, 100.0)
        self.point_size.setDecimals(2)
        self.point_size.setSpecialValueText("Automatic")
        self.point_size.setValue(3.0)
        self.preview_dpi = DeferredSpinBox()
        self.preview_dpi.setRange(60, 300)
        self.preview_dpi.setValue(100)
        self.preview_dpi.setSuffix(" dpi")
        self.saved_figure_dpi = DeferredSpinBox()
        self.saved_figure_dpi.setRange(72, 1200)
        self.saved_figure_dpi.setValue(300)
        self.saved_figure_dpi.setSuffix(" dpi")
        self.saved_figure_dpi.setVisible(False)
        self.preview_point_limit = DeferredSpinBox()
        self.preview_point_limit.setRange(0, 5_000_000)
        self.preview_point_limit.setSingleStep(1000)
        self.preview_point_limit.setSpecialValueText("Off")
        self.preview_point_limit.setValue(12_000)
        self.preview_point_limit.setSuffix(" pts")
        self.preview_point_limit.setToolTip(
            "Maximum sources drawn interactively. The exact scientific subset, counts, history, and exports remain full resolution."
        )
        self.preview_point_limit.setFixedWidth(104)
        display_form.addRow("Layout", layout_row)
        display_form.addRow("Coordinates", self.coordinate_system)
        display_form.addRow("Text size", self.text_size_preset)
        display_form.addRow("Color by", self.color_combo)
        display_form.addRow("Colormap", cmap_row)
        display_form.addRow(self.reverse_cmap)
        display_form.addRow("Figure theme", self.theme_combo)
        display_form.addRow("Point size", self.point_size)
        display_form.addRow("Preview DPI", self.preview_dpi)
        display_form.addRow("Preview point cap", self.preview_point_limit)
        outer.addWidget(display_group)

        visibility_group = QGroupBox("View options")
        visibility_layout = QVBoxLayout(visibility_group)
        self.show_stations = QCheckBox("Show LMA stations")
        self.show_stations.setChecked(True)
        self.show_station_labels = QCheckBox("Show station labels")
        self.show_station_labels.setChecked(False)
        self.show_stations_in_vertical_projections = QCheckBox(
            "Show stations in vertical panels"
        )
        self.show_stations_in_vertical_projections.setChecked(False)
        self.show_stations_in_vertical_projections.setToolTip(
            "Plan-view stations are shown by default. Enable this to also draw station altitude in the vertical spatial projections."
        )
        self.show_colorbar = QCheckBox("Show shared colorbar")
        self.show_colorbar.setChecked(True)
        self.show_grid = QCheckBox("Show grid lines")
        self.show_grid.setChecked(True)
        self.show_legend = QCheckBox("Show legend")
        self.show_legend.setChecked(False)
        self.show_panel_labels = QCheckBox("Show panel labels")
        self.show_panel_labels.setChecked(False)
        self.relative_time_from_window_start = QCheckBox("Relative time from window start")
        self.relative_time_from_window_start.setToolTip(
            "Label the time axis in adaptive elapsed units from the fixed start of the windowed record."
        )
        self.true_aspect = QCheckBox("True spatial aspect (1 km = 1 km)")
        self.true_aspect.setToolTip(
            "Use strict physical-distance scaling in every spatial panel. Time-altitude is unchanged."
        )
        self.show_map_underlay = QCheckBox("Map underlay")
        self.show_map_underlay.setToolTip(
            "Draw county/state boundaries, national borders, and coastlines. True Aspect is required."
        )
        for widget in (
            self.show_stations,
            self.show_station_labels,
            self.show_stations_in_vertical_projections,
            self.show_colorbar,
            self.show_grid,
            self.show_legend,
            self.show_panel_labels,
            self.relative_time_from_window_start,
            self.true_aspect,
            self.show_map_underlay,
        ):
            visibility_layout.addWidget(widget)
        outer.addWidget(visibility_group)

        linked_group = QGroupBox("Linked zoom behavior")
        linked_layout = QVBoxLayout(linked_group)
        self.auto_fit_spatial = QCheckBox("Auto-fit spatial panels")
        self.auto_fit_spatial.setToolTip("Auto-fit linked spatial panels to the selected sources.")
        self.auto_fit_spatial.setChecked(True)
        self.remap_time_colors = QCheckBox("Remap colormap")
        self.remap_time_colors.setToolTip("Remap the color range to the currently selected points.")
        self.remap_time_colors.setChecked(True)
        linked_layout.addWidget(self.auto_fit_spatial)
        linked_layout.addWidget(self.remap_time_colors)
        outer.addWidget(linked_group)

        limits_group = QGroupBox("Interactive plot limits")
        limits_layout = QVBoxLayout(limits_group)
        self.interactive_time = InteractiveTimeRange()
        self.interactive_x = InteractiveDoubleRange("W ← (km) → E", decimals=3)
        self.interactive_y = InteractiveDoubleRange("S ← (km) → N", decimals=3)
        self.interactive_altitude = InteractiveDoubleRange("Altitude (km MSL)", decimals=3)
        for widget in (
            self.interactive_time,
            self.interactive_x,
            self.interactive_y,
            self.interactive_altitude,
        ):
            limits_layout.addWidget(widget)
        outer.addWidget(limits_group)

        viewpoints_group = QGroupBox("Viewpoints")
        viewpoints_form = QFormLayout(viewpoints_group)
        viewpoints_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)
        viewpoints_form.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        viewpoints_form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        viewpoints_form.setHorizontalSpacing(7)
        viewpoints_form.setVerticalSpacing(5)
        self.north_south_viewpoint = QComboBox()
        self.north_south_viewpoint.addItem("View from South", "south")
        self.north_south_viewpoint.addItem("View from North", "north")
        self.show_north_south_title = QCheckBox("Label")
        self.show_north_south_title.setToolTip("Show the north/south viewpoint title.")
        ns_row = QWidget()
        ns_layout = QHBoxLayout(ns_row)
        ns_layout.setContentsMargins(0, 0, 0, 0)
        ns_layout.setSpacing(5)
        ns_layout.addWidget(self.north_south_viewpoint)
        ns_layout.addWidget(self.show_north_south_title)
        ns_layout.addStretch(1)
        viewpoints_form.addRow("North / South", ns_row)

        self.east_west_viewpoint = QComboBox()
        self.east_west_viewpoint.addItem("View from East", "east")
        self.east_west_viewpoint.addItem("View from West", "west")
        self.show_east_west_title = QCheckBox("Label")
        self.show_east_west_title.setToolTip("Show the east/west viewpoint title.")
        ew_row = QWidget()
        ew_layout = QHBoxLayout(ew_row)
        ew_layout.setContentsMargins(0, 0, 0, 0)
        ew_layout.setSpacing(5)
        ew_layout.addWidget(self.east_west_viewpoint)
        ew_layout.addWidget(self.show_east_west_title)
        ew_layout.addStretch(1)
        viewpoints_form.addRow("East / West", ew_row)

        self.depth_mode = QComboBox()
        self.depth_mode.addItem("Time", "time")
        self.depth_mode.addItem("Spatial", "spatial")
        self.depth_mode.setCurrentIndex(1)
        viewpoints_form.addRow("Depth", self.depth_mode)
        outer.addWidget(viewpoints_group)

        quality_group = QGroupBox("Quality filters")
        quality_layout = QVBoxLayout(quality_group)
        quality_layout.setSpacing(4)

        quality_form = QFormLayout()
        quality_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)
        quality_form.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        quality_form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        quality_form.setHorizontalSpacing(7)
        quality_form.setVerticalSpacing(4)

        self.min_stations = DeferredSpinBox()
        self.min_stations.setRange(0, 100)
        self.min_stations.setValue(6)
        self.min_stations.setFixedWidth(58)
        quality_form.addRow("Stations Min", self.min_stations)

        self.max_chi2 = DeferredDoubleSpinBox()
        self.max_chi2.setRange(0.0, 1000.0)
        self.max_chi2.setDecimals(2)
        self.max_chi2.setSingleStep(0.1)
        self.max_chi2.setValue(1.0)
        self.max_chi2.setFixedWidth(68)
        quality_form.addRow("χ² Max", self.max_chi2)
        quality_layout.addLayout(quality_form)

        quality_layout.addWidget(QLabel("Source power (dBW)"))
        power_row = QHBoxLayout()
        power_row.setSpacing(4)
        power_row.addWidget(QLabel("Min"))
        self.min_power = DeferredDoubleSpinBox()
        self.min_power.setRange(-1000.0, 1000.0)
        self.min_power.setDecimals(2)
        self.min_power.setFixedWidth(84)
        power_row.addWidget(self.min_power)
        power_row.addSpacing(5)
        power_row.addWidget(QLabel("Max"))
        self.max_power = DeferredDoubleSpinBox()
        self.max_power.setRange(-1000.0, 1000.0)
        self.max_power.setDecimals(2)
        self.max_power.setFixedWidth(84)
        power_row.addWidget(self.max_power)
        power_row.addStretch(1)
        quality_layout.addLayout(power_row)
        outer.addWidget(quality_group)

        visualization_group = QGroupBox("3D visualization")
        visualization_layout = QVBoxLayout(visualization_group)
        visualization_layout.setSpacing(5)
        visualization_form = QFormLayout()
        visualization_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint
        )
        visualization_form.setFormAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        visualization_form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        visualization_form.setHorizontalSpacing(7)
        visualization_form.setVerticalSpacing(4)

        self.three_d_display_mode = QComboBox()
        self.three_d_display_mode.addItem("Cumulative", "cumulative")
        self.three_d_display_mode.addItem("Trail", "trail")
        self.three_d_display_mode.addItem("Trail + afterimage", "trail-afterimage")
        self.three_d_display_mode.setMaximumWidth(176)
        visualization_form.addRow("Display", self.three_d_display_mode)

        transition_row = QWidget()
        transition_layout = QHBoxLayout(transition_row)
        transition_layout.setContentsMargins(0, 0, 0, 0)
        transition_layout.setSpacing(4)
        self.three_d_transition_ms = DeferredDoubleSpinBox()
        self.three_d_transition_ms.setRange(0.001, 1.0e9)
        self.three_d_transition_ms.setDecimals(3)
        self.three_d_transition_ms.setValue(30.0)
        self.three_d_transition_ms.setSuffix(" ms")
        self.three_d_transition_ms.setFixedWidth(86)
        # Backward-compatible widget aliases.  The GUI intentionally exposes one
        # transition duration while PlotSpec retains both backend fields.
        self.three_d_trail_ms = self.three_d_transition_ms
        self.three_d_afterimage_ms = self.three_d_transition_ms
        transition_layout.addWidget(QLabel("Transition"))
        transition_layout.addWidget(self.three_d_transition_ms)
        transition_layout.addStretch(1)
        visualization_form.addRow("Timing", transition_row)

        playback_row = QWidget()
        playback_layout = QHBoxLayout(playback_row)
        playback_layout.setContentsMargins(0, 0, 0, 0)
        playback_layout.setSpacing(5)
        self.three_d_playback_fps = DeferredSpinBox()
        self.three_d_playback_fps.setRange(1, 240)
        self.three_d_playback_fps.setValue(30)
        self.three_d_playback_fps.setSuffix(" fps")
        self.three_d_playback_fps.setFixedWidth(82)
        self.three_d_playback_duration_s = DeferredDoubleSpinBox()
        self.three_d_playback_duration_s.setRange(0.1, 3600.0)
        self.three_d_playback_duration_s.setDecimals(1)
        self.three_d_playback_duration_s.setValue(15.0)
        self.three_d_playback_duration_s.setSuffix(" s")
        self.three_d_playback_duration_s.setFixedWidth(82)
        playback_layout.addWidget(self.three_d_playback_fps)
        playback_layout.addWidget(self.three_d_playback_duration_s)
        playback_layout.addStretch(1)
        visualization_form.addRow("Playback", playback_row)

        self.three_d_interaction_mode = QComboBox()
        self.three_d_interaction_mode.addItem("Z-axis orbit", "z-orbit")
        self.three_d_interaction_mode.addItem("Full 3D", "full-3d")
        self.three_d_interaction_mode.setMaximumWidth(132)
        visualization_form.addRow("Camera", self.three_d_interaction_mode)

        self.three_d_show_grid_and_labels = QCheckBox("Show 3D base grid and labels")
        self.three_d_show_grid_and_labels.setChecked(True)
        visualization_form.addRow("", self.three_d_show_grid_and_labels)
        visualization_layout.addLayout(visualization_form)

        projection_action_row = QHBoxLayout()
        projection_action_row.setContentsMargins(0, 0, 0, 0)
        projection_action_row.setSpacing(5)
        projection_view_button = QPushButton("View proj.")
        projection_save_button = QPushButton("Save proj.")
        projection_view_button.setToolTip("Open interactive linked projection animation")
        projection_save_button.setToolTip("Save linked projection animation")
        projection_action_row.addWidget(projection_view_button, 1)
        projection_action_row.addWidget(projection_save_button, 1)
        visualization_layout.addLayout(projection_action_row)

        three_d_action_row = QHBoxLayout()
        three_d_action_row.setContentsMargins(0, 0, 0, 0)
        three_d_action_row.setSpacing(5)
        viewer_button = QPushButton("View 3D")
        animation_button = QPushButton("Save 3D")
        viewer_button.setToolTip("Open interactive 3D viewer")
        animation_button.setToolTip("Save 3D animation")
        three_d_action_row.addWidget(viewer_button, 1)
        three_d_action_row.addWidget(animation_button, 1)
        visualization_layout.addLayout(three_d_action_row)
        outer.addWidget(visualization_group)

        auto_note = QLabel("Changes apply automatically.")
        auto_note.setWordWrap(True)
        outer.addWidget(auto_note)

        reset_button = QPushButton("Reset")
        outer.addWidget(reset_button)
        action_row = QHBoxLayout()
        save_button = QPushButton("Save figure")
        detach_button = QPushButton("Detach")
        action_row.addWidget(save_button, 1)
        action_row.addWidget(detach_button)
        outer.addLayout(action_row)
        outer.addStretch(1)

        # Keep the controls sidebar genuinely compact.  Popup menus retain the
        # full item text, while the closed controls use bounded, left-aligned
        # widths rather than stretching across the form column.
        combo_widths = {
            self.layout_combo: 122,
            self.coordinate_system: 150,
            self.text_size_preset: 150,
            self.color_combo: 150,
            self.cmap_combo: 122,
            self.theme_combo: 150,
            self.depth_mode: 110,
        }
        for combo, width in combo_widths.items():
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(8)
            combo.setMaximumWidth(width)
            combo.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        # Keep viewpoint selectors compact but wide enough for “View from South”.
        self.north_south_viewpoint.setFixedWidth(142)
        self.east_west_viewpoint.setFixedWidth(142)

        self.point_size.setFixedWidth(105)
        self.preview_dpi.setFixedWidth(88)
        self.saved_figure_dpi.setFixedWidth(88)

        reset_button.clicked.connect(self.reset_requested)
        save_button.clicked.connect(self.save_requested)
        detach_button.clicked.connect(self.detach_requested)
        projection_view_button.clicked.connect(
            self.interactive_projection_animation_requested
        )
        projection_save_button.clicked.connect(self.save_projection_animation_requested)
        viewer_button.clicked.connect(self.interactive_3d_requested)
        animation_button.clicked.connect(self.save_animation_requested)
        self.three_d_display_mode.currentIndexChanged.connect(
            self._update_three_d_controls
        )
        self._connect_auto_redraw()
        self._update_remap_control()
        self._update_layout_controls()
        self._update_three_d_controls()

    def _update_three_d_controls(self, *_args) -> None:
        mode = str(self.three_d_display_mode.currentData() or "cumulative")
        self.three_d_transition_ms.setEnabled(mode in {"trail", "trail-afterimage"})

    @contextmanager
    def _block_changes(self):
        self._change_block_depth += 1
        try:
            yield
        finally:
            self._change_block_depth -= 1

    def _request_redraw(self, preserve_view: bool) -> None:
        if self._change_block_depth == 0 and self._project is not None:
            self.redraw_requested.emit(bool(preserve_view))

    def _connect_auto_redraw(self) -> None:
        self.layout_combo.currentIndexChanged.connect(self._layout_changed)
        self.coordinate_system.currentIndexChanged.connect(self._layout_changed)
        self.show_histogram.toggled.connect(lambda *_: self._request_redraw(True))
        self.text_size_preset.currentIndexChanged.connect(lambda *_: self._request_redraw(True))
        for editor in (self.min_stations, self.max_chi2):
            editor.valueChanged.connect(lambda *_: self._request_redraw(True))
        self.min_power.valueChanged.connect(self._minimum_power_changed)
        self.max_power.valueChanged.connect(self._maximum_power_changed)

        self.color_combo.currentIndexChanged.connect(self._color_changed)
        self.cmap_combo.currentIndexChanged.connect(lambda *_: self._request_redraw(True))
        self.log_cmap.toggled.connect(lambda *_: self._request_redraw(True))
        self.reverse_cmap.toggled.connect(lambda *_: self._request_redraw(True))
        self.theme_combo.currentIndexChanged.connect(lambda *_: self._request_redraw(True))
        self.point_size.valueChanged.connect(lambda *_: self._request_redraw(True))
        self.preview_dpi.valueChanged.connect(lambda *_: self._request_redraw(True))
        self.preview_point_limit.valueChanged.connect(lambda *_: self._request_redraw(True))
        self.show_stations.toggled.connect(self._station_visibility_changed)
        self.show_station_labels.toggled.connect(lambda *_: self._request_redraw(True))
        self.show_stations_in_vertical_projections.toggled.connect(
            lambda *_: self._request_redraw(True)
        )
        self.show_colorbar.toggled.connect(lambda *_: self._request_redraw(True))
        self.show_grid.toggled.connect(lambda *_: self._request_redraw(True))
        self.show_legend.toggled.connect(lambda *_: self._request_redraw(True))
        self.show_panel_labels.toggled.connect(lambda *_: self._request_redraw(True))
        self.relative_time_from_window_start.toggled.connect(lambda *_: self._request_redraw(True))
        self.true_aspect.toggled.connect(self._true_aspect_changed)
        self.show_map_underlay.toggled.connect(self._map_underlay_changed)
        self.auto_fit_spatial.toggled.connect(self._emit_linked_behavior)
        self.remap_time_colors.toggled.connect(self._emit_linked_behavior)
        self.north_south_viewpoint.currentIndexChanged.connect(self._viewpoint_changed)
        self.east_west_viewpoint.currentIndexChanged.connect(self._viewpoint_changed)
        self.show_north_south_title.toggled.connect(lambda *_: self._request_redraw(True))
        self.show_east_west_title.toggled.connect(lambda *_: self._request_redraw(True))
        self.depth_mode.currentIndexChanged.connect(lambda *_: self._request_redraw(True))
        for widget in (
            self.interactive_time,
            self.interactive_x,
            self.interactive_y,
            self.interactive_altitude,
        ):
            widget.changed.connect(self._emit_interactive_limits)

    def _true_aspect_changed(self, checked: bool) -> None:
        # Maps require undistorted geometry, but the user must always be able
        # to leave True Aspect directly.  Turning it off therefore disables
        # the map underlay in the same action instead of trapping a disabled
        # checkbox in the checked state.
        if not bool(checked) and self.show_map_underlay.isChecked():
            blocked = self.show_map_underlay.blockSignals(True)
            self.show_map_underlay.setChecked(False)
            self.show_map_underlay.blockSignals(blocked)
        self._request_redraw(True)

    def _map_underlay_changed(self, *_args) -> None:
        enabled = self.show_map_underlay.isChecked()
        if enabled and not self.true_aspect.isChecked():
            blocked = self.true_aspect.blockSignals(True)
            self.true_aspect.setChecked(True)
            self.true_aspect.blockSignals(blocked)
        # Keep True Aspect interactive.  Unchecking it while a map is active
        # disables the map, preserving the invariant without locking the user.
        self.true_aspect.setEnabled(True)
        self._request_redraw(True)

    def _station_visibility_changed(self, *_args) -> None:
        enabled = self.show_stations.isChecked()
        self.show_station_labels.setEnabled(enabled)
        self.show_stations_in_vertical_projections.setEnabled(enabled)
        self._request_redraw(True)

    def _minimum_power_changed(self, *_args) -> None:
        if self._change_block_depth == 0:
            self._minimum_power_auto = False
        self._request_redraw(True)

    def _maximum_power_changed(self, *_args) -> None:
        if self._change_block_depth == 0:
            self._maximum_power_auto = False
        self._request_redraw(True)

    def _layout_changed(self, *_args) -> None:
        self._update_layout_controls()
        self._request_redraw(True)

    def _viewpoint_changed(self, *_args) -> None:
        self._update_layout_controls()
        self._request_redraw(True)

    def _emit_linked_behavior(self, *_args) -> None:
        if self._change_block_depth == 0 and self._project is not None:
            self.linked_behavior_changed.emit(
                self.auto_fit_spatial.isChecked(),
                self.remap_time_colors.isChecked(),
            )

    def _update_layout_controls(self) -> None:
        is_landscape = self.layout_combo.currentData() in (None, "intfs")
        is_local = self.coordinate_system.currentData() in (None, "local")
        self.show_histogram.setEnabled(is_landscape)
        for widget in (
            self.north_south_viewpoint,
            self.east_west_viewpoint,
            self.show_north_south_title,
            self.show_east_west_title,
            self.depth_mode,
        ):
            widget.setEnabled(is_landscape and is_local)
        if is_local:
            if is_landscape:
                x_label = (
                    "E ← (km) → W"
                    if self.north_south_viewpoint.currentData() == "north"
                    else "W ← (km) → E"
                )
                y_label = (
                    "N ← (km) → S"
                    if self.east_west_viewpoint.currentData() == "west"
                    else "S ← (km) → N"
                )
            else:
                x_label, y_label = "W ← (km) → E", "S ← (km) → N"
        else:
            x_label, y_label = "Longitude (degrees)", "Latitude (degrees)"
        self.interactive_x.set_label(x_label)
        self.interactive_y.set_label(y_label)

    def _emit_interactive_limits(self, *_args) -> None:
        if self._change_block_depth != 0 or self._project is None:
            return
        try:
            payload = self.interactive_limits()
        except (TypeError, ValueError):
            return
        self.interactive_limits_changed.emit(payload)

    def interactive_limits(self) -> dict[str, tuple[float, float]]:
        time_limits = tuple(sorted(self.interactive_time.values()))
        x_limits = tuple(sorted(self.interactive_x.values()))
        y_limits = tuple(sorted(self.interactive_y.values()))
        altitude_limits = tuple(sorted(self.interactive_altitude.values()))
        if any(high <= low for low, high in (time_limits, x_limits, y_limits, altitude_limits)):
            raise ValueError("Interactive plot-limit minima must be below maxima")
        is_local = self.coordinate_system.currentData() in (None, "local")
        is_landscape = self.layout_combo.currentData() in (None, "intfs")
        if is_local:
            x_name = "west" if is_landscape and self.north_south_viewpoint.currentData() == "north" else "east"
            y_name = "south" if is_landscape and self.east_west_viewpoint.currentData() == "west" else "north"
        else:
            x_name, y_name = "longitude", "latitude"
        return {
            "time": time_limits,
            x_name: x_limits,
            y_name: y_limits,
            "altitude": altitude_limits,
        }

    def update_interactive_limits(self, limits: dict[str, tuple[float, float]] | None) -> None:
        if not limits:
            return
        is_landscape = self.layout_combo.currentData() in (None, "intfs")
        is_local = self.coordinate_system.currentData() in (None, "local")
        if is_local:
            x_name = "west" if is_landscape and self.north_south_viewpoint.currentData() == "north" else "east"
            y_name = "south" if is_landscape and self.east_west_viewpoint.currentData() == "west" else "north"
        else:
            x_name, y_name = "longitude", "latitude"
        with self._block_changes():
            if "time" in limits:
                self.interactive_time.set_values(*limits["time"])
            if x_name in limits:
                self.interactive_x.set_values(*limits[x_name])
            if y_name in limits:
                self.interactive_y.set_values(*limits[y_name])
            if "altitude" in limits:
                self.interactive_altitude.set_values(*limits["altitude"])

    def set_view_counts(
        self,
        visible: int,
        in_view: int,
        loaded: int,
        displayed: int | None = None,
    ) -> None:
        visible = int(visible)
        displayed = visible if displayed is None else int(displayed)
        if displayed < visible:
            prefix = f"<b>{displayed:,}</b> displayed of {visible:,} visible"
        else:
            prefix = f"<b>{visible:,}</b> visible"
        self.view_count.setText(
            f"{prefix} of {int(in_view):,} sources in view "
            f"({int(loaded):,} loaded)"
        )

    def _color_changed(self, *_args) -> None:
        self._update_remap_control()
        self._request_redraw(True)

    def _update_remap_control(self) -> None:
        mode = self.color_combo.currentData()
        is_time = mode in (None, "time")
        is_charge = mode == "charge"
        is_group = mode == "group"
        is_categorical = is_charge or is_group
        is_power = mode == "power"
        self.cmap_combo.setEnabled(not is_categorical)
        self.reverse_cmap.setEnabled(not is_categorical)
        self.remap_time_colors.setEnabled(not is_categorical)
        log_available = not (is_time or is_categorical or is_power)
        self.log_cmap.setEnabled(log_available)
        if is_power:
            self.log_cmap.setToolTip(
                "Unavailable for Source Power because dBW is already logarithmic."
            )
        elif not log_available:
            self.log_cmap.setToolTip("Logarithmic normalization is unavailable for this color quantity.")
        else:
            self.log_cmap.setToolTip("Use logarithmic normalization for positive finite values.")
        if not log_available and self.log_cmap.isChecked():
            blocked = self.log_cmap.blockSignals(True)
            self.log_cmap.setChecked(False)
            self.log_cmap.blockSignals(blocked)

    def set_project(self, project: LMAProject) -> None:
        self._project = project
        self._base_filters = project.filters.validated()
        start, end = project.time_limits
        alt = altitude_km(project.dataset)
        x, y = event_local_coordinates(
            project.dataset,
            project.reference_longitude,
            project.reference_latitude,
        )
        with self._block_changes():
            self.summary.setText(
                f"<b>{project.name}</b><br>"
                f"{project.event_count:,} sources<br>"
                f"{str(start).replace('T', ' ')} to {str(end).replace('T', ' ')} UTC<br>"
                f"Reference: {project.reference_latitude:.5f}, {project.reference_longitude:.5f}<br>"
                "Altitude: km MSL (no ground subtraction)"
            )
            self.color_combo.clear()
            labels = {
                "time": "Source time",
                "altitude": "Altitude",
                "power": "Source Power",
                "stations": "Contributing stations",
                "chi2": "Reduced χ²",
                "charge": "Charge",
                "group": "Source group",
            }
            for mode in project.available_color_fields:
                self.color_combo.addItem(labels[mode], mode)
            self._power_available = "event_power" in project.dataset
            self.min_power.setEnabled(self._power_available)
            self.max_power.setEnabled(self._power_available)
            self.set_specs(project.filters, project.plot)
            view = project.view_filters.validated()
            view_start = np.datetime64(view.start_time, "ns") if view.start_time else start
            view_end = np.datetime64(view.end_time, "ns") if view.end_time else end
            time_numbers = mdates.date2num(
                np.asarray([view_start, view_end]).astype("datetime64[us]").astype(object)
            )
            self.interactive_time.set_values(float(time_numbers[0]), float(time_numbers[1]))
            if self.coordinate_system.currentData() in (None, "local"):
                centre = station_center_local_km(
                    project.dataset,
                    project.reference_longitude,
                    project.reference_latitude,
                )
                centre_east, centre_north = centre if centre is not None else (0.0, 0.0)
                east_limits = (
                    (float(view.minimum_x_km), float(view.maximum_x_km))
                    if view.minimum_x_km is not None and view.maximum_x_km is not None
                    else (centre_east - 200.0, centre_east + 200.0)
                )
                north_limits = (
                    (float(view.minimum_y_km), float(view.maximum_y_km))
                    if view.minimum_y_km is not None and view.maximum_y_km is not None
                    else (centre_north - 200.0, centre_north + 200.0)
                )
                if self.layout_combo.currentData() in (None, "intfs"):
                    if self.north_south_viewpoint.currentData() == "north":
                        east_limits = (-east_limits[1], -east_limits[0])
                    if self.east_west_viewpoint.currentData() == "west":
                        north_limits = (-north_limits[1], -north_limits[0])
                self.interactive_x.set_values(*east_limits)
                self.interactive_y.set_values(*north_limits)
            else:
                centre = station_center_latlon(project.dataset)
                centre_lon, centre_lat = centre if centre is not None else (
                    project.reference_longitude,
                    project.reference_latitude,
                )
                lat_half = 200.0 / 111.195
                lon_half = 200.0 / max(111.195 * np.cos(np.deg2rad(centre_lat)), 1.0e-6)
                self.interactive_x.set_values(centre_lon - lon_half, centre_lon + lon_half)
                self.interactive_y.set_values(centre_lat - lat_half, centre_lat + lat_half)
            altitude_limits = (
                float(view.minimum_altitude_km) if view.minimum_altitude_km is not None else -0.75,
                float(view.maximum_altitude_km) if view.maximum_altitude_km is not None else 30.0,
            )
            self.interactive_altitude.set_values(*altitude_limits)
        self._update_remap_control()
        self._update_layout_controls()

    def set_specs(self, filters: FilterSpec, plot: PlotSpec) -> None:
        with self._block_changes():
            spec = filters.validated()
            self._base_filters = spec
            self._minimum_power_auto = spec.minimum_power is None
            self._maximum_power_auto = spec.maximum_power is None
            self.min_stations.setValue(int(spec.minimum_stations or 0))
            self.max_chi2.setValue(float(spec.maximum_chi2 if spec.maximum_chi2 is not None else 1.0))
            if spec.minimum_power is not None:
                self.min_power.setValue(float(spec.minimum_power))
            if spec.maximum_power is not None:
                self.max_power.setValue(float(spec.maximum_power))
            self._show_auto_power_extrema()
            plot = plot.validated()
            self._set_combo(self.layout_combo, plot.layout)
            self._set_combo(self.coordinate_system, plot.coordinate_system)
            self.show_histogram.setChecked(plot.show_histogram)
            self._set_combo(self.text_size_preset, plot.text_size_preset)
            self._set_combo(self.color_combo, plot.color_by)
            self._set_combo_text(self.cmap_combo, plot.cmap)
            self._set_combo(self.theme_combo, plot.theme)
            self.point_size.setValue(plot.point_size)
            self.preview_dpi.setValue(plot.dpi)
            self.saved_figure_dpi.setValue(plot.saved_figure_dpi)
            self.preview_point_limit.setValue(plot.preview_point_limit)
            self.show_stations.setChecked(plot.show_stations)
            self.show_station_labels.setChecked(plot.show_station_labels)
            self.show_stations_in_vertical_projections.setChecked(
                plot.show_stations_in_vertical_projections
            )
            self.show_station_labels.setEnabled(plot.show_stations)
            self.show_stations_in_vertical_projections.setEnabled(plot.show_stations)
            self.show_colorbar.setChecked(plot.show_colorbar)
            self.show_grid.setChecked(plot.show_grid)
            self.show_legend.setChecked(plot.show_legend)
            self.show_panel_labels.setChecked(plot.show_panel_labels)
            self.relative_time_from_window_start.setChecked(plot.relative_time_from_window_start)
            self.show_map_underlay.setChecked(plot.show_map_underlay)
            self.true_aspect.setChecked(plot.true_aspect)
            self.true_aspect.setEnabled(True)
            self.reverse_cmap.setChecked(plot.reverse_cmap)
            self.log_cmap.setChecked(plot.log_color_scale)
            self.auto_fit_spatial.setChecked(plot.auto_fit_spatial)
            self.remap_time_colors.setChecked(plot.remap_time_colors)
            self._set_combo(self.north_south_viewpoint, plot.north_south_viewpoint)
            self._set_combo(self.east_west_viewpoint, plot.east_west_viewpoint)
            self.show_north_south_title.setChecked(plot.show_north_south_title)
            self.show_east_west_title.setChecked(plot.show_east_west_title)
            self._set_combo(self.depth_mode, plot.depth_mode)
            self._set_combo(self.three_d_display_mode, plot.three_d_display_mode)
            self.three_d_transition_ms.setValue(plot.three_d_trail_ms)
            self.three_d_playback_fps.setValue(plot.three_d_playback_fps)
            self.three_d_playback_duration_s.setValue(plot.three_d_playback_duration_s)
            self._three_d_hold_end_s = float(plot.three_d_hold_end_s)
            self._three_d_orbit_speed_deg_s = float(plot.three_d_orbit_speed_deg_s)
            self._set_combo(self.three_d_interaction_mode, plot.three_d_interaction_mode)
            self.three_d_show_grid_and_labels.setChecked(plot.three_d_show_grid_and_labels)
        self._update_remap_control()
        self._update_layout_controls()
        self._update_three_d_controls()

    @staticmethod
    def _set_combo(combo: QComboBox, value: str) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    @staticmethod
    def _set_combo_text(combo: QComboBox, value: str) -> None:
        index = combo.findText(value)
        if index < 0:
            combo.addItem(value)
            index = combo.findText(value)
        combo.setCurrentIndex(index)

    def filters(self) -> FilterSpec:
        return FilterSpec(
            start_time=None,
            end_time=None,
            minimum_stations=int(self.min_stations.value()),
            maximum_chi2=float(self.max_chi2.value()),
            minimum_altitude_km=None,
            maximum_altitude_km=None,
            minimum_power=(
                None
                if not self._power_available or self._minimum_power_auto
                else float(self.min_power.value())
            ),
            maximum_power=(
                None
                if not self._power_available or self._maximum_power_auto
                else float(self.max_power.value())
            ),
            minimum_x_km=None,
            maximum_x_km=None,
            minimum_y_km=None,
            maximum_y_km=None,
        ).validated()

    def _show_auto_power_extrema(self) -> None:
        """Display rounded data extrema without converting Auto into a filter."""

        if not self._power_available or self._project is None:
            return
        power = np.asarray(self._project.dataset["event_power"].values, dtype=float)
        finite = power[np.isfinite(power)]
        if finite.size == 0:
            return
        if self._minimum_power_auto:
            self.min_power.setValue(float(np.floor(np.min(finite) / 5.0) * 5.0))
        if self._maximum_power_auto:
            self.max_power.setValue(float(np.ceil(np.max(finite) / 5.0) * 5.0))

    def view_filters(self) -> FilterSpec:
        """Return the live linked-view constraints in canonical coordinates."""
        time_low, time_high = tuple(sorted(self.interactive_time.values()))
        x_low, x_high = tuple(sorted(self.interactive_x.values()))
        y_low, y_high = tuple(sorted(self.interactive_y.values()))
        alt_low, alt_high = tuple(sorted(self.interactive_altitude.values()))
        start_text = self.interactive_time._format(time_low)
        end_text = self.interactive_time._format(time_high)
        if self.coordinate_system.currentData() in (None, "local"):
            if self.layout_combo.currentData() in (None, "intfs"):
                if self.north_south_viewpoint.currentData() == "north":
                    x_low, x_high = -x_high, -x_low
                if self.east_west_viewpoint.currentData() == "west":
                    y_low, y_high = -y_high, -y_low
            return FilterSpec(
                start_time=start_text,
                end_time=end_text,
                minimum_stations=None,
                maximum_chi2=None,
                minimum_altitude_km=alt_low,
                maximum_altitude_km=alt_high,
                minimum_x_km=x_low,
                maximum_x_km=x_high,
                minimum_y_km=y_low,
                maximum_y_km=y_high,
            ).validated()
        return FilterSpec(
            start_time=start_text,
            end_time=end_text,
            minimum_stations=None,
            maximum_chi2=None,
            minimum_altitude_km=alt_low,
            maximum_altitude_km=alt_high,
        ).validated()

    def plot_spec(self) -> PlotSpec:
        return PlotSpec(
            layout=str(self.layout_combo.currentData()),
            coordinate_system=str(self.coordinate_system.currentData()),
            show_histogram=self.show_histogram.isChecked(),
            text_size_preset=str(self.text_size_preset.currentData()),
            color_by=str(self.color_combo.currentData()),
            cmap=self.cmap_combo.currentText(),
            theme=str(self.theme_combo.currentData()),
            point_size=float(self.point_size.value()),
            show_stations=self.show_stations.isChecked(),
            show_station_labels=self.show_station_labels.isChecked(),
            show_stations_in_vertical_projections=(
                self.show_stations_in_vertical_projections.isChecked()
            ),
            show_colorbar=self.show_colorbar.isChecked(),
            show_grid=self.show_grid.isChecked(),
            show_legend=self.show_legend.isChecked(),
            show_panel_labels=self.show_panel_labels.isChecked(),
            relative_time_from_window_start=self.relative_time_from_window_start.isChecked(),
            true_aspect=self.true_aspect.isChecked(),
            show_map_underlay=self.show_map_underlay.isChecked(),
            reverse_cmap=self.reverse_cmap.isChecked(),
            log_color_scale=self.log_cmap.isChecked(),
            auto_fit_spatial=self.auto_fit_spatial.isChecked(),
            remap_time_colors=self.remap_time_colors.isChecked(),
            north_south_viewpoint=str(self.north_south_viewpoint.currentData()),
            east_west_viewpoint=str(self.east_west_viewpoint.currentData()),
            show_north_south_title=self.show_north_south_title.isChecked(),
            show_east_west_title=self.show_east_west_title.isChecked(),
            depth_mode=str(self.depth_mode.currentData()),
            dpi=int(self.preview_dpi.value()),
            saved_figure_dpi=int(self.saved_figure_dpi.value()),
            preview_point_limit=int(self.preview_point_limit.value()),
            three_d_display_mode=str(self.three_d_display_mode.currentData()),
            three_d_trail_ms=float(self.three_d_transition_ms.value()),
            three_d_afterimage_ms=float(self.three_d_transition_ms.value()),
            three_d_playback_fps=int(self.three_d_playback_fps.value()),
            three_d_playback_duration_s=float(self.three_d_playback_duration_s.value()),
            three_d_hold_end_s=float(self._three_d_hold_end_s),
            three_d_orbit_speed_deg_s=float(self._three_d_orbit_speed_deg_s),
            three_d_interaction_mode=str(self.three_d_interaction_mode.currentData()),
            three_d_show_grid_and_labels=self.three_d_show_grid_and_labels.isChecked(),
        ).validated()

    def reset_for_project(self) -> None:
        if self._project is None:
            return
        self.set_specs(FilterSpec(), PlotSpec())
