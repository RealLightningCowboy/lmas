from __future__ import annotations

from pathlib import Path
import importlib.util
import subprocess
import sys
import tempfile

from PySide6.QtCore import QProcess, QSettings, QTimer, Qt
from PySide6.QtGui import QAction, QActionGroup, QCloseEvent, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .. import __version__
from ..errors import ArchiveMemberSelectionRequired, LMASError
from ..io.backends import normalize_reader_backend, reader_backend_statuses
from ..model import LMAProject
from ..output_naming import default_output_path, display_mode_label, project_output_directory
from ..paths import is_lmas_development_path, user_documents_directory
from ..overlays.satellite.manager import SatelliteOverlayManager
from ..overlays.satellite.rendering import SatelliteOverlayRenderer
from ..overlays.network.manager import NetworkOverlayManager
from ..overlays.network.rendering import NetworkOverlayRenderer
from ..profile_defaults import profile_save_directory
from ..help_docs import (
    CHANGELOG,
    DEVELOPMENT_PROVENANCE,
    KNOWN_LIMITATIONS,
    NETWORK_OVERLAYS,
    LINEAGE_AND_ATTRIBUTION,
    RELEASE_NOTES,
    POLARITY_PRODUCT_FORMAT,
    USER_MANUAL,
    WHAT_LMAS_CAN_DO,
)
from ..profiles import (
    BUILTIN_STARTUP_NAME,
    ProfileStore,
    profile_from_specs,
    startup_profile,
)
from .icon import (
    application_icon,
    charge_analysis_icon,
    precision_crosshair_icon,
    network_overlay_icon,
    satellite_overlay_icon,
    selection_lasso_icon,
)
from .controls import ControlPanel
from .file_browser import LMAFileBrowserDock
from .plot_widget import DetachedPlotWindow, FigureHost
from .window_geometry import Rect, clamp_frame_to_work_area, full_height_tool_geometry
from .shortcuts import ShortcutManager

SETTINGS_ORGANIZATION = "Langmuir Laboratory"
SETTINGS_APPLICATION = "LMAS"
class MainWindow(QMainWindow):
    def __init__(self, parent: QWidget | None = None, *, profile_name: str | None = None, reader_backend: str = "auto") -> None:
        super().__init__(parent)
        self.project: LMAProject | None = None
        self._detached: list[DetachedPlotWindow] = []
        self._tool_windows: list[QWidget] = []
        self._help_windows: list[QWidget] = []
        self._animation_processes: list[subprocess.Popen] = []
        # Interactive projection viewers are independent top-level windows.
        # Retain them explicitly so deferred scene preparation cannot be lost
        # to garbage collection, and so close events can remove them cleanly.
        self._projection_animation_windows: list[QWidget] = []
        self._precision_window: PrecisionModeWindow | None = None
        self._selection_window: SourceSelectionWindow | None = None
        self._satellite_overlay_window: SatelliteOverlayWindow | None = None
        self.satellite_overlays = SatelliteOverlayManager()
        self._satellite_renderer = SatelliteOverlayRenderer(self.satellite_overlays)
        self._network_overlay_window: NetworkOverlayWindow | None = None
        self.network_overlays = NetworkOverlayManager()
        self._network_renderer = NetworkOverlayRenderer(self.network_overlays)
        self._data_header_window: DataHeaderWindow | None = None
        self._source_distributions_window: SourceDistributionsWindow | None = None
        self.settings = QSettings(SETTINGS_ORGANIZATION, SETTINGS_APPLICATION)
        self._migrate_release_preferences()
        self.profile_store = ProfileStore()
        self.active_profile_name = BUILTIN_STARTUP_NAME
        self.requested_profile_name = profile_name
        requested_reader = normalize_reader_backend(reader_backend)
        if requested_reader == "auto":
            requested_reader = normalize_reader_backend(
                str(self.settings.value("reader/backend", "auto"))
            )
        self.reader_backend = requested_reader
        self._pending_preserve_view: bool | None = None
        self._project_home_limits: dict[str, tuple[float, float]] | None = None
        # During saved-project startup the first linked controller reports its
        # full-record limits before the saved view can be applied.  Suppress
        # only that transient control-panel synchronization so the saved limits
        # remain authoritative through the first render.
        self._suspend_view_limit_sync = False
        self._auto_redraw_timer = QTimer(self)
        self._auto_redraw_timer.setSingleShot(True)
        self._auto_redraw_timer.setInterval(100)
        self._auto_redraw_timer.timeout.connect(self._perform_auto_redraw)
        self._satellite_refresh_timer = QTimer(self)
        self._satellite_refresh_timer.setSingleShot(True)
        self._satellite_refresh_timer.setInterval(160)
        self._satellite_refresh_timer.timeout.connect(
            self._perform_satellite_overlay_refresh
        )
        self._satellite_refresh_diagnostics = False
        self._network_refresh_timer = QTimer(self)
        self._network_refresh_timer.setSingleShot(True)
        self._network_refresh_timer.setInterval(160)
        self._network_refresh_timer.timeout.connect(self._perform_network_overlay_refresh)
        self._network_refresh_diagnostics = False

        self.setWindowTitle(f"Lightning Mapping Array Suite — LMAS {__version__}")
        self.setWindowIcon(application_icon())
        self.resize(1600, 960)

        self.controls = ControlPanel()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.controls)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(235)
        scroll.setMaximumWidth(16777215)
        scroll.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        self.figure_host = FigureHost()
        self.welcome = QLabel(
            "<h1>Lightning Mapping Array Suite</h1>"
            "<p>Open solved LMA data or a supported archive bundle, open an "
            "optional LMAS project, or use "
            "the demonstration dataset.</p>"
            "<p>The built-in native solved-source reader and Startup profile "
            "are applied automatically. pyxlma is optional.</p>"
        )
        self.welcome.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.welcome.setWordWrap(True)
        figure_container = QWidget()
        self.figure_layout = QVBoxLayout(figure_container)
        self.figure_layout.setContentsMargins(0, 0, 0, 0)
        self.figure_layout.addWidget(self.welcome, 1)
        self.figure_layout.addWidget(self.figure_host, 1)
        self.figure_host.hide()

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(scroll)
        self.splitter.addWidget(figure_container)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setCollapsible(0, False)
        self.splitter.setHandleWidth(7)
        self.splitter.setSizes([300, 1300])
        self.setCentralWidget(self.splitter)

        browser_root = self._existing_directory(
            self.settings.value("browser/root", self._last_directory()),
            fallback=self._last_directory(),
        )
        self.file_browser = LMAFileBrowserDock(browser_root, self)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.file_browser)
        self.file_browser.open_requested.connect(self._open_browser_path)
        self.file_browser.root_changed.connect(self._browser_root_changed)
        self.statusBar().showMessage("Ready")

        self.controls.redraw_requested.connect(self.schedule_redraw)
        self.controls.linked_behavior_changed.connect(self._set_linked_behavior)
        self.controls.interactive_limits_changed.connect(self._set_interactive_limits)
        self.figure_host.view_state_changed.connect(self._view_state_changed)
        self.figure_host.project_home_requested.connect(self.restore_project_home)
        self.figure_host.figure_changed.connect(self._precision_figure_changed)
        self.controls.reset_requested.connect(self.reset_view)
        self.controls.save_requested.connect(self.save_figure)
        self.controls.detach_requested.connect(self.detach_figure)
        self.controls.interactive_projection_animation_requested.connect(
            self.open_projection_animation
        )
        self.controls.save_projection_animation_requested.connect(
            self.save_projection_animation
        )
        self.controls.interactive_3d_requested.connect(self.open_interactive_3d)
        self.controls.save_animation_requested.connect(self.save_3d_animation)
        self.shortcut_manager = ShortcutManager(self, self.settings)
        self._register_shortcuts()
        self._build_actions()
        self._restore_window_preferences()

    def _register_shortcuts(self) -> None:
        manager = self.shortcut_manager
        manager.register("open_data", self.choose_lma_files)
        manager.register("save_project", self.save_project)
        manager.register("save_figure", self.save_figure)
        manager.register("quit", self.close)
        manager.register("full_view", self.figure_host.restore_full_view)
        manager.register("history_back", self.figure_host.history_back)
        manager.register("history_forward", self.figure_host.history_forward)
        manager.register("rectangle_zoom", self._activate_rectangle_zoom)
        manager.register("pan_drag", self._activate_pan_drag)
        manager.register("precision_mode", self.open_precision_mode)
        manager.register("source_selection", self.open_source_selection)
        manager.register("precision_cursor_a", lambda: self._precision_select_cursor("A"))
        manager.register("precision_cursor_b", lambda: self._precision_select_cursor("B"))
        manager.register("precision_previous", lambda: self._precision_step(-1))
        manager.register("precision_next", lambda: self._precision_step(1))
        manager.register("precision_previous_10", lambda: self._precision_step(-10))
        manager.register("precision_next_10", lambda: self._precision_step(10))
        manager.register("precision_swap", self._precision_swap)
        manager.register("precision_undo", self._analysis_undo)
        manager.register("precision_clear", self._precision_clear)
        manager.register("precision_clear_all", self._precision_clear_all)
        manager.register("precision_copy", self._precision_copy)
        manager.register("toggle_grid", self.controls.show_grid.toggle)
        manager.register("toggle_colorbar", self.controls.show_colorbar.toggle)
        manager.register("toggle_stations", self.controls.show_stations.toggle)
        manager.register("toggle_station_labels", self.controls.show_station_labels.toggle)
        manager.register("toggle_auto_fit", self.controls.auto_fit_spatial.toggle)
        manager.register("toggle_remap", self.controls.remap_time_colors.toggle)
        manager.register("layout_landscape", lambda: self._set_layout("intfs"))
        manager.register("layout_portrait", lambda: self._set_layout("xlma"))
        manager.register("open_3d", self.open_interactive_3d)
        manager.register("fullscreen", self._toggle_fullscreen)
        manager.register("keybind_help", self.open_keybind_reference)

    def _activate_rectangle_zoom(self) -> None:
        if self._selection_window is not None:
            self._selection_window.suspend_selection()
        if self.figure_host.activate_rectangle_zoom():
            self.statusBar().showMessage(
                "Rectangle zoom active — drag a box on a scientific panel", 4000
            )

    def _activate_pan_drag(self) -> None:
        if self._selection_window is not None:
            self._selection_window.suspend_selection()
        if self.figure_host.activate_pan_drag():
            self.statusBar().showMessage(
                "Pan active — click and drag a scientific panel", 4000
            )

    @staticmethod
    def _restore_and_focus_window(window) -> None:
        """Restore, raise, and focus one persistent analysis window."""

        if window.isMinimized():
            window.showNormal()
        elif not window.isVisible():
            window.show()
        else:
            window.show()
        window.raise_()
        window.activateWindow()

    def _ensure_precision_window(self) -> PrecisionModeWindow | None:
        if self.figure_host.figure is None:
            self.statusBar().showMessage(
                "Open LMA data before starting Precision Mode", 5000
            )
            return None
        if self._precision_window is None:
            
            self._precision_window = PrecisionModeWindow(self.figure_host)
        return self._precision_window

    def open_precision_mode(self) -> None:
        if self._selection_window is not None:
            self._selection_window.suspend_selection()
        window = self._ensure_precision_window()
        if window is None:
            return
        window.bind_current_figure()
        window.activate_for_placement()
        self._restore_and_focus_window(window)
        QTimer.singleShot(
            0, lambda selected=window: self._initialize_precision_window_geometry(selected)
        )
        self.statusBar().showMessage(
            "Precision Mode active — click places the active cursor; Shift+click places B",
            6000,
        )

    def _initialize_precision_window_geometry(self, window: PrecisionModeWindow) -> None:
        """Open Precision Mode at the same safe usable height as Charge Analysis."""

        if bool(window.property("lmasPrecisionHeightInitialized")):
            return
        screen = QApplication.screenAt(self.frameGeometry().center()) or self.screen()
        if screen is None:
            return
        main_geometry = self.frameGeometry()
        current_frame = window.frameGeometry()
        current_client = window.geometry()
        available_geometry = screen.availableGeometry()

        frame_left = max(0, current_client.x() - current_frame.x())
        frame_top = max(0, current_client.y() - current_frame.y())
        frame_extra_width = max(0, current_frame.width() - current_client.width())
        frame_extra_height = max(0, current_frame.height() - current_client.height())

        fitted = full_height_tool_geometry(
            Rect(
                main_geometry.x(), main_geometry.y(),
                main_geometry.width(), main_geometry.height(),
            ),
            Rect(
                current_frame.x(), current_frame.y(),
                current_frame.width(), current_frame.height(),
            ),
            Rect(
                available_geometry.x(), available_geometry.y(),
                available_geometry.width(), available_geometry.height(),
            ),
            minimum_width=max(700 + frame_extra_width, current_frame.width()),
            minimum_height=max(520 + frame_extra_height, current_frame.height()),
            work_area_margin=18,
        )
        safe_frame_height = max(
            520 + frame_extra_height,
            fitted.height - 24,
        )
        client_width = max(window.minimumWidth(), fitted.width - frame_extra_width)
        client_height = max(
            window.minimumHeight(), safe_frame_height - frame_extra_height
        )
        window.setGeometry(
            fitted.x + frame_left,
            fitted.y + frame_top,
            client_width,
            client_height,
        )
        window.setProperty("lmasPrecisionHeightInitialized", True)
        QTimer.singleShot(
            0, lambda selected=window: self._clamp_precision_window(selected)
        )
        QTimer.singleShot(
            120, lambda selected=window: self._clamp_precision_window(selected)
        )

    def _clamp_precision_window(self, window: PrecisionModeWindow) -> None:
        """Clamp the realized Precision Mode frame into the monitor work area."""

        if window is None or not window.isVisible():
            return
        screen = (
            QApplication.screenAt(self.frameGeometry().center())
            or window.screen()
            or self.screen()
        )
        if screen is None:
            return
        frame = window.frameGeometry()
        available = screen.availableGeometry()
        clamped = clamp_frame_to_work_area(
            Rect(frame.x(), frame.y(), frame.width(), frame.height()),
            Rect(
                available.x(),
                available.y(),
                available.width(),
                available.height(),
            ),
            margin=18,
            minimum_width=max(520, window.minimumWidth()),
            minimum_height=max(420, window.minimumHeight()),
        )
        width_delta = max(0, frame.width() - clamped.width)
        height_delta = max(0, frame.height() - clamped.height)
        if width_delta or height_delta:
            window.resize(
                max(window.minimumWidth(), window.width() - width_delta),
                max(window.minimumHeight(), window.height() - height_delta),
            )
        window.move(
            window.x() + (clamped.x - frame.x()),
            window.y() + (clamped.y - frame.y()),
        )

    def _precision_figure_changed(self, figure) -> None:
        if self._precision_window is not None:
            self._precision_window.bind_figure(figure)
        if self._selection_window is not None:
            self._selection_window.bind_figure(figure)
        self._refresh_source_distributions()
        if self.satellite_overlays.has_data:
            self._queue_satellite_overlay_refresh(60, diagnostics=True)

    def _ensure_selection_window(self) -> SourceSelectionWindow | None:
        if self.figure_host.figure is None:
            self.statusBar().showMessage(
                "Open LMA data before starting Source Selection", 5000
            )
            return None
        if self._selection_window is None:
            
            self._selection_window = SourceSelectionWindow(self.figure_host)
            self._selection_window.charge_default_requested.connect(
                self._default_charge_coloring
            )
            self._selection_window.selection_state_changed.connect(
                self._selection_state_changed
            )
            self._selection_window.polarity_export_requested.connect(
                self.export_polarity_product
            )
            self._selection_window.polarity_import_requested.connect(
                self.import_polarity_product
            )
            if self.project is not None:
                self._selection_window.restore_project_state(
                    self.project.source_selection_state
                )
        return self._selection_window

    def _default_charge_coloring(self) -> None:
        """Select Charge coloring once when Charge Analysis is first activated."""

        index = self.controls.color_combo.findData("charge")
        if index < 0 or self.controls.color_combo.currentData() == "charge":
            return
        self.controls.color_combo.setCurrentIndex(index)

    def _selection_state_changed(self) -> None:
        if self.project is None or self._selection_window is None:
            return
        state = self._selection_window.project_state()
        self.project.source_selection_state = state
        # Charge assignments are scientific base colors, not tab-local
        # overlays. Update the current scatter arrays immediately while the
        # ordinary rebuilt figure is queued.
        from ..source_selection import refresh_charge_source_colors

        refresh_charge_source_colors(self.figure_host.figure, state, draw=True)
        if self.controls.color_combo.currentData() in {"charge", "group"}:
            self.schedule_redraw(True)
        self._refresh_source_distributions()

    def open_source_selection(self) -> None:
        window = self._ensure_selection_window()
        if window is None:
            return
        window.bind_current_figure()
        window.activate_for_selection()
        self._restore_and_focus_window(window)
        self.statusBar().showMessage(
            "Source Selection active — lasso and point edit use the selected Default tool action; Shift removes from the active group, Alt removes, Ctrl intersects",
            7000,
        )

    def open_charge_analysis(self) -> None:
        window = self._ensure_selection_window()
        if window is None:
            return
        window.bind_current_figure()
        window.activate_for_charge_analysis()
        self._restore_and_focus_window(window)
        QTimer.singleShot(0, lambda selected=window: self._initialize_charge_window_geometry(selected))
        self.statusBar().showMessage(
            "Charge Analysis active — select source groups and assign Unassigned, Positive, or Negative",
            7000,
        )

    def _initialize_charge_window_geometry(self, window: SourceSelectionWindow) -> None:
        """Open Charge Analysis tall, but keep its realized frame on-screen."""

        if bool(window.property("lmasChargeHeightInitialized")):
            return
        screen = QApplication.screenAt(self.frameGeometry().center()) or self.screen()
        if screen is None:
            return
        main_geometry = self.frameGeometry()
        current_frame = window.frameGeometry()
        current_client = window.geometry()
        available_geometry = screen.availableGeometry()

        # Qt positions top-level widgets by client geometry while the title bar
        # lives in the surrounding frame. Fit the frame first, then translate
        # back to client coordinates so the title bar can never land off-screen.
        frame_left = max(0, current_client.x() - current_frame.x())
        frame_top = max(0, current_client.y() - current_frame.y())
        frame_extra_width = max(0, current_frame.width() - current_client.width())
        frame_extra_height = max(0, current_frame.height() - current_client.height())

        fitted = full_height_tool_geometry(
            Rect(
                main_geometry.x(), main_geometry.y(),
                main_geometry.width(), main_geometry.height(),
            ),
            Rect(
                current_frame.x(), current_frame.y(),
                current_frame.width(), current_frame.height(),
            ),
            Rect(
                available_geometry.x(), available_geometry.y(),
                available_geometry.width(), available_geometry.height(),
            ),
            minimum_width=max(610 + frame_extra_width, current_frame.width()),
            minimum_height=max(520 + frame_extra_height, current_frame.height()),
            work_area_margin=18,
        )
        # Leave a small reserve beyond the nominal work-area inset. Some
        # Windows window managers finalize title-bar dimensions only after the
        # first move/resize, and an exact full-height fit can otherwise place
        # the title bar a few pixels above the screen.
        safe_frame_height = max(
            520 + frame_extra_height,
            fitted.height - 24,
        )
        client_width = max(window.minimumWidth(), fitted.width - frame_extra_width)
        client_height = max(
            window.minimumHeight(), safe_frame_height - frame_extra_height
        )
        window.setGeometry(
            fitted.x + frame_left,
            fitted.y + frame_top,
            client_width,
            client_height,
        )
        window.setProperty("lmasChargeHeightInitialized", True)
        # Clamp twice: once as soon as Qt applies the requested geometry and
        # once after the native window manager has finalized frame decorations.
        QTimer.singleShot(0, lambda selected=window: self._clamp_charge_window(selected))
        QTimer.singleShot(120, lambda selected=window: self._clamp_charge_window(selected))

    def _clamp_charge_window(self, window: SourceSelectionWindow) -> None:
        """Clamp the realized Charge Analysis frame into the monitor work area."""

        if window is None or not window.isVisible():
            return
        screen = (
            QApplication.screenAt(self.frameGeometry().center())
            or window.screen()
            or self.screen()
        )
        if screen is None:
            return
        frame = window.frameGeometry()
        available = screen.availableGeometry()
        clamped = clamp_frame_to_work_area(
            Rect(frame.x(), frame.y(), frame.width(), frame.height()),
            Rect(
                available.x(),
                available.y(),
                available.width(),
                available.height(),
            ),
            margin=18,
            minimum_width=max(520, window.minimumWidth()),
            minimum_height=max(420, window.minimumHeight()),
        )
        width_delta = max(0, frame.width() - clamped.width)
        height_delta = max(0, frame.height() - clamped.height)
        if width_delta or height_delta:
            window.resize(
                max(window.minimumWidth(), window.width() - width_delta),
                max(window.minimumHeight(), window.height() - height_delta),
            )
        window.move(
            window.x() + (clamped.x - frame.x()),
            window.y() + (clamped.y - frame.y()),
        )

    def _ensure_satellite_overlay_window(self) -> SatelliteOverlayWindow | None:
        if self.figure_host.figure is None or self.project is None:
            self.statusBar().showMessage(
                "Open LMA data before starting Satellite Overlays", 5000
            )
            return None
        if self._satellite_overlay_window is None:
            
            self._satellite_overlay_window = SatelliteOverlayWindow(
                self.satellite_overlays,
                self.figure_host,
                lambda: self.project,
            )
            self._satellite_overlay_window.overlays_changed.connect(
                self._refresh_satellite_overlays
            )
        return self._satellite_overlay_window

    def open_satellite_overlays(self) -> None:
        window = self._ensure_satellite_overlay_window()
        if window is None:
            return
        window.bind_current_figure()
        window.position_next_to(self)
        self._restore_and_focus_window(window)
        self.statusBar().showMessage(
            "Satellite Overlays active — load GLM files and enable event footprints or centroids",
            7000,
        )

    def _queue_satellite_overlay_refresh(
        self, delay_ms: int = 160, *, diagnostics: bool = False
    ) -> None:
        """Collapse rapid linked-view events into one GLM refresh."""
        if self.project is None or self.figure_host.figure is None:
            return
        self._satellite_refresh_diagnostics = (
            self._satellite_refresh_diagnostics or bool(diagnostics)
        )
        self._satellite_refresh_timer.start(max(0, int(delay_ms)))

    def _refresh_satellite_overlays(self) -> None:
        # User control changes should feel prompt but still collapse spin-box
        # and checkbox signal bursts into a single renderer update.
        self._queue_satellite_overlay_refresh(25, diagnostics=True)

    def _perform_satellite_overlay_refresh(self) -> None:
        if self.project is None or self.figure_host.figure is None:
            return
        refresh_diagnostics = self._satellite_refresh_diagnostics
        self._satellite_refresh_diagnostics = False
        try:
            self._satellite_renderer.bind(self.figure_host.figure, self.project)
        except Exception as exc:
            self.statusBar().showMessage(f"Satellite overlay not updated: {exc}", 6000)
            return
        if refresh_diagnostics and self._satellite_overlay_window is not None:
            # The workspace internally skips this redraw when only spatial
            # limits changed and the time window is unchanged.
            self._satellite_overlay_window.refresh_diagnostics()
        rendered = sum(item.rendered_events for item in self._satellite_renderer.summaries)
        groups = sum(item.visible_groups for item in self._satellite_renderer.summaries)
        elapsed_ms = 1000.0 * sum(
            item.total_seconds for item in self._satellite_renderer.summaries
        )
        if self.satellite_overlays.has_data:
            self.statusBar().showMessage(
                f"Satellite overlays updated — {rendered:,} footprints, "
                f"{groups:,} groups ({elapsed_ms:.0f} ms preparation)",
                5000,
            )


    def _ensure_network_overlay_window(self) -> NetworkOverlayWindow | None:
        if self.figure_host.figure is None or self.project is None:
            self.statusBar().showMessage(
                "Open LMA data before starting Network Overlays", 5000
            )
            return None
        if self._network_overlay_window is None:
            
            self._network_overlay_window = NetworkOverlayWindow(
                self.network_overlays,
                self.figure_host,
                lambda: self.project,
            )
            self._network_overlay_window.overlays_changed.connect(
                self._refresh_network_overlays
            )
        return self._network_overlay_window

    def open_network_overlays(self) -> None:
        window = self._ensure_network_overlay_window()
        if window is None:
            return
        window.bind_current_figure()
        window.position_next_to(self)
        self._restore_and_focus_window(window)
        self.statusBar().showMessage(
            "Network Overlays active — load ENTLN or generic ground-network CSV files",
            7000,
        )

    def _queue_network_overlay_refresh(
        self, delay_ms: int = 160, *, diagnostics: bool = False
    ) -> None:
        if self.project is None or self.figure_host.figure is None:
            return
        self._network_refresh_diagnostics = (
            self._network_refresh_diagnostics or bool(diagnostics)
        )
        self._network_refresh_timer.start(max(0, int(delay_ms)))

    def _refresh_network_overlays(self) -> None:
        self._queue_network_overlay_refresh(25, diagnostics=True)

    def _perform_network_overlay_refresh(self) -> None:
        if self.project is None or self.figure_host.figure is None:
            return
        refresh_diagnostics = self._network_refresh_diagnostics
        self._network_refresh_diagnostics = False
        try:
            self._network_renderer.bind(self.figure_host.figure, self.project)
        except Exception as exc:
            self.statusBar().showMessage(f"Network overlay not updated: {exc}", 6000)
            return
        if refresh_diagnostics and self._network_overlay_window is not None:
            self._network_overlay_window.refresh_diagnostics()
        rendered = sum(item.rendered_events for item in self._network_renderer.summaries)
        ellipses = sum(item.visible_ellipses for item in self._network_renderer.summaries)
        elapsed_ms = 1000.0 * sum(item.total_seconds for item in self._network_renderer.summaries)
        if self.network_overlays.has_data:
            self.statusBar().showMessage(
                f"Network overlays updated — {rendered:,} events, "
                f"{ellipses:,} uncertainty ellipses ({elapsed_ms:.0f} ms preparation)",
                5000,
            )


    def open_export_product(self) -> None:
        """Open the general, extensible scientific-product export interface."""

        from .export_product_dialog import ExportProductDialog

        if self.project is None:
            self._show_error(
                "No data", LMASError("Open LMA data before exporting a product")
            )
            return
        dialog = ExportProductDialog(self)
        if not dialog.exec():
            return
        options = dialog.options()
        self.export_polarity_product(options.format_name, options.scope)

    def open_data_file_header(self) -> None:
        """Show literal DAT headers or equivalent dataset metadata."""

        from .data_header_window import DataHeaderWindow

        if self.project is None:
            self._show_error(
                "No data", LMASError("Open LMA data before viewing its header")
            )
            return
        if (
            self._data_header_window is None
            or self._data_header_window.project is not self.project
        ):
            if self._data_header_window is not None:
                self._data_header_window.close()
            self._data_header_window = DataHeaderWindow(self.project, self)
        self._data_header_window.show()
        self._data_header_window.raise_()
        self._data_header_window.activateWindow()

    def export_polarity_product(self, format_name: str, scope: str) -> None:
        from ..polarity_product import export_polarity_csv, export_polarity_netcdf

        if self.project is None:
            self._show_error(
                "No data", LMASError("Open LMA data before exporting polarity data")
            )
            return
        self._commit_live_project_state()
        format_value = str(format_name).strip().lower()
        if format_value == "csv":
            suffix = ".csv"
            caption = "Export polarity source table"
            file_filter = "CSV table (*.csv)"
        else:
            suffix = ".nc"
            caption = "Export complete LMAS polarity product"
            file_filter = "NetCDF polarity product (*.nc *.netcdf)"
        directory = self.project.output_directory or self._last_directory()
        default = Path(directory) / f"{self.project.output_stem}_polarity{suffix}"
        selected, _ = QFileDialog.getSaveFileName(
            self, caption, str(default), file_filter
        )
        if not selected:
            return
        try:
            if format_value == "csv":
                destination = export_polarity_csv(self.project, selected, scope=scope)
            else:
                destination = export_polarity_netcdf(self.project, selected, scope=scope)
        except Exception as exc:
            self._show_error("Could not export polarity data", exc)
            return
        self._remember_directory(destination)
        self.statusBar().showMessage(
            f"Exported polarity data to {destination}", 9000
        )

    def import_polarity_product(self) -> None:
        from ..polarity_product import import_polarity_netcdf

        if self.project is None:
            self._show_error(
                "No data",
                LMASError("Open the matching LMA dataset before importing polarity data"),
            )
            return
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Import complete LMAS polarity product",
            str(self._last_directory()),
            "NetCDF polarity product (*.nc *.netcdf)",
        )
        if not selected:
            return
        answer = QMessageBox.question(
            self,
            "Replace charge-analysis groups?",
            "Importing this product will replace the current named source groups and "
            "charge assignments after verifying the dataset fingerprint. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            state = import_polarity_netcdf(self.project, selected)
        except Exception as exc:
            self._show_error("Could not import polarity product", exc)
            return
        self.project.source_selection_state = state
        window = self._ensure_selection_window()
        if window is not None:
            window.restore_project_state(state)
            window.bind_current_figure()
        self.schedule_redraw(True)
        self._remember_directory(Path(selected))
        self.statusBar().showMessage(
            f"Imported polarity assignments from {Path(selected).name}", 9000
        )

    def _visible_precision_window(self) -> PrecisionModeWindow | None:
        window = self._precision_window
        return window if window is not None and window.isVisible() else None

    def _precision_select_cursor(self, name: str) -> None:
        if self._selection_window is not None:
            self._selection_window.suspend_selection()
        window = self._visible_precision_window()
        if window is None:
            return
        window.select_cursor(name)
        self.figure_host.activate_precision_mode()

    def _precision_step(self, amount: int) -> None:
        window = self._visible_precision_window()
        if window is not None:
            window.step_active(amount)

    def _precision_swap(self) -> None:
        window = self._visible_precision_window()
        if window is not None:
            window.swap_cursors()

    def _analysis_undo(self) -> None:
        if self.figure_host.selection_mode_active and self._selection_window is not None:
            self._selection_window.undo_last_action()
            return
        window = self._visible_precision_window()
        if window is not None:
            window.undo_last_action()

    def _precision_undo(self) -> None:
        self._analysis_undo()

    def _precision_clear(self) -> None:
        window = self._visible_precision_window()
        if window is not None:
            window.clear_active()

    def _precision_clear_all(self) -> None:
        window = self._visible_precision_window()
        if window is not None:
            window.clear_all()

    def _precision_copy(self) -> None:
        window = self._visible_precision_window()
        if window is not None:
            window.copy_measurements()

    def _set_layout(self, layout: str) -> None:
        index = self.controls.layout_combo.findData(str(layout))
        if index >= 0:
            self.controls.layout_combo.setCurrentIndex(index)

    def _toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def _refresh_source_distributions(self) -> None:
        window = self._source_distributions_window
        if window is None or not window.isVisible():
            return
        manager = self._selection_window.manager if self._selection_window is not None else None
        filters = self.controls.filters() if self.project is not None else None
        subset_filters = self.controls.view_filters() if self.project is not None else None
        theme = self.controls.plot_spec().theme if self.project is not None else "space"
        window.update_project(
            self.project, filters, manager, subset_filters=subset_filters, theme=theme
        )

    def open_source_distributions(self) -> None:
        
        if self.project is None:
            self.statusBar().showMessage("Open LMA data before viewing source distributions", 5000)
            return
        if self._source_distributions_window is None:
            self._source_distributions_window = SourceDistributionsWindow(self)
        manager = self._selection_window.manager if self._selection_window is not None else None
        self._source_distributions_window.update_project(
            self.project,
            self.controls.filters(),
            manager,
            subset_filters=self.controls.view_filters(),
            theme=self.controls.plot_spec().theme,
        )
        self._source_distributions_window.show()
        self._source_distributions_window.raise_()
        self._source_distributions_window.activateWindow()

    def open_shortcut_settings(self) -> None:
        from .shortcuts import ShortcutSettingsDialog

        ShortcutSettingsDialog(self.shortcut_manager, self).exec()

    def open_keybind_reference(self) -> None:
        from .shortcuts import ShortcutReferenceDialog

        ShortcutReferenceDialog(self.shortcut_manager, self).exec()

    def _build_actions(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        open_files = QAction("Open LMA data", self)
        open_files.triggered.connect(self.choose_lma_files)
        demo_action = QAction("Open demonstration", self)
        demo_action.triggered.connect(self.open_demo)
        save_figure_action = QAction("Save figure", self)
        save_figure_action.triggered.connect(self.save_figure)
        self.export_product_action = QAction("Export Product…", self)
        self.export_product_action.setToolTip(
            "Export a scientific data product from the current Project"
        )
        self.export_product_action.setEnabled(self.project is not None)
        self.export_product_action.triggered.connect(self.open_export_product)
        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(open_files)
        file_menu.addAction(demo_action)
        file_menu.addSeparator()
        file_menu.addAction(save_figure_action)
        file_menu.addAction(self.export_product_action)
        file_menu.addSeparator()
        file_menu.addAction(quit_action)

        self.profile_menu = self.menuBar().addMenu("&Profiles")
        self.profile_menu.setToolTipsVisible(True)
        self.use_profile_menu = self.profile_menu.addMenu("Apply Profile")
        save_profile_action = QAction("Save Profile", self)
        save_profile_action.setToolTip("Save reusable settings; profiles do not bind to one dataset or exact view")
        save_profile_action.triggered.connect(self.save_current_profile)
        import_profile_action = QAction("Import Profile", self)
        import_profile_action.triggered.connect(self.import_profile)
        export_profile_action = QAction("Export Active Profile", self)
        export_profile_action.triggered.connect(self.export_active_profile)
        delete_profile_action = QAction("Delete Profile", self)
        delete_profile_action.triggered.connect(self.delete_profile)
        self.profile_menu.addSeparator()
        for action in (
            save_profile_action,
            import_profile_action,
            export_profile_action,
            delete_profile_action,
        ):
            self.profile_menu.addAction(action)
        self._refresh_profile_menu()

        # Projects are exact data-bound sessions, deliberately separate from
        # reusable Profiles.  Keep this menu immediately beside Profiles.
        self.projects_menu = self.menuBar().addMenu("&Projects")
        self.projects_menu.setToolTipsVisible(True)
        open_project_action = QAction("Open Project", self)
        open_project_action.setToolTip("Open an exact dataset and saved linked view")
        open_project_action.triggered.connect(self.choose_project)
        save_project_action = QAction("Save Project", self)
        save_project_action.setToolTip("Update the current project with the exact dataset and view")
        save_project_action.triggered.connect(self.save_project)
        save_project_as_action = QAction("Save Project As", self)
        save_project_as_action.triggered.connect(self.save_project_as)
        self.projects_menu.addAction(open_project_action)
        self.projects_menu.addSeparator()
        self.projects_menu.addAction(save_project_action)
        self.projects_menu.addAction(save_project_as_action)
        self.projects_menu.addSeparator()
        project_home_action = QAction("Project Home", self)
        project_home_action.setShortcut(QKeySequence("Ctrl+Shift+0"))
        project_home_action.setToolTip("Return to the project's saved starting bounds")
        project_home_action.triggered.connect(self.restore_project_home)
        set_project_home_action = QAction("Set Current View as Project Home", self)
        set_project_home_action.setToolTip(
            "Use the current linked view as the project starting view when next saved"
        )
        set_project_home_action.triggered.connect(self.set_current_view_as_project_home)
        self.projects_menu.addAction(project_home_action)
        self.projects_menu.addAction(set_project_home_action)

        view_menu = self.menuBar().addMenu("&View")
        redraw_action = QAction("Redraw now", self)
        redraw_action.setShortcut(QKeySequence("Ctrl+R"))
        redraw_action.triggered.connect(
            lambda: self.redraw(preserve_view=True, show_errors=True)
        )
        home_action = QAction("Home linked view", self)
        home_action.setShortcut(QKeySequence("Ctrl+0"))
        home_action.triggered.connect(self.figure_host.restore_full_view)
        reset_action = QAction("Reset", self)
        reset_action.triggered.connect(self.reset_view)
        self.data_header_action = QAction("Data File Header…", self)
        self.data_header_action.setToolTip(
            "View the original DAT/DAT.GZ header or an equivalent metadata summary"
        )
        self.data_header_action.setEnabled(self.project is not None)
        self.data_header_action.triggered.connect(self.open_data_file_header)
        self.source_distributions_action = QAction("Source Distributions…", self)
        self.source_distributions_action.setToolTip(
            "Inspect χ², source-power, and station-count distributions with active filter diagnostics"
        )
        self.source_distributions_action.setEnabled(self.project is not None)
        self.source_distributions_action.triggered.connect(self.open_source_distributions)
        self.precision_action = QAction(
            precision_crosshair_icon(), "Precision Mode", self
        )
        self.precision_action.setToolTip(
            "Open Precision Mode (scope mode) for source, free, and axis cursors — P"
        )
        self.precision_action.setStatusTip(
            "Measure source or free positions, separations, axis intervals, bearings, and apparent speeds"
        )
        self.precision_action.setEnabled(self.project is not None)
        self.precision_action.triggered.connect(self.open_precision_mode)
        self.selection_action = QAction(
            selection_lasso_icon(), "Source Selection", self
        )
        self.selection_action.setToolTip(
            "Open linked lasso and point source selection — L"
        )
        self.selection_action.setStatusTip(
            "Create stable named source groups from any linked scientific panel"
        )
        self.selection_action.setEnabled(self.project is not None)
        self.selection_action.triggered.connect(self.open_source_selection)
        self.charge_action = QAction(
            charge_analysis_icon(), "Charge Analysis", self
        )
        self.charge_action.setToolTip(
            "Open manual polarity assignment and storm-scale charge groups"
        )
        self.charge_action.setStatusTip(
            "Assign polarity to linked source groups"
        )
        self.charge_action.setEnabled(self.project is not None)
        self.charge_action.triggered.connect(self.open_charge_analysis)
        self.satellite_overlay_action = QAction(
            satellite_overlay_icon(), "Satellite Overlays", self
        )
        self.satellite_overlay_action.setToolTip(
            "Overlay GLM and future satellite lightning-imager observations"
        )
        self.satellite_overlay_action.setStatusTip(
            "Load independent satellite datasets and display linked event footprints and group centroids"
        )
        self.satellite_overlay_action.setEnabled(self.project is not None)
        self.satellite_overlay_action.triggered.connect(self.open_satellite_overlays)
        self.network_overlay_action = QAction(
            network_overlay_icon(), "Network Overlays", self
        )
        self.network_overlay_action.setToolTip(
            "Overlay ENTLN and other ground lightning-location-network observations"
        )
        self.network_overlay_action.setStatusTip(
            "Load independent ground-network event tables with linked symbols, uncertainty ellipses, and time rails"
        )
        self.network_overlay_action.setEnabled(self.project is not None)
        self.network_overlay_action.triggered.connect(self.open_network_overlays)
        detach_action = QAction("Detach figure", self)
        detach_action.triggered.connect(self.detach_figure)
        interactive_projection_action = QAction("Interactive projection animation", self)
        interactive_projection_action.triggered.connect(self.open_projection_animation)
        save_projection_action = QAction("Save projection animation", self)
        save_projection_action.triggered.connect(self.save_projection_animation)
        interactive_3d_action = QAction("Interactive 3D Viewer", self)
        interactive_3d_action.triggered.connect(self.open_interactive_3d)
        save_animation_action = QAction("Save 3D animation", self)
        save_animation_action.triggered.connect(self.save_3d_animation)
        view_menu.addAction(redraw_action)
        view_menu.addAction(home_action)
        view_menu.addAction(reset_action)
        view_menu.addAction(self.data_header_action)
        view_menu.addAction(self.source_distributions_action)
        view_menu.addSeparator()
        view_menu.addAction(self.precision_action)
        view_menu.addAction(self.selection_action)
        view_menu.addAction(self.charge_action)
        view_menu.addAction(self.satellite_overlay_action)
        view_menu.addAction(self.network_overlay_action)
        view_menu.addSeparator()
        view_menu.addAction(detach_action)
        view_menu.addSeparator()
        view_menu.addAction(interactive_projection_action)
        view_menu.addAction(save_projection_action)
        view_menu.addSeparator()
        view_menu.addAction(interactive_3d_action)
        view_menu.addAction(save_animation_action)
        view_menu.addSeparator()
        view_menu.addAction(self.file_browser.toggleViewAction())

        self.analysis_toolbar = QToolBar("Analysis", self)
        self.analysis_toolbar.setObjectName("lmasAnalysisToolbar")
        self.analysis_toolbar.setMovable(True)
        self.analysis_toolbar.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.analysis_toolbar.addAction(self.precision_action)
        self.analysis_toolbar.addAction(self.selection_action)
        self.analysis_toolbar.addAction(self.charge_action)
        self.analysis_toolbar.addAction(self.satellite_overlay_action)
        self.analysis_toolbar.addAction(self.network_overlay_action)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.analysis_toolbar)

        options_menu = self.menuBar().addMenu("&Options")
        preferences_action = QAction("Preferences", self)
        preferences_action.triggered.connect(self.open_preferences)
        options_menu.addAction(preferences_action)
        options_menu.addSeparator()
        keyboard_action = QAction("Keyboard Shortcuts", self)
        keyboard_action.triggered.connect(self.open_shortcut_settings)
        options_menu.addAction(keyboard_action)
        reader_menu = options_menu.addMenu("Reader Backend")
        self.reader_action_group = QActionGroup(self)
        self.reader_action_group.setExclusive(True)
        statuses = reader_backend_statuses()
        menu_entries = [("auto", "Auto — prefer LMAS native", True, None)]
        menu_entries.extend(
            (status.name, status.label, status.available, status.version)
            for status in statuses
        )
        for backend_name, label, available, version in menu_entries:
            suffix = "" if version in (None, "") else f" ({version})"
            action = QAction(label + suffix, self)
            action.setCheckable(True)
            action.setData(backend_name)
            action.setChecked(self.reader_backend == backend_name)
            if not available:
                action.setEnabled(False)
                action.setToolTip(f"Install the optional {label} backend to enable it")
            action.triggered.connect(
                lambda checked=False, name=backend_name: self._set_reader_backend(name) if checked else None
            )
            self.reader_action_group.addAction(action)
            reader_menu.addAction(action)

        array_information_action = QAction("Array Info", self)
        array_information_action.triggered.connect(self.open_array_information)
        self.menuBar().addAction(array_information_action)

        help_menu = self.menuBar().addMenu("&Help")
        capabilities = QAction("What LMAS can do", self)
        capabilities.triggered.connect(
            lambda: self.open_help_document(WHAT_LMAS_CAN_DO)
        )
        manual = QAction("User Manual", self)
        manual.triggered.connect(lambda: self.open_help_document(USER_MANUAL))
        network_guide = QAction("Network Overlays Guide", self)
        network_guide.triggered.connect(lambda: self.open_help_document(NETWORK_OVERLAYS))
        lineage = QAction("Lineage and attribution", self)
        lineage.triggered.connect(
            lambda: self.open_help_document(LINEAGE_AND_ATTRIBUTION)
        )
        development_provenance = QAction("Development provenance", self)
        development_provenance.triggered.connect(
            lambda: self.open_help_document(DEVELOPMENT_PROVENANCE)
        )
        release_notes = QAction("Release notes", self)
        release_notes.triggered.connect(
            lambda: self.open_help_document(RELEASE_NOTES)
        )
        polarity_product_format = QAction("Polarity product format", self)
        polarity_product_format.triggered.connect(
            lambda: self.open_help_document(POLARITY_PRODUCT_FORMAT)
        )
        known_limitations = QAction("Known limitations", self)
        known_limitations.triggered.connect(
            lambda: self.open_help_document(KNOWN_LIMITATIONS)
        )
        changelog = QAction("Changelog", self)
        changelog.triggered.connect(lambda: self.open_help_document(CHANGELOG))
        keybinds = QAction("Keybinds", self)
        keybinds.triggered.connect(self.open_keybind_reference)
        about = QAction("About LMAS", self)
        about.triggered.connect(self.show_about)
        help_menu.addAction(capabilities)
        help_menu.addAction(manual)
        help_menu.addAction(network_guide)
        help_menu.addAction(lineage)
        help_menu.addAction(development_provenance)
        help_menu.addAction(release_notes)
        help_menu.addAction(polarity_product_format)
        help_menu.addAction(known_limitations)
        help_menu.addAction(changelog)
        help_menu.addAction(keybinds)
        help_menu.addSeparator()
        help_menu.addAction(about)

    def _set_reader_backend(self, backend: str) -> None:
        self.reader_backend = normalize_reader_backend(backend)
        self.settings.setValue("reader/backend", self.reader_backend)
        self.statusBar().showMessage(
            f"Reader backend set to {self.reader_backend}; applies to the next data load",
            6000,
        )

    @staticmethod
    def _setting_bool(value, default: bool = False) -> bool:
        if value is None:
            return bool(default)
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    def _migrate_release_preferences(self) -> None:
        """Remove development-tree directories from persistent release settings."""

        # Keep the original dev19 migration marker for compatibility, then run
        # the broader RC migration once.  No drive letter, user name, host name,
        # or laboratory root is embedded in this policy.
        if not self._setting_bool(
            self.settings.value("preferences/release_path_migration_v1", False), False
        ):
            raw_output = str(
                self.settings.value("preferences/output_directory", "")
            ).strip()
            try:
                parts = [part.casefold() for part in Path(raw_output).expanduser().parts]
            except (OSError, RuntimeError, ValueError):
                parts = []
            if len(parts) >= 3 and parts[-3:] == ["lmas", "development", "outputs"]:
                self.settings.setValue("preferences/output_mode", "input")
                self.settings.remove("preferences/output_directory")
            self.settings.setValue("preferences/release_path_migration_v1", True)

        if self._setting_bool(
            self.settings.value("preferences/release_path_migration_rc1", False), False
        ):
            return

        directory_keys = (
            "preferences/data_directory",
            "preferences/output_directory",
            "last_directory",
            "browser/root",
            "profiles/last_directory",
        )
        contaminated: set[str] = set()
        for key in directory_keys:
            value = self.settings.value(key, "")
            if is_lmas_development_path(value):
                contaminated.add(key)

        if "preferences/output_directory" in contaminated:
            self.settings.setValue("preferences/output_mode", "input")
        for key in contaminated:
            self.settings.remove(key)

        self.settings.setValue("preferences/release_path_migration_rc1", True)
        self.settings.sync()

    @staticmethod
    def _existing_directory(value, *, fallback: Path | None = None) -> Path:
        """Return an existing directory, falling back portably when stale."""

        try:
            candidate = Path(str(value)).expanduser()
            if candidate.exists() and candidate.is_dir():
                return candidate.resolve()
        except (OSError, RuntimeError, ValueError):
            pass
        if fallback is not None:
            try:
                candidate = Path(fallback).expanduser()
                if candidate.exists() and candidate.is_dir():
                    return candidate.resolve()
            except (OSError, RuntimeError, ValueError):
                pass
        return user_documents_directory()

    def _default_data_directory(self) -> Path:
        value = self.settings.value(
            "preferences/data_directory", str(user_documents_directory())
        )
        return self._existing_directory(value, fallback=user_documents_directory())

    def _last_directory(self) -> Path:
        remember = self._setting_bool(
            self.settings.value("preferences/remember_last_data_directory", True), True
        )
        value = (
            self.settings.value("last_directory", str(self._default_data_directory()))
            if remember
            else self._default_data_directory()
        )
        return self._existing_directory(value, fallback=self._default_data_directory())

    def _output_directory_override(self) -> Path | None:
        mode = str(self.settings.value("preferences/output_mode", "input"))
        raw = str(self.settings.value("preferences/output_directory", "")).strip()
        return Path(raw).expanduser() if mode == "custom" and raw else None

    def _preferred_output_directory(self) -> Path:
        override = self._output_directory_override()
        if override is not None:
            return override
        if self.project is not None and self.project.output_directory is not None:
            return self.project.output_directory
        return self._last_directory()

    def open_preferences(self) -> None:
        
        raw_output = str(self.settings.value("preferences/output_directory", "")).strip()
        dialog = PreferencesDialog(
            data_directory=self._default_data_directory(),
            remember_last_data_directory=self._setting_bool(
                self.settings.value("preferences/remember_last_data_directory", True), True
            ),
            output_mode=str(self.settings.value("preferences/output_mode", "input")),
            output_directory=Path(raw_output).expanduser() if raw_output else None,
            parent=self,
        )
        if not dialog.exec():
            return
        values = dialog.preferences()
        self.settings.setValue("preferences/data_directory", str(values.data_directory))
        self.settings.setValue(
            "preferences/remember_last_data_directory",
            values.remember_last_data_directory,
        )
        self.settings.setValue("preferences/output_mode", values.output_mode)
        self.settings.setValue(
            "preferences/output_directory",
            "" if values.output_directory is None else str(values.output_directory),
        )
        if not values.remember_last_data_directory:
            self.settings.remove("last_directory")
        self.file_browser.set_root(self._last_directory())
        self.statusBar().showMessage("Preferences saved", 5000)

    def _remember_directory(
        self, path: Path, *, update_browser_root: bool = True
    ) -> None:
        directory = path if path.is_dir() else path.parent
        remember = self._setting_bool(
            self.settings.value("preferences/remember_last_data_directory", True), True
        )
        if remember:
            self.settings.setValue("last_directory", str(directory))
        if update_browser_root:
            self.file_browser.set_root(directory)

    def _browser_root_changed(self, path: Path) -> None:
        self.settings.setValue("browser/root", str(path))

    def _restore_window_preferences(self) -> None:
        geometry = self.settings.value("window/geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        splitter_state = self.settings.value("window/splitter_state")
        if splitter_state is not None:
            self.splitter.restoreState(splitter_state)
        # v0.2.6 removes the former 285 px hard lock.  Give existing users one
        # migration to a width that exposes the complete compact control rows;
        # subsequent launches preserve whatever width they choose.
        migrated = str(
            self.settings.value("window/control_width_v026_migrated", "false")
        ).lower() == "true"
        if not migrated:
            sizes = self.splitter.sizes()
            if len(sizes) >= 2:
                total = max(sum(sizes), 600)
                control_width = max(315, sizes[0])
                self.splitter.setSizes([control_width, max(300, total - control_width)])
            self.settings.setValue("window/control_width_v026_migrated", True)
        collapsed = str(self.settings.value("browser/collapsed", "false")).lower() == "true"
        width = int(self.settings.value("browser/expanded_width", 310))
        self.file_browser.restore_state(collapsed=collapsed, expanded_width=width)

    def choose_lma_files(self) -> None:
        selected, _ = QFileDialog.getOpenFileNames(
            self,
            "Open LMA data",
            str(self._last_directory()),
            "LMA data (*.dat *.dat.gz *.tar *.tar.gz *.tgz *.nc *.netcdf);;All files (*)",
        )
        if selected:
            self.open_files([Path(value) for value in selected])

    def choose_project(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Open LMAS Project",
            str(self._last_directory()),
            "LMAS projects (*.lmas-project.yaml *.lmas-project.yml *.lmas.yaml *.lmas.yml)",
        )
        if selected:
            self.open_project(Path(selected))

    def _run_load(self, message: str, function) -> None:
        self.statusBar().showMessage(message)
        try:
            QApplication.processEvents()
            project = function()
            # Figure/viewer construction is part of opening a Project. Keep it
            # inside the same guarded path so GUI wiring errors surface in the
            # normal load-error dialog instead of appearing to do nothing.
            self.set_project(project)
        except Exception as exc:
            self._show_error("Could not open LMA data or Project", exc)
            self.statusBar().showMessage("Load failed", 5000)
            return

    def _load_files_with_archive_choice(self, paths: list[Path]) -> LMAProject:
        from ..io.readers import load_lma_files

        selections: dict[Path, str] = {}
        while True:
            try:
                return load_lma_files(paths, archive_members=selections, reader_backend=self.reader_backend)
            except ArchiveMemberSelectionRequired as exc:
                choices = ["Load all contained LMA datasets", *exc.members]
                choice, accepted = QInputDialog.getItem(
                    self,
                    "Choose LMA dataset",
                    f"{Path(exc.archive).name} contains multiple plausible datasets:",
                    choices,
                    editable=False,
                )
                if not accepted:
                    raise LMASError("Archive loading was cancelled")
                selections[Path(exc.archive)] = "__all__" if choice == choices[0] else choice

    def open_files(self, paths: list[Path]) -> None:
        if not paths:
            return
        self._remember_directory(paths[0])
        self._run_load(
            "Loading LMA data…",
            lambda: self._load_files_with_archive_choice(paths),
        )

    def _open_browser_path(self, path: Path) -> None:
        name = path.name.lower()
        if name.endswith((".lmas-project.yaml", ".lmas-project.yml", ".lmas.yaml", ".lmas.yml")):
            # Preserve the useful browser root while the double-click is being
            # handled. Resetting QFileSystemModel here invalidates the active
            # index and can require a second double-click on Windows.
            self.open_project(path, preserve_browser_root=True)
        else:
            self.open_files([path])

    def _locate_project_source(self, reference: SourceFileReference) -> Path | None:
        expected = reference.filename or Path(reference.saved_path).name
        start = self._default_data_directory()
        selected, _ = QFileDialog.getOpenFileName(
            self,
            f"Locate Project source — {expected}",
            str(start),
            (
                f"Expected source ({expected});;"
                "LMA data (*.dat *.dat.gz *.tar *.tar.gz *.tgz *.nc *.netcdf);;"
                "All files (*)"
            ),
        )
        return Path(selected) if selected else None

    def open_project(
        self, path: Path, *, preserve_browser_root: bool = False
    ) -> None:
        from ..io.project import load_project

        self._remember_directory(
            path, update_browser_root=not preserve_browser_root
        )
        data_roots = (self._default_data_directory(), self._last_directory())
        self._run_load(
            "Loading LMAS project…",
            lambda: load_project(
                path,
                reader_backend=self.reader_backend,
                data_roots=data_roots,
                source_locator=self._locate_project_source,
            ),
        )

    def open_demo(self) -> None:
        from ..demo import demo_project

        project = demo_project()
        self.active_profile_name = "Packaged hybrid demonstration"
        self.set_project(project)
        # The installed demo is a template. Saving should prompt for a new user
        # Project rather than attempting to overwrite package resources.
        if self.project is not None:
            self.project.project_path = None

    def set_project(self, project: LMAProject) -> None:
        loaded_source_selection_state = dict(project.source_selection_state or {})
        if self.project is not None:
            self.project.satellite_overlay_state = self.satellite_overlays.project_state()
            self.project.network_overlay_state = self.network_overlays.project_state()
        self._auto_redraw_timer.stop()
        self._pending_preserve_view = None
        # Raw data receive a reusable profile. Saved projects retain their exact
        # data-bound settings and view state.
        if project.project_path is None:
            profile = self.profile_store.get(self.requested_profile_name or BUILTIN_STARTUP_NAME)
            project.filters = profile.filters
            project.plot = profile.plot
            self.active_profile_name = profile.name
        self.project = project
        self.satellite_overlays.restore_project_state(
            project.satellite_overlay_state,
            project_directory=(project.project_path.parent if project.project_path is not None else None),
        )
        self.network_overlays.restore_project_state(
            project.network_overlay_state,
            project_directory=(project.project_path.parent if project.project_path is not None else None),
        )
        if hasattr(self, "precision_action"):
            self.precision_action.setEnabled(True)
        if hasattr(self, "selection_action"):
            self.selection_action.setEnabled(True)
        if hasattr(self, "charge_action"):
            self.charge_action.setEnabled(True)
        if hasattr(self, "satellite_overlay_action"):
            self.satellite_overlay_action.setEnabled(True)
        if hasattr(self, "network_overlay_action"):
            self.network_overlay_action.setEnabled(True)
        if hasattr(self, "export_product_action"):
            self.export_product_action.setEnabled(True)
        if hasattr(self, "data_header_action"):
            self.data_header_action.setEnabled(True)
        if hasattr(self, "source_distributions_action"):
            self.source_distributions_action.setEnabled(True)
        if self._data_header_window is not None:
            self._data_header_window.close()
            self._data_header_window = None
        if self._selection_window is not None:
            self._selection_window.setProperty("lmasChargeHeightInitialized", False)
        # Once a source or saved Project is resolved, make its actual source-data
        # directory authoritative for browsing and default output placement.
        # This corrects stale remembered/browser paths and ensures opening a
        # Project does not leave the Project-file directory ahead of its data.
        if project.output_directory is not None:
            self._remember_directory(project.output_directory)
        self.controls.set_project(project)
        saved_limits = None
        if project.project_path is not None:
            # Saved project bounds are a non-destructive starting view.  Exact
            # source membership from older project files is deliberately ignored
            # so zooming or panning outward can reveal the complete dataset.
            project.selected_source_ids = None
            saved_limits = self.controls.interactive_limits()
        if project.project_home_limits:
            saved_limits = dict(project.project_home_limits)
        self._project_home_limits = (
            None if saved_limits is None else dict(saved_limits)
        )
        self.welcome.hide()
        self.figure_host.show()
        if saved_limits is None:
            self.redraw(preserve_view=False, show_errors=True)
        else:
            applied = False
            self._suspend_view_limit_sync = True
            try:
                self.redraw(preserve_view=False, show_errors=True)
                applied = self.figure_host.set_interactive_limits(
                    saved_limits,
                    initialize_all_matching_axes=True,
                    soft_startup_view=True,
                )
            finally:
                self._suspend_view_limit_sync = False
            controller = self.figure_host.linked_view
            if controller is not None:
                controller._notify_state()
            if not applied:
                self.statusBar().showMessage(
                    "Saved project view was not applied; no sources fall inside the requested subset",
                    5000,
                )
        project.source_selection_state = loaded_source_selection_state
        if self._selection_window is not None:
            self._selection_window.restore_project_state(
                loaded_source_selection_state
            )
        self.setWindowTitle(f"{project.name} — LMAS {__version__}")
        self.statusBar().showMessage(
            f"Loaded with {project.reader_backend} reader {project.reader_backend_version or 'unknown'}",
            7000,
        )

    def _refresh_profile_menu(self) -> None:
        self.use_profile_menu.clear()
        for profile in self.profile_store.list():
            action = QAction(profile.name, self)
            action.setCheckable(True)
            action.setChecked(profile.name == self.active_profile_name)
            action.triggered.connect(
                lambda _checked=False, name=profile.name: self.apply_profile(name)
            )
            self.use_profile_menu.addAction(action)

    def apply_profile(self, name: str) -> None:
        profile = self.profile_store.get(name)
        self.active_profile_name = profile.name
        if self.project is not None:
            self.controls.set_specs(profile.filters, profile.plot)
            # Profiles alter reusable settings, never the current data-bound
            # linked subset. Projects own exact time/spatial membership.
            self.redraw(preserve_view=True, show_errors=True)
        self._refresh_profile_menu()

    def save_current_profile(self) -> None:
        
        if self.project is None:
            self._show_error("No data", LMASError("Open LMA data before saving a profile"))
            return
        suggested_name = (
            self.active_profile_name
            if self.active_profile_name != BUILTIN_STARTUP_NAME
            else f"{self.project.name} profile"
        )
        default_directory = profile_save_directory(self.project, self._last_directory())
        dialog = SaveProfileDialog(
            default_name=suggested_name,
            default_directory=default_directory,
            parent=self,
        )
        if not dialog.exec():
            return
        try:
            options = dialog.options()
            if options.path.exists():
                answer = QMessageBox.question(
                    self,
                    "Replace profile?",
                    f"{options.path} already exists. Replace it?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if answer != QMessageBox.StandardButton.Yes:
                    return
            profile = profile_from_specs(
                options.name, self.controls.filters(), self.controls.plot_spec()
            )
            destination = self.profile_store.save(
                profile, path=options.path, overwrite=True, register=True
            )
        except Exception as exc:
            self._show_error("Could not save profile", exc)
            return
        self.settings.setValue("profiles/last_directory", str(destination.parent))
        self.active_profile_name = profile.name
        self._refresh_profile_menu()
        self.statusBar().showMessage(f"Saved profile to {destination}", 7000)

    def import_profile(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Import LMAS profile",
            str(self._last_directory()),
            "LMAS profiles (*.lmas-profile.yaml *.lmas-profile.yml *.yaml *.yml)",
        )
        if not selected:
            return
        try:
            self.profile_store.import_file(selected, overwrite=True)
        except Exception as exc:
            self._show_error("Could not import profile", exc)
            return
        self._refresh_profile_menu()

    def export_active_profile(self) -> None:
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Export LMAS profile",
            str(self._last_directory() / f"{self.active_profile_name}.lmas-profile.yaml"),
            "LMAS profiles (*.lmas-profile.yaml)",
        )
        if not selected:
            return
        try:
            self.profile_store.export(self.active_profile_name, selected)
        except Exception as exc:
            self._show_error("Could not export profile", exc)

    def delete_profile(self) -> None:
        names = [name for name in self.profile_store.names() if name != BUILTIN_STARTUP_NAME]
        if not names:
            QMessageBox.information(self, "Profiles", "There are no custom profiles to delete.")
            return
        name, accepted = QInputDialog.getItem(
            self, "Delete LMAS profile", "Profile", names, editable=False
        )
        if not accepted:
            return
        try:
            self.profile_store.delete(name)
        except Exception as exc:
            self._show_error("Could not delete profile", exc)
            return
        if self.active_profile_name == name:
            self.active_profile_name = BUILTIN_STARTUP_NAME
        self._refresh_profile_menu()

    def schedule_redraw(self, preserve_view: bool) -> None:
        if self.project is None:
            return
        if self._pending_preserve_view is None:
            self._pending_preserve_view = bool(preserve_view)
        else:
            self._pending_preserve_view = self._pending_preserve_view and bool(preserve_view)
        self._auto_redraw_timer.start()

    def _perform_auto_redraw(self) -> None:
        preserve = bool(self._pending_preserve_view)
        self._pending_preserve_view = None
        self.redraw(preserve_view=preserve, show_errors=False)

    def redraw(self, *, preserve_view: bool = False, show_errors: bool = True) -> None:
        from ..plotting import create_lma_figure, update_lma_figure_in_place

        if self.project is None:
            return
        self.statusBar().showMessage("Rendering LMA view…")
        try:
            filters = self.controls.filters()
            plot = self.controls.plot_spec()
            current_figure = self.figure_host.figure
            changed = (
                update_lma_figure_in_place(
                    current_figure,
                    self.project,
                    filters=filters,
                    plot=plot,
                )
                if preserve_view and current_figure is not None
                else None
            )
            if changed is not None:
                self.project.filters = filters
                self.project.plot = plot
                linked = self.figure_host.linked_view
                if linked is not None:
                    linked.set_behavior(
                        auto_fit_spatial=plot.auto_fit_spatial,
                        remap_time_colors=plot.remap_time_colors,
                    )
                    linked.refresh_display(
                        preview_point_limit=(
                            plot.preview_point_limit
                            if "preview_point_limit" in changed
                            else None
                        ),
                        update_subset=False,
                        notify=bool(
                            {"preview_point_limit", "title"} & set(changed)
                        ),
                        redraw=False,
                    )
                if changed:
                    current_figure.canvas.draw_idle()
                    self.statusBar().showMessage(
                        "Updated LMA view without rebuilding the figure", 2500
                    )
                else:
                    self.statusBar().showMessage("LMA view is up to date", 1500)
                return

            figure = create_lma_figure(self.project, filters=filters, plot=plot)
        except Exception as exc:
            if show_errors:
                self._show_error("Could not render LMA view", exc)
            self.statusBar().showMessage(f"View not updated: {exc}", 5000)
            return
        self.project.filters = filters
        self.project.plot = plot
        self.figure_host.set_figure(figure, preserve_view=preserve_view)
        self.figure_host.set_linked_zoom_behavior(
            auto_fit_spatial=plot.auto_fit_spatial,
            remap_time_colors=plot.remap_time_colors,
        )
        self._satellite_refresh_timer.stop()
        self._satellite_refresh_diagnostics = False
        self._satellite_renderer.bind(figure, self.project)
        self._network_refresh_timer.stop()
        self._network_refresh_diagnostics = False
        self._network_renderer.bind(figure, self.project)
        if self._satellite_overlay_window is not None:
            self._satellite_overlay_window.bind_current_figure()
        if self._network_overlay_window is not None:
            self._network_overlay_window.bind_current_figure()
        if plot.show_map_underlay:
            metadata = getattr(figure, "_lmas_metadata", {})
            status = metadata.get("map_status") if isinstance(metadata, dict) else None
            self.statusBar().showMessage(
                str(status or "Map underlay requested but no map source is available"),
                7000,
            )

    def _set_linked_behavior(self, auto_fit_spatial: bool, remap_time_colors: bool) -> None:
        if self.project is None:
            return
        self.figure_host.set_linked_zoom_behavior(
            auto_fit_spatial=bool(auto_fit_spatial),
            remap_time_colors=bool(remap_time_colors),
        )
        try:
            self.project.plot = self.controls.plot_spec()
        except Exception:
            return

    def _set_interactive_limits(self, limits: object) -> None:
        if not isinstance(limits, dict):
            return
        if not self.figure_host.set_interactive_limits(limits):
            self.statusBar().showMessage(
                "Interactive limits were not applied; no sources fall inside the requested view",
                5000,
            )

    def _view_state_changed(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        visible = int(payload.get("visible_count", 0))
        displayed = int(payload.get("displayed_count", visible))
        in_view = int(payload.get("in_view_count", visible))
        filtered = int(payload.get("filtered_count", visible))
        loaded = int(payload.get("loaded_count", filtered))
        self.controls.set_view_counts(visible, in_view, loaded, displayed)
        limits = payload.get("interactive_limits")
        if isinstance(limits, dict) and not self._suspend_view_limit_sync:
            self.controls.update_interactive_limits(limits)
        if displayed < visible:
            status = (
                f"Displaying {displayed:,} of {visible:,} quality-filtered sources; "
                f"{in_view:,} sources in scientific view ({loaded:,} loaded)"
            )
        else:
            status = (
                f"Showing {visible:,} quality-filtered of {in_view:,} sources in view "
                f"({loaded:,} loaded)"
            )
        self.statusBar().showMessage(status)
        self._refresh_source_distributions()
        if self.satellite_overlays.has_data:
            self._queue_satellite_overlay_refresh(160, diagnostics=True)
        if self.network_overlays.has_data:
            self._queue_network_overlay_refresh(160, diagnostics=True)

    def restore_project_home(self) -> None:
        """Return to the active Project's saved non-destructive starting view."""

        if self.project is None or not self._project_home_limits:
            self.statusBar().showMessage("No Project Home view is available", 4000)
            return
        self._auto_redraw_timer.stop()
        self._pending_preserve_view = None
        controller = self.figure_host.linked_view
        if controller is None:
            self.statusBar().showMessage("No linked view is available", 4000)
            return
        # Clear any exact-membership fallback or transient linked constraints
        # before applying the saved home bounds.  Project Home must always be
        # able to recover the complete quality-filtered source population.
        controller.restore_full(record_history=False)
        applied = controller.apply_interactive_limits(
            dict(self._project_home_limits),
            initialize_all_matching_axes=True,
            soft_startup_view=True,
        )
        if applied:
            self.controls.update_interactive_limits(self._project_home_limits)
            self.statusBar().showMessage("Returned to Project Home", 4000)
        else:
            self.statusBar().showMessage(
                "Project Home could not be applied to the current layout", 5000
            )

    def set_current_view_as_project_home(self) -> None:
        """Stage the current linked bounds as Project Home for the next save."""

        if self.project is None:
            return
        controller = self.figure_host.linked_view
        if controller is None:
            return
        limits = controller.current_interactive_limits()
        if not limits:
            self.statusBar().showMessage("No linked view is available", 4000)
            return
        self._project_home_limits = {
            str(key): (float(value[0]), float(value[1]))
            for key, value in limits.items()
        }
        self.controls.update_interactive_limits(self._project_home_limits)
        self.project.project_home_limits = dict(self._project_home_limits)
        self.project.view_filters = self.controls.view_filters()
        self.project.selected_source_ids = None
        self.statusBar().showMessage(
            "Current bounds staged as Project Home; save the Project to keep them",
            6000,
        )

    def reset_view(self) -> None:
        if self.project is None:
            return
        self._auto_redraw_timer.stop()
        self._pending_preserve_view = None
        startup = startup_profile()
        self.active_profile_name = startup.name
        self.controls.set_specs(startup.filters, startup.plot)
        self.redraw(preserve_view=False, show_errors=True)
        self._refresh_profile_menu()

    def save_figure(self) -> None:
        from ..figure_batch import FigureBatchManifest, write_figure_batch_manifest
        from ..figure_export import (
            default_custom_title,
            save_exact_view,
            save_theme_variants,
            theme_variant_paths,
        )
        
        if self.project is None or self.figure_host.figure is None:
            return
        # Save the exact visible analysis state, including active group overlays.
        if self._selection_window is not None:
            self.project.source_selection_state = self._selection_window.project_state()
        self.project.satellite_overlay_state = self.satellite_overlays.project_state()
        self.project.network_overlay_state = self.network_overlays.project_state()
        plot = self.controls.plot_spec()
        default = default_output_path(
            self.project, self._last_directory(), "projection", plot.theme, extension=".png",
            output_directory=self._output_directory_override(),
        )
        metadata = getattr(self.figure_host.figure, "_lmas_metadata", {})
        title_artist = metadata.get("title_artist") if isinstance(metadata, dict) else None
        current_title = title_artist.get_text() if title_artist is not None else self.project.name
        controller = self.figure_host.linked_view
        current_limits = None
        if controller is not None:
            current_limits = controller.current_interactive_limits().get("time")
        default_title = default_custom_title(current_title, current_limits)
        dialog = SaveFigureDialog(
            default_path=default,
            default_dpi=plot.saved_figure_dpi,
            default_title=default_title,
            current_theme=plot.theme,
            current_color_by=plot.color_by,
            current_maximum_chi2=float(self.controls.filters().maximum_chi2 or 1.0),
            current_log_color_scale=bool(plot.log_color_scale),
            available_color_fields=tuple(self.project.available_color_fields),
            parent=self,
        )
        if not dialog.exec():
            return
        options = dialog.options()
        self.controls.saved_figure_dpi.setValue(int(options.dpi))
        if options.mode == "batch":
            try:
                temp_project, _plot, stem = self._prepare_projection_animation_project()
                root = Path(tempfile.gettempdir()) / "lmas" / "figure-batches"
                root.mkdir(parents=True, exist_ok=True)
                manifest_path = root / f"{stem}_figure_batch.json"
                manifest = FigureBatchManifest(
                    project_path=str(temp_project),
                    jobs=options.batch_jobs,
                    dpi=options.dpi,
                    dynamic_titles=options.dynamic_titles,
                    custom_title=options.title,
                    overwrite_policy=options.overwrite_policy,
                    continue_on_error=options.continue_on_error,
                )
                write_figure_batch_manifest(manifest, manifest_path)
            except Exception as exc:
                self._show_error("Could not prepare figure batch", exc)
                return
            if not self._start_lmas_with_terminal_output(
                ["batch-figures", "--manifest", str(manifest_path)]
            ):
                self._show_error(
                    "Could not launch figure batch",
                    LMASError("The LMAS figure batch process did not start"),
                )
                return
            self._remember_directory(options.path)
            self.statusBar().showMessage(
                f"Queued {len(options.batch_jobs)} figure jobs; progress is printed in the launching terminal",
                9000,
            )
            return

        if not options.batch_theme_export and options.path.exists():
            answer = QMessageBox.question(
                self,
                "Replace existing figure?",
                f"The file already exists:\n\n{options.path}\n\nReplace it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        old_title = title_artist.get_text() if title_artist is not None else None
        try:
            if options.batch_theme_export:
                paths = theme_variant_paths(options.path, options.themes)
                existing = [path for path in paths.values() if path.exists()]
                if existing:
                    names = "\n".join(path.name for path in existing)
                    answer = QMessageBox.question(
                        self,
                        "Replace existing figures?",
                        "The following files already exist:\n\n"
                        f"{names}\n\nReplace them?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.No,
                    )
                    if answer != QMessageBox.StandardButton.Yes:
                        return
                view_state = controller.capture_view_state() if controller is not None else None
                source_group_overlays_visible = bool(
                    self._selection_window is not None
                    and self._selection_window.isVisible()
                )
                destinations = save_theme_variants(
                    self.project,
                    filters=self.controls.filters(),
                    plot=plot,
                    view_state=view_state,
                    path=options.path,
                    dpi=options.dpi,
                    title=options.title,
                    themes=options.themes,
                    source_group_overlays_visible=source_group_overlays_visible,
                )
            else:
                view_state = controller.capture_view_state() if controller is not None else None
                source_group_overlays_visible = bool(
                    self._selection_window is not None
                    and self._selection_window.isVisible()
                )
                destinations = (
                    save_exact_view(
                        self.project,
                        filters=self.controls.filters(),
                        plot=plot,
                        view_state=view_state,
                        path=options.path,
                        dpi=options.dpi,
                        title=options.title,
                        source_group_overlays_visible=source_group_overlays_visible,
                    ),
                )
        except Exception as exc:
            self._show_error("Could not save figure", exc)
            return
        finally:
            if title_artist is not None and old_title is not None:
                title_artist.set_text(old_title)
                self.figure_host.figure.canvas.draw_idle()
        if destinations:
            self._remember_directory(destinations[0])
            if len(destinations) == 1:
                message = f"Saved {destinations[0]}"
            else:
                message = f"Saved {len(destinations)} themed figures to {destinations[0].parent}"
            self.statusBar().showMessage(message, 7000)

    def _commit_live_project_state(self) -> None:
        if self.project is None:
            return
        self.project.filters = self.controls.filters()
        self.project.plot = self.controls.plot_spec()
        if self._selection_window is not None:
            self.project.source_selection_state = (
                self._selection_window.project_state()
            )
        self.project.satellite_overlay_state = self.satellite_overlays.project_state()
        self.project.network_overlay_state = self.network_overlays.project_state()
        controller = self.figure_host.linked_view
        if controller is None:
            if self._project_home_limits is None:
                self._project_home_limits = dict(self.controls.interactive_limits())
                self.project.view_filters = self.controls.view_filters()
            self.project.project_home_limits = dict(self._project_home_limits or {})
            self.project.selected_source_ids = None
            self.project.color_norm_limits = None
            return
        state = controller.capture_view_state() or {}
        limits = state.get("interactive_limits")
        if isinstance(limits, dict) and limits:
            # Saving a Project defines its starting view.  Capture the actual
            # current linked bounds every time rather than preserving an older
            # staged Project Home until the user invokes a separate command.
            # This makes Save Project and Save Project As behave as expected:
            # reopening the file and pressing Project Home returns to the view
            # that was on screen when that project was last saved.
            self._project_home_limits = {
                str(key): (float(value[0]), float(value[1]))
                for key, value in limits.items()
            }
            self.controls.update_interactive_limits(self._project_home_limits)
            self.project.view_filters = self.controls.view_filters()
        self.project.project_home_limits = dict(self._project_home_limits or {})
        # Project files persist starting bounds, never a destructive exact
        # membership mask.  Source groups preserve intentional memberships.
        self.project.selected_source_ids = None
        norm_limits = state.get("norm_limits")
        self.project.color_norm_limits = (
            None
            if norm_limits is None
            else (float(norm_limits[0]), float(norm_limits[1]))
        )

    def save_project(self) -> None:
        from ..io.project import save_project

        if self.project is None:
            self._show_error("No project", LMASError("Open LMA data before saving a project"))
            return
        if self.project.project_path is None:
            self.save_project_as()
            return
        try:
            self._commit_live_project_state()
            destination = save_project(self.project, self.project.project_path)
        except Exception as exc:
            self._show_error("Could not save project", exc)
            return
        self._remember_directory(destination)
        self.statusBar().showMessage(f"Saved project {destination}", 5000)

    def save_project_as(self) -> None:
        from ..io.project import save_project

        if self.project is None:
            self._show_error("No project", LMASError("Open LMA data before saving a project"))
            return
        default_directory = project_output_directory(self.project, self._last_directory())
        default_name = f"{self.project.output_stem}.lmas-project.yaml"
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Save exact LMAS project",
            str(default_directory / default_name),
            "LMAS project (*.lmas-project.yaml)",
        )
        if not selected:
            return
        try:
            self._commit_live_project_state()
            destination = save_project(self.project, selected)
        except Exception as exc:
            self._show_error("Could not save project", exc)
            return
        self._remember_directory(destination)
        self.statusBar().showMessage(f"Saved exact project to {destination}", 5000)

    def detach_figure(self) -> None:
        from ..plotting import create_lma_figure

        if self.project is None:
            return
        try:
            filters = self.controls.filters()
            plot = self.controls.plot_spec()
        except Exception as exc:
            self._show_error("Could not detach view", exc)
            return
        window = DetachedPlotWindow(
            f"{self.project.name} — {plot.layout}",
            lambda: create_lma_figure(self.project, filters=filters, plot=plot),
            self,
        )
        self._detached.append(window)
        window.destroyed.connect(
            lambda *_args, w=window: self._detached.remove(w) if w in self._detached else None
        )
        window.show()

    def _visualization_dependencies_available(self) -> bool:
        missing = [
            name
            for name in ("pyvista", "vtk")
            if importlib.util.find_spec(name) is None
        ]
        if not missing:
            return True
        QMessageBox.warning(
            self,
            "3D visualization dependencies",
            "Interactive 3D viewing and animation require PyVista and VTK.\n\n"
            "Install them in the active LMAS environment with:\n"
            "mamba install -c conda-forge pyvista vtk imageio imageio-ffmpeg ffmpeg\n\n"
            "Then restart LMAS.",
        )
        return False

    def _current_visualization_snapshot(self):
        if self.project is None:
            raise LMASError("Open LMA data before launching the 3D viewer")
        from ..visualization.projection_animation import combined_filter_spec
        from ..visualization.snapshot import build_visualization_snapshot

        # Combine the exact live linked bounds with the current quality filters
        # without redefining Project Home merely because a 3D view was opened.
        selected_ids = None
        controller = self.figure_host.linked_view
        if controller is not None:
            state = controller.capture_view_state()
            if isinstance(state, dict):
                values = state.get("selected_source_ids")
                if values is not None:
                    selected_ids = values
        plot = self.controls.plot_spec()
        filters = combined_filter_spec(
            self.controls.filters(), self.controls.view_filters()
        )
        return build_visualization_snapshot(
            self.project,
            filters=filters,
            plot=plot,
            selected_source_ids=selected_ids,
        )

    @staticmethod
    def _start_detached_lmas(arguments: list[str]) -> bool:
        result = QProcess.startDetached(sys.executable, ["-m", "lmas", *arguments])
        if isinstance(result, tuple):
            return bool(result[0])
        return bool(result)

    def _start_lmas_with_terminal_output(self, arguments: list[str]) -> bool:
        """Launch an isolated animation worker while forwarding output.

        The worker does not inherit the GUI console input handle. This prevents
        ffmpeg or backend shutdown from retaining the parent Python process on
        Windows after a projection animation has finished.
        """
        self._animation_processes = [
            process for process in self._animation_processes if process.poll() is None
        ]
        kwargs: dict = {
            "stdin": subprocess.DEVNULL,
            "close_fds": True,
        }
        if sys.platform == "win32":
            kwargs["creationflags"] = getattr(
                subprocess, "CREATE_NEW_PROCESS_GROUP", 0
            )
        try:
            process = subprocess.Popen(
                [sys.executable, "-m", "lmas", *arguments], **kwargs
            )
        except OSError:
            return False
        self._animation_processes.append(process)
        return True

    def _prepare_projection_animation_project(self):
        from ..io.project import save_project

        if self.project is None:
            raise LMASError("Open LMA data before opening a projection animation")
        plot = self.controls.plot_spec()
        self._commit_live_project_state()
        stem = self.project.output_stem
        root = Path(tempfile.gettempdir()) / "lmas" / "projection-animation"
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix=f"{stem}_", suffix=".lmas-project.yaml", dir=root, delete=False
        ) as handle:
            temp_project = Path(handle.name)
        previous_project_path = self.project.project_path
        try:
            save_project(self.project, temp_project)
        finally:
            self.project.project_path = previous_project_path
        return temp_project, plot, stem

    def open_projection_animation(self) -> None:
        if self.project is None:
            self._show_error(
                "No data",
                LMASError("Open LMA data before opening a projection animation"),
            )
            return
        try:
            self._commit_live_project_state()
            project = self.project
            plot = self.controls.plot_spec()
        except Exception as exc:
            self._show_error("Could not prepare the projection animation", exc)
            return

        # The viewer creates a lightweight loading shell immediately and then
        # prepares its Matplotlib scene on the next event-loop turn. Keeping the
        # viewer in this process reuses the loaded Project and avoids a second
        # Python startup, temporary project file, and source-file reload.
        self.statusBar().showMessage("Preparing interactive projection animation…")

        def create_viewer() -> None:
            try:
                from .projection_animation_viewer import ProjectionAnimationViewer

                window = ProjectionAnimationViewer(
                    project,
                    display_mode=plot.three_d_display_mode,
                    trail_ms=plot.three_d_trail_ms,
                    afterimage_ms=plot.three_d_afterimage_ms,
                    fps=plot.three_d_playback_fps,
                    duration_s=plot.three_d_playback_duration_s,
                    point_limit=max(0, int(plot.preview_point_limit)),
                    parent=self,
                )
            except Exception as exc:
                self._show_error("Could not prepare the projection animation", exc)
                self.statusBar().showMessage("Projection animation failed", 5000)
                return
            self._projection_animation_windows.append(window)
            window.destroyed.connect(
                lambda *_args, w=window: self._projection_animation_windows.remove(w)
                if w in self._projection_animation_windows
                else None
            )
            window.show()
            window.raise_()
            window.activateWindow()
            self.statusBar().showMessage(
                "Preparing interactive projection animation…", 7000
            )

        QTimer.singleShot(0, create_viewer)

    def open_interactive_3d(self) -> None:
        if not self._visualization_dependencies_available():
            return
        try:
            snapshot = self._current_visualization_snapshot()
            plot = self.controls.plot_spec()
        except Exception as exc:
            self._show_error("Could not prepare the 3D viewer", exc)
            return
        camera_output = self._preferred_output_directory() / (
            self.project.name.replace(" ", "_").replace("/", "-") + ".camera.json"
        )
        args = [
            "view-3d",
            str(snapshot.path),
            "--display-mode",
            plot.three_d_display_mode,
            "--trail-ms",
            f"{plot.three_d_trail_ms:g}",
            "--afterimage-ms",
            f"{plot.three_d_afterimage_ms:g}",
            "--point-size",
            f"{max(plot.point_size, 0.1):g}",
            "--cmap",
            plot.cmap,
            "--theme",
            plot.theme,
            "--interaction-mode",
            plot.three_d_interaction_mode,
            "--fps",
            str(plot.three_d_playback_fps),
            "--duration-s",
            f"{plot.three_d_playback_duration_s:g}",
            "--point-limit",
            str(max(0, int(plot.preview_point_limit))),
            "--camera-output",
            str(camera_output),
        ]
        if not plot.three_d_show_grid_and_labels:
            args.append("--hide-axes")
        if plot.reverse_cmap:
            args.append("--reverse-cmap")
        if not self._start_detached_lmas(args):
            self._show_error(
                "Could not launch the 3D viewer",
                LMASError("The detached LMAS visualization process did not start"),
            )
            return
        self.statusBar().showMessage(
            f"Opened Interactive 3D Viewer for {snapshot.source_count:,} sources",
            7000,
        )

    def save_projection_animation(self) -> None:
        from ..animation_batch import AnimationBatchManifest, write_batch_manifest
        from ..figure_export import default_custom_title
        
        if self.project is None:
            self._show_error("No data", LMASError("Open LMA data before saving an animation"))
            return
        if importlib.util.find_spec("imageio") is None:
            QMessageBox.warning(
                self,
                "Projection animation dependency",
                "Projection animation requires imageio and imageio-ffmpeg.\n\n"
                "Install them from base with:\n"
                "mamba install -c conda-forge imageio imageio-ffmpeg ffmpeg",
            )
            return
        try:
            temp_project, plot, stem = self._prepare_projection_animation_project()
        except Exception as exc:
            self._show_error("Could not prepare the projection animation", exc)
            return
        default = default_output_path(
            self.project,
            self._last_directory(),
            "projection",
            "development",
            display_mode_label(plot.three_d_display_mode),
            plot.theme,
            extension=".mp4",
            output_directory=self._output_directory_override(),
        )
        metadata = getattr(self.figure_host.figure, "_lmas_metadata", {})
        title_artist = metadata.get("title_artist") if isinstance(metadata, dict) else None
        current_title = title_artist.get_text() if title_artist is not None else self.project.name
        controller = self.figure_host.linked_view
        current_limits = None
        if controller is not None:
            current_limits = controller.current_interactive_limits().get("time")
        default_title = default_custom_title(current_title, current_limits).split(" — ", 1)[0]
        dialog = SaveProjectionAnimationDialog(
            default_path=default, plot=plot, default_title=default_title, parent=self
        )
        if not dialog.exec():
            return
        try:
            options = dialog.options()
        except Exception as exc:
            self._show_error("Could not prepare the projection animation", exc)
            return
        if options.batch_mode:
            try:
                root = Path(tempfile.gettempdir()) / "lmas" / "animation-batches"
                root.mkdir(parents=True, exist_ok=True)
                manifest_path = root / f"{stem}_projection_animation_batch.json"
                manifest = AnimationBatchManifest(
                    jobs=options.batch_jobs,
                    project_path=str(temp_project),
                    overwrite_policy=options.overwrite_policy,
                    continue_on_error=options.continue_on_error,
                    trail_ms=options.trail_ms,
                    afterimage_ms=options.afterimage_ms,
                    fps=options.fps,
                    duration_s=options.duration_s,
                    hold_end_s=options.hold_end_s,
                    width=options.width,
                    height=options.height,
                    video_quality=options.video_quality,
                    point_size=max(plot.point_size, 0.1),
                    cmap=plot.cmap,
                    reverse_cmap=plot.reverse_cmap,
                    custom_title=options.custom_title,
                )
                write_batch_manifest(manifest, manifest_path)
            except Exception as exc:
                self._show_error("Could not prepare projection animation batch", exc)
                return
            if not self._start_lmas_with_terminal_output(
                ["batch-animations", "--manifest", str(manifest_path)]
            ):
                self._show_error(
                    "Could not launch projection animation batch",
                    LMASError("The LMAS animation batch process did not start"),
                )
                return
            self._remember_directory(options.output_path)
            self.statusBar().showMessage(
                f"Queued {len(options.batch_jobs)} projection animations; progress is printed in the launching terminal",
                9000,
            )
            return
        args = [
            "animate-projections",
            "--project",
            str(temp_project),
            "--output",
            str(options.output_path),
            "--display-mode",
            options.display_mode,
            "--trail-ms",
            f"{options.trail_ms:g}",
            "--afterimage-ms",
            f"{options.afterimage_ms:g}",
            "--fps",
            str(options.fps),
            "--duration-s",
            f"{options.duration_s:g}",
            "--hold-end-s",
            f"{options.hold_end_s:g}",
            "--width",
            str(options.width),
            "--height",
            str(options.height),
            "--video-quality",
            str(options.video_quality),
        ]
        if options.custom_title:
            args.extend(["--title", options.custom_title])
        if not self._start_lmas_with_terminal_output(args):
            self._show_error(
                "Could not launch projection animation",
                LMASError("The LMAS animation process did not start"),
            )
            return
        self._remember_directory(options.output_path)
        self.statusBar().showMessage(
            f"Rendering {options.output_path.name}; progress is printed in the launching terminal",
            9000,
        )

    def save_3d_animation(self) -> None:
        from ..animation_batch import AnimationBatchManifest, write_batch_manifest
        from .animation_dialog import SaveAnimationDialog

        if not self._visualization_dependencies_available():
            return
        if self.project is None:
            self._show_error("No data", LMASError("Open LMA data before saving an animation"))
            return
        plot = self.controls.plot_spec()
        default = default_output_path(
            self.project,
            self._last_directory(),
            "3d",
            "development",
            display_mode_label(plot.three_d_display_mode),
            plot.theme,
            extension=".mp4",
            output_directory=self._output_directory_override(),
        )
        dialog = SaveAnimationDialog(default_path=default, plot=plot, parent=self)
        if not dialog.exec():
            return
        try:
            options = dialog.options()
            snapshot = self._current_visualization_snapshot()
        except Exception as exc:
            self._show_error("Could not prepare the 3D animation", exc)
            return
        if options.batch_mode:
            try:
                root = Path(tempfile.gettempdir()) / "lmas" / "animation-batches"
                root.mkdir(parents=True, exist_ok=True)
                manifest_path = root / f"{self.project.output_stem}_3d_animation_batch.json"
                manifest = AnimationBatchManifest(
                    jobs=options.batch_jobs,
                    snapshot_path=str(snapshot.path),
                    overwrite_policy=options.overwrite_policy,
                    continue_on_error=options.continue_on_error,
                    trail_ms=options.trail_ms,
                    afterimage_ms=options.afterimage_ms,
                    fps=options.fps,
                    duration_s=options.duration_s,
                    hold_end_s=options.hold_end_s,
                    width=options.width,
                    height=options.height,
                    video_quality=options.video_quality,
                    orbit_speed_deg_s=options.orbit_speed_deg_s,
                    point_size=options.point_size,
                    cmap=plot.cmap,
                    reverse_cmap=plot.reverse_cmap,
                    render_profile=options.render_profile,
                    camera_path=None if options.camera_path is None else str(options.camera_path),
                    show_grid_and_labels=plot.three_d_show_grid_and_labels,
                )
                write_batch_manifest(manifest, manifest_path)
            except Exception as exc:
                self._show_error("Could not prepare 3D animation batch", exc)
                return
            if not self._start_lmas_with_terminal_output(
                ["batch-animations", "--manifest", str(manifest_path)]
            ):
                self._show_error(
                    "Could not launch 3D animation batch",
                    LMASError("The LMAS animation batch process did not start"),
                )
                return
            self._remember_directory(options.output_path)
            self.statusBar().showMessage(
                f"Queued {len(options.batch_jobs)} 3D animations; progress is printed in the launching terminal",
                9000,
            )
            return
        args = [
            "animate-3d",
            str(snapshot.path),
            "--output",
            str(options.output_path),
            "--mode",
            options.mode,
            "--display-mode",
            options.display_mode,
            "--trail-ms",
            f"{options.trail_ms:g}",
            "--afterimage-ms",
            f"{options.afterimage_ms:g}",
            "--point-size",
            f"{options.point_size:g}",
            "--cmap",
            plot.cmap,
            "--theme",
            plot.theme,
            "--render-profile",
            options.render_profile,
            "--fps",
            str(options.fps),
            "--duration-s",
            f"{options.duration_s:g}",
            "--hold-end-s",
            f"{options.hold_end_s:g}",
            "--orbit-speed-deg-s",
            f"{options.orbit_speed_deg_s:g}",
            "--video-quality",
            str(options.video_quality),
            "--width",
            str(options.width),
            "--height",
            str(options.height),
        ]
        if not plot.three_d_show_grid_and_labels:
            args.append("--hide-axes")
        if plot.reverse_cmap:
            args.append("--reverse-cmap")
        if options.camera_path is not None:
            args.extend(("--camera", str(options.camera_path)))
        if not self._start_lmas_with_terminal_output(args):
            self._show_error(
                "Could not launch animation rendering",
                LMASError("The LMAS animation process did not start"),
            )
            return
        self._remember_directory(options.output_path)
        self.statusBar().showMessage(
            f"Rendering {options.output_path.name}; progress is printed in the launching terminal",
            9000,
        )

    def open_array_information(self) -> None:
        from .array_information_window import ArrayInformationWindow

        if self.project is None:
            self._show_error(
                "No data",
                LMASError("Open LMA data before viewing array information"),
            )
            return
        window = ArrayInformationWindow(self.project, self)
        self._tool_windows.append(window)
        window.destroyed.connect(
            lambda *_args, w=window: self._tool_windows.remove(w)
            if w in self._tool_windows
            else None
        )
        window.show()
        window.raise_()
        window.activateWindow()

    def open_help_document(self, document_name: str) -> None:
        from .help_window import HelpDocumentWindow

        try:
            window = HelpDocumentWindow(document_name, self)
        except Exception as exc:
            self._show_error("Could not open LMAS help", exc)
            return
        self._help_windows.append(window)
        window.destroyed.connect(
            lambda *_args, w=window: self._help_windows.remove(w)
            if w in self._help_windows
            else None
        )
        window.show()
        window.raise_()
        window.activateWindow()

    def show_about(self) -> None:
        QMessageBox.about(
            self,
            "About LMAS",
            f"<h2>Lightning Mapping Array Suite {__version__}</h2>"
            "<p>A modular viewer for solved LMA source data.</p>"
            "<p>Author and maintainer: R. Stetson Reger.</p>"
            "<p>LMAS is intended to provide a modern, open, Python-based "
            "alternative to the legacy IDL program XLMA.</p>"
            "<p>LMAS includes its own native solved-LMA reader and does not "
            "require pyxlma. Its development draws on the xlma-python / pyxlma "
            "software lineage developed by Eric Bruning and collaborators, "
            "with later methods and workflows developed by R. Stetson Reger.</p>",
        )

    def _show_error(self, title: str, error: Exception) -> None:
        QMessageBox.critical(self, title, str(error))

    def closeEvent(self, event: QCloseEvent) -> None:
        # Analysis workspaces are deliberately independent top-level windows so
        # they can move behind LMAS and receive normal taskbar entries. Hide them
        # when the main application closes so they do not keep the Qt event loop
        # alive after the primary window is gone.
        if self._satellite_overlay_window is not None:
            self._satellite_overlay_window.hide()
        if self._network_overlay_window is not None:
            self._network_overlay_window.hide()
        if self._precision_window is not None:
            self._precision_window.hide_precision_mode()
        if self._selection_window is not None:
            self._selection_window.hide_selection_mode()
        for window in list(self._projection_animation_windows):
            try:
                window.close()
            except Exception:
                pass
        self._projection_animation_windows.clear()
        self.settings.setValue("window/geometry", self.saveGeometry())
        self.settings.setValue("window/splitter_state", self.splitter.saveState())
        self.settings.setValue("browser/root", str(self.file_browser.root))
        self.settings.setValue("browser/collapsed", self.file_browser.collapsed)
        self.settings.setValue("browser/expanded_width", self.file_browser.expanded_width)
        self.settings.sync()
        super().closeEvent(event)
