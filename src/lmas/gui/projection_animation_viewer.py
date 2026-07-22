from __future__ import annotations

import time

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT

from ..model import LMAProject
from ..visualization.animation import animation_frame_times
from ..visualization.projection_animation import (
    DisplayMode,
    ProjectionAnimationScene,
    build_projection_animation_scene,
)


class ProjectionAnimationViewer(QMainWindow):
    """Responsive interactive playback in linked 2D projections.

    Scene construction is deferred until the window has received its first Qt
    paint. This gives immediate visual feedback and, when launched from the main
    LMAS window, reuses the already-loaded Project instead of starting another
    Python interpreter and reading the source files again.
    """

    def __init__(
        self,
        project: LMAProject,
        *,
        display_mode: DisplayMode = "cumulative",
        trail_ms: float = 30.0,
        afterimage_ms: float = 30.0,
        fps: int = 30,
        duration_s: float = 15.0,
        point_limit: int | None = None,
        parent=None,
    ) -> None:
        # Keep the interactive animation as an independent top-level window.
        # MainWindow retains a strong reference, while the owner hint is kept
        # separately so Windows does not force transient always-above behavior.
        super().__init__(None, Qt.WindowType.Window)
        self._owner_window = parent
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setAttribute(Qt.WidgetAttribute.WA_QuitOnClose, False)
        self.setWindowTitle(f"{project.name} — Projection animation")
        self.resize(1500, 940)

        self._project = project
        self._initial_display_mode: DisplayMode = display_mode
        self._point_limit = point_limit
        self.scene: ProjectionAnimationScene | None = None
        self.canvas: FigureCanvasQTAgg | None = None
        self.toolbar: NavigationToolbar2QT | None = None
        self.play_button: QPushButton | None = None
        self.restart_button: QPushButton | None = None
        self.mode_combo: QComboBox | None = None
        self.loop_checkbox: QCheckBox | None = None
        self.time_label: QLabel | None = None
        self.timeline: QSlider | None = None

        self.trail_ms = float(trail_ms)
        self.afterimage_ms = float(afterimage_ms)
        self.fps = max(1, int(fps))
        self.duration_s = max(0.1, float(duration_s))
        self.frame_times = []
        self._playing = False
        self._updating_slider = False
        self._scrubbing = False
        self._pending_scrub_index: int | None = None
        self._play_epoch_s = 0.0
        self._play_epoch_frame = 0
        self._current_frame = -1
        self._last_title_update_s = 0.0
        self._blit_background = None
        self._recaching_blit = False
        self._closed = False
        self._preparing = False
        self._pending_space_toggle = False

        # A window-level shortcut works before the Play button has ever held
        # focus. If Space is pressed while the scene is still preparing, remember
        # the request and begin playback as soon as the controls are installed.
        self._play_shortcut = QShortcut(QKeySequence("Space"), self)
        self._play_shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        self._play_shortcut.setAutoRepeat(False)
        self._play_shortcut.activated.connect(self._toggle_playback)

        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.setInterval(max(1, int(round(1000.0 / self.fps))))
        self.timer.timeout.connect(self._advance)

        self.scrub_timer = QTimer(self)
        self.scrub_timer.setSingleShot(True)
        self.scrub_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.scrub_timer.setInterval(max(16, int(round(1000.0 / min(self.fps, 30)))))
        self.scrub_timer.timeout.connect(self._render_pending_scrub)

        loading = QWidget(self)
        layout = QVBoxLayout(loading)
        layout.addStretch(1)
        self._loading_label = QLabel("Preparing animation…", loading)
        self._loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._loading_label)
        layout.addStretch(1)
        self.setCentralWidget(loading)

        # The zero-delay callback runs after show(), allowing the window shell to
        # appear before the Matplotlib scene is prepared.
        QTimer.singleShot(35, self._prepare_scene)

    @property
    def display_mode(self) -> DisplayMode:
        if self.mode_combo is None:
            return self._initial_display_mode
        return str(self.mode_combo.currentData() or "cumulative")  # type: ignore[return-value]

    def _prepare_scene(self) -> None:
        if self._closed or self._preparing or self.scene is not None:
            return
        self._preparing = True
        try:
            scene = build_projection_animation_scene(
                self._project,
                interactive=True,
                point_limit=self._point_limit,
                preserve_depth_order=False,
            )
        except Exception as exc:
            self._loading_label.setText("Could not prepare the animation.")
            QMessageBox.critical(self, "Could not prepare projection animation", str(exc))
            self._preparing = False
            return
        if self._closed:
            scene.close()
            return
        self.scene = scene
        self._install_viewer_ui()
        self._preparing = False

    def _install_viewer_ui(self) -> None:
        scene = self.scene
        if scene is None:
            return
        central = QWidget(self)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(5)

        self.canvas = FigureCanvasQTAgg(scene.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        outer.addWidget(self.toolbar)
        outer.addWidget(self.canvas, 1)

        controls = QHBoxLayout()
        controls.setSpacing(6)
        self.play_button = QPushButton("Play")
        self.restart_button = QPushButton("Restart")
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Cumulative", "cumulative")
        self.mode_combo.addItem("Trail", "trail")
        self.mode_combo.addItem("Trail + afterimage", "trail-afterimage")
        mode_index = self.mode_combo.findData(self._initial_display_mode)
        self.mode_combo.setCurrentIndex(max(0, mode_index))
        self.loop_checkbox = QCheckBox("Loop")
        self.loop_checkbox.setChecked(False)
        self.time_label = QLabel("")
        controls.addWidget(self.play_button)
        controls.addWidget(self.restart_button)
        controls.addWidget(QLabel("Display"))
        controls.addWidget(self.mode_combo)
        controls.addWidget(self.loop_checkbox)
        controls.addStretch(1)
        controls.addWidget(self.time_label)
        outer.addLayout(controls)

        self.timeline = QSlider(Qt.Orientation.Horizontal)
        self.timeline.setTracking(True)
        outer.addWidget(self.timeline)

        old = self.centralWidget()
        self.setCentralWidget(central)
        if old is not None and old is not central:
            old.deleteLater()

        self.play_button.clicked.connect(self._toggle_playback)
        self.restart_button.clicked.connect(self._restart)
        self.timeline.valueChanged.connect(self._slider_changed)
        self.timeline.sliderPressed.connect(self._slider_pressed)
        self.timeline.sliderReleased.connect(self._slider_released)
        self.mode_combo.currentIndexChanged.connect(self._mode_changed)
        self.canvas.mpl_connect("draw_event", self._canvas_drawn)
        self.canvas.mpl_connect("resize_event", self._canvas_resized)

        for artist in scene.dynamic_artists:
            try:
                artist.set_animated(True)
            except Exception:
                pass

        self._rebuild_frame_times(preserve_fraction=False)
        self._show_frame(0, force=True, force_title=True)
        if self._pending_space_toggle:
            self._pending_space_toggle = False
            self._toggle_playback()

    def _rebuild_frame_times(self, *, preserve_fraction: bool) -> None:
        if self.scene is None or self.timeline is None:
            return
        previous_max = max(1, self.timeline.maximum())
        fraction = float(self.timeline.value()) / previous_max if preserve_fraction else 0.0
        self.frame_times = animation_frame_times(
            self.scene.first_time_ms,
            self.scene.final_time_ms,
            fps=self.fps,
            duration_s=self.duration_s,
            display_mode=self.display_mode,
            afterimage_ms=self.afterimage_ms,
        )
        self._updating_slider = True
        try:
            self.timeline.setRange(0, max(0, len(self.frame_times) - 1))
            self.timeline.setValue(int(round(fraction * self.timeline.maximum())))
        finally:
            self._updating_slider = False

    def _canvas_resized(self, _event) -> None:
        self._blit_background = None

    def _canvas_drawn(self, _event) -> None:
        if self._recaching_blit or self.canvas is None or self.scene is None:
            return
        try:
            self._blit_background = self.canvas.copy_from_bbox(self.scene.figure.bbox)
            self._blit_dynamic_artists()
        except Exception:
            self._blit_background = None

    def _recache_blit(self) -> None:
        if self.canvas is None or self.scene is None:
            return
        self._recaching_blit = True
        try:
            self.canvas.draw()
            self._blit_background = self.canvas.copy_from_bbox(self.scene.figure.bbox)
        finally:
            self._recaching_blit = False
        self._blit_dynamic_artists()

    def _blit_dynamic_artists(self) -> None:
        if self._blit_background is None or self.canvas is None or self.scene is None:
            return
        try:
            self.canvas.restore_region(self._blit_background)
            for artist in self.scene.dynamic_artists:
                self.scene.figure.draw_artist(artist)
            self.canvas.blit(self.scene.figure.bbox)
            self.canvas.flush_events()
        except Exception:
            self._blit_background = None
            self.canvas.draw_idle()

    def _show_frame(
        self,
        index: int,
        *,
        force: bool = False,
        force_title: bool = False,
    ) -> None:
        if (
            self.scene is None
            or self.timeline is None
            or self.time_label is None
            or len(self.frame_times) == 0
        ):
            return
        index = max(0, min(int(index), len(self.frame_times) - 1))
        if not force and index == self._current_frame:
            return
        current = float(self.frame_times[index])
        now = time.perf_counter()
        update_title = force_title or not self._playing or (now - self._last_title_update_s) >= 0.10
        visible = self.scene.update(
            current,
            display_mode=self.display_mode,
            trail_ms=self.trail_ms,
            afterimage_ms=self.afterimage_ms,
            update_title=update_title,
        )
        if update_title:
            self._last_title_update_s = now
        self._current_frame = index
        self._updating_slider = True
        try:
            self.timeline.setValue(index)
        finally:
            self._updating_slider = False

        displayed = self.scene.displayed_source_count
        total = max(displayed, int(self.scene.total_source_count))
        if displayed < total:
            population = f"{visible:,}/{displayed:,} visible · {total:,} total"
        else:
            population = f"{visible:,}/{total:,} visible"
        self.time_label.setText(
            f"Source time: {current:.3f} ms · {population}"
        )

        if self._blit_background is None:
            self._recache_blit()
        else:
            self._blit_dynamic_artists()

    def _toggle_playback(self) -> None:
        if self.timeline is None or self.play_button is None:
            if not self._closed:
                self._pending_space_toggle = not self._pending_space_toggle
            return
        if self._playing:
            self._pause(force_title=True)
            return
        if self.timeline.value() >= self.timeline.maximum():
            self._show_frame(0, force=True, force_title=True)
        self._playing = True
        self.play_button.setText("Pause")
        self._play_epoch_frame = int(self.timeline.value())
        self._play_epoch_s = time.perf_counter()
        self.timer.start()

    def _pause(self, *, force_title: bool = False) -> None:
        self._playing = False
        self.timer.stop()
        if self.play_button is not None:
            self.play_button.setText("Play")
        if force_title and self._current_frame >= 0:
            self._show_frame(self._current_frame, force=True, force_title=True)

    def _restart(self) -> None:
        self._pause()
        self._show_frame(0, force=True, force_title=True)

    def _advance(self) -> None:
        count = len(self.frame_times)
        if not self._playing or count <= 0 or self.loop_checkbox is None:
            return
        elapsed_frames = int((time.perf_counter() - self._play_epoch_s) * self.fps)
        absolute = self._play_epoch_frame + elapsed_frames
        if self.loop_checkbox.isChecked():
            target = absolute % count
        elif absolute >= count:
            self._show_frame(count - 1, force=True, force_title=True)
            self._pause()
            return
        else:
            target = absolute
        self._show_frame(target)

    def _slider_pressed(self) -> None:
        self._scrubbing = True
        self._pause()

    def _slider_released(self) -> None:
        if self.timeline is None:
            return
        self._scrubbing = False
        self.scrub_timer.stop()
        self._pending_scrub_index = None
        self._show_frame(self.timeline.value(), force=True, force_title=True)

    def _slider_changed(self, value: int) -> None:
        if self._updating_slider:
            return
        self._pending_scrub_index = int(value)
        if not self.scrub_timer.isActive():
            self.scrub_timer.start()

    def _render_pending_scrub(self) -> None:
        if self._pending_scrub_index is None:
            return
        value = self._pending_scrub_index
        self._pending_scrub_index = None
        self._show_frame(value, force=True, force_title=not self._scrubbing)
        if self._pending_scrub_index is not None:
            self.scrub_timer.start()

    def _mode_changed(self, _index: int) -> None:
        if self.timeline is None:
            return
        self._pause()
        fraction = (
            float(self.timeline.value()) / max(1, self.timeline.maximum())
            if self.timeline.maximum() > 0
            else 0.0
        )
        self._rebuild_frame_times(preserve_fraction=False)
        target = int(round(fraction * self.timeline.maximum()))
        self._current_frame = -1
        self._show_frame(target, force=True, force_title=True)

    def closeEvent(self, event) -> None:
        self._closed = True
        self._pause()
        self.scrub_timer.stop()
        if self.scene is not None:
            self.scene.close()
        super().closeEvent(event)


def run_projection_animation_viewer(
    project: LMAProject,
    *,
    display_mode: DisplayMode = "cumulative",
    trail_ms: float = 30.0,
    afterimage_ms: float = 30.0,
    fps: int = 30,
    duration_s: float = 15.0,
    point_limit: int | None = None,
) -> int:
    app = QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QApplication([])
    window = ProjectionAnimationViewer(
        project,
        display_mode=display_mode,
        trail_ms=trail_ms,
        afterimage_ms=afterimage_ms,
        fps=fps,
        duration_s=duration_s,
        point_limit=point_limit,
    )
    window.show()
    if owns_app:
        return int(app.exec())
    return 0


__all__ = ["ProjectionAnimationViewer", "run_projection_animation_viewer"]
