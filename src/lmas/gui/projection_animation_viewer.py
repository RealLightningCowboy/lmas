from __future__ import annotations

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
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
    """Interactive playback of 3D source development in linked 2D projections."""

    def __init__(
        self,
        project: LMAProject,
        *,
        display_mode: DisplayMode = "cumulative",
        trail_ms: float = 30.0,
        afterimage_ms: float = 30.0,
        fps: int = 30,
        duration_s: float = 15.0,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle(f"{project.name} — Projection animation")
        self.resize(1500, 940)

        self.scene: ProjectionAnimationScene = build_projection_animation_scene(project)
        self.trail_ms = float(trail_ms)
        self.afterimage_ms = float(afterimage_ms)
        self.fps = max(1, int(fps))
        self.duration_s = max(0.1, float(duration_s))
        self._playing = False
        self._updating_slider = False

        central = QWidget(self)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(5)

        self.canvas = FigureCanvasQTAgg(self.scene.figure)
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
        mode_index = self.mode_combo.findData(display_mode)
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
        self.setCentralWidget(central)

        self.timer = QTimer(self)
        self.timer.setInterval(max(1, int(round(1000.0 / self.fps))))
        self.timer.timeout.connect(self._advance)
        self.play_button.clicked.connect(self._toggle_playback)
        self.restart_button.clicked.connect(self._restart)
        self.timeline.valueChanged.connect(self._slider_changed)
        self.mode_combo.currentIndexChanged.connect(self._mode_changed)

        self._rebuild_frame_times(preserve_fraction=False)
        self._show_frame(0)

    @property
    def display_mode(self) -> DisplayMode:
        return str(self.mode_combo.currentData() or "cumulative")  # type: ignore[return-value]

    def _rebuild_frame_times(self, *, preserve_fraction: bool) -> None:
        previous_max = max(1, self.timeline.maximum()) if hasattr(self, "timeline") else 1
        fraction = (
            float(self.timeline.value()) / previous_max
            if preserve_fraction and hasattr(self, "timeline")
            else 0.0
        )
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

    def _show_frame(self, index: int) -> None:
        if len(self.frame_times) == 0:
            return
        index = max(0, min(int(index), len(self.frame_times) - 1))
        current = float(self.frame_times[index])
        visible = self.scene.update(
            current,
            display_mode=self.display_mode,
            trail_ms=self.trail_ms,
            afterimage_ms=self.afterimage_ms,
        )
        self._updating_slider = True
        try:
            self.timeline.setValue(index)
        finally:
            self._updating_slider = False
        self.time_label.setText(
            f"{current:.3f} ms · {visible:,}/{self.scene.colors.size:,} visible"
        )
        self.canvas.draw_idle()

    def _toggle_playback(self) -> None:
        if self._playing:
            self._pause()
        else:
            if self.timeline.value() >= self.timeline.maximum():
                self.timeline.setValue(0)
            self._playing = True
            self.play_button.setText("Pause")
            self.timer.start()

    def _pause(self) -> None:
        self._playing = False
        self.timer.stop()
        self.play_button.setText("Play")

    def _restart(self) -> None:
        self._pause()
        self._show_frame(0)

    def _advance(self) -> None:
        next_index = self.timeline.value() + 1
        if next_index > self.timeline.maximum():
            if self.loop_checkbox.isChecked():
                next_index = 0
            else:
                self._pause()
                return
        self._show_frame(next_index)

    def _slider_changed(self, value: int) -> None:
        if not self._updating_slider:
            self._show_frame(value)

    def _mode_changed(self, _index: int) -> None:
        fraction = (
            float(self.timeline.value()) / max(1, self.timeline.maximum())
            if self.timeline.maximum() > 0
            else 0.0
        )
        self._rebuild_frame_times(preserve_fraction=False)
        target = int(round(fraction * self.timeline.maximum()))
        self._show_frame(target)

    def closeEvent(self, event) -> None:
        self._pause()
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
    )
    window.show()
    if owns_app:
        return int(app.exec())
    return 0


__all__ = ["ProjectionAnimationViewer", "run_projection_animation_viewer"]
