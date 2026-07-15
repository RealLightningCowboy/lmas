from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QPushButton,
    QTabWidget, QVBoxLayout, QWidget,
)

from ..animation_batch import (
    AnimationBatchJob,
    THREE_D_BATCH_DEFAULT_AFTERIMAGE_MS,
    THREE_D_BATCH_DEFAULT_ANIMATION_MODES,
    THREE_D_BATCH_DEFAULT_CONTINUE_ON_ERROR,
    THREE_D_BATCH_DEFAULT_DISPLAY_MODES,
    THREE_D_BATCH_DEFAULT_DURATION_S,
    THREE_D_BATCH_DEFAULT_EXTENSION,
    THREE_D_BATCH_DEFAULT_FPS,
    THREE_D_BATCH_DEFAULT_HOLD_END_S,
    THREE_D_BATCH_DEFAULT_ORBIT_SPEED_DEG_S,
    THREE_D_BATCH_DEFAULT_OVERWRITE,
    THREE_D_BATCH_DEFAULT_POINT_SIZE,
    THREE_D_BATCH_DEFAULT_RENDER_PROFILE,
    THREE_D_BATCH_DEFAULT_THEMES,
    THREE_D_BATCH_DEFAULT_TRAIL_MS,
    THREE_D_BATCH_DEFAULT_VIDEO_QUALITY,
    THREE_D_BATCH_DEFAULT_WINDOW,
    three_d_batch_jobs,
)
from ..errors import ConfigurationError
from ..model import PlotSpec
from ..output_naming import display_mode_label, safe_output_stem
from .numeric_editors import DeferredDoubleSpinBox, DeferredSpinBox


@dataclass(frozen=True)
class AnimationOptions:
    batch_mode: bool
    output_path: Path
    mode: str
    display_mode: str
    trail_ms: float
    afterimage_ms: float
    fps: int
    duration_s: float
    hold_end_s: float
    orbit_speed_deg_s: float
    point_size: float
    render_profile: str
    width: int
    height: int
    video_quality: int
    camera_path: Path | None
    batch_jobs: tuple[AnimationBatchJob, ...] = ()
    overwrite_policy: str = "replace"
    continue_on_error: bool = True


class SaveAnimationDialog(QDialog):
    def __init__(self, *, default_path: str | Path, plot: PlotSpec, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Save LMAS 3D animation")
        self.setMinimumSize(620, 520)
        self.resize(760, 700)
        spec = plot.validated()
        self._default_path = Path(default_path).expanduser()
        marker = "_3d_"
        raw_stem = self._default_path.stem
        self._output_base = raw_stem.split(marker, 1)[0] if marker in raw_stem else raw_stem
        self._output_theme = spec.theme
        self._path_tracks_options = True

        outer = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._single_tab(spec), "Single")
        self.tabs.addTab(self._batch_tab(spec), "Batch")
        outer.addWidget(self.tabs, 1)
        outer.addWidget(self._common_settings(spec))
        self._active_shared_tab = 0
        self._shared_state_by_tab = {
            0: self._capture_shared_state(),
            1: self._batch_shared_defaults(),
        }
        note = QLabel(
            "Batch rendering runs one isolated job at a time. The final hold freezes source development while an enabled orbit continues."
        )
        note.setWordWrap(True); outer.addWidget(note)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept_checked); buttons.rejected.connect(self.reject); outer.addWidget(buttons)
        self.mode.currentIndexChanged.connect(self._update_enabled)
        self.display_mode.currentIndexChanged.connect(self._update_enabled)
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self._update_enabled(); self._refresh_batch_preview()

    def _single_tab(self, spec: PlotSpec) -> QWidget:
        tab=QWidget(); outer=QVBoxLayout(tab)
        destination=QGroupBox("Output"); row=QHBoxLayout(destination)
        self.output_path=QLineEdit(str(self._default_path)); self.output_path.textEdited.connect(self._output_path_edited)
        browse=QPushButton("Browse…"); browse.clicked.connect(self._browse_output)
        row.addWidget(self.output_path,1); row.addWidget(browse); outer.addWidget(destination)
        animation=QGroupBox("Animation"); form=QFormLayout(animation); form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.mode=QComboBox(); self.mode.addItem("Flash development","develop"); self.mode.addItem("Camera orbit","orbit"); self.mode.addItem("Development + orbit","develop-orbit")
        form.addRow("Mode",self.mode)
        self.display_mode=QComboBox(); self.display_mode.addItem("Cumulative","cumulative"); self.display_mode.addItem("Trail","trail"); self.display_mode.addItem("Trail + afterimage","trail-afterimage")
        self.display_mode.setCurrentIndex(max(0,self.display_mode.findData(spec.three_d_display_mode)))
        form.addRow("Display",self.display_mode); outer.addWidget(animation); outer.addStretch(1)
        return tab

    def _batch_tab(self, spec: PlotSpec) -> QWidget:
        tab=QWidget(); outer=QVBoxLayout(tab)
        destination=QGroupBox("Batch destination"); form=QFormLayout(destination)
        self.batch_directory=QLineEdit(str(self._default_path.parent.resolve()))
        browse=QPushButton("Browse…"); browse.clicked.connect(self._browse_batch_directory)
        row=QWidget(); layout=QHBoxLayout(row); layout.setContentsMargins(0,0,0,0); layout.addWidget(self.batch_directory,1); layout.addWidget(browse)
        form.addRow("Directory",row)
        self.batch_base=QLineEdit(safe_output_stem(self._output_base)); form.addRow("Base filename",self.batch_base)
        self.batch_extension=QComboBox(); self.batch_extension.addItem("MP4 video",".mp4"); self.batch_extension.addItem("Animated GIF",".gif")
        self.batch_extension.setCurrentIndex(max(0,self.batch_extension.findData(THREE_D_BATCH_DEFAULT_EXTENSION)))
        form.addRow("Format",self.batch_extension); outer.addWidget(destination)

        combinations=QGroupBox("Combinations"); cform=QFormLayout(combinations)
        labels={"dark":"Dark","light":"Light","space":"Space"}
        self.batch_theme_checks=self._checks(("dark","light","space"),THREE_D_BATCH_DEFAULT_THEMES,labels)
        cform.addRow("Themes",self._check_row(self.batch_theme_checks))
        labels={"develop":"Development","orbit":"Orbit","develop-orbit":"Development + orbit"}
        self.batch_mode_checks=self._checks(("develop","orbit","develop-orbit"),THREE_D_BATCH_DEFAULT_ANIMATION_MODES,labels)
        cform.addRow("Animation kinds",self._check_row(self.batch_mode_checks))
        labels={"cumulative":"Cumulative","trail":"Trail","trail-afterimage":"Trail + afterimage"}
        self.batch_display_checks=self._checks(("cumulative","trail","trail-afterimage"),THREE_D_BATCH_DEFAULT_DISPLAY_MODES,labels)
        cform.addRow("Display modes",self._check_row(self.batch_display_checks)); outer.addWidget(combinations)

        behavior=QGroupBox("Batch behavior"); bform=QFormLayout(behavior)
        self.batch_overwrite=QComboBox(); self.batch_overwrite.addItem("Skip existing files","skip"); self.batch_overwrite.addItem("Replace existing files","replace"); self.batch_overwrite.addItem("Stop when a file exists","fail"); self.batch_overwrite.setCurrentIndex(self.batch_overwrite.findData(THREE_D_BATCH_DEFAULT_OVERWRITE))
        bform.addRow("Existing files",self.batch_overwrite)
        self.batch_continue=QCheckBox("Continue after an individual job fails"); self.batch_continue.setChecked(THREE_D_BATCH_DEFAULT_CONTINUE_ON_ERROR); bform.addRow("",self.batch_continue)
        outer.addWidget(behavior)
        self.batch_count=QLabel(); self.batch_preview=QPlainTextEdit(); self.batch_preview.setReadOnly(True); self.batch_preview.setMaximumHeight(125)
        outer.addWidget(self.batch_count); outer.addWidget(self.batch_preview)
        for widget in (self.batch_directory,self.batch_base): widget.textChanged.connect(self._refresh_batch_preview)
        self.batch_extension.currentIndexChanged.connect(self._refresh_batch_preview)
        for check in (*self.batch_theme_checks.values(),*self.batch_mode_checks.values(),*self.batch_display_checks.values()): check.toggled.connect(self._refresh_batch_preview)
        return tab

    def _common_settings(self,spec:PlotSpec)->QWidget:
        group=QGroupBox("Shared render settings"); form=QFormLayout(group); form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        transition=QWidget(); row=QHBoxLayout(transition); row.setContentsMargins(0,0,0,0)
        self.transition_ms=DeferredDoubleSpinBox(); self.transition_ms.setRange(.001,1e9); self.transition_ms.setDecimals(3); self.transition_ms.setValue(spec.three_d_trail_ms); self.transition_ms.setSuffix(" ms")
        self.trail_ms=self.transition_ms; self.afterimage_ms=self.transition_ms
        row.addWidget(QLabel("Transition")); row.addWidget(self.transition_ms); row.addStretch(1); form.addRow("Timing",transition)
        timing=QWidget(); row=QHBoxLayout(timing); row.setContentsMargins(0,0,0,0)
        self.fps=DeferredSpinBox(); self.fps.setRange(1,240); self.fps.setValue(spec.three_d_playback_fps); self.fps.setSuffix(" fps")
        self.duration_s=DeferredDoubleSpinBox(); self.duration_s.setRange(.1,3600); self.duration_s.setDecimals(1); self.duration_s.setValue(spec.three_d_playback_duration_s); self.duration_s.setSuffix(" s")
        self.hold_end_s=DeferredDoubleSpinBox(); self.hold_end_s.setRange(0,120); self.hold_end_s.setDecimals(1); self.hold_end_s.setValue(spec.three_d_hold_end_s); self.hold_end_s.setSuffix(" s hold")
        row.addWidget(self.fps); row.addWidget(self.duration_s); row.addWidget(self.hold_end_s); row.addStretch(1); form.addRow("Playback",timing)
        self.orbit_speed=DeferredDoubleSpinBox(); self.orbit_speed.setRange(-720,720); self.orbit_speed.setDecimals(1); self.orbit_speed.setValue(spec.three_d_orbit_speed_deg_s); self.orbit_speed.setSuffix(" °/s"); form.addRow("Orbit camera speed",self.orbit_speed)
        self.preset=QComboBox(); self.preset.addItem("Fast — 960 × 640",(960,640,"compatible",5)); self.preset.addItem("Standard — 1400 × 900",(1400,900,"compatible",7)); self.preset.addItem("HD — 1920 × 1080",(1920,1080,"compatible",8)); self.preset.addItem("2K — 2560 × 1440",(2560,1440,"quality",9)); self.preset.setCurrentIndex(1); form.addRow("Preset",self.preset)
        self.point_size=DeferredDoubleSpinBox(); self.point_size.setRange(.1,100); self.point_size.setDecimals(2); self.point_size.setValue(max(.1,spec.point_size)); form.addRow("Point size",self.point_size)
        camera=QWidget(); row=QHBoxLayout(camera); row.setContentsMargins(0,0,0,0)
        self.camera_path=QLineEdit(); self.camera_path.setPlaceholderText("Optional saved camera JSON")
        browse=QPushButton("Browse…"); browse.clicked.connect(self._browse_camera); row.addWidget(self.camera_path,1); row.addWidget(browse); form.addRow("Camera",camera)
        return group

    def _capture_shared_state(self) -> dict:
        return {
            "trail_ms": float(self.transition_ms.value()),
            "afterimage_ms": float(self.transition_ms.value()),
            "fps": int(self.fps.value()),
            "duration_s": float(self.duration_s.value()),
            "hold_end_s": float(self.hold_end_s.value()),
            "orbit_speed_deg_s": float(self.orbit_speed.value()),
            "preset_index": int(self.preset.currentIndex()),
            "point_size": float(self.point_size.value()),
            "camera_path": self.camera_path.text(),
        }

    def _batch_shared_defaults(self) -> dict:
        width, height = THREE_D_BATCH_DEFAULT_WINDOW
        preset_index = 0
        for index in range(self.preset.count()):
            item = self.preset.itemData(index)
            if item and tuple(item[:2]) == (width, height):
                preset_index = index
                break
        return {
            "trail_ms": THREE_D_BATCH_DEFAULT_TRAIL_MS,
            "afterimage_ms": THREE_D_BATCH_DEFAULT_AFTERIMAGE_MS,
            "fps": THREE_D_BATCH_DEFAULT_FPS,
            "duration_s": THREE_D_BATCH_DEFAULT_DURATION_S,
            "hold_end_s": THREE_D_BATCH_DEFAULT_HOLD_END_S,
            "orbit_speed_deg_s": THREE_D_BATCH_DEFAULT_ORBIT_SPEED_DEG_S,
            "preset_index": preset_index,
            "point_size": THREE_D_BATCH_DEFAULT_POINT_SIZE,
            "camera_path": "",
        }

    def _load_shared_state(self, state: dict) -> None:
        self.transition_ms.setValue(float(state["trail_ms"]))
        self.fps.setValue(int(state["fps"]))
        self.duration_s.setValue(float(state["duration_s"]))
        self.hold_end_s.setValue(float(state["hold_end_s"]))
        self.orbit_speed.setValue(float(state["orbit_speed_deg_s"]))
        self.preset.setCurrentIndex(int(state["preset_index"]))
        self.point_size.setValue(float(state["point_size"]))
        self.camera_path.setText(str(state.get("camera_path", "")))

    def _on_tab_changed(self, index: int) -> None:
        old = int(getattr(self, "_active_shared_tab", 0))
        self._shared_state_by_tab[old] = self._capture_shared_state()
        self._active_shared_tab = int(index)
        self._load_shared_state(self._shared_state_by_tab[int(index)])
        self._update_enabled()

    @staticmethod
    def _checks(values,selected,labels):
        result={}
        for value in values:
            check=QCheckBox(labels[value]); check.setChecked(value in selected); result[value]=check
        return result
    @staticmethod
    def _check_row(checks):
        row=QWidget(); layout=QHBoxLayout(row); layout.setContentsMargins(0,0,0,0)
        for check in checks.values(): layout.addWidget(check)
        layout.addStretch(1); return row
    def _selected(self,checks): return tuple(value for value,check in checks.items() if check.isChecked())
    def _output_path_edited(self,_value): self._path_tracks_options=False
    def _refresh_default_output(self):
        if not self._path_tracks_options:return
        mode=display_mode_label(str(self.mode.currentData())); display=display_mode_label(str(self.display_mode.currentData()))
        parts = [safe_output_stem(self._output_base), "3d", mode]
        if str(self.mode.currentData())!="orbit":parts.append(display)
        parts.append(self._output_theme); self.output_path.setText(str(self._default_path.parent/("_".join(parts)+self._default_path.suffix.lower())))
    def _browse_output(self):
        selected,_=QFileDialog.getSaveFileName(self,"Save LMAS 3D animation",self.output_path.text(),"Video (*.mp4);;Animated GIF (*.gif)")
        if selected:self._path_tracks_options=False;self.output_path.setText(selected)
    def _browse_batch_directory(self):
        selected=QFileDialog.getExistingDirectory(self,"Choose batch output directory",self.batch_directory.text())
        if selected:self.batch_directory.setText(selected)
    def _browse_camera(self):
        selected,_=QFileDialog.getOpenFileName(self,"Use saved LMAS camera",str(self._default_path.parent),"Camera JSON (*.json);;All files (*)")
        if selected:self.camera_path.setText(selected)
    def _update_enabled(self):
        display=str(self.display_mode.currentData()); mode=str(self.mode.currentData()); batch=self.tabs.currentIndex()==1
        self.transition_ms.setEnabled(batch or display in {"trail","trail-afterimage"}); self.orbit_speed.setEnabled(batch or mode in {"orbit","develop-orbit"}); self.display_mode.setEnabled(mode!="orbit"); self._refresh_default_output()
    def _batch_jobs(self):
        return three_d_batch_jobs(output_directory=self.batch_directory.text().strip(),base_stem=self.batch_base.text().strip(),extension=str(self.batch_extension.currentData()),themes=self._selected(self.batch_theme_checks),display_modes=self._selected(self.batch_display_checks),animation_modes=self._selected(self.batch_mode_checks))
    def _refresh_batch_preview(self,*_args):
        try:jobs=self._batch_jobs()
        except Exception as exc:self.batch_count.setText(f"Batch not ready: {exc}");self.batch_preview.setPlainText("");return
        self.batch_count.setText(f"{len(jobs)} animation job(s) will be queued")
        names=[Path(job.output_path).name for job in jobs];preview=names[:10]
        if len(names)>10:preview.append(f"… and {len(names)-10} more")
        self.batch_preview.setPlainText("\n".join(preview))
    def _accept_checked(self):
        try:self.options()
        except Exception as exc:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self,"Invalid animation settings",str(exc));return
        self.accept()
    def options(self)->AnimationOptions:
        width,height,profile,quality=self.preset.currentData(); camera_text=self.camera_path.text().strip(); camera=Path(camera_text).expanduser().resolve() if camera_text else None
        if camera is not None and not camera.is_file():raise ConfigurationError(f"Camera file does not exist: {camera}")
        common=dict(trail_ms=float(self.transition_ms.value()),afterimage_ms=float(self.transition_ms.value()),fps=int(self.fps.value()),duration_s=float(self.duration_s.value()),hold_end_s=float(self.hold_end_s.value()),orbit_speed_deg_s=float(self.orbit_speed.value()),point_size=float(self.point_size.value()),render_profile=str(profile),width=int(width),height=int(height),video_quality=int(quality),camera_path=camera)
        if self.tabs.currentIndex()==1:
            jobs=self._batch_jobs();return AnimationOptions(True,Path(self.batch_directory.text().strip()).expanduser().resolve(),"develop","cumulative",batch_jobs=jobs,overwrite_policy=str(self.batch_overwrite.currentData()),continue_on_error=bool(self.batch_continue.isChecked()),**common)
        output=Path(self.output_path.text().strip()).expanduser()
        if output.suffix.lower() not in {".mp4",".gif"}:raise ConfigurationError("Animation output must end in .mp4 or .gif")
        return AnimationOptions(False,output.resolve(),str(self.mode.currentData()),str(self.display_mode.currentData()),**common)


__all__=["AnimationOptions","SaveAnimationDialog"]
