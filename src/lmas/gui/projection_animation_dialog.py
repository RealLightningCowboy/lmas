from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QPushButton,
    QTabWidget, QVBoxLayout, QWidget,
)

from ..animation_batch import AnimationBatchJob, projection_batch_jobs
from ..errors import ConfigurationError
from ..model import PlotSpec
from ..output_naming import display_mode_label, safe_output_stem
from .numeric_editors import DeferredDoubleSpinBox, DeferredSpinBox


@dataclass(frozen=True)
class ProjectionAnimationOptions:
    batch_mode: bool
    output_path: Path
    display_mode: str
    trail_ms: float
    afterimage_ms: float
    fps: int
    duration_s: float
    hold_end_s: float
    width: int
    height: int
    video_quality: int
    custom_title: str = ""
    batch_jobs: tuple[AnimationBatchJob, ...] = ()
    overwrite_policy: str = "replace"
    continue_on_error: bool = True


class SaveProjectionAnimationDialog(QDialog):
    def __init__(
        self,
        *,
        default_path: str | Path,
        plot: PlotSpec,
        default_title: str = "",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Save linked projection animation")
        self.setMinimumSize(600, 470)
        self.resize(720, 620)
        spec = plot.validated()
        self._default_path = Path(default_path).expanduser()
        marker = "_projection_"
        raw_stem = self._default_path.stem
        self._output_base = raw_stem.split(marker, 1)[0] if marker in raw_stem else raw_stem
        self._output_theme = spec.theme
        self._default_title = str(default_title or spec.title or "").strip()
        self._path_tracks_options = True

        outer = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._single_tab(spec), "Single")
        self.tabs.addTab(self._batch_tab(spec), "Batch")
        outer.addWidget(self.tabs, 1)
        outer.addWidget(self._common_settings(spec))

        note = QLabel(
            "Batch jobs use the exact committed project view. Progress is printed per job in the launching terminal."
        )
        note.setWordWrap(True); outer.addWidget(note)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept_checked); buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)
        self.tabs.currentChanged.connect(self._update_enabled)
        self._update_enabled()
        self._refresh_batch_preview()

    def _single_tab(self, spec: PlotSpec) -> QWidget:
        tab = QWidget(); outer = QVBoxLayout(tab)
        output_group = QGroupBox("Output"); output_layout = QHBoxLayout(output_group)
        self.output_path = QLineEdit(str(self._default_path)); self.output_path.textEdited.connect(self._output_path_edited)
        browse = QPushButton("Browse…"); browse.clicked.connect(self._browse)
        output_layout.addWidget(self.output_path, 1); output_layout.addWidget(browse)
        outer.addWidget(output_group)
        group = QGroupBox("Projection animation"); form = QFormLayout(group)
        self.display_mode = self._display_combo(spec.three_d_display_mode)
        form.addRow("Display", self.display_mode)
        self.custom_title = QLineEdit(self._default_title)
        self.custom_title.setClearButtonEnabled(True)
        self.custom_title.setToolTip(
            "Leading title text; LMAS appends live source counts and source time."
        )
        form.addRow("Custom title", self.custom_title)
        outer.addWidget(group); outer.addStretch(1)
        self.display_mode.currentIndexChanged.connect(self._update_enabled)
        return tab

    def _batch_tab(self, spec: PlotSpec) -> QWidget:
        tab = QWidget(); outer = QVBoxLayout(tab)
        destination = QGroupBox("Batch destination"); form = QFormLayout(destination)
        self.batch_directory = QLineEdit(str(self._default_path.parent.resolve()))
        browse = QPushButton("Browse…"); browse.clicked.connect(self._browse_batch_directory)
        row = QWidget(); layout = QHBoxLayout(row); layout.setContentsMargins(0,0,0,0)
        layout.addWidget(self.batch_directory,1); layout.addWidget(browse)
        form.addRow("Directory", row)
        self.batch_base = QLineEdit(safe_output_stem(self._output_base)); form.addRow("Base filename", self.batch_base)
        self.batch_extension = QComboBox(); self.batch_extension.addItem("MP4 video", ".mp4"); self.batch_extension.addItem("Animated GIF", ".gif")
        if self._default_path.suffix.lower() == ".gif": self.batch_extension.setCurrentIndex(1)
        form.addRow("Format", self.batch_extension); outer.addWidget(destination)

        combinations = QGroupBox("Combinations"); cform = QFormLayout(combinations)
        self.batch_theme_checks = self._checks(("dark","light","space"), (spec.theme,), {"dark":"Dark","light":"Light","space":"Space"})
        cform.addRow("Themes", self._check_row(self.batch_theme_checks))
        displays = ("cumulative","trail","trail-afterimage")
        labels = {"cumulative":"Cumulative","trail":"Trail","trail-afterimage":"Trail + afterimage"}
        self.batch_display_checks = self._checks(displays, (spec.three_d_display_mode,), labels)
        cform.addRow("Display modes", self._check_row(self.batch_display_checks)); outer.addWidget(combinations)

        behavior = QGroupBox("Batch behavior"); bform = QFormLayout(behavior)
        self.batch_overwrite = QComboBox(); self.batch_overwrite.addItem("Skip existing files","skip"); self.batch_overwrite.addItem("Replace existing files","replace"); self.batch_overwrite.addItem("Stop when a file exists","fail"); self.batch_overwrite.setCurrentIndex(self.batch_overwrite.findData("replace"))
        bform.addRow("Existing files", self.batch_overwrite)
        self.batch_continue = QCheckBox("Continue after an individual job fails"); self.batch_continue.setChecked(True)
        bform.addRow("", self.batch_continue)
        self.batch_custom_title = QLineEdit(self._default_title)
        self.batch_custom_title.setClearButtonEnabled(True)
        self.batch_custom_title.setToolTip(
            "Leading title text shared by all queued projection animations; "
            "LMAS appends live source counts and source time."
        )
        bform.addRow("Custom title", self.batch_custom_title)
        outer.addWidget(behavior)
        self.batch_count = QLabel(); self.batch_preview = QPlainTextEdit(); self.batch_preview.setReadOnly(True); self.batch_preview.setMaximumHeight(125)
        outer.addWidget(self.batch_count); outer.addWidget(self.batch_preview)
        for widget in (self.batch_directory, self.batch_base): widget.textChanged.connect(self._refresh_batch_preview)
        self.batch_extension.currentIndexChanged.connect(self._refresh_batch_preview)
        for check in (*self.batch_theme_checks.values(), *self.batch_display_checks.values()): check.toggled.connect(self._refresh_batch_preview)
        return tab

    def _common_settings(self, spec: PlotSpec) -> QWidget:
        group = QGroupBox("Shared render settings"); form = QFormLayout(group); form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        transition = QWidget(); row = QHBoxLayout(transition); row.setContentsMargins(0,0,0,0)
        self.transition_ms = DeferredDoubleSpinBox(); self.transition_ms.setRange(.001,1e9); self.transition_ms.setDecimals(3); self.transition_ms.setValue(spec.three_d_trail_ms); self.transition_ms.setSuffix(" ms")
        self.trail_ms = self.transition_ms; self.afterimage_ms = self.transition_ms
        row.addWidget(QLabel("Transition")); row.addWidget(self.transition_ms); row.addStretch(1)
        form.addRow("Timing", transition)
        timing = QWidget(); row = QHBoxLayout(timing); row.setContentsMargins(0,0,0,0)
        self.fps = DeferredSpinBox(); self.fps.setRange(1,240); self.fps.setValue(spec.three_d_playback_fps); self.fps.setSuffix(" fps")
        self.duration_s = DeferredDoubleSpinBox(); self.duration_s.setRange(.1,3600); self.duration_s.setDecimals(1); self.duration_s.setValue(spec.three_d_playback_duration_s); self.duration_s.setSuffix(" s")
        self.hold_end_s = DeferredDoubleSpinBox(); self.hold_end_s.setRange(0,120); self.hold_end_s.setDecimals(1); self.hold_end_s.setValue(spec.three_d_hold_end_s); self.hold_end_s.setSuffix(" s hold")
        row.addWidget(self.fps); row.addWidget(self.duration_s); row.addWidget(self.hold_end_s); row.addStretch(1)
        form.addRow("Playback", timing)
        self.preset = QComboBox(); self.preset.addItem("Fast — 1280 × 720",(1280,720,6)); self.preset.addItem("Standard — 1600 × 900",(1600,900,8)); self.preset.addItem("HD — 1920 × 1080",(1920,1080,8)); self.preset.addItem("2K — 2560 × 1440",(2560,1440,9)); self.preset.setCurrentIndex(1)
        form.addRow("Quality", self.preset)
        return group

    @staticmethod
    def _display_combo(current: str) -> QComboBox:
        combo=QComboBox(); combo.addItem("Cumulative","cumulative"); combo.addItem("Trail","trail"); combo.addItem("Trail + afterimage","trail-afterimage")
        combo.setCurrentIndex(max(0, combo.findData(current))); return combo

    @staticmethod
    def _checks(values, selected, labels):
        result={}
        for value in values:
            check=QCheckBox(labels[value]); check.setChecked(value in selected); result[value]=check
        return result

    @staticmethod
    def _check_row(checks):
        row=QWidget(); layout=QHBoxLayout(row); layout.setContentsMargins(0,0,0,0)
        for check in checks.values(): layout.addWidget(check)
        layout.addStretch(1); return row

    def _output_path_edited(self, _value: str) -> None: self._path_tracks_options=False
    def _refresh_default_output(self) -> None:
        if not self._path_tracks_options: return
        display=display_mode_label(str(self.display_mode.currentData()))
        filename="_".join([safe_output_stem(self._output_base), "projection", "development", display, self._output_theme])+self._default_path.suffix.lower()
        self.output_path.setText(str(self._default_path.parent/filename))
    def _browse(self) -> None:
        selected,_=QFileDialog.getSaveFileName(self,"Save linked projection animation",self.output_path.text(),"Video (*.mp4);;Animated GIF (*.gif)")
        if selected: self._path_tracks_options=False; self.output_path.setText(selected)
    def _browse_batch_directory(self) -> None:
        selected=QFileDialog.getExistingDirectory(self,"Choose batch output directory",self.batch_directory.text())
        if selected: self.batch_directory.setText(selected)
    def _update_enabled(self) -> None:
        mode=str(self.display_mode.currentData()); self.transition_ms.setEnabled(mode in {"trail","trail-afterimage"} or self.tabs.currentIndex()==1); self._refresh_default_output()
    def _selected(self, checks): return tuple(value for value,check in checks.items() if check.isChecked())
    def _batch_jobs(self):
        return projection_batch_jobs(output_directory=self.batch_directory.text().strip(),base_stem=self.batch_base.text().strip(),extension=str(self.batch_extension.currentData()),themes=self._selected(self.batch_theme_checks),display_modes=self._selected(self.batch_display_checks))
    def _refresh_batch_preview(self,*_args) -> None:
        try: jobs=self._batch_jobs()
        except Exception as exc: self.batch_count.setText(f"Batch not ready: {exc}"); self.batch_preview.setPlainText(""); return
        self.batch_count.setText(f"{len(jobs)} animation job(s) will be queued")
        names=[Path(job.output_path).name for job in jobs]; preview=names[:10]
        if len(names)>10: preview.append(f"… and {len(names)-10} more")
        self.batch_preview.setPlainText("\n".join(preview))
    def _accept_checked(self) -> None:
        try: self.options()
        except Exception as exc:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self,"Invalid animation settings",str(exc)); return
        self.accept()
    def options(self) -> ProjectionAnimationOptions:
        width,height,quality=self.preset.currentData()
        common=dict(trail_ms=float(self.transition_ms.value()),afterimage_ms=float(self.transition_ms.value()),fps=int(self.fps.value()),duration_s=float(self.duration_s.value()),hold_end_s=float(self.hold_end_s.value()),width=int(width),height=int(height),video_quality=int(quality))
        if self.tabs.currentIndex()==1:
            jobs=self._batch_jobs()
            return ProjectionAnimationOptions(True,Path(self.batch_directory.text().strip()).expanduser().resolve(),"cumulative",custom_title=self.batch_custom_title.text().strip(),batch_jobs=jobs,overwrite_policy=str(self.batch_overwrite.currentData()),continue_on_error=bool(self.batch_continue.isChecked()),**common)
        output=Path(self.output_path.text().strip()).expanduser()
        if output.suffix.lower() not in {".mp4",".gif"}: raise ConfigurationError("Animation output must end in .mp4 or .gif")
        return ProjectionAnimationOptions(False,output.resolve(),str(self.display_mode.currentData()),custom_title=self.custom_title.text().strip(),**common)


__all__=["ProjectionAnimationOptions","SaveProjectionAnimationDialog"]
