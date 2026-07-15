from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..figure_batch import FigureBatchJob, figure_batch_jobs
from ..figure_export import SUPPORTED_EXPORT_THEMES
from ..output_naming import safe_output_stem
from .numeric_editors import DeferredSpinBox


@dataclass(frozen=True)
class FigureSaveOptions:
    mode: str
    path: Path
    dpi: int
    title: str
    themes: tuple[str, ...]
    batch_theme_export: bool
    batch_jobs: tuple[FigureBatchJob, ...] = ()
    overwrite_policy: str = "replace"
    continue_on_error: bool = True
    dynamic_titles: bool = True


class SaveFigureDialog(QDialog):
    def __init__(
        self,
        *,
        default_path: Path,
        default_dpi: int,
        default_title: str,
        current_theme: str,
        current_color_by: str = "time",
        current_maximum_chi2: float = 1.0,
        current_log_color_scale: bool = False,
        available_color_fields: tuple[str, ...] = (
            "time", "altitude", "power", "stations", "chi2", "charge", "group"
        ),
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Save LMAS figure")
        self.setMinimumSize(620, 300)
        self.resize(700, 390)
        self.current_theme = str(current_theme).strip().lower()
        self.current_color_by = str(current_color_by).strip().lower()
        self.available_color_fields = tuple(str(value).strip().lower() for value in available_color_fields)
        self._default_path = Path(default_path).expanduser()
        raw_stem = self._default_path.stem
        self._base_stem = safe_output_stem(raw_stem.split("_projection", 1)[0])

        outer = QVBoxLayout(self)
        self.tabs = QTabWidget()
        outer.addWidget(self.tabs, 1)
        self.tabs.addTab(self._build_single_tab(default_dpi, default_title), "Single")
        self.tabs.addTab(
            self._build_batch_tab(
                default_dpi, default_title, current_maximum_chi2, current_log_color_scale
            ),
            "Batch",
        )

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept_checked)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

        self.tabs.currentChanged.connect(lambda *_: self._grow_to_fit())
        QTimer.singleShot(0, self._grow_to_fit)

    def _build_single_tab(self, default_dpi: int, default_title: str) -> QWidget:
        tab = QWidget()
        outer = QVBoxLayout(tab)
        form = QFormLayout()

        self.path_edit = QLineEdit(str(self._default_path))
        browse = QPushButton("Browse…")
        path_row = QWidget()
        path_layout = QHBoxLayout(path_row)
        path_layout.setContentsMargins(0, 0, 0, 0)
        path_layout.addWidget(self.path_edit, 1)
        path_layout.addWidget(browse)
        form.addRow("Filename", path_row)

        self.format_combo = self._format_combo(self._default_path.suffix)
        form.addRow("Format", self.format_combo)

        self.dpi_spin = DeferredSpinBox()
        self.dpi_spin.setRange(72, 1200)
        self.dpi_spin.setValue(int(default_dpi))
        self.dpi_spin.setSuffix(" dpi")
        form.addRow("DPI", self.dpi_spin)

        self.title_edit = QLineEdit(default_title)
        self.title_edit.setClearButtonEnabled(True)
        form.addRow("Custom title", self.title_edit)

        self.multi_theme_check = QCheckBox("Save multiple themes")
        self.multi_theme_check.setChecked(False)
        form.addRow("", self.multi_theme_check)

        self.theme_row, self.theme_checks = self._theme_selector(all_checked=True)
        self.theme_row.setVisible(False)
        form.addRow("Themes", self.theme_row)

        self.theme_note = QLabel(
            "Theme batching preserves the exact current subset, limits, title, point size, "
            "viewpoints, overlays, and color normalization."
        )
        self.theme_note.setWordWrap(True)
        self.theme_note.setVisible(False)

        outer.addLayout(form)
        outer.addWidget(self.theme_note)
        outer.addStretch(1)

        browse.clicked.connect(self._browse_single)
        self.format_combo.currentIndexChanged.connect(self._single_format_changed)
        self.multi_theme_check.toggled.connect(self._multi_theme_toggled)
        return tab

    def _build_batch_tab(
        self,
        default_dpi: int,
        default_title: str,
        current_maximum_chi2: float,
        current_log_color_scale: bool,
    ) -> QWidget:
        tab = QWidget()
        outer = QVBoxLayout(tab)

        destination = QGroupBox("Batch destination")
        form = QFormLayout(destination)
        self.batch_directory = QLineEdit(str(self._default_path.parent.resolve()))
        browse = QPushButton("Browse…")
        row = QWidget(); layout = QHBoxLayout(row); layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.batch_directory, 1); layout.addWidget(browse)
        form.addRow("Directory", row)
        self.batch_base_stem = QLineEdit(self._base_stem)
        form.addRow("Base filename", self.batch_base_stem)
        self.batch_format = self._format_combo(self._default_path.suffix)
        form.addRow("Format", self.batch_format)
        self.batch_dpi = DeferredSpinBox(); self.batch_dpi.setRange(72, 1200)
        self.batch_dpi.setValue(int(default_dpi)); self.batch_dpi.setSuffix(" dpi")
        form.addRow("DPI", self.batch_dpi)
        outer.addWidget(destination)

        combinations = QGroupBox("Combinations")
        combo_form = QFormLayout(combinations)
        self.batch_theme_row, self.batch_theme_checks = self._theme_selector(all_checked=False)
        self.batch_theme_checks.get(self.current_theme, next(iter(self.batch_theme_checks.values()))).setChecked(True)
        combo_form.addRow("Themes", self.batch_theme_row)

        color_labels = {
            "time": "Time", "altitude": "Altitude", "power": "Source Power",
            "stations": "Stations", "chi2": "χ²", "charge": "Charge", "group": "Group",
            "log-chi2": "log₁₀(χ²)",
        }
        self.batch_color_checks: dict[str, QCheckBox] = {}
        color_row = QWidget(); color_layout = QHBoxLayout(color_row)
        color_layout.setContentsMargins(0, 0, 0, 0); color_layout.setSpacing(12)
        for value in ("time", "altitude", "power", "stations", "chi2", "charge", "group", "log-chi2"):
            check = QCheckBox(color_labels[value])
            required_field = "chi2" if value == "log-chi2" else value
            check.setEnabled(required_field in self.available_color_fields)
            selected_value = (
                "log-chi2"
                if self.current_color_by == "chi2" and current_log_color_scale
                else self.current_color_by
            )
            check.setChecked(value == selected_value and check.isEnabled())
            self.batch_color_checks[value] = check
            color_layout.addWidget(check)
        if not any(check.isChecked() for check in self.batch_color_checks.values()):
            self.batch_color_checks["time"].setChecked(True)
        color_layout.addStretch(1)
        combo_form.addRow("Color by", color_row)

        self.batch_chi2_values = QLineEdit(f"{float(current_maximum_chi2):g}")
        self.batch_chi2_values.setPlaceholderText("Examples: 0.5, 0.75, 1.0, 2.0")
        self.batch_chi2_values.setToolTip(
            "Comma-, semicolon-, or space-separated maximum reduced χ² values."
        )
        combo_form.addRow("Maximum χ² values", self.batch_chi2_values)
        outer.addWidget(combinations)

        policy = QGroupBox("Batch behavior")
        policy_form = QFormLayout(policy)
        self.batch_overwrite = QComboBox()
        self.batch_overwrite.addItem("Skip existing files", "skip")
        self.batch_overwrite.addItem("Replace existing files", "replace")
        self.batch_overwrite.addItem("Stop when a file exists", "fail")
        self.batch_overwrite.setCurrentIndex(self.batch_overwrite.findData("replace"))
        policy_form.addRow("Existing files", self.batch_overwrite)
        self.batch_continue = QCheckBox("Continue after an individual job fails")
        self.batch_continue.setChecked(True)
        policy_form.addRow("", self.batch_continue)
        self.batch_dynamic_titles = QCheckBox("Generate the live title separately for each job")
        self.batch_dynamic_titles.setChecked(True)
        policy_form.addRow("", self.batch_dynamic_titles)
        self.batch_custom_title = QLineEdit(default_title)
        self.batch_custom_title.setEnabled(False)
        policy_form.addRow("Custom title", self.batch_custom_title)
        outer.addWidget(policy)

        self.batch_count_label = QLabel()
        self.batch_preview = QPlainTextEdit()
        self.batch_preview.setReadOnly(True)
        self.batch_preview.setMaximumHeight(115)
        outer.addWidget(self.batch_count_label)
        outer.addWidget(self.batch_preview)

        browse.clicked.connect(self._browse_batch_directory)
        self.batch_dynamic_titles.toggled.connect(
            lambda checked: self.batch_custom_title.setEnabled(not checked)
        )
        for widget in (
            self.batch_directory, self.batch_base_stem, self.batch_chi2_values
        ):
            widget.textChanged.connect(self._refresh_batch_preview)
        self.batch_format.currentIndexChanged.connect(self._refresh_batch_preview)
        for check in (*self.batch_theme_checks.values(), *self.batch_color_checks.values()):
            check.toggled.connect(self._refresh_batch_preview)
        self._refresh_batch_preview()
        return tab

    @staticmethod
    def _format_combo(suffix: str) -> QComboBox:
        combo = QComboBox()
        combo.addItem("PNG image", ".png")
        combo.addItem("PDF document", ".pdf")
        combo.addItem("SVG image", ".svg")
        index = combo.findData(suffix.lower() if suffix.lower() in {".png", ".pdf", ".svg"} else ".png")
        combo.setCurrentIndex(max(index, 0))
        return combo

    @staticmethod
    def _theme_selector(*, all_checked: bool) -> tuple[QWidget, dict[str, QCheckBox]]:
        row = QWidget(); layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(14)
        checks: dict[str, QCheckBox] = {}
        labels = {"dark": "Dark", "light": "Light", "space": "Space"}
        for theme in SUPPORTED_EXPORT_THEMES:
            checkbox = QCheckBox(labels[theme]); checkbox.setChecked(all_checked)
            checks[theme] = checkbox; layout.addWidget(checkbox)
        layout.addStretch(1)
        return row, checks

    def _grow_to_fit(self) -> None:
        """Apply layout hints without ever shrinking an already usable dialog."""
        old_width, old_height = self.width(), self.height()
        layout = self.layout()
        if layout is not None:
            layout.activate()
            hint = layout.sizeHint()
            self.resize(max(old_width, hint.width(), self.minimumWidth()),
                        max(old_height, hint.height(), self.minimumHeight()))

    def _multi_theme_toggled(self, enabled: bool) -> None:
        old_width, old_height = self.width(), self.height()
        self.theme_row.setVisible(bool(enabled))
        self.theme_note.setVisible(bool(enabled))
        layout = self.layout()
        if layout is not None:
            layout.activate()
            hint = layout.sizeHint()
            self.resize(max(old_width, hint.width(), self.minimumWidth()),
                        max(old_height, hint.height(), self.minimumHeight()))

    def _single_format_changed(self) -> None:
        suffix = str(self.format_combo.currentData())
        self.path_edit.setText(str(Path(self.path_edit.text()).expanduser().with_suffix(suffix)))

    def _browse_single(self) -> None:
        selected, _ = QFileDialog.getSaveFileName(
            self, "Choose LMAS figure filename", self.path_edit.text(),
            "PNG image (*.png);;PDF document (*.pdf);;SVG image (*.svg)",
            options=QFileDialog.Option.DontConfirmOverwrite,
        )
        if selected:
            suffix = Path(selected).suffix.lower()
            index = self.format_combo.findData(suffix)
            if index >= 0:
                self.format_combo.setCurrentIndex(index)
            self.path_edit.setText(selected)

    def _browse_batch_directory(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self, "Choose LMAS batch output directory", self.batch_directory.text()
        )
        if selected:
            self.batch_directory.setText(selected)

    def _selected_batch_themes(self) -> tuple[str, ...]:
        return tuple(name for name, check in self.batch_theme_checks.items() if check.isChecked())

    def _selected_batch_colors(self) -> tuple[str, ...]:
        return tuple(name for name, check in self.batch_color_checks.items() if check.isChecked())

    def _parsed_chi2_values(self) -> tuple[float, ...]:
        tokens = [token for token in re.split(r"[,;\s]+", self.batch_chi2_values.text().strip()) if token]
        return tuple(float(token) for token in tokens)

    def _batch_jobs(self) -> tuple[FigureBatchJob, ...]:
        return figure_batch_jobs(
            output_directory=self.batch_directory.text().strip(),
            base_stem=self.batch_base_stem.text().strip(),
            extension=str(self.batch_format.currentData()),
            themes=self._selected_batch_themes(),
            color_by_options=self._selected_batch_colors(),
            maximum_chi2_values=self._parsed_chi2_values(),
        )

    def _refresh_batch_preview(self, *_args) -> None:
        try:
            jobs = self._batch_jobs()
        except Exception as exc:
            self.batch_count_label.setText(f"Batch not ready: {exc}")
            self.batch_preview.setPlainText("")
            return
        self.batch_count_label.setText(f"{len(jobs)} figure job(s) will be queued")
        names = [Path(job.output_path).name for job in jobs]
        preview = names[:10]
        if len(names) > len(preview):
            preview.append(f"… and {len(names) - len(preview)} more")
        self.batch_preview.setPlainText("\n".join(preview))

    def _accept_checked(self) -> None:
        try:
            self.options()
        except Exception as exc:
            QMessageBox.warning(self, "Invalid figure settings", str(exc))
            return
        self.accept()

    def options(self) -> FigureSaveOptions:
        if self.tabs.currentIndex() == 1:
            jobs = self._batch_jobs()
            return FigureSaveOptions(
                mode="batch",
                path=Path(self.batch_directory.text().strip()).expanduser().resolve(),
                dpi=int(self.batch_dpi.value()),
                title=self.batch_custom_title.text().strip(),
                themes=self._selected_batch_themes(),
                batch_theme_export=False,
                batch_jobs=jobs,
                overwrite_policy=str(self.batch_overwrite.currentData()),
                continue_on_error=bool(self.batch_continue.isChecked()),
                dynamic_titles=bool(self.batch_dynamic_titles.isChecked()),
            )

        suffix = str(self.format_combo.currentData())
        path = Path(self.path_edit.text()).expanduser()
        if path.suffix.lower() != suffix:
            path = path.with_suffix(suffix)
        if self.multi_theme_check.isChecked():
            themes = tuple(
                theme for theme, checkbox in self.theme_checks.items() if checkbox.isChecked()
            )
            if not themes:
                raise ValueError("Select at least one theme to save")
        else:
            themes = (self.current_theme,)
        return FigureSaveOptions(
            mode="single",
            path=path.resolve(),
            dpi=int(self.dpi_spin.value()),
            title=self.title_edit.text().strip(),
            themes=themes,
            batch_theme_export=bool(self.multi_theme_check.isChecked()),
        )
