from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
    QFormLayout, QHBoxLayout, QLineEdit, QPushButton, QVBoxLayout, QWidget,
)


@dataclass(frozen=True)
class DirectoryPreferences:
    data_directory: Path
    remember_last_data_directory: bool
    output_mode: str
    output_directory: Path | None


class PreferencesDialog(QDialog):
    def __init__(
        self,
        *,
        data_directory: Path,
        remember_last_data_directory: bool,
        output_mode: str,
        output_directory: Path | None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("LMAS Preferences")
        self.setMinimumWidth(620)
        outer = QVBoxLayout(self)
        form = QFormLayout()

        self.data_directory = QLineEdit(str(Path(data_directory).expanduser()))
        data_browse = QPushButton("Browse…")
        data_browse.clicked.connect(self._browse_data)
        data_row = QWidget(); data_layout = QHBoxLayout(data_row)
        data_layout.setContentsMargins(0, 0, 0, 0)
        data_layout.addWidget(self.data_directory, 1); data_layout.addWidget(data_browse)
        form.addRow("Default data directory", data_row)

        self.remember_last = QCheckBox("Remember the last successfully opened data directory")
        self.remember_last.setChecked(bool(remember_last_data_directory))
        form.addRow("", self.remember_last)

        self.output_mode = QComboBox()
        self.output_mode.addItem("Same directory as input data", "input")
        self.output_mode.addItem("Custom directory", "custom")
        index = self.output_mode.findData(str(output_mode))
        self.output_mode.setCurrentIndex(index if index >= 0 else 0)
        form.addRow("Default output location", self.output_mode)

        self.output_directory = QLineEdit("" if output_directory is None else str(output_directory))
        output_browse = QPushButton("Browse…")
        output_browse.clicked.connect(self._browse_output)
        output_row = QWidget(); output_layout = QHBoxLayout(output_row)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.addWidget(self.output_directory, 1); output_layout.addWidget(output_browse)
        form.addRow("Custom output directory", output_row)
        self._output_row = output_row
        self.output_mode.currentIndexChanged.connect(self._update_enabled)
        self._update_enabled()

        outer.addLayout(form)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    def _update_enabled(self, *_args) -> None:
        enabled = str(self.output_mode.currentData()) == "custom"
        self._output_row.setEnabled(enabled)

    def _browse_data(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self, "Choose default LMA data directory", self.data_directory.text()
        )
        if selected:
            self.data_directory.setText(selected)

    def _browse_output(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self, "Choose default LMAS output directory", self.output_directory.text()
        )
        if selected:
            self.output_directory.setText(selected)

    def preferences(self) -> DirectoryPreferences:
        data = Path(self.data_directory.text().strip()).expanduser()
        mode = str(self.output_mode.currentData() or "input")
        raw_output = self.output_directory.text().strip()
        output = Path(raw_output).expanduser() if mode == "custom" and raw_output else None
        return DirectoryPreferences(
            data_directory=data,
            remember_last_data_directory=self.remember_last.isChecked(),
            output_mode=mode,
            output_directory=output,
        )


__all__ = ["DirectoryPreferences", "PreferencesDialog"]
