from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..errors import ConfigurationError


@dataclass(frozen=True)
class SaveProfileOptions:
    name: str
    path: Path


def _profile_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip()).strip("._") or "profile"
    return f"{safe}.lmas-profile.yaml"


class SaveProfileDialog(QDialog):
    def __init__(
        self,
        *,
        default_name: str,
        default_directory: str | Path,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Save LMAS profile")
        self.setMinimumWidth(520)
        self._path_tracks_name = True

        outer = QVBoxLayout(self)
        form = QFormLayout()
        self.name_edit = QLineEdit(default_name)
        form.addRow("Profile name", self.name_edit)

        path_widget = QWidget()
        path_layout = QHBoxLayout(path_widget)
        path_layout.setContentsMargins(0, 0, 0, 0)
        self.path_edit = QLineEdit(
            str(Path(default_directory).expanduser() / _profile_filename(default_name))
        )
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        path_layout.addWidget(self.path_edit, 1)
        path_layout.addWidget(browse)
        form.addRow("Save path", path_widget)
        outer.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept_checked)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

        self.name_edit.textEdited.connect(self._name_changed)
        self.path_edit.textEdited.connect(self._path_edited)

    def _name_changed(self, value: str) -> None:
        if not self._path_tracks_name:
            return
        current = Path(self.path_edit.text().strip() or ".").expanduser()
        self.path_edit.setText(str(current.parent / _profile_filename(value)))

    def _path_edited(self, _value: str) -> None:
        self._path_tracks_name = False

    def _browse(self) -> None:
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Save LMAS profile",
            self.path_edit.text(),
            "LMAS profiles (*.lmas-profile.yaml *.lmas-profile.yml *.yaml *.yml)",
        )
        if selected:
            self._path_tracks_name = False
            self.path_edit.setText(selected)

    def _accept_checked(self) -> None:
        try:
            self.options()
        except ConfigurationError as exc:
            QMessageBox.warning(self, "Invalid profile", str(exc))
            return
        self.accept()

    def options(self) -> SaveProfileOptions:
        name = self.name_edit.text().strip()
        if not name:
            raise ConfigurationError("Profile name cannot be empty")
        raw_path = self.path_edit.text().strip()
        if not raw_path:
            raise ConfigurationError("Choose where to save the profile")
        path = Path(raw_path).expanduser()
        if not path.name.lower().endswith((".yaml", ".yml")):
            path = path.with_name(path.name + ".lmas-profile.yaml")
        return SaveProfileOptions(name=name, path=path.resolve())


__all__ = ["SaveProfileDialog", "SaveProfileOptions"]
