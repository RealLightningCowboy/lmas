from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PySide6.QtWidgets import QFileDialog, QWidget


def _dialog(parent: QWidget | None, title: str, start: str | Path) -> QFileDialog:
    """Return a detailed, non-native data browser.

    Windows' native directory picker hides files and omits the filename field.
    LMAS deliberately uses Qt's detailed browser so scientific filenames remain
    visible while users navigate or type a path.
    """

    dialog = QFileDialog(parent, title, str(Path(start).expanduser()))
    dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
    dialog.setOption(QFileDialog.Option.ReadOnly, False)
    dialog.setViewMode(QFileDialog.ViewMode.Detail)
    dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)
    return dialog


def choose_existing_files(
    parent: QWidget | None,
    title: str,
    start: str | Path,
    name_filters: Iterable[str],
) -> list[Path]:
    dialog = _dialog(parent, title, start)
    dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
    filters = [str(value) for value in name_filters]
    if filters:
        dialog.setNameFilters(filters)
    if not dialog.exec():
        return []
    return [Path(value).expanduser().resolve() for value in dialog.selectedFiles()]


def choose_directory_with_files_visible(
    parent: QWidget | None,
    title: str,
    start: str | Path,
    name_filters: Iterable[str] = (),
) -> Path | None:
    """Choose a directory without hiding the files that identify it.

    Qt's non-native Directory mode retains the location/filename field and a
    detailed file listing.  A typed path is accepted directly.  If a platform
    returns a selected file despite Directory mode, LMAS uses its parent.
    """

    dialog = _dialog(parent, title, start)
    dialog.setFileMode(QFileDialog.FileMode.Directory)
    dialog.setOption(QFileDialog.Option.ShowDirsOnly, False)
    filters = [str(value) for value in name_filters]
    if filters:
        dialog.setNameFilters(filters)
    if not dialog.exec():
        return None
    selected = dialog.selectedFiles()
    if not selected:
        return None
    path = Path(selected[0]).expanduser()
    if path.is_file():
        path = path.parent
    return path.resolve()


__all__ = ["choose_existing_files", "choose_directory_with_files_visible"]
