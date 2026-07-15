from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QDir, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QFileSystemModel,
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QToolButton,
    QTreeView,
    QVBoxLayout,
    QWidget,
)


SUPPORTED_SUFFIXES = (
    ".dat",
    ".dat.gz",
    ".tar.gz",
    ".tgz",
    ".tar",
    ".nc",
    ".netcdf",
    ".lmas-project.yaml",
    ".lmas-project.yml",
    ".lmas.yaml",
    ".lmas.yml",
)


def is_supported_lmas_path(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in SUPPORTED_SUFFIXES)


class LMAFileBrowserDock(QDockWidget):
    """Collapsible filesystem browser for supported LMAS inputs."""

    open_requested = Signal(Path)
    root_changed = Signal(Path)
    collapsed_changed = Signal(bool)

    def __init__(self, root: Path, parent=None) -> None:
        super().__init__("LMA files", parent)
        self.root = Path(root).expanduser()
        self.setObjectName("lmaFileBrowserDock")
        self.toggleViewAction().setText("LMA file browser")
        self._collapsed = False
        self._expanded_width = 310
        self._expanded_features = self.features()

        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(6, 6, 6, 6)
        header_row = QHBoxLayout()
        header_row.addWidget(QLabel("Path"))
        header_row.addStretch(1)
        collapse = QToolButton()
        collapse.setText("◀")
        collapse.setToolTip("Collapse the LMA file browser to the left")
        collapse.setAutoRaise(True)
        header_row.addWidget(collapse)
        layout.addLayout(header_row)

        self.root_edit = QLineEdit(str(self.root))
        self.root_edit.setClearButtonEnabled(True)
        layout.addWidget(self.root_edit)

        button_row = QHBoxLayout()
        browse = QPushButton("Browse…")
        refresh = QPushButton("Refresh")
        button_row.addWidget(browse)
        button_row.addWidget(refresh)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        self.model = QFileSystemModel(self)
        self.model.setFilter(QDir.Filter.AllDirs | QDir.Filter.Files | QDir.Filter.NoDotAndDotDot)
        self.model.setNameFilters(
            [
                "*.dat",
                "*.dat.gz",
                "*.tar.gz",
                "*.tgz",
                "*.tar",
                "*.nc",
                "*.netcdf",
                "*.lmas-project.yaml",
                "*.lmas-project.yml",
                "*.lmas.yaml",
                "*.lmas.yml",
            ]
        )
        self.model.setNameFilterDisables(False)
        self.tree = QTreeView()
        self.tree.setModel(self.model)
        self.tree.setAlternatingRowColors(True)
        self.tree.setAnimated(False)
        self.tree.setSortingEnabled(True)
        self.tree.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tree.sortByColumn(0, Qt.SortOrder.AscendingOrder)
        self.tree.setColumnWidth(0, 245)
        for column in (1, 2, 3):
            self.tree.hideColumn(column)
        layout.addWidget(self.tree, 1)

        collapsed_strip = QWidget()
        collapsed_strip.setObjectName("lmaFileBrowserCollapsedStrip")
        collapsed_layout = QVBoxLayout(collapsed_strip)
        collapsed_layout.setContentsMargins(2, 4, 2, 4)
        expand = QToolButton()
        expand.setText("▶")
        expand.setToolTip("Expand the LMA file browser to the right")
        expand.setAutoRaise(True)
        expand.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        collapsed_layout.addWidget(expand)
        collapsed_layout.addStretch(1)

        self._stack = QStackedWidget()
        self._stack.addWidget(body)
        self._stack.addWidget(collapsed_strip)
        self._expanded_body = body
        self._collapsed_strip = collapsed_strip
        self._collapsed_title_bar = QWidget()
        self.setWidget(self._stack)

        browse.clicked.connect(self._browse)
        refresh.clicked.connect(self.refresh)
        collapse.clicked.connect(self.collapse_to_strip)
        expand.clicked.connect(self.expand_from_strip)
        self.toggleViewAction().triggered.connect(self._view_action_triggered)
        self.root_edit.returnPressed.connect(self._edit_root)
        self.tree.doubleClicked.connect(self._open_index)
        self.set_root(self.root)

    @property
    def collapsed(self) -> bool:
        return self._collapsed

    @property
    def expanded_width(self) -> int:
        return int(self._expanded_width)

    def set_root(self, root: Path) -> None:
        path = Path(root).expanduser()
        if path.is_file():
            path = path.parent
        if not path.exists():
            return
        self.root = path.resolve()
        self.root_edit.setText(str(self.root))
        index = self.model.setRootPath(str(self.root))
        self.tree.setRootIndex(index)
        self.root_changed.emit(self.root)

    def refresh(self) -> None:
        # QFileSystemModel watches the directory, but resetting the root forces a
        # prompt refresh after an external extraction or download.
        current = self.root
        self.model.setRootPath("")
        self.set_root(current)

    def _browse(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select LMA data root",
            str(self.root),
        )
        if selected:
            self.set_root(Path(selected))

    def _edit_root(self) -> None:
        self.set_root(Path(self.root_edit.text()))

    def _open_index(self, index) -> None:
        path = Path(self.model.filePath(index))
        if path.is_file() and is_supported_lmas_path(path):
            self.open_requested.emit(path)

    def collapse_to_strip(self) -> None:
        if self._collapsed:
            return
        self._expanded_width = max(self.width(), 240)
        self._expanded_features = self.features()
        self._collapsed = True
        self._stack.setCurrentWidget(self._collapsed_strip)
        self.setTitleBarWidget(self._collapsed_title_bar)
        self.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        self.setMinimumWidth(28)
        self.setMaximumWidth(34)
        owner = self.parentWidget()
        if isinstance(owner, QMainWindow):
            owner.resizeDocks([self], [32], Qt.Orientation.Horizontal)
        self.collapsed_changed.emit(True)

    def expand_from_strip(self) -> None:
        if not self._collapsed:
            return
        self._collapsed = False
        self.setMinimumWidth(180)
        self.setMaximumWidth(16777215)
        self.setFeatures(self._expanded_features)
        self.setTitleBarWidget(None)
        self._stack.setCurrentWidget(self._expanded_body)
        owner = self.parentWidget()
        if isinstance(owner, QMainWindow):
            owner.resizeDocks([self], [max(self._expanded_width, 240)], Qt.Orientation.Horizontal)
        self.collapsed_changed.emit(False)

    def restore_state(self, *, collapsed: bool, expanded_width: int | None = None) -> None:
        if expanded_width is not None and int(expanded_width) > 0:
            self._expanded_width = int(expanded_width)
        if collapsed:
            self.collapse_to_strip()
        else:
            self.expand_from_strip()

    def _view_action_triggered(self, checked: bool) -> None:
        if checked and self._collapsed:
            self.expand_from_strip()
