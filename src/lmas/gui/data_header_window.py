"""Read-only, searchable LMA data-header viewer."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QAction, QFontDatabase, QKeySequence, QTextCursor, QTextDocument
from PySide6.QtWidgets import (
    QFileDialog,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from ..data_header import DataHeaderDocument, data_header_documents
from .icon import application_icon


class DataHeaderWindow(QMainWindow):
    """Show literal DAT headers or an equivalent metadata summary."""

    def __init__(self, project, parent=None) -> None:
        super().__init__(parent)
        self.project = project
        self.documents = data_header_documents(project)
        self._document_index = 0

        self.setWindowTitle("Data File Header — LMAS")
        self.setWindowIcon(application_icon())
        self.resize(920, 720)

        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.source_label = QLabel()
        self.source_label.setWordWrap(True)
        layout.addWidget(self.source_label)

        document_row = QHBoxLayout()
        document_row.addWidget(QLabel("Document"))
        self.document_combo = QComboBox()
        for document in self.documents:
            self.document_combo.addItem(document.title)
        self.document_combo.currentIndexChanged.connect(self._show_document)
        document_row.addWidget(self.document_combo, 1)
        layout.addLayout(document_row)

        find_row = QHBoxLayout()
        find_row.addWidget(QLabel("Find"))
        self.find_edit = QLineEdit()
        self.find_edit.setClearButtonEnabled(True)
        self.find_edit.returnPressed.connect(self.find_next)
        find_row.addWidget(self.find_edit, 1)
        previous = QPushButton("Previous")
        previous.clicked.connect(self.find_previous)
        next_button = QPushButton("Next")
        next_button.clicked.connect(self.find_next)
        find_row.addWidget(previous)
        find_row.addWidget(next_button)
        layout.addLayout(find_row)

        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.text.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.text.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        layout.addWidget(self.text, 1)
        self.setCentralWidget(central)

        toolbar = QToolBar("Header", self)
        toolbar.setMovable(False)
        copy_all = QAction("Copy all", self)
        copy_all.triggered.connect(self.copy_all)
        save_text = QAction("Save as text", self)
        save_text.triggered.connect(self.save_as_text)
        toolbar.addAction(copy_all)
        toolbar.addAction(save_text)
        self.addToolBar(toolbar)

        focus_find = QAction(self)
        focus_find.setShortcut(QKeySequence.StandardKey.Find)
        focus_find.triggered.connect(self._focus_find)
        self.addAction(focus_find)
        self._show_document(0)

    @property
    def current_document(self) -> DataHeaderDocument:
        return self.documents[self._document_index]

    def _show_document(self, index: int) -> None:
        if not self.documents:
            return
        self._document_index = max(0, min(int(index), len(self.documents) - 1))
        document = self.current_document
        kind_label = "Original DAT header" if document.kind == "dat-header" else "Metadata summary"
        self.source_label.setText(
            f"<b>{kind_label}</b><br>{document.source}<br>"
            f"Reader: {self.project.reader_backend} "
            f"{self.project.reader_backend_version or 'unknown'}"
        )
        self.text.setPlainText(document.text)
        self.text.moveCursor(QTextCursor.MoveOperation.Start)

    def _focus_find(self) -> None:
        self.find_edit.setFocus()
        self.find_edit.selectAll()

    def _find(self, *, backward: bool) -> None:
        query = self.find_edit.text()
        if not query:
            self._focus_find()
            return
        find_flags = (
            QTextDocument.FindFlag.FindBackward
            if backward
            else QTextDocument.FindFlag(0)
        )
        if not self.text.find(query, find_flags):
            cursor = self.text.textCursor()
            cursor.movePosition(
                QTextCursor.MoveOperation.End if backward else QTextCursor.MoveOperation.Start
            )
            self.text.setTextCursor(cursor)
            self.text.find(query, find_flags)

    def find_next(self) -> None:
        self._find(backward=False)

    def find_previous(self) -> None:
        self._find(backward=True)

    def copy_all(self) -> None:
        from PySide6.QtWidgets import QApplication

        QApplication.clipboard().setText(self.current_document.text)
        self.statusBar().showMessage("Header copied to the clipboard", 4000)

    def save_as_text(self) -> None:
        base = Path(self.current_document.title).name
        default_name = Path(base).with_suffix(".header.txt").name
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Save data header as text",
            default_name,
            "Text files (*.txt);;All files (*)",
        )
        if not selected:
            return
        try:
            Path(selected).expanduser().write_text(
                self.current_document.text, encoding="utf-8"
            )
        except OSError as exc:
            QMessageBox.critical(self, "Could not save header", str(exc))
            return
        self.statusBar().showMessage(f"Saved header to {selected}", 5000)


__all__ = ["DataHeaderWindow"]
