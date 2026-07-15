from __future__ import annotations

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QTextBrowser, QVBoxLayout

from ..help_docs import (
    CHANGELOG,
    DEVELOPMENT_PROVENANCE,
    KNOWN_LIMITATIONS,
    NETWORK_OVERLAYS,
    LINEAGE_AND_ATTRIBUTION,
    POLARITY_PRODUCT_FORMAT,
    RELEASE_NOTES,
    USER_MANUAL,
    WHAT_LMAS_CAN_DO,
    read_help_document,
)


_TITLES = {
    WHAT_LMAS_CAN_DO: "What LMAS can do",
    USER_MANUAL: "LMAS User Manual",
    NETWORK_OVERLAYS: "LMAS Network Overlays",
    LINEAGE_AND_ATTRIBUTION: "LMAS lineage and attribution",
    DEVELOPMENT_PROVENANCE: "LMAS development provenance",
    RELEASE_NOTES: "LMAS release notes",
    KNOWN_LIMITATIONS: "LMAS known limitations",
    CHANGELOG: "LMAS changelog",
    POLARITY_PRODUCT_FORMAT: "LMAS polarity product format",
}


class HelpDocumentWindow(QDialog):
    def __init__(self, document_name: str, parent=None) -> None:
        super().__init__(parent)
        self.resize(900, 700)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)

        outer = QVBoxLayout(self)
        self.browser = QTextBrowser()
        self.browser.setOpenLinks(False)
        self.browser.setOpenExternalLinks(True)
        self.browser.anchorClicked.connect(self._open_link)
        outer.addWidget(self.browser, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.close)
        buttons.accepted.connect(self.close)
        outer.addWidget(buttons)
        self.open_document(document_name)

    def open_document(self, document_name: str) -> None:
        self._document_name = document_name
        self.setWindowTitle(_TITLES.get(document_name, "LMAS Help"))
        self.browser.setMarkdown(read_help_document(document_name))
        self.browser.moveCursor(QTextCursor.MoveOperation.Start)

    def _open_link(self, url: QUrl) -> None:
        text = url.toString()
        if text.startswith("lmas-doc:"):
            name = text.split(":", 1)[1]
            if name in _TITLES:
                self.open_document(name)
                return
        self.browser.setSource(url)


__all__ = ["HelpDocumentWindow"]
