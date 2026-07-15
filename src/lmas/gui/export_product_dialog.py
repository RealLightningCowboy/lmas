"""General product-export dialog for the LMAS GUI."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
)

from ..product_export import EXPORT_PRODUCTS, EXPORT_SCOPES, export_product_by_key


@dataclass(frozen=True)
class ExportProductOptions:
    product_key: str
    format_name: str
    scope: str


class ExportProductDialog(QDialog):
    """Choose one available scientific product and its source scope."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export Product — LMAS")
        self.resize(520, 250)

        layout = QVBoxLayout(self)
        intro = QLabel(
            "Export a scientific product from the current LMAS Project. "
            "Additional product types can use this same interface in later releases."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QFormLayout()
        self.product_combo = QComboBox()
        for product in EXPORT_PRODUCTS:
            self.product_combo.addItem(product.label, product.key)
        self.product_combo.currentIndexChanged.connect(self._refresh_description)
        form.addRow("Product", self.product_combo)

        self.scope_combo = QComboBox()
        for scope in EXPORT_SCOPES:
            self.scope_combo.addItem(scope.label, scope.key)
            self.scope_combo.setItemData(
                self.scope_combo.count() - 1, scope.description, 3
            )
        form.addRow("Source scope", self.scope_combo)
        layout.addLayout(form)

        self.description = QLabel()
        self.description.setWordWrap(True)
        self.description.setMinimumHeight(54)
        layout.addWidget(self.description)

        note = QLabel(
            "All loaded sources is the authoritative default for a complete "
            "round-trip polarity product."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._refresh_description()

    def _refresh_description(self, _index: int = -1) -> None:
        try:
            product = export_product_by_key(str(self.product_combo.currentData()))
        except KeyError:
            self.description.setText("")
            return
        self.description.setText(product.description)

    def options(self) -> ExportProductOptions:
        product = export_product_by_key(str(self.product_combo.currentData()))
        return ExportProductOptions(
            product_key=product.key,
            format_name=product.format_name,
            scope=str(self.scope_combo.currentData() or "all"),
        )


__all__ = ["ExportProductDialog", "ExportProductOptions"]
