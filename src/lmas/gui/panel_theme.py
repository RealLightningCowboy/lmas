from __future__ import annotations

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication


def apply_dark_palette(app: QApplication) -> None:
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(45, 45, 45))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(235, 235, 235))
    palette.setColor(QPalette.ColorRole.Base, QColor(28, 28, 28))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(52, 52, 52))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(235, 235, 235))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(20, 20, 20))
    palette.setColor(QPalette.ColorRole.Text, QColor(235, 235, 235))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(155, 155, 155))
    palette.setColor(QPalette.ColorRole.Button, QColor(52, 52, 52))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(235, 235, 235))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 95, 95))
    palette.setColor(QPalette.ColorRole.Link, QColor(90, 170, 255))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(55, 120, 190))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Light, QColor(75, 75, 75))
    palette.setColor(QPalette.ColorRole.Midlight, QColor(62, 62, 62))
    palette.setColor(QPalette.ColorRole.Mid, QColor(42, 42, 42))
    palette.setColor(QPalette.ColorRole.Dark, QColor(24, 24, 24))
    palette.setColor(QPalette.ColorRole.Shadow, QColor(10, 10, 10))
    for role in (
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
        QPalette.ColorRole.PlaceholderText,
    ):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor(135, 135, 135))
    app.setPalette(palette)
