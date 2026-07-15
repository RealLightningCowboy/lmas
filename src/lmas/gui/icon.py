from __future__ import annotations

from importlib.resources import as_file, files

from PySide6.QtCore import QLineF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap


def application_icon() -> QIcon:
    """Return the packaged LMAS lightning-bolt icon."""
    resource = files("lmas.resources").joinpath("lmas_bolt.svg")
    with as_file(resource) as path:
        return QIcon(str(path))


def precision_crosshair_icon(size: int = 64) -> QIcon:
    """Return a scalable-looking crosshairs icon for Precision Mode."""

    extent = max(24, int(size))
    pixmap = QPixmap(extent, extent)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    color = QColor("#42a5f5")
    pen = QPen(color)
    pen.setWidthF(max(1.8, extent / 24.0))
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    painter.setPen(pen)
    center = extent / 2.0
    radius = extent * 0.22
    gap = radius * 0.52
    margin = extent * 0.11
    painter.drawEllipse(QRectF(center - radius, center - radius, 2 * radius, 2 * radius))
    painter.drawLine(QLineF(center, margin, center, center - gap))
    painter.drawLine(QLineF(center, center + gap, center, extent - margin))
    painter.drawLine(QLineF(margin, center, center - gap, center))
    painter.drawLine(QLineF(center + gap, center, extent - margin, center))
    painter.setBrush(color)
    painter.drawEllipse(QRectF(center - 2.0, center - 2.0, 4.0, 4.0))
    painter.end()
    return QIcon(pixmap)


def selection_lasso_icon(size: int = 64) -> QIcon:
    """Return a compact polygon/lasso icon for linked source selection."""

    extent = max(24, int(size))
    pixmap = QPixmap(extent, extent)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    color = QColor("#d65ad1")
    pen = QPen(color)
    pen.setWidthF(max(1.6, extent / 28.0))
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    points = [
        (0.18, 0.31),
        (0.39, 0.14),
        (0.72, 0.22),
        (0.84, 0.52),
        (0.67, 0.80),
        (0.31, 0.83),
        (0.12, 0.58),
    ]
    from PySide6.QtCore import QPointF
    polygon = [QPointF(extent * x, extent * y) for x, y in points]
    for first, second in zip(polygon, polygon[1:] + polygon[:1]):
        painter.drawLine(QLineF(first, second))
    painter.setBrush(color)
    radius = max(1.6, extent / 32.0)
    for point in polygon:
        painter.drawEllipse(QRectF(point.x() - radius, point.y() - radius, 2 * radius, 2 * radius))
    painter.end()
    return QIcon(pixmap)



def satellite_overlay_icon(size: int = 64) -> QIcon:
    """Return a satellite-and-footprint icon for Satellite Overlays."""

    extent = max(24, int(size))
    pixmap = QPixmap(extent, extent)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    cyan = QColor("#4dd0e1")
    lime = QColor("#76ff03")
    pen = QPen(cyan)
    pen.setWidthF(max(1.5, extent / 30.0))
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(QColor("#263238"))
    body = QRectF(extent * 0.38, extent * 0.20, extent * 0.24, extent * 0.22)
    painter.drawRoundedRect(body, extent * 0.03, extent * 0.03)
    painter.drawLine(QLineF(extent * 0.50, extent * 0.42, extent * 0.50, extent * 0.58))
    painter.drawLine(QLineF(extent * 0.20, extent * 0.31, extent * 0.38, extent * 0.31))
    painter.drawLine(QLineF(extent * 0.62, extent * 0.31, extent * 0.80, extent * 0.31))
    painter.setBrush(QColor("#1565c0"))
    painter.drawRect(QRectF(extent * 0.10, extent * 0.22, extent * 0.10, extent * 0.18))
    painter.drawRect(QRectF(extent * 0.80, extent * 0.22, extent * 0.10, extent * 0.18))
    painter.setPen(QPen(lime, max(1.5, extent / 32.0)))
    painter.setBrush(QColor(118, 255, 3, 60))
    footprint = QPainterPath()
    footprint.moveTo(extent * 0.32, extent * 0.58)
    footprint.lineTo(extent * 0.68, extent * 0.58)
    footprint.lineTo(extent * 0.82, extent * 0.88)
    footprint.lineTo(extent * 0.18, extent * 0.88)
    footprint.closeSubpath()
    painter.drawPath(footprint)
    painter.end()
    return QIcon(pixmap)



def network_overlay_icon(size: int = 64) -> QIcon:
    """Return a compact ground-network nodes icon."""

    extent = max(24, int(size))
    pixmap = QPixmap(extent, extent)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    cyan = QColor("#26c6da")
    amber = QColor("#ffca28")
    pen = QPen(cyan)
    pen.setWidthF(max(1.5, extent / 30.0))
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    painter.setPen(pen)
    nodes = [(0.20, 0.72), (0.39, 0.43), (0.68, 0.62), (0.80, 0.28)]
    for first, second in zip(nodes[:-1], nodes[1:]):
        painter.drawLine(QLineF(extent * first[0], extent * first[1], extent * second[0], extent * second[1]))
    painter.drawLine(QLineF(extent * 0.20, extent * 0.72, extent * 0.68, extent * 0.62))
    painter.setBrush(cyan)
    radius = extent * 0.055
    for x, y in nodes:
        painter.drawEllipse(QRectF(extent * x - radius, extent * y - radius, 2 * radius, 2 * radius))
    bolt = QPainterPath()
    bolt.moveTo(extent * 0.52, extent * 0.10)
    bolt.lineTo(extent * 0.39, extent * 0.34)
    bolt.lineTo(extent * 0.51, extent * 0.34)
    bolt.lineTo(extent * 0.43, extent * 0.55)
    bolt.lineTo(extent * 0.66, extent * 0.27)
    bolt.lineTo(extent * 0.54, extent * 0.27)
    bolt.closeSubpath()
    painter.setPen(QPen(amber, max(1.2, extent / 38.0)))
    painter.setBrush(amber)
    painter.drawPath(bolt)
    painter.end()
    return QIcon(pixmap)

def charge_analysis_icon(size: int = 64) -> QIcon:
    """Return a red/blue polarity icon for Charge Analysis."""

    extent = max(24, int(size))
    pixmap = QPixmap(extent, extent)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    radius = extent * 0.18
    x = extent * 0.50
    positive_y = extent * 0.32
    negative_y = extent * 0.68
    for y, color in (
        (positive_y, QColor("#d62728")),
        (negative_y, QColor("#0077ff")),
    ):
        pen = QPen(color)
        pen.setWidthF(max(1.4, extent / 30.0))
        painter.setPen(pen)
        painter.setBrush(color)
        painter.drawEllipse(QRectF(x - radius, y - radius, 2 * radius, 2 * radius))
    painter.setPen(QPen(QColor("#ffffff"), max(1.2, extent / 36.0)))
    painter.drawLine(QLineF(x - extent * 0.08, positive_y, x + extent * 0.08, positive_y))
    painter.drawLine(QLineF(x, positive_y - extent * 0.08, x, positive_y + extent * 0.08))
    painter.drawLine(QLineF(x - extent * 0.08, negative_y, x + extent * 0.08, negative_y))
    painter.end()
    return QIcon(pixmap)


__all__ = ["application_icon", "charge_analysis_icon", "network_overlay_icon", "precision_crosshair_icon", "satellite_overlay_icon", "selection_lasso_icon"]
