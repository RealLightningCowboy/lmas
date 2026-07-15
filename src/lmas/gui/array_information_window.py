from __future__ import annotations

import math

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..array_info import ArrayInformation, BaselineStatistics, build_array_information
from ..model import LMAProject


def _number(value: float, decimals: int = 3) -> str:
    return "" if not math.isfinite(value) else f"{value:.{decimals}f}"


def _table(headers: list[str], rows: list[list[str]]) -> QTableWidget:
    widget = QTableWidget(len(rows), len(headers))
    widget.setHorizontalHeaderLabels(headers)
    widget.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    widget.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    widget.setAlternatingRowColors(True)
    for row_index, row in enumerate(rows):
        for column_index, value in enumerate(row):
            item = QTableWidgetItem(value)
            if column_index > 0:
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            widget.setItem(row_index, column_index, item)
    widget.resizeColumnsToContents()
    if headers:
        # Qt can initially size the first CSV-viewer column a few pixels too
        # narrowly, clipping the leading “S” in headers such as Station A.
        widget.setColumnWidth(0, max(widget.columnWidth(0) + 14, 112))
    widget.setSortingEnabled(True)
    return widget


def _statistics_html(title: str, stats: BaselineStatistics | None) -> str:
    if stats is None:
        return f"<h3>{title}</h3><p>No finite baselines available.</p>"
    return (
        f"<h3>{title}</h3>"
        "<table cellspacing='4'>"
        f"<tr><td><b>Count</b></td><td>{stats.count:,}</td></tr>"
        f"<tr><td><b>Minimum</b></td><td>{stats.minimum_km:.3f} km</td></tr>"
        f"<tr><td><b>25th percentile</b></td><td>{stats.first_quartile_km:.3f} km</td></tr>"
        f"<tr><td><b>Median</b></td><td>{stats.median_km:.3f} km</td></tr>"
        f"<tr><td><b>Mean</b></td><td>{stats.mean_km:.3f} km</td></tr>"
        f"<tr><td><b>75th percentile</b></td><td>{stats.third_quartile_km:.3f} km</td></tr>"
        f"<tr><td><b>Maximum</b></td><td>{stats.maximum_km:.3f} km</td></tr>"
        f"<tr><td><b>Standard deviation</b></td><td>{stats.standard_deviation_km:.3f} km</td></tr>"
        "</table>"
    )


class ArrayInformationWindow(QDialog):
    def __init__(self, project: LMAProject, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Array information — {project.data_source_stem}")
        self.resize(980, 650)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        information = build_array_information(project)

        outer = QVBoxLayout(self)
        tabs = QTabWidget()
        tabs.addTab(self._summary_tab(information), "Summary")
        tabs.addTab(self._stations_tab(information), "Stations")
        tabs.addTab(self._baselines_tab(information), "Baselines")
        outer.addWidget(tabs, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.close)
        buttons.accepted.connect(self.close)
        outer.addWidget(buttons)

    @staticmethod
    def _summary_tab(information: ArrayInformation) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        lines = [
            f"<h2>{information.network_name}</h2>",
            f"<p><b>Reference latitude:</b> {information.reference_latitude:.7f}°<br>",
            f"<b>Reference longitude:</b> {information.reference_longitude:.7f}°<br>",
            f"<b>Stations:</b> {len(information.stations):,}<br>",
            f"<b>Baselines:</b> {len(information.baselines):,}</p>",
            _statistics_html(
                "Horizontal baseline statistics",
                information.horizontal_baseline_statistics,
            ),
            _statistics_html(
                "3D baseline statistics",
                information.three_d_baseline_statistics,
            ),
        ]
        if not information.stations:
            lines.append(
                "<p>This dataset does not contain station latitude/longitude metadata.</p>"
            )
        label = QLabel("".join(lines))
        label.setTextFormat(Qt.TextFormat.RichText)
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(label)
        layout.addStretch(1)
        return page

    @staticmethod
    def _stations_tab(information: ArrayInformation) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        rows = [
            [
                station.code,
                str(station.index),
                _number(station.latitude, 6),
                _number(station.longitude, 6),
                _number(station.altitude_km, 3),
                _number(station.east_km, 3),
                _number(station.north_km, 3),
            ]
            for station in information.stations
        ]
        layout.addWidget(
            _table(
                [
                    "Station",
                    "Index",
                    "Latitude (°N)",
                    "Longitude (°E)",
                    "Altitude (km MSL)",
                    "Easting (km)",
                    "Northing (km)",
                ],
                rows,
            )
        )
        return page

    @staticmethod
    def _baselines_tab(information: ArrayInformation) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        rows = [
            [
                baseline.station_a,
                baseline.station_b,
                _number(baseline.horizontal_length_km, 3),
                _number(baseline.three_d_length_km, 3),
                _number(baseline.azimuth_deg, 2),
            ]
            for baseline in information.baselines
        ]
        layout.addWidget(
            _table(
                [
                    "Station A",
                    "Station B",
                    "Horizontal length (km)",
                    "3D length (km)",
                    "Azimuth A→B (°)",
                ],
                rows,
            )
        )
        return page


__all__ = ["ArrayInformationWindow"]
