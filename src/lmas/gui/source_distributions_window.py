"""Scientific source-distribution diagnostics for loaded LMA products."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import csv

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..model import FilterSpec, LMAProject
from ..plotting.common import apply_figure_theme, theme_values
from ..selection import event_selection_mask
from ..source_selection import SourceSelectionManager
from .icon import application_icon


_FIELDS = (
    ("Source χ²", "chi2", "event_chi2", "Reduced χ²"),
    ("Source power", "power", "event_power", "Source power (dBW)"),
    ("Contributing stations", "stations", "event_stations", "Contributing stations"),
)


class SourceDistributionsWindow(QMainWindow):
    """Interactive histogram window for source-quality and source-value fields."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_QuitOnClose, False)
        self.setWindowTitle("Source Distributions — LMAS")
        self.setWindowIcon(application_icon())
        self.resize(980, 720)
        self._project: LMAProject | None = None
        self._filters = FilterSpec()
        self._subset_filters = FilterSpec(minimum_stations=None, maximum_chi2=None)
        self._theme = "space"
        self._selection_manager = None
        self._histogram_rows: list[dict[str, float | int]] = []

        central = QWidget(self)
        layout = QVBoxLayout(central)
        controls = QHBoxLayout()

        controls.addWidget(QLabel("Variable"))
        self.variable = QComboBox()
        self.variable.currentIndexChanged.connect(self.refresh)
        controls.addWidget(self.variable)

        controls.addWidget(QLabel("Set"))
        self.source_set = QComboBox()
        self.source_set.addItem("Full dataset", "full")
        self.source_set.addItem("Selected subset (current view)", "subset")
        self.source_set.addItem("Active source group", "active_group")
        self.source_set.currentIndexChanged.connect(self.refresh)
        controls.addWidget(self.source_set)

        controls.addWidget(QLabel("Display"))
        self.scope = QComboBox()
        self.scope.addItem("Filter diagnostic: before vs accepted", "diagnostic")
        self.scope.addItem("Accepted by all active filters", "filtered")
        self.scope.addItem("Unfiltered values in set", "raw")
        self.scope.currentIndexChanged.connect(self.refresh)
        controls.addWidget(self.scope, 1)

        controls.addWidget(QLabel("Bins"))
        self.bins = QSpinBox()
        self.bins.setRange(5, 500)
        self.bins.setValue(50)
        self.bins.valueChanged.connect(self.refresh)
        controls.addWidget(self.bins)

        self.log_x = QCheckBox("Log x")
        self.log_x.toggled.connect(self.refresh)
        controls.addWidget(self.log_x)
        self.log_count = QCheckBox("Log count")
        self.log_count.toggled.connect(self.refresh)
        controls.addWidget(self.log_count)
        layout.addLayout(controls)

        self.summary = QLabel("Open LMA data to view source distributions.")
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.summary)

        self.figure = Figure(figsize=(9, 6), constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)

        buttons = QHBoxLayout()
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.refresh)
        export_button = QPushButton("Export Histogram Data")
        export_button.clicked.connect(self.export_histogram_data)
        save_button = QPushButton("Save Figure")
        save_button.clicked.connect(self.save_figure)
        buttons.addWidget(refresh_button)
        buttons.addStretch(1)
        buttons.addWidget(export_button)
        buttons.addWidget(save_button)
        layout.addLayout(buttons)
        self.setCentralWidget(central)

    def update_project(
        self,
        project: LMAProject | None,
        filters: FilterSpec | None,
        selection_manager=None,
        *,
        subset_filters: FilterSpec | None = None,
        theme: str | None = None,
    ) -> None:
        self._project = project
        self._filters = filters or (project.filters if project is not None else FilterSpec())
        self._subset_filters = subset_filters or (
            project.view_filters if project is not None else FilterSpec()
        )
        self._theme = str(theme or (project.plot.theme if project is not None else "space"))
        if selection_manager is None and project is not None and project.source_selection_state:
            temporary_manager = SourceSelectionManager()
            state = dict(project.source_selection_state or {})
            temporary_manager.load_groups(
                state.get("groups") or (), active_name=state.get("active_group")
            )
            self._selection_manager = temporary_manager
        else:
            self._selection_manager = selection_manager
        current = self.variable.currentData()
        self.variable.blockSignals(True)
        self.variable.clear()
        if project is not None:
            for label, key, field, _axis_label in _FIELDS:
                if field in project.source_store:
                    self.variable.addItem(label, key)
        if current is not None:
            index = self.variable.findData(current)
            if index >= 0:
                self.variable.setCurrentIndex(index)
        self.variable.blockSignals(False)
        self.refresh()

    def _field(self):
        key = str(self.variable.currentData() or "")
        return next((item for item in _FIELDS if item[1] == key), None)

    def _filters_without_selected_field(self, key: str) -> FilterSpec:
        if key == "chi2":
            return replace(self._filters, maximum_chi2=None)
        if key == "power":
            return replace(self._filters, minimum_power=None, maximum_power=None)
        if key == "stations":
            return replace(self._filters, minimum_stations=None)
        return self._filters

    def _active_group_mask(self, project: LMAProject) -> np.ndarray:
        store = project.source_store
        mask = np.zeros(store.event_count, dtype=bool)
        manager = self._selection_manager
        group = getattr(manager, "active_group", None) if manager is not None else None
        if group is None or not group.source_ids:
            return mask
        ids = np.asarray(store.event_array("event_source_index"), dtype=np.int64)
        return np.isin(ids, np.fromiter(group.source_ids, dtype=np.int64), assume_unique=False)

    def _base_set_mask(self, project: LMAProject) -> tuple[np.ndarray, str]:
        selected = str(self.source_set.currentData() or "full")
        if selected == "subset":
            return event_selection_mask(project, self._subset_filters), "Selected subset"
        if selected == "active_group":
            return self._active_group_mask(project), "Active source group"
        return np.ones(project.source_store.event_count, dtype=bool), "Full dataset"

    @staticmethod
    def _stats(values: np.ndarray) -> str:
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return "n=0"
        p05, p25, p50, p75, p95 = np.percentile(values, [5, 25, 50, 75, 95])
        return (
            f"n={values.size:,}; mean={np.mean(values):.4g}; median={p50:.4g}; "
            f"σ={np.std(values):.4g}; P05/P25/P75/P95="
            f"{p05:.4g}/{p25:.4g}/{p75:.4g}/{p95:.4g}; "
            f"range={np.min(values):.4g}–{np.max(values):.4g}"
        )

    def refresh(self, *_args) -> None:
        self.figure.clear()
        self._histogram_rows = []
        ax = self.figure.add_subplot(111)
        apply_figure_theme(self.figure, (ax,), self._theme, show_grid=True)
        project = self._project
        field_spec = self._field()
        if project is None or field_spec is None:
            ax.text(0.5, 0.5, "Open LMA data with χ², power, or station fields.", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            self.summary.setText("No compatible source field is available.")
            self.canvas.draw_idle()
            return

        label, key, field, axis_label = field_spec
        values = np.asarray(project.source_store.event_array(field), dtype=float)
        finite = np.isfinite(values)
        scope = str(self.scope.currentData() or "diagnostic")
        base_mask, set_label = self._base_set_mask(project)
        accepted_mask = base_mask & event_selection_mask(project, self._filters) & finite
        series: list[tuple[str, np.ndarray]] = []
        if scope == "diagnostic":
            before_mask = (
                base_mask
                & event_selection_mask(project, self._filters_without_selected_field(key))
                & finite
            )
            series.append((f"{set_label}: before this variable's filter", values[before_mask]))
            series.append((f"{set_label}: accepted", values[accepted_mask]))
        elif scope == "filtered":
            series.append((f"{set_label}: accepted by active filters", values[accepted_mask]))
        else:
            series.append((f"{set_label}: unfiltered", values[base_mask & finite]))

        combined = np.concatenate([item[1] for item in series if item[1].size]) if any(item[1].size for item in series) else np.array([], dtype=float)
        if combined.size == 0:
            ax.text(0.5, 0.5, "No finite sources in the selected scope.", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            self.summary.setText("No finite values are available for this variable and scope.")
            self.canvas.draw_idle()
            return

        use_log_x = bool(self.log_x.isChecked())
        if use_log_x:
            positive = combined[combined > 0]
            if positive.size < 2 or np.min(positive) == np.max(positive):
                use_log_x = False
                self.log_x.blockSignals(True)
                self.log_x.setChecked(False)
                self.log_x.blockSignals(False)
            else:
                edges = np.geomspace(np.min(positive), np.max(positive), int(self.bins.value()) + 1)
                series = [(name, vals[vals > 0]) for name, vals in series]
        if not use_log_x:
            if key == "stations":
                low = int(np.floor(np.min(combined)))
                high = int(np.ceil(np.max(combined)))
                edges = np.arange(low - 0.5, high + 1.5, 1.0)
            else:
                edges = np.histogram_bin_edges(combined, bins=int(self.bins.value()))
                if edges.size < 2 or not np.all(np.isfinite(edges)) or edges[0] == edges[-1]:
                    centre = float(combined[0])
                    edges = np.linspace(centre - 0.5, centre + 0.5, int(self.bins.value()) + 1)

        counts_by_name: dict[str, np.ndarray] = {}
        for index, (name, vals) in enumerate(series):
            counts, _ = np.histogram(vals, bins=edges)
            counts_by_name[name] = counts
            ax.hist(vals, bins=edges, histtype="stepfilled" if index == 0 else "step", alpha=0.28 if index == 0 else 1.0, linewidth=1.8, label=f"{name} (n={vals.size:,})")

        thresholds: list[tuple[float, str]] = []
        if key == "chi2" and self._filters.maximum_chi2 is not None:
            thresholds.append((float(self._filters.maximum_chi2), f"χ² max = {self._filters.maximum_chi2:g}"))
        elif key == "power":
            if self._filters.minimum_power is not None:
                thresholds.append((float(self._filters.minimum_power), f"Power min = {self._filters.minimum_power:g} dBW"))
            if self._filters.maximum_power is not None:
                thresholds.append((float(self._filters.maximum_power), f"Power max = {self._filters.maximum_power:g} dBW"))
        elif key == "stations" and self._filters.minimum_stations is not None:
            thresholds.append((float(self._filters.minimum_stations), f"Stations min = {self._filters.minimum_stations:d}"))
        for value, threshold_label in thresholds:
            if not use_log_x or value > 0:
                ax.axvline(
                    value,
                    linestyle="--",
                    linewidth=1.5,
                    color=theme_values(self._theme)["text"],
                    label=threshold_label,
                )

        ax.set_title(f"{label} distribution — {set_label}")
        ax.set_xlabel(axis_label)
        ax.set_ylabel("Source count")
        if use_log_x:
            ax.set_xscale("log")
        if self.log_count.isChecked():
            ax.set_yscale("log")
        apply_figure_theme(self.figure, (ax,), self._theme, show_grid=True)
        legend = ax.legend(loc="best")
        if legend is not None:
            colors = theme_values(self._theme)
            legend.get_frame().set_facecolor(colors["axes"])
            legend.get_frame().set_edgecolor(colors["text"])
            for text in legend.get_texts():
                text.set_color(colors["text"])

        centres = 0.5 * (edges[:-1] + edges[1:])
        for i, centre in enumerate(centres):
            row: dict[str, float | int] = {
                "bin_left": float(edges[i]),
                "bin_right": float(edges[i + 1]),
                "bin_center": float(centre),
            }
            for name, counts in counts_by_name.items():
                row[name] = int(counts[i])
            self._histogram_rows.append(row)

        summary_parts = [f"{name}: {self._stats(vals)}" for name, vals in series]
        if scope == "diagnostic" and len(series) == 2 and series[0][1].size:
            fraction = 100.0 * series[1][1].size / series[0][1].size
            summary_parts.append(f"Accepted after this variable's filter: {fraction:.1f}%")
        self.summary.setText("\n".join(summary_parts))
        self.canvas.draw_idle()

    def export_histogram_data(self) -> None:
        if not self._histogram_rows:
            QMessageBox.information(self, "No histogram data", "Refresh a non-empty distribution before exporting.")
            return
        selected, _ = QFileDialog.getSaveFileName(self, "Export histogram data", "source_distribution.csv", "CSV file (*.csv)")
        if not selected:
            return
        path = Path(selected)
        if path.suffix.lower() != ".csv":
            path = path.with_suffix(".csv")
        fieldnames = list(self._histogram_rows[0])
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._histogram_rows)

    def save_figure(self) -> None:
        selected, _ = QFileDialog.getSaveFileName(self, "Save source-distribution figure", "source_distribution.png", "PNG image (*.png);;PDF document (*.pdf);;SVG image (*.svg)")
        if not selected:
            return
        self.figure.savefig(selected, dpi=300, bbox_inches="tight")

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        event.ignore()
        self.hide()


__all__ = ["SourceDistributionsWindow"]
