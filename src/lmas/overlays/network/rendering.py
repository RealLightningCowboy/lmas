"""Responsive linked rendering for ground lightning-location networks."""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import matplotlib.dates as mdates
from matplotlib.collections import LineCollection
from matplotlib.transforms import blended_transform_factory
import numpy as np

from ...coordinates import EARTH_RADIUS_KM, latlon_to_local_km
from ...plotting.figures import refresh_figure_legend
from .manager import NetworkDatasetRecord, NetworkOverlayManager


@dataclass(frozen=True, slots=True)
class NetworkRenderSummary:
    dataset_key: str
    visible_events: int
    rendered_events: int
    visible_ellipses: int
    truncated: bool
    total_seconds: float


class _Bundle:
    def __init__(self, plan, time_axis=None) -> None:
        self.plan = plan
        self.time_axis = time_axis
        self.categories: dict[str, object] = {}
        self.ellipses = None
        self.time_categories: dict[str, object] = {}
        self.time_label = None


_CATEGORY_ORDER = ("NEGATIVE_CG", "POSITIVE_CG", "UNKNOWN_CG", "IC", "OTHER")
_CATEGORY_MARKERS = {
    "NEGATIVE_CG": "v",
    "POSITIVE_CG": "^",
    "UNKNOWN_CG": "s",
    "IC": "x",
    "OTHER": "D",
}
_CATEGORY_LABELS = {
    "NEGATIVE_CG": "negative CG",
    "POSITIVE_CG": "positive CG",
    "UNKNOWN_CG": "CG, polarity unknown",
    "IC": "IC",
    "OTHER": "other/unknown",
}


class NetworkOverlayRenderer:
    def __init__(self, manager: NetworkOverlayManager, *, for_export: bool = False) -> None:
        self.manager = manager
        self.for_export = bool(for_export)
        self.figure = None
        self.project = None
        self._bundles: dict[str, _Bundle] = {}
        self.artists: list[object] = []
        self.summaries: tuple[NetworkRenderSummary, ...] = ()
        self.manager.set_renderer(self)

    def clear(self) -> None:
        for artist in tuple(self.artists):
            try:
                artist.remove()
            except (ValueError, AttributeError):
                try:
                    artist.set_visible(False)
                except AttributeError:
                    pass
        self.artists.clear()
        self._bundles.clear()
        self.summaries = ()
        self._draw_idle()

    def bind(self, figure, project) -> None:
        if figure is not self.figure:
            self.clear()
        self.figure = figure
        self.project = project
        self.refresh()

    def refresh(self) -> tuple[NetworkRenderSummary, ...]:
        if self.figure is None or self.project is None:
            self.summaries = ()
            return self.summaries
        metadata = getattr(self.figure, "_lmas_metadata", {}) or {}
        axes = metadata.get("axes") or {}
        plan = axes.get("plan")
        time_axis = axes.get("time_altitude")
        if plan is None:
            self.summaries = ()
            return self.summaries
        plan_names = self._plan_coordinate_names(metadata, plan)
        if plan_names is None:
            self.summaries = ()
            return self.summaries
        x_name, y_name = plan_names

        active = {record.key for record in self.manager.records}
        for key in tuple(self._bundles):
            bundle = self._bundles[key]
            if key not in active or bundle.plan is not plan or bundle.time_axis is not time_axis:
                self._remove_bundle(key)

        summaries: list[NetworkRenderSummary] = []
        enabled_records = [record for record in self.manager.records if record.style.enabled]
        for lane, record in enumerate(enabled_records):
            summaries.append(
                self._render_record(record, plan, time_axis, x_name=x_name, y_name=y_name, lane=lane)
            )
        for record in self.manager.records:
            if not record.style.enabled:
                self._hide_record(record.key)
        self.summaries = tuple(summaries)
        self._refresh_figure_legend()
        self._draw_idle()
        return self.summaries

    def _render_record(self, record, plan, time_axis, *, x_name: str, y_name: str, lane: int) -> NetworkRenderSummary:
        started = perf_counter()
        style = record.style.validated()
        observation = record.observation
        time_range = self._time_range_ns(time_axis)
        selection = observation.select(
            time_range_ns=time_range,
            minimum_absolute_peak_current_ka=style.minimum_absolute_peak_current_ka,
            minimum_sensor_count=style.minimum_sensor_count,
        )
        indices = selection.event_indices
        x, y = self._point_coordinates(
            observation.events.longitude_deg[indices],
            observation.events.latitude_deg[indices],
            x_name=x_name,
            y_name=y_name,
        )
        finite = np.isfinite(x) & np.isfinite(y)
        if style.follow_spatial_view:
            xlim = sorted(plan.get_xlim())
            ylim = sorted(plan.get_ylim())
            finite &= (x >= xlim[0]) & (x <= xlim[1]) & (y >= ylim[0]) & (y <= ylim[1])
        indices = indices[finite]
        x = x[finite]
        y = y[finite]
        # Apply scientific category/polarity visibility before interactive
        # sampling so hidden events cannot leak into uncertainty ellipses or
        # consume the event cap.
        categories = self._categories(record, indices)
        active = categories != "HIDDEN"
        indices = indices[active]
        x = x[active]
        y = y[active]
        categories = categories[active]
        visible_events = int(indices.size)
        truncated = False
        limit = 0 if self.for_export else style.maximum_interactive_events
        if limit and indices.size > limit:
            sample = np.linspace(0, indices.size - 1, limit, dtype=np.int64)
            indices = indices[sample]
            x = x[sample]
            y = y[sample]
            categories = categories[sample]
            truncated = True

        bundle = self._bundles.get(record.key)
        if bundle is None:
            bundle = _Bundle(plan, time_axis)
            self._bundles[record.key] = bundle

        theme = getattr(self.figure, "_lmas_theme", {}) or {}
        edge = "black" if str(theme.get("axes", "black")).lower() in {"white", "#ffffff"} else "white"
        for category in _CATEGORY_ORDER:
            mask = categories == category
            offsets = np.column_stack((x[mask], y[mask])) if np.any(mask) else np.empty((0, 2))
            self._update_category(bundle, record, category, offsets, indices[mask], style, edge)

        ellipse_count = self._update_ellipses(bundle, record, plan, indices, x, y, style, x_name=x_name)
        self._update_time_rail(bundle, record, time_axis, indices, categories, style, edge, lane)
        return NetworkRenderSummary(
            dataset_key=record.key,
            visible_events=visible_events,
            rendered_events=int(indices.size),
            visible_ellipses=ellipse_count,
            truncated=truncated,
            total_seconds=perf_counter() - started,
        )

    def _categories(self, record: NetworkDatasetRecord, indices: np.ndarray) -> np.ndarray:
        style = record.style
        events = record.observation.events
        result = np.full(indices.size, "OTHER", dtype="U16")
        types = np.char.upper(events.event_type[indices].astype("U16"))
        polarity = events.polarity[indices]
        cg = types == "CG"
        ic = types == "IC"
        result[cg & (polarity < 0)] = "NEGATIVE_CG"
        result[cg & (polarity > 0)] = "POSITIVE_CG"
        result[cg & (polarity == 0)] = "UNKNOWN_CG"
        result[ic] = "IC"
        allowed = np.ones(indices.size, dtype=bool)
        allowed &= ~((polarity > 0) & (not style.show_positive))
        allowed &= ~((polarity < 0) & (not style.show_negative))
        allowed &= ~((polarity == 0) & (not style.show_unknown_polarity))
        allowed &= ~(cg & (not style.show_cg))
        allowed &= ~(ic & (not style.show_ic))
        allowed &= ~((~cg & ~ic) & (not style.show_other_types))
        result[~allowed] = "HIDDEN"
        return result

    @staticmethod
    def _category_color(style, category: str) -> str:
        if category == "NEGATIVE_CG":
            return style.negative_color
        if category == "POSITIVE_CG":
            return style.positive_color
        if category == "IC":
            return style.intracloud_color
        return style.unknown_color

    def _update_category(self, bundle, record, category, offsets, indices, style, edge) -> None:
        artist = bundle.categories.get(category)
        if artist is None:
            artist = bundle.plan.scatter([], [], marker=_CATEGORY_MARKERS[category], rasterized=False)
            bundle.categories[category] = artist
            self.artists.append(artist)
        visible = bool(style.show_events and offsets.size)
        artist.set_visible(visible)
        artist.set_offsets(offsets if offsets.size else np.empty((0, 2)))
        artist.set_label(
            f"{record.display_name} — {_CATEGORY_LABELS[category]}"
            if visible and style.show_legend else "_nolegend_"
        )
        if not visible:
            return
        color = self._category_color(style, category)
        size = np.full(offsets.shape[0], style.marker_size, dtype=float)
        if style.scale_by_peak_current:
            current = np.abs(record.observation.events.peak_current_ka[indices])
            finite = np.isfinite(current)
            if np.any(finite):
                scale = np.ones(current.size, dtype=float)
                reference = max(float(np.nanpercentile(current[finite], 90)), 1.0)
                scale[finite] = np.clip(np.sqrt(current[finite] / reference), 0.55, 2.2)
                size *= scale
        artist.set_sizes(size)
        if category == "IC":
            artist.set_color(color)
            artist.set_linewidth(max(1.0, style.marker_edge_width))
        else:
            artist.set_facecolor(color)
            artist.set_edgecolor(edge)
            artist.set_linewidth(style.marker_edge_width)
        artist.set_alpha(style.marker_alpha)
        artist.set_zorder(style.event_zorder)

    def _update_ellipses(self, bundle, record, plan, indices, x, y, style, *, x_name: str) -> int:
        if bundle.ellipses is None:
            bundle.ellipses = LineCollection([], linewidths=style.ellipse_line_width)
            plan.add_collection(bundle.ellipses, autolim=False)
            self.artists.append(bundle.ellipses)
        artist = bundle.ellipses
        if not style.show_uncertainty or not indices.size:
            artist.set_segments([])
            artist.set_visible(False)
            artist.set_label("_nolegend_")
            return 0
        events = record.observation.events
        major = events.ellipse_major_km[indices]
        minor = events.ellipse_minor_km[indices]
        angle = events.ellipse_angle_deg[indices]
        keep = np.isfinite(major) & np.isfinite(minor) & (major > 0) & (minor > 0)
        ellipse_indices = np.flatnonzero(keep)
        limit = 0 if self.for_export else style.maximum_interactive_ellipses
        if limit and ellipse_indices.size > limit:
            ellipse_indices = ellipse_indices[np.linspace(0, ellipse_indices.size - 1, limit, dtype=np.int64)]
        theta = np.linspace(0.0, 2.0 * np.pi, 49)
        segments: list[np.ndarray] = []
        for pos in ellipse_indices:
            a = major[pos] / 2.0
            b = minor[pos] / 2.0
            phi = np.deg2rad(angle[pos] if np.isfinite(angle[pos]) else 0.0)
            dx = a * np.cos(theta) * np.cos(phi) - b * np.sin(theta) * np.sin(phi)
            dy = a * np.cos(theta) * np.sin(phi) + b * np.sin(theta) * np.cos(phi)
            if x_name == "longitude":
                lat = record.observation.events.latitude_deg[indices[pos]]
                lon_scale = EARTH_RADIUS_KM * np.cos(np.deg2rad(lat)) * np.pi / 180.0
                lat_scale = EARTH_RADIUS_KM * np.pi / 180.0
                dx = dx / max(abs(lon_scale), 1.0e-9)
                dy = dy / lat_scale
            segments.append(np.column_stack((x[pos] + dx, y[pos] + dy)))
        artist.set_segments(segments)
        artist.set_visible(bool(segments))
        color = style.unknown_color if str(style.ellipse_color).lower() == "auto" else style.ellipse_color
        artist.set_color(color)
        artist.set_alpha(style.ellipse_alpha)
        artist.set_linewidth(style.ellipse_line_width)
        artist.set_zorder(style.ellipse_zorder)
        artist.set_label(
            f"{record.display_name} — location uncertainty"
            if segments and style.show_legend else "_nolegend_"
        )
        return len(segments)

    def _update_time_rail(self, bundle, record, time_axis, indices, categories, style, edge, lane: int) -> None:
        for category in _CATEGORY_ORDER:
            artist = bundle.time_categories.get(category)
            if time_axis is not None and artist is None:
                transform = blended_transform_factory(time_axis.transData, self.figure.transFigure)
                artist = time_axis.scatter(
                    [], [], marker=_CATEGORY_MARKERS[category], transform=transform, clip_on=False
                )
                bundle.time_categories[category] = artist
                self.artists.append(artist)
            if artist is None:
                continue
            mask = categories == category
            visible = bool(style.show_time_rail and np.any(mask))
            artist.set_visible(visible)
            if not visible:
                artist.set_offsets(np.empty((0, 2)))
                continue
            times = mdates.date2num(
                record.observation.events.time_ns[indices[mask]].astype("datetime64[ns]").astype("datetime64[us]").astype(object)
            )
            axes_top = float(time_axis.get_position().y1)
            lane_y = min(0.985, axes_top + 0.010 + lane * 0.016)
            artist.set_offsets(np.column_stack((times, np.full(times.size, lane_y))))
            artist.set_sizes(np.full(times.size, style.time_rail_marker_size, dtype=float))
            color = self._category_color(style, category)
            if category == "IC":
                artist.set_color(color)
                artist.set_linewidth(max(1.0, style.marker_edge_width))
            else:
                artist.set_facecolor(color)
                artist.set_edgecolor(edge)
                artist.set_linewidth(style.marker_edge_width)
            artist.set_alpha(style.marker_alpha)
            artist.set_zorder(style.time_rail_zorder)
            artist.set_label("_nolegend_")
        if time_axis is None:
            return
        if bundle.time_label is None:
            label_transform = blended_transform_factory(time_axis.transAxes, self.figure.transFigure)
            bundle.time_label = time_axis.text(
                0.005, 0.0, "", transform=label_transform, clip_on=False,
                ha="left", va="center", fontsize="x-small", zorder=style.time_rail_zorder + 0.1,
            )
            self.artists.append(bundle.time_label)
        axes_top = float(time_axis.get_position().y1)
        label_y = min(0.985, axes_top + 0.010 + lane * 0.016)
        bundle.time_label.set_position((0.005, label_y))
        bundle.time_label.set_text(record.display_name)
        bundle.time_label.set_visible(bool(style.show_time_rail and indices.size))

    def _point_coordinates(self, lon, lat, *, x_name: str, y_name: str):
        if x_name == "longitude" and y_name == "latitude":
            return np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)
        return latlon_to_local_km(
            lon, lat, self.project.reference_longitude, self.project.reference_latitude
        )

    def _remove_bundle(self, key: str) -> None:
        bundle = self._bundles.pop(key, None)
        if bundle is None:
            return
        owned = list(bundle.categories.values()) + list(bundle.time_categories.values())
        owned.extend([bundle.ellipses, bundle.time_label])
        for artist in owned:
            if artist is None:
                continue
            try:
                artist.remove()
            except (ValueError, AttributeError):
                pass
            if artist in self.artists:
                self.artists.remove(artist)

    def _hide_record(self, key: str) -> None:
        bundle = self._bundles.get(key)
        if bundle is None:
            return
        for artist in list(bundle.categories.values()) + list(bundle.time_categories.values()):
            artist.set_visible(False)
        if bundle.ellipses is not None:
            bundle.ellipses.set_visible(False)
        if bundle.time_label is not None:
            bundle.time_label.set_visible(False)

    def _refresh_figure_legend(self) -> None:
        if self.figure is None or self.project is None:
            return
        metadata = getattr(self.figure, "_lmas_metadata", {}) or {}
        axes = tuple(metadata.get("axis_order") or ())
        plot = metadata.get("plot_spec") or getattr(self.project, "plot", None)
        if not axes or plot is None or not hasattr(plot, "show_legend"):
            return
        # Satellite Overlays may already have expanded the bottom gutter for a
        # TOE colorbar. Preserve that clearance while rebuilding one combined
        # LMA/GLM/network figure legend.
        is_portrait = str(metadata.get("layout") or "") == "xlma"
        satellite_bottom = getattr(self.figure, "_lmas_satellite_original_subplot_bottom", None)
        extra = (
            0.52
            if not is_portrait
            and satellite_bottom is not None
            and self.figure.subplotpars.bottom >= 0.20
            else 0.0
        )
        refresh_figure_legend(self.figure, axes, plot, extra_clearance_inches=extra)

    def _draw_idle(self) -> None:
        if self.figure is not None and getattr(self.figure, "canvas", None) is not None:
            self.figure.canvas.draw_idle()

    @staticmethod
    def _time_range_ns(axis):
        if axis is None:
            return None
        low, high = sorted(axis.get_xlim())
        start = np.datetime64(mdates.num2date(low).replace(tzinfo=None), "ns")
        end = np.datetime64(mdates.num2date(high).replace(tzinfo=None), "ns")
        return int(start.astype(np.int64)), int(end.astype(np.int64))

    @staticmethod
    def _plan_coordinate_names(metadata, plan):
        axis_order = tuple(metadata.get("axis_order") or ())
        coordinate_names = tuple(metadata.get("coordinate_names") or ())
        for axis, names in zip(axis_order, coordinate_names):
            if axis is plan and len(names) == 2:
                return str(names[0]), str(names[1])
        return None


__all__ = ["NetworkOverlayRenderer", "NetworkRenderSummary"]
