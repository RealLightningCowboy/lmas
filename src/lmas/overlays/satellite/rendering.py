"""Responsive Matplotlib rendering for linked satellite overlays.

The dev5 renderer retains one Matplotlib collection per scientific layer,
updates those collections in place, adds a non-altitude GLM group-time rail,
and uses dedicated bottom-canvas colorbar axes rather than inset colorbars.
The same renderer is used by the live viewer and full-resolution figure export.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Iterable

import matplotlib.dates as mdates
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm, Normalize
from matplotlib.path import Path as MplPath
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter

from ...coordinates import latlon_to_local_km
from ...plotting.common import style_colorbar
from ...plotting.figures import refresh_figure_legend
from .colormaps import satellite_colormap
from .manager import SatelliteDatasetRecord, SatelliteOverlayManager

_PLATFORM_COLORS = {
    "east": "chartreuse",
    "west": "crimson",
}
_TIME_RAIL_Y = {
    "east": 0.965,
    "west": 0.905,
}
_TIME_RAIL_LABEL = {
    "east": "GLM E",
    "west": "GLM W",
}


def configure_group_energy_time_axis(axis) -> None:
    """Apply compact UTC major labels plus visible linear/log minor ticks."""

    locator = mdates.AutoDateLocator(minticks=3, maxticks=6, interval_multiples=False)
    formatter = mdates.ConciseDateFormatter(locator, show_offset=True)
    formatter.formats = ["%Y", "%b", "%d", "%H:%M", "%H:%M:%S", "%S.%f"]
    formatter.zero_formats = ["", "%Y", "%b %d", "%H:%M", "%H:%M:%S", "%S.%f"]
    formatter.offset_formats = [
        "", "%Y", "%Y-%m", "%Y-%m-%d", "%Y-%m-%d", "%Y-%m-%d %H:%M UTC"
    ]
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(formatter)
    axis.xaxis.set_minor_locator(AutoMinorLocator(2))
    axis.yaxis.set_minor_locator(
        LogLocator(base=10.0, subs=np.arange(2.0, 10.0) * 0.1, numticks=100)
    )
    axis.yaxis.set_minor_formatter(NullFormatter())
    axis.tick_params(axis="x", which="major", labelsize=8, pad=2)
    axis.tick_params(axis="x", which="minor", length=3.0, width=0.6)
    axis.tick_params(axis="y", which="minor", length=3.0, width=0.6)


@dataclass(slots=True)
class RenderSummary:
    dataset_key: str
    visible_events: int
    rendered_events: int
    visible_groups: int
    visible_flashes: int
    truncated: bool
    time_rail_groups: int = 0
    selection_seconds: float = 0.0
    geometry_seconds: float = 0.0
    artist_seconds: float = 0.0
    total_seconds: float = 0.0


@dataclass(slots=True)
class _ArtistBundle:
    axis: object
    time_axis: object | None = None
    footprint: object | None = None
    groups: object | None = None
    flashes: object | None = None
    maximum: object | None = None
    time_groups: object | None = None
    time_label: object | None = None

    def layer_artists(self):
        return tuple(
            artist
            for artist in (
                self.footprint,
                self.groups,
                self.flashes,
                self.maximum,
                self.time_groups,
                self.time_label,
            )
            if artist is not None
        )


@dataclass(slots=True)
class SatelliteOverlayRenderer:
    manager: SatelliteOverlayManager
    figure: object | None = None
    project: object | None = None
    for_export: bool = False
    artists: list[object] = field(default_factory=list)
    colorbars: list[object] = field(default_factory=list)
    summaries: list[RenderSummary] = field(default_factory=list)
    _bundles: dict[str, _ArtistBundle] = field(default_factory=dict, init=False, repr=False)
    _bottom_colorbar_axes: list[object] = field(default_factory=list, init=False, repr=False)
    _bottom_colorbar_mappables: list[ScalarMappable] = field(default_factory=list, init=False, repr=False)

    def bind(self, figure, project) -> None:
        if figure is not self.figure:
            self.clear(draw=False)
        self.figure = figure
        self.project = project
        self.manager.set_renderer(self)
        self.refresh()

    def clear(self, *, draw: bool = True) -> None:
        for key in tuple(self._bundles):
            self._remove_bundle(key)
        self._remove_bottom_colorbars()
        self.artists.clear()
        self.colorbars.clear()
        self.summaries.clear()
        if draw and self.figure is not None and getattr(self.figure, "canvas", None) is not None:
            self.figure.canvas.draw_idle()

    def _remove_bundle(self, key: str) -> None:
        bundle = self._bundles.pop(key, None)
        if bundle is None:
            return
        for artist in bundle.layer_artists():
            try:
                artist.remove()
            except Exception:
                pass
            try:
                self.artists.remove(artist)
            except ValueError:
                pass

    def _hide_bundle(self, bundle: _ArtistBundle) -> None:
        for artist in bundle.layer_artists():
            try:
                artist.set_visible(False)
            except Exception:
                pass

    def _remove_bottom_colorbars(self) -> None:
        for colorbar in tuple(self.colorbars):
            try:
                colorbar.remove()
            except Exception:
                pass
        for axis in tuple(self._bottom_colorbar_axes):
            try:
                axis.remove()
            except Exception:
                pass
        self.colorbars.clear()
        self._bottom_colorbar_axes.clear()
        self._bottom_colorbar_mappables.clear()

    def refresh(self) -> tuple[RenderSummary, ...]:
        self.summaries.clear()
        if self.figure is None or self.project is None or not self.manager.has_data:
            for bundle in self._bundles.values():
                self._hide_bundle(bundle)
            self._update_bottom_colorbars(())
            self._refresh_figure_legend(False)
            self._draw_idle()
            return ()
        metadata = getattr(self.figure, "_lmas_metadata", None)
        if not isinstance(metadata, dict):
            return ()
        axes = metadata.get("axes") or {}
        plan = axes.get("plan")
        time_axis = axes.get("time_altitude")
        if plan is None:
            return ()
        plan_names = self._plan_coordinate_names(metadata, plan)
        if plan_names is None:
            return ()
        x_name, y_name = plan_names

        # A full LMAS redraw replaces the axes. Remove bundles attached to the
        # prior axes, but retain/update them throughout ordinary zoom/pan.
        for key, bundle in tuple(self._bundles.items()):
            if bundle.axis is not plan or bundle.time_axis is not time_axis:
                self._remove_bundle(key)

        time_range = self._time_range_ns(time_axis)
        records = self.manager.records
        record_keys = {record.key for record in records}
        for key in tuple(self._bundles):
            if key not in record_keys:
                self._remove_bundle(key)
        enabled = [record for record in records if record.style.enabled]
        enabled_keys = {record.key for record in enabled}
        for key, bundle in self._bundles.items():
            if key not in enabled_keys:
                self._hide_bundle(bundle)

        shared_norm = (
            self._shared_event_norm(enabled, time_range)
            if self.manager.shared_energy_scale
            else None
        )
        colorbar_specs: list[tuple[str, object, object]] = []
        for record in enabled:
            summary, colorbar_spec = self._render_record(
                record,
                plan,
                time_axis=time_axis,
                x_name=x_name,
                y_name=y_name,
                time_range=time_range,
                shared_norm=shared_norm,
            )
            self.summaries.append(summary)
            if colorbar_spec is not None:
                colorbar_specs.append(colorbar_spec)
        self._update_bottom_colorbars(tuple(colorbar_specs))
        self._refresh_figure_legend(bool(colorbar_specs))
        self._draw_idle()
        return tuple(self.summaries)

    def _render_record(
        self,
        record: SatelliteDatasetRecord,
        plan,
        *,
        time_axis,
        x_name: str,
        y_name: str,
        time_range,
        shared_norm,
    ) -> tuple[RenderSummary, tuple[str, object, object] | None]:
        started = perf_counter()
        style = record.style.validated()
        observation = record.observation
        selection_started = perf_counter()
        selection = observation.select(time_range_ns=time_range)
        all_event_idx = selection.event_indices
        time_group_idx = selection.group_indices
        group_idx = time_group_idx.copy()
        flash_idx = selection.flash_indices

        # Scientific visibility counts use the exact axes bounds. Footprint
        # rendering uses a padded center-selection box so event pixels whose
        # centers lie just outside the axes can still fill the clipped edge.
        _event_x, _event_y, exact_event_keep = self._event_coordinates(
            record,
            all_event_idx,
            x_name=x_name,
            y_name=y_name,
            plan=plan,
            padding_fraction=0.0,
        )
        _pad_x, _pad_y, padded_event_keep = self._event_coordinates(
            record,
            all_event_idx,
            x_name=x_name,
            y_name=y_name,
            plan=plan,
            padding_fraction=style.footprint_render_padding_fraction,
        )
        visible_event_idx = all_event_idx[exact_event_keep]
        event_idx = all_event_idx[padded_event_keep]
        visible_events = int(visible_event_idx.size)
        limit = 0 if self.for_export else int(style.maximum_interactive_events)
        truncated = bool(limit and event_idx.size > limit)
        if truncated:
            # Preserve all centered-in-view events whenever the cap permits,
            # then spend the remaining budget on the highest-energy padded
            # neighbors. This prevents the edge pad from displacing the
            # scientific core of the current view.
            if visible_event_idx.size >= limit:
                energy = observation.events.energy_j[visible_event_idx]
                finite_energy = np.nan_to_num(energy, nan=-np.inf)
                chosen = np.argpartition(finite_energy, -limit)[-limit:]
                event_idx = visible_event_idx[chosen]
            else:
                outside_idx = all_event_idx[padded_event_keep & ~exact_event_keep]
                remaining = limit - visible_event_idx.size
                if outside_idx.size > remaining:
                    energy = observation.events.energy_j[outside_idx]
                    finite_energy = np.nan_to_num(energy, nan=-np.inf)
                    chosen = np.argpartition(finite_energy, -remaining)[-remaining:]
                    outside_idx = outside_idx[chosen]
                event_idx = np.concatenate((visible_event_idx, outside_idx))
        selection_seconds = perf_counter() - selection_started

        norm = shared_norm or self._event_norm(
            observation.events.energy_j[event_idx],
            logarithmic=style.logarithmic_energy,
        )
        cmap = satellite_colormap(style.colormap)
        role = observation.identity.operational_role.lower()
        role_color = _PLATFORM_COLORS.get(role, "deepskyblue")
        group_color = (
            role_color
            if str(style.group_marker_color).strip().lower() == "auto"
            else str(style.group_marker_color)
        )
        theme = getattr(self.figure, "_lmas_theme", {}) or {}
        edge = "black" if str(theme.get("axes", "black")).lower() in {"white", "#ffffff"} else "white"
        bundle = self._bundles.get(record.key)
        if bundle is None:
            bundle = _ArtistBundle(plan, time_axis=time_axis)
            self._bundles[record.key] = bundle

        geometry_started = perf_counter()
        polygons = np.empty((0, 4, 2), dtype=float)
        polygon_energy = np.empty(0, dtype=float)
        if style.show_event_footprints and event_idx.size and norm is not None:
            polygons_lonlat = observation.geometry.event_corners_lonlat(event_idx)
            polygons = self._transform_polygons(polygons_lonlat, x_name=x_name, y_name=y_name)
            polygon_keep = self._polygons_in_plan(polygons, plan)
            polygons = polygons[polygon_keep]
            polygon_energy = observation.events.energy_j[event_idx][polygon_keep] * 1.0e15
        geometry_seconds = perf_counter() - geometry_started

        artist_started = perf_counter()
        self._update_footprints(
            bundle,
            plan,
            polygons,
            polygon_energy,
            style=style,
            cmap=cmap,
            norm=norm,
            label=f"{observation.identity.legend_prefix} — GLM event footprints",
        )

        group_x, group_y, group_keep = self._point_coordinates(
            observation.groups.longitude_deg[group_idx],
            observation.groups.latitude_deg[group_idx],
            x_name=x_name,
            y_name=y_name,
            plan=plan,
        )
        group_idx = group_idx[group_keep]
        group_offsets = np.column_stack((group_x[group_keep], group_y[group_keep]))
        self._update_groups(
            bundle,
            plan,
            group_offsets,
            style=style,
            role_color=group_color,
            edge=edge,
            label=f"{observation.identity.legend_prefix} — GLM group centroids",
        )

        flash_offsets = np.empty((0, 2), dtype=float)
        if style.show_flash_centroids and flash_idx.size:
            flash_x, flash_y, flash_keep = self._point_coordinates(
                observation.flashes.longitude_deg[flash_idx],
                observation.flashes.latitude_deg[flash_idx],
                x_name=x_name,
                y_name=y_name,
                plan=plan,
            )
            flash_offsets = np.column_stack((flash_x[flash_keep], flash_y[flash_keep]))
        self._update_flashes(
            bundle,
            plan,
            flash_offsets,
            style=style,
            role_color=role_color,
        )

        maximum_offsets = np.empty((0, 2), dtype=float)
        if style.show_maximum_group and group_idx.size:
            energy = observation.groups.energy_j[group_idx]
            finite = np.isfinite(energy)
            if np.any(finite):
                candidate = group_idx[np.flatnonzero(finite)[np.argmax(energy[finite])]]
                star_x, star_y, star_keep = self._point_coordinates(
                    observation.groups.longitude_deg[[candidate]],
                    observation.groups.latitude_deg[[candidate]],
                    x_name=x_name,
                    y_name=y_name,
                    plan=plan,
                )
                if star_keep[0]:
                    maximum_offsets = np.column_stack((star_x, star_y))
        self._update_maximum(
            bundle, plan, maximum_offsets, style=style, group_color=group_color
        )

        time_count = self._update_time_rail(
            bundle,
            time_axis,
            observation.groups.time_ns[time_group_idx],
            observation.groups.energy_j[time_group_idx],
            style=style,
            role=role,
            role_color=role_color,
            edge=edge,
        )
        artist_seconds = perf_counter() - artist_started
        colorbar_spec = None
        if (
            style.show_colorbar
            and style.show_event_footprints
            and polygon_energy.size
            and norm is not None
        ):
            colorbar_spec = (observation.identity.display_name, cmap, norm)
        return RenderSummary(
            dataset_key=record.key,
            visible_events=visible_events,
            rendered_events=int(polygons.shape[0]),
            visible_groups=int(group_idx.size),
            visible_flashes=int(flash_idx.size),
            truncated=truncated,
            time_rail_groups=time_count,
            selection_seconds=selection_seconds,
            geometry_seconds=geometry_seconds,
            artist_seconds=artist_seconds,
            total_seconds=perf_counter() - started,
        ), colorbar_spec

    def _update_footprints(self, bundle, plan, polygons, energies, *, style, cmap, norm, label) -> None:
        visible = bool(style.show_event_footprints and len(polygons) and norm is not None)
        if bundle.footprint is None:
            bundle.footprint = PolyCollection([], cmap=cmap, norm=norm, rasterized=True)
            plan.add_collection(bundle.footprint, autolim=False)
            self.artists.append(bundle.footprint)
        artist = bundle.footprint
        artist.set_visible(visible)
        artist.set_label(label if visible else "_nolegend_")
        if not visible:
            artist.set_verts([])
            artist.set_array(np.empty(0, dtype=float))
            return
        artist.set_verts(polygons)
        artist.set_array(np.asarray(energies, dtype=float))
        artist.set_cmap(cmap)
        artist.set_norm(norm)
        artist.set_alpha(style.footprint_alpha)
        artist.set_linewidth(style.footprint_edge_width)
        artist.set_edgecolor((1.0, 1.0, 1.0, 0.35))
        artist.set_zorder(style.footprint_zorder)

    def _update_groups(self, bundle, plan, offsets, *, style, role_color, edge, label) -> None:
        if bundle.groups is None:
            bundle.groups = plan.scatter([], [], marker="o", label=label, rasterized=False)
            self.artists.append(bundle.groups)
        artist = bundle.groups
        visible = bool(style.show_group_centroids and offsets.size)
        artist.set_visible(visible)
        artist.set_label(label if visible else "_nolegend_")
        artist.set_offsets(offsets if offsets.size else np.empty((0, 2)))
        if visible:
            artist.set_sizes(np.full(offsets.shape[0], style.group_marker_size, dtype=float))
            artist.set_facecolor(role_color)
            artist.set_edgecolor(edge)
            artist.set_linewidth(style.group_edge_width)
            artist.set_alpha(1.0)
            artist.set_zorder(style.group_zorder)
            artist.set_label(label)

    def _update_flashes(self, bundle, plan, offsets, *, style, role_color) -> None:
        if bundle.flashes is None:
            bundle.flashes = plan.scatter([], [], s=70.0, marker="D", facecolors="none")
            self.artists.append(bundle.flashes)
        artist = bundle.flashes
        visible = bool(style.show_flash_centroids and offsets.size)
        artist.set_visible(visible)
        artist.set_offsets(offsets if offsets.size else np.empty((0, 2)))
        if visible:
            artist.set_sizes(np.full(offsets.shape[0], 70.0, dtype=float))
            artist.set_facecolor("none")
            artist.set_edgecolor(role_color)
            artist.set_linewidth(1.0)
            artist.set_zorder(style.group_zorder + 0.05)

    def _update_maximum(self, bundle, plan, offsets, *, style, group_color) -> None:
        if bundle.maximum is None:
            bundle.maximum = plan.scatter([], [], marker=MplPath.unit_regular_star(4))
            self.artists.append(bundle.maximum)
        artist = bundle.maximum
        visible = bool(style.show_maximum_group and offsets.size)
        artist.set_visible(visible)
        artist.set_offsets(offsets if offsets.size else np.empty((0, 2)))
        if visible:
            artist.set_sizes(np.full(offsets.shape[0], style.maximum_group_size, dtype=float))
            artist.set_facecolor(group_color)
            artist.set_edgecolor("black")
            artist.set_linewidth(0.9)
            artist.set_zorder(style.group_zorder + 0.10)

    def _update_time_rail(
        self,
        bundle,
        time_axis,
        time_ns,
        energy_j,
        *,
        style,
        role,
        role_color,
        edge,
    ) -> int:
        if time_axis is None:
            if bundle.time_groups is not None:
                bundle.time_groups.set_visible(False)
            if bundle.time_label is not None:
                bundle.time_label.set_visible(False)
            return 0
        if bundle.time_groups is None:
            bundle.time_groups = time_axis.scatter(
                [], [],
                marker="o",
                transform=time_axis.get_xaxis_transform(),
                clip_on=True,
                rasterized=False,
            )
            self.artists.append(bundle.time_groups)
        if bundle.time_label is None:
            bundle.time_label = time_axis.text(
                0.004,
                _TIME_RAIL_Y.get(role, 0.935),
                _TIME_RAIL_LABEL.get(role, "GLM"),
                transform=time_axis.transAxes,
                ha="left",
                va="center",
                fontsize=7,
                fontweight="bold",
                clip_on=True,
            )
            self.artists.append(bundle.time_label)

        times = np.asarray(time_ns, dtype="datetime64[ns]")
        energy = np.asarray(energy_j, dtype=float) * 1.0e15
        keep = (~np.isnat(times)) & np.isfinite(energy) & (energy > 0)
        visible = bool(style.show_group_time_rail and np.any(keep))
        artist = bundle.time_groups
        label = bundle.time_label
        artist.set_visible(visible)
        label.set_visible(bool(visible and style.show_time_rail_labels))
        if not visible:
            artist.set_offsets(np.empty((0, 2)))
            return 0

        x = mdates.date2num(times[keep].astype("datetime64[us]").astype(object))
        y_value = _TIME_RAIL_Y.get(role, 0.935)
        offsets = np.column_stack((x, np.full(x.size, y_value, dtype=float)))
        values = np.log10(energy[keep])
        if values.size > 1 and np.nanmax(values) > np.nanmin(values):
            low, high = np.nanpercentile(values, (5.0, 95.0))
            if high <= low:
                scaled = np.full(values.size, 0.5)
            else:
                scaled = np.clip((values - low) / (high - low), 0.0, 1.0)
        else:
            scaled = np.full(values.size, 0.5)
        sizes = style.time_rail_marker_size * (0.55 + 1.20 * scaled)
        artist.set_offsets(offsets)
        artist.set_sizes(sizes)
        artist.set_facecolor(role_color)
        artist.set_edgecolor(edge)
        artist.set_linewidth(0.45)
        artist.set_alpha(0.95)
        artist.set_zorder(style.time_rail_zorder)
        label.set_position((0.004, y_value))
        label.set_text(_TIME_RAIL_LABEL.get(role, "GLM"))
        label.set_color(role_color)
        label.set_zorder(style.time_rail_zorder + 0.1)
        return int(x.size)

    def _update_bottom_colorbars(self, specs: tuple[tuple[str, object, object], ...]) -> None:
        if self.figure is None:
            return

        metadata = getattr(self.figure, "_lmas_metadata", {}) or {}
        layout_text = str(metadata.get("layout", "")).strip().lower()
        is_portrait = "xlma" in layout_text or "portrait" in layout_text

        # Landscape keeps the established bottom caption gutter unchanged.
        # Portrait uses a left-side vertical TOE bar and therefore does not
        # reserve any additional space beneath the scientific axes.
        self._set_bottom_gutter(bool(specs) if not is_portrait else False)

        # A shared scale and matching colormap produce one authoritative bar.
        # Otherwise each independently styled dataset gets its own bar.
        if specs and self.manager.shared_energy_scale:
            first_name, first_cmap, first_norm = specs[0]
            if all(
                getattr(cmap, "name", None) == getattr(first_cmap, "name", None)
                for _, cmap, _ in specs
            ):
                specs = (("GLM Total Optical Energy (fJ)", first_cmap, first_norm),)

        if not specs:
            for axis in self._bottom_colorbar_axes:
                axis.set_visible(False)
            return

        axis_map = metadata.get("axes") or {}
        scientific_axes = []
        for axis in axis_map.values():
            if axis is not None and axis not in scientific_axes:
                scientific_axes.append(axis)
        if not scientific_axes:
            return

        self.figure.canvas.draw()
        positions = [axis.get_position() for axis in scientific_axes]
        left = min(position.x0 for position in positions)
        right = max(position.x1 for position in positions)
        bottom = min(position.y0 for position in positions)
        top = max(position.y1 for position in positions)
        count = len(specs)

        expected_orientation = "vertical" if is_portrait else "horizontal"
        orientation_changed = any(
            getattr(colorbar, "orientation", None) != expected_orientation
            for colorbar in self.colorbars
        )

        if len(self._bottom_colorbar_axes) != count or orientation_changed:
            self._remove_bottom_colorbars()

            if is_portrait:
                figure_width = max(float(self.figure.get_figwidth()), 1.0)
                # Place the TOE bar well inside the enlarged left Portrait margin,
                # leaving room for ticks and the full vertical label.
                bar_width_inches = 0.18 if count == 1 else 0.13
                bar_gap_inches = 0.08 if count > 1 else 0.0
                axes_gap_inches = 0.85
                bar_width = bar_width_inches / figure_width
                bar_gap = bar_gap_inches / figure_width
                axes_gap = axes_gap_inches / figure_width
                total_width = count * bar_width + (count - 1) * bar_gap
                x_start = max(0.012, left - axes_gap - total_width)

                for index in range(count):
                    axis = self.figure.add_axes(
                        [
                            x_start + index * (bar_width + bar_gap),
                            bottom,
                            bar_width,
                            top - bottom,
                        ]
                    )
                    mappable = ScalarMappable()
                    colorbar = self.figure.colorbar(
                        mappable,
                        cax=axis,
                        orientation="vertical",
                    )
                    self._bottom_colorbar_axes.append(axis)
                    self._bottom_colorbar_mappables.append(mappable)
                    self.colorbars.append(colorbar)
            else:
                # Raise the Landscape TOE bar conservatively by half of its
                # own height while preserving the established gutter geometry.
                height = 0.014
                y = max(0.055, bottom - 0.100) + 0.5 * height
                gap = 0.018 if count > 1 else 0.0
                width = (right - left - gap * (count - 1)) / count

                for _ in range(count):
                    axis = self.figure.add_axes([left, y, width, height])
                    mappable = ScalarMappable()
                    colorbar = self.figure.colorbar(
                        mappable,
                        cax=axis,
                        orientation="horizontal",
                    )
                    self._bottom_colorbar_axes.append(axis)
                    self._bottom_colorbar_mappables.append(mappable)
                    self.colorbars.append(colorbar)

        theme = str(
            metadata.get("theme")
            or getattr(getattr(self.project, "plot", None), "theme", "dark")
        )

        # Match TOE colorbar typography to the surrounding scientific axes
        # in both layouts rather than using smaller hard-coded text.
        reference_axis = scientific_axes[0]
        axis_label_sizes = [
            float(reference_axis.xaxis.label.get_fontsize()),
            float(reference_axis.yaxis.label.get_fontsize()),
        ]
        reference_label_fontsize = max(axis_label_sizes)
        reference_tick_labels = (
            list(reference_axis.get_xticklabels())
            + list(reference_axis.get_yticklabels())
        )
        reference_tick_fontsize = (
            float(reference_tick_labels[0].get_fontsize())
            if reference_tick_labels
            else reference_label_fontsize
        )

        if is_portrait:
            figure_width = max(float(self.figure.get_figwidth()), 1.0)
            bar_width_inches = 0.18 if count == 1 else 0.13
            bar_gap_inches = 0.08 if count > 1 else 0.0
            axes_gap_inches = 0.85
            bar_width = bar_width_inches / figure_width
            bar_gap = bar_gap_inches / figure_width
            axes_gap = axes_gap_inches / figure_width
            total_width = count * bar_width + (count - 1) * bar_gap
            x_start = max(0.012, left - axes_gap - total_width)

            for index, ((name, cmap, norm), axis, mappable, colorbar) in enumerate(
                zip(
                    specs,
                    self._bottom_colorbar_axes,
                    self._bottom_colorbar_mappables,
                    self.colorbars,
                    strict=True,
                )
            ):
                axis.set_position(
                    [
                        x_start + index * (bar_width + bar_gap),
                        bottom,
                        bar_width,
                        top - bottom,
                    ]
                )
                axis.set_visible(True)
                mappable.set_cmap(cmap)
                mappable.set_norm(norm)
                colorbar.update_normal(mappable)
                label = name if count > 1 else "GLM Total Optical Energy (fJ)"
                colorbar.set_label(
                    label,
                    fontsize=reference_label_fontsize,
                    labelpad=8,
                )
                colorbar.ax.yaxis.set_ticks_position("left")
                colorbar.ax.yaxis.set_label_position("left")
                colorbar.ax.tick_params(
                    labelsize=reference_tick_fontsize,
                    pad=2.0,
                )
                style_colorbar(colorbar, theme)
        else:
            # Keep the working Landscape orientation and width, but lift the
            # TOE bar by half its height to make room for full-size text.
            height = 0.014
            y = max(0.055, bottom - 0.100) + 0.5 * height
            gap = 0.018 if count > 1 else 0.0
            width = (right - left - gap * (count - 1)) / count

            for index, ((name, cmap, norm), axis, mappable, colorbar) in enumerate(
                zip(
                    specs,
                    self._bottom_colorbar_axes,
                    self._bottom_colorbar_mappables,
                    self.colorbars,
                    strict=True,
                )
            ):
                axis.set_position([left + index * (width + gap), y, width, height])
                axis.set_visible(True)
                mappable.set_cmap(cmap)
                mappable.set_norm(norm)
                colorbar.update_normal(mappable)
                label = name if count > 1 else "GLM Total Optical Energy (fJ)"
                colorbar.set_label(
                    label,
                    fontsize=reference_label_fontsize,
                    labelpad=2,
                )
                colorbar.ax.xaxis.set_label_position("top")
                colorbar.ax.tick_params(
                    labelsize=reference_tick_fontsize,
                    pad=1.5,
                )
                style_colorbar(colorbar, theme)

    def _set_bottom_gutter(self, visible: bool) -> None:
        if self.figure is None:
            return
        original = getattr(self.figure, "_lmas_satellite_original_subplot_bottom", None)
        if original is None:
            original = float(self.figure.subplotpars.bottom)
            self.figure._lmas_satellite_original_subplot_bottom = original
        target = max(float(original), 0.20) if visible else float(original)
        if abs(float(self.figure.subplotpars.bottom) - target) > 1.0e-6:
            try:
                self.figure.subplots_adjust(bottom=target)
            except Exception:
                pass

    def _refresh_figure_legend(self, colorbar_visible: bool) -> None:
        if self.figure is None or self.project is None:
            return
        metadata = getattr(self.figure, "_lmas_metadata", {}) or {}
        axes = tuple(metadata.get("axis_order") or ())
        plot = metadata.get("plot_spec") or getattr(self.project, "plot", None)
        if not axes or plot is None or not hasattr(plot, "show_legend"):
            return

        layout_text = str(metadata.get("layout", "")).strip().lower()
        is_portrait = "xlma" in layout_text or "portrait" in layout_text

        # Portrait's TOE bar is vertical on the left, so it requires no extra
        # bottom clearance. Preserve the established Landscape clearance.
        extra_clearance = 0.0 if is_portrait else (0.52 if colorbar_visible else 0.0)
        refresh_figure_legend(
            self.figure,
            axes,
            plot,
            extra_clearance_inches=extra_clearance,
        )

    def _shared_event_norm(self, records: Iterable[SatelliteDatasetRecord], time_range):
        values = []
        logarithmic = True
        for record in records:
            selection = record.observation.select(time_range_ns=time_range)
            if selection.event_indices.size:
                values.append(record.observation.events.energy_j[selection.event_indices])
            logarithmic = logarithmic and record.style.logarithmic_energy
        if not values:
            return None
        return self._event_norm(np.concatenate(values), logarithmic=logarithmic)

    @staticmethod
    def _event_norm(values, logarithmic: bool = True):
        energy_fj = np.asarray(values, dtype=float) * 1.0e15
        finite = energy_fj[np.isfinite(energy_fj)]
        if logarithmic:
            finite = finite[finite > 0]
        if finite.size == 0:
            return None
        low, high = float(np.min(finite)), float(np.max(finite))
        if high <= low:
            high = low * 1.01 if low > 0 else low + 1.0
        return LogNorm(low, high) if logarithmic else Normalize(low, high)

    def _event_coordinates(
        self, record, indices, *, x_name, y_name, plan, padding_fraction: float = 0.0
    ):
        return self._point_coordinates(
            record.observation.events.longitude_deg[indices],
            record.observation.events.latitude_deg[indices],
            x_name=x_name,
            y_name=y_name,
            plan=plan,
            padding_fraction=padding_fraction,
        )

    def _point_coordinates(
        self,
        lon,
        lat,
        *,
        x_name: str,
        y_name: str,
        plan,
        padding_fraction: float = 0.0,
    ):
        x, y = self._transform_lonlat(lon, lat, x_name=x_name, y_name=y_name)
        xlim = sorted(plan.get_xlim())
        ylim = sorted(plan.get_ylim())
        padding_fraction = float(np.clip(float(padding_fraction), 0.0, 1.0))
        if padding_fraction:
            xpad = (xlim[1] - xlim[0]) * padding_fraction
            ypad = (ylim[1] - ylim[0]) * padding_fraction
            xlim = [xlim[0] - xpad, xlim[1] + xpad]
            ylim = [ylim[0] - ypad, ylim[1] + ypad]
        keep = (
            np.isfinite(x) & np.isfinite(y)
            & (x >= xlim[0]) & (x <= xlim[1])
            & (y >= ylim[0]) & (y <= ylim[1])
        )
        return x, y, keep

    def _transform_polygons(self, polygons_lonlat, *, x_name: str, y_name: str):
        lon = polygons_lonlat[:, :, 0]
        lat = polygons_lonlat[:, :, 1]
        x, y = self._transform_lonlat(lon, lat, x_name=x_name, y_name=y_name)
        return np.stack((x, y), axis=-1)

    def _transform_lonlat(self, lon, lat, *, x_name: str, y_name: str):
        if x_name == "longitude" and y_name == "latitude":
            return np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)
        east, north = latlon_to_local_km(
            lon,
            lat,
            self.project.reference_longitude,
            self.project.reference_latitude,
        )
        x = -east if x_name == "west" else east
        y = -north if y_name == "south" else north
        return x, y

    @staticmethod
    def _polygons_in_plan(polygons, plan):
        if polygons.size == 0:
            return np.zeros(0, dtype=bool)
        xlim = sorted(plan.get_xlim())
        ylim = sorted(plan.get_ylim())
        x = polygons[:, :, 0]
        y = polygons[:, :, 1]
        return (
            np.all(np.isfinite(polygons), axis=(1, 2))
            & (np.max(x, axis=1) >= xlim[0]) & (np.min(x, axis=1) <= xlim[1])
            & (np.max(y, axis=1) >= ylim[0]) & (np.min(y, axis=1) <= ylim[1])
        )

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
        return start, end

    @staticmethod
    def _plan_coordinate_names(metadata, plan):
        axis_order = tuple(metadata.get("axis_order") or ())
        coordinate_names = tuple(metadata.get("coordinate_names") or ())
        for axis, names in zip(axis_order, coordinate_names):
            if axis is plan and len(names) == 2:
                return str(names[0]), str(names[1])
        return None


__all__ = ["RenderSummary", "SatelliteOverlayRenderer", "configure_group_energy_time_axis"]
