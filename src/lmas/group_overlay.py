"""Reusable source-group overlay rendering for live and saved figures.

The interactive Source Selection window and the saved-figure renderer both use
stable source IDs.  This module supplies the export-side composition so a saved
figure reproduces visible Recolor/Halo/Outline/Hull group displays instead of
silently dropping them when the base projection is rebuilt.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from matplotlib.collections import PolyCollection

from .source_selection import (
    SourceSelectionGroup,
    charge_group_overlay_visible,
    effective_group_display_style,
    projection_hull_geometry,
    source_mask_in_linked_limits,
)


def _bounded_mask(mask: np.ndarray, maximum: int) -> np.ndarray:
    values = np.asarray(mask, dtype=bool)
    indices = np.flatnonzero(values)
    if indices.size <= maximum:
        return values
    positions = np.linspace(0, indices.size - 1, int(maximum), dtype=np.int64)
    result = np.zeros(values.shape, dtype=bool)
    result[indices[positions]] = True
    return result


def _view_mask(metadata: Mapping[str, Any], payload: Mapping[str, Any]) -> np.ndarray:
    ids = np.asarray(payload.get("source_ids", ()), dtype=np.int64)
    if ids.size == 0:
        return np.ones(ids.shape, dtype=bool)
    axes = tuple(metadata.get("axis_order", ()))
    names = tuple(metadata.get("coordinate_names", ()))
    pairs = tuple(payload.get("coordinate_pairs", ()))
    limits: dict[str, tuple[float, float]] = {}
    for axis, pair_names in zip(axes, names, strict=False):
        if len(pair_names) != 2:
            continue
        limits.setdefault(str(pair_names[0]), tuple(sorted(axis.get_xlim())))
        limits.setdefault(str(pair_names[1]), tuple(sorted(axis.get_ylim())))
    return source_mask_in_linked_limits(ids, names, pairs, limits)


def _center_kwargs(metadata: Mapping[str, Any], axis_index: int, mask: np.ndarray) -> dict[str, Any]:
    values = np.asarray(metadata.get("color_values", ()), dtype=float)
    scatters = tuple(metadata.get("scatters", ()))
    if values.shape != mask.shape:
        return {}
    kwargs: dict[str, Any] = {"c": values[mask], "norm": metadata.get("norm")}
    if axis_index < len(scatters):
        kwargs["cmap"] = scatters[axis_index].get_cmap()
    return kwargs


def _draw_subset(
    metadata: Mapping[str, Any],
    group: SourceSelectionGroup,
    mask: np.ndarray,
    *,
    active: bool,
) -> list[Any]:
    axes = tuple(metadata.get("axis_order", ()))
    pairs = tuple((metadata.get("selection_scopes") or {}).get("filtered", {}).get("coordinate_pairs", ()))
    scatters = tuple(metadata.get("scatters", ()))
    if not axes or len(pairs) != len(axes) or not np.any(mask):
        return []
    artists: list[Any] = []
    point_mask = _bounded_mask(mask, 50_000)
    color_by = str(metadata.get("color_by") or "").strip().lower()
    color_by_group = color_by == "group"
    style = effective_group_display_style(group.display_style, color_by=color_by)
    for axis_index, (axis, pair) in enumerate(zip(axes, pairs, strict=False)):
        x_all = np.asarray(pair[0], dtype=float)
        y_all = np.asarray(pair[1], dtype=float)
        if x_all.shape != point_mask.shape or y_all.shape != point_mask.shape:
            continue
        x = x_all[point_mask]
        y = y_all[point_mask]
        valid = np.isfinite(x) & np.isfinite(y)
        if not np.any(valid):
            continue
        x = x[valid]
        y = y[valid]
        base_z = float(scatters[axis_index].get_zorder()) if axis_index < len(scatters) else 1.0
        if color_by_group:
            fill = axis.scatter(
                x, y, s=10 if active else 7.5, facecolors=group.color,
                edgecolors="none", alpha=0.98 if active else 0.88,
                zorder=base_z + 0.20, clip_on=True,
            )
            artists.append(fill)
            if style == "recolor":
                continue
        if style == "halo":
            artists.append(axis.scatter(
                x, y, s=24 if active else 19, facecolors=group.color,
                edgecolors="none", alpha=0.34 if active else 0.24,
                zorder=base_z - 0.05, clip_on=True,
            ))
            center_mask = point_mask.copy()
            center_mask[point_mask] = valid
            artists.append(axis.scatter(
                x, y, s=6.5 if active else 5.0, edgecolors="none",
                alpha=1.0, zorder=base_z + 0.15, clip_on=True,
                **_center_kwargs(metadata, axis_index, center_mask),
            ))
        elif style == "recolor":
            artists.append(axis.scatter(
                x, y, s=11 if active else 8, facecolors=group.color,
                edgecolors="none", alpha=0.95 if active else 0.82,
                zorder=base_z + 0.20, clip_on=True,
            ))
        elif style == "outline":
            artists.append(axis.scatter(
                x, y, s=20 if active else 15, facecolors="none",
                edgecolors=group.color, linewidths=1.0 if active else 0.75,
                alpha=0.95 if active else 0.75, zorder=base_z + 0.25,
                clip_on=True,
            ))
        elif style in {"convex_hull", "concave_hull", "clustered_hulls"}:
            geometry = projection_hull_geometry(np.column_stack((x, y)), style)
            if geometry.faces:
                collection = PolyCollection(
                    geometry.faces, facecolors=group.color, edgecolors="none",
                    alpha=0.18 if active else 0.12, zorder=base_z - 0.08,
                    clip_on=True,
                )
                axis.add_collection(collection)
                artists.append(collection)
            for boundary in geometry.boundaries:
                if boundary.shape[0] < 2:
                    continue
                artists.append(axis.plot(
                    boundary[:, 0], boundary[:, 1], color=group.color,
                    linewidth=1.35 if active else 0.95,
                    alpha=0.95 if active else 0.72,
                    zorder=base_z + 0.18, clip_on=True,
                )[0])
    return artists


def _draw_ghost_subset(
    metadata: Mapping[str, Any],
    group: SourceSelectionGroup,
    payload: Mapping[str, Any],
    mask: np.ndarray,
    *,
    active: bool,
) -> list[Any]:
    axes = tuple(metadata.get("axis_order", ()))
    pairs = tuple(payload.get("coordinate_pairs", ()))
    scatters = tuple(metadata.get("scatters", ()))
    ghost_mask = _bounded_mask(mask, 25_000)
    if not axes or len(pairs) != len(axes) or not np.any(ghost_mask):
        return []
    artists: list[Any] = []
    for axis_index, (axis, pair) in enumerate(zip(axes, pairs, strict=False)):
        x_all = np.asarray(pair[0], dtype=float)
        y_all = np.asarray(pair[1], dtype=float)
        if x_all.shape != ghost_mask.shape or y_all.shape != ghost_mask.shape:
            continue
        x = x_all[ghost_mask]
        y = y_all[ghost_mask]
        valid = np.isfinite(x) & np.isfinite(y)
        if not np.any(valid):
            continue
        base_z = float(scatters[axis_index].get_zorder()) if axis_index < len(scatters) else 1.0
        artists.append(
            axis.scatter(
                x[valid], y[valid],
                s=14 if active else 11,
                facecolors="none",
                edgecolors=group.color,
                linewidths=0.65 if active else 0.5,
                alpha=0.42 if active else 0.30,
                zorder=base_z + 0.1,
                clip_on=True,
            )
        )
    return artists


def apply_saved_source_group_overlays(figure, selection_state: Mapping[str, Any] | None) -> tuple[Any, ...]:
    """Compose visible source-group overlays onto a rebuilt export figure.

    The exact interactive view limits must already have been restored before
    this function is called.
    """
    state = dict(selection_state or {})
    metadata = getattr(figure, "_lmas_metadata", {})
    if not isinstance(metadata, Mapping) or not metadata.get("linked_view"):
        return ()
    scopes = metadata.get("selection_scopes") or {}
    filtered = scopes.get("filtered") if isinstance(scopes, Mapping) else None
    all_payload = scopes.get("all") if isinstance(scopes, Mapping) else None
    if not isinstance(filtered, Mapping):
        return ()
    ids = np.asarray(filtered.get("source_ids", ()), dtype=np.int64)
    if ids.size == 0:
        return ()
    in_view = _view_mask(metadata, filtered)
    display_mode = str(state.get("member_display_scope") or "filtered")
    active_name = str(state.get("active_group") or state.get("active_name") or "")
    category_visibility = dict(state.get("category_visibility") or {})
    show_charge = bool(state.get("show_charge_overlays_with_other_color_modes", False))
    artists: list[Any] = []
    groups = [
        raw if isinstance(raw, SourceSelectionGroup) else SourceSelectionGroup.from_dict(raw)
        for raw in (state.get("groups") or ())
    ]
    active_domain = str(state.get("active_domain") or "").strip().lower()
    if active_domain not in {"custom", "charge"}:
        active_group = next((item for item in groups if item.name == active_name), None)
        active_domain = active_group.domain if active_group is not None else "custom"
    # Match the live renderer: the active group is composed last so overlap
    # precedence and hull/outline visibility are deterministic in exports.
    groups.sort(key=lambda item: item.name == active_name)
    for group in groups:
        if group.domain != active_domain:
            continue
        if not group.visible or group.display_style == "hidden" or not group.source_ids:
            continue
        if not category_visibility.get(group.charge_category, True):
            continue
        if not charge_group_overlay_visible(
            group.charge_category,
            color_by=metadata.get("color_by"),
            show_with_other_color_modes=show_charge,
        ):
            continue
        requested = np.fromiter(group.source_ids, dtype=np.int64)
        membership = np.isin(ids, requested, assume_unique=False) & in_view
        if display_mode != "filtered_out":
            artists.extend(_draw_subset(metadata, group, membership, active=group.name == active_name))
        if display_mode in {"all", "filtered_out"} and isinstance(all_payload, Mapping):
            all_ids = np.asarray(all_payload.get("source_ids", ()), dtype=np.int64)
            if all_ids.size:
                all_selected = np.isin(all_ids, requested, assume_unique=False)
                if display_mode == "filtered_out":
                    ghost = all_selected & ~np.isin(all_ids, ids, assume_unique=False)
                else:
                    normally_drawn = ids[membership]
                    ghost = all_selected & ~np.isin(all_ids, normally_drawn, assume_unique=False)
                artists.extend(
                    _draw_ghost_subset(
                        metadata,
                        group,
                        all_payload,
                        ghost,
                        active=group.name == active_name,
                    )
                )
    metadata = dict(metadata)
    metadata["saved_group_overlay_artists"] = tuple(artists)
    figure._lmas_metadata = metadata
    return tuple(artists)


__all__ = ["apply_saved_source_group_overlays"]
