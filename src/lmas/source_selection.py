"""Stable source-selection, charge-region-polarity, and hull primitives.

The module deliberately has no Qt dependency.  Source Selection, Charge
Analysis share these source-ID based groups,
serialization rules, undo history, and responsive projection-hull geometry.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from collections.abc import MutableMapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np
from matplotlib.path import Path as MplPath
from scipy.spatial import ConvexHull, Delaunay, QhullError, cKDTree

from . import __version__

SelectionOperation = Literal["replace", "add", "subtract", "intersect", "toggle"]
SelectionDisplayStyle = Literal[
    "halo",
    "recolor",
    "outline",
    "convex_hull",
    "concave_hull",
    "clustered_hulls",
    "hidden",
]
ChargeCategory = Literal["unassigned", "positive", "negative"]
SelectionDomain = Literal["custom", "charge"]

DISPLAY_STYLES: tuple[SelectionDisplayStyle, ...] = (
    "halo",
    "recolor",
    "outline",
    "convex_hull",
    "concave_hull",
    "clustered_hulls",
    "hidden",
)
SELECTION_DOMAINS: tuple[SelectionDomain, ...] = ("custom", "charge")

CHARGE_CATEGORIES: tuple[ChargeCategory, ...] = (
    "unassigned",
    "positive",
    "negative",
)
CHARGE_COLORS: dict[ChargeCategory, str] = {
    "unassigned": "#8a8a8a",
    "positive": "#d62728",
    "negative": "#0077ff",
}



CHARGE_REGION_LABELS: dict[str, str] = {
    "leader_polarity": "Leader polarity",
    "charge_region_polarity": "Charge region polarity",
}
DEFAULT_CHARGE_REGION_LABEL = "leader_polarity"


def charge_region_label(selection_state: Mapping[str, Any] | None) -> str:
    """Return the user-facing polarity label saved with a selection state.

    Older Projects do not contain a label preference and therefore migrate to
    the release default, ``Leader polarity``. Internal polarity/category schema
    identifiers remain unchanged.
    """

    payload = dict(selection_state or {})
    key = str(payload.get("charge_region_label") or DEFAULT_CHARGE_REGION_LABEL)
    return CHARGE_REGION_LABELS.get(key, CHARGE_REGION_LABELS[DEFAULT_CHARGE_REGION_LABEL])

CHARGE_NUMERIC_VALUES: dict[ChargeCategory, float] = {
    "negative": -1.0,
    "unassigned": 0.0,
    "positive": 1.0,
}


def effective_group_display_style(
    display_style: str,
    *,
    color_by: str | None,
) -> SelectionDisplayStyle:
    """Resolve a group overlay style against the active source color mode.

    ``Recolor`` is a true fill replacement only while the main plot is
    explicitly colored by Source group.  Charge coloring is already a
    categorical base-color mode derived from persisted polarity assignments, so
    selection overlays must not replace those red/blue/gray fills.  In Charge
    and all continuous color modes, Recolor therefore becomes an outline.  The
    stored group style is not changed.
    """

    style = _normalized_display_style(display_style)
    active_color = str(color_by or "time").strip().lower().replace("_", "-")
    if style == "recolor" and active_color != "group":
        return "outline"
    return style


def charge_group_overlay_visible(
    charge_category: str,
    *,
    color_by: str | None,
    show_with_other_color_modes: bool,
) -> bool:
    """Return whether a charge-assigned group overlay should be visible.

    Positive and Negative groups are already encoded in the source colors while
    the main figure uses ``Color by Charge``.  In other source-color modes their
    extra halos/outlines are hidden by default so the requested time, altitude,
    power, station-count, or chi-squared colors remain visually primary.  The
    user may explicitly enable those overlays from Charge Analysis.

    Unassigned groups remain visible because they may also be ordinary Source
    Selection groups rather than completed polarity assignments.
    """

    category = _normalized_charge_category(str(charge_category))
    active_color = str(color_by or "time").strip().lower().replace("_", "-")
    if active_color in {"charge", "group"} or category == "unassigned":
        return True
    return bool(show_with_other_color_modes)


def charge_values_for_source_ids(
    source_ids: Sequence[int] | np.ndarray,
    selection_state: Mapping[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return categorical charge values and a positive/negative conflict mask.

    The result uses ``-1`` for Negative, ``0`` for Unassigned, and ``+1`` for
    Positive. Opposite-polarity overlap is reported through the conflict mask,
    while the most recently modified charge group supplies the visible category.
    Group display colors do not alter the category colors used by
    ``Color by: Charge``.
    """

    ids = np.asarray(source_ids, dtype=np.int64)
    values = np.zeros(ids.shape, dtype=float)
    conflicts = np.zeros(ids.shape, dtype=bool)
    if ids.size == 0:
        return values, conflicts

    payload = dict(selection_state or {})
    raw_groups = payload.get("groups") or ()
    groups: list[tuple[int, SourceSelectionGroup]] = []
    for order, raw in enumerate(raw_groups):
        group = raw if isinstance(raw, SourceSelectionGroup) else SourceSelectionGroup.from_dict(raw)
        if group.domain == "charge" and group.charge_category in {"positive", "negative"}:
            groups.append((order, group))

    # Charge groups are editable scientific classifications.  A source may be
    # present in opposite-polarity groups in an older project or after a former
    # Add operation.  Preserve that conflict for audit, but display the most
    # recently modified assignment rather than silently reverting the source to
    # Unassigned gray.  Project order is the deterministic fallback for legacy
    # groups with identical or missing timestamps.
    groups.sort(key=lambda item: (str(item[1].modified_utc or ""), item[0]))
    assigned = np.zeros(ids.shape, dtype=bool)
    previous = np.zeros(ids.shape, dtype=float)
    for _order, group in groups:
        if not group.source_ids:
            continue
        mask = np.isin(
            ids,
            np.fromiter(group.source_ids, dtype=np.int64),
            assume_unique=False,
        )
        category_value = float(CHARGE_NUMERIC_VALUES[group.charge_category])
        conflicts |= mask & assigned & (previous != category_value)
        values[mask] = category_value
        previous[mask] = category_value
        assigned |= mask
    return values, conflicts


def refresh_charge_source_colors(
    figure: Any,
    selection_state: Mapping[str, Any] | None,
    *,
    draw: bool = True,
) -> bool:
    """Refresh an existing ``Color by Charge`` figure from persisted state.

    Source Selection overlays are transient editing aids.  The red/blue/gray
    charge colors are part of the underlying scientific figure and must remain
    correct when tabs change or the Source Selection window closes.  This
    helper updates the existing scatter arrays immediately while the ordinary
    full redraw is pending.

    Returns ``True`` when a charge-colored linked figure was updated.
    """

    if figure is None:
        return False
    metadata = getattr(figure, "_lmas_metadata", None)
    if not isinstance(metadata, Mapping):
        return False
    if str(metadata.get("color_by") or "").strip().lower() != "charge":
        return False

    source_ids = np.asarray(metadata.get("source_ids", ()), dtype=np.int64)
    scatters = tuple(metadata.get("scatters", ()))
    if source_ids.size == 0 or not scatters:
        return False

    values, _conflicts = charge_values_for_source_ids(source_ids, selection_state)
    raw_orders = tuple(metadata.get("scatter_orders", ()))
    updated = False
    for index, scatter in enumerate(scatters):
        order = (
            np.asarray(raw_orders[index], dtype=np.int64)
            if index < len(raw_orders)
            else np.arange(source_ids.size, dtype=np.int64)
        )
        try:
            offsets = np.asarray(scatter.get_offsets())
            expected = int(offsets.shape[0]) if offsets.ndim >= 1 else int(order.size)
        except Exception:
            expected = int(order.size)
        if order.ndim != 1 or order.size != expected:
            # Older figures did not persist their preview/depth ordering.  A
            # complete unthinned scatter can still be updated safely.
            if expected == source_ids.size:
                order = np.arange(source_ids.size, dtype=np.int64)
            else:
                continue
        if order.size and (np.min(order) < 0 or np.max(order) >= source_ids.size):
            continue
        try:
            scatter.set_array(np.asarray(values[order], dtype=float))
            updated = True
        except Exception:
            continue

    if not updated:
        return False
    # The linked-view controller and Source Selection window retain a direct
    # reference to the figure metadata dictionary created with the plot.  Do
    # not replace that dictionary here: doing so leaves those consumers holding
    # stale ``color_values`` and the next linked-view update can repaint newly
    # assigned Negative/Positive sources as Unassigned gray.  Mutate the shared
    # authoritative mapping in place whenever possible.
    if isinstance(metadata, MutableMapping):
        refreshed = metadata
        refreshed["color_values"] = np.asarray(values, dtype=float)
    else:
        # Defensive fallback for nonstandard externally supplied figures.
        refreshed = dict(metadata)
        refreshed["color_values"] = np.asarray(values, dtype=float)
        figure._lmas_metadata = refreshed
    callback = refreshed.get("colorbar_update_callback")
    if callable(callback):
        try:
            callback()
        except Exception:
            pass
    if draw:
        canvas = getattr(figure, "canvas", None)
        if canvas is not None:
            try:
                canvas.draw_idle()
            except Exception:
                pass
    return True


def group_values_for_source_ids(
    source_ids: Sequence[int] | np.ndarray,
    selection_state: Mapping[str, Any] | None,
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...], np.ndarray]:
    """Return categorical source-group values, colors, labels, and overlaps.

    Zero represents sources that do not belong to a visible group. Visible
    groups receive stable positive integer codes in project order. When a
    source belongs to more than one visible group, the active group wins;
    otherwise the later group in project order wins. This keeps custom group
    colors directly viewable while making overlap resolution deterministic.

    Group visibility and the ``Hidden`` display style are honored. Exact group
    membership is not changed by this display mapping.
    """

    ids = np.asarray(source_ids, dtype=np.int64)
    values = np.zeros(ids.shape, dtype=float)
    overlap_count = np.zeros(ids.shape, dtype=np.int16)
    payload = dict(selection_state or {})
    parsed: list[SourceSelectionGroup] = []
    for raw in payload.get("groups") or ():
        group = raw if isinstance(raw, SourceSelectionGroup) else SourceSelectionGroup.from_dict(raw)
        if not group.visible or group.display_style == "hidden" or not group.source_ids:
            continue
        parsed.append(group)

    active_name = str(payload.get("active_group") or payload.get("active_name") or "")
    if active_name:
        parsed.sort(key=lambda group: group.name == active_name)

    colors: list[str] = [CHARGE_COLORS["unassigned"]]
    labels: list[str] = ["Ungrouped"]
    for code, group in enumerate(parsed, start=1):
        colors.append(str(group.color))
        labels.append(str(group.name))
        membership = np.isin(
            ids,
            np.fromiter(group.source_ids, dtype=np.int64),
            assume_unique=False,
        )
        overlap_count[membership] += 1
        values[membership] = float(code)
    return values, tuple(colors), tuple(labels), overlap_count > 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _normalized_display_style(value: str | None) -> SelectionDisplayStyle:
    candidate = str(value or "recolor").strip().lower()
    aliases = {
        "convex": "convex_hull",
        "concave": "concave_hull",
        "clustered": "clustered_hulls",
        "clustered_hull": "clustered_hulls",
        "hull": "concave_hull",
    }
    candidate = aliases.get(candidate, candidate)
    if candidate not in DISPLAY_STYLES:
        return "recolor"
    return candidate  # type: ignore[return-value]


def _normalized_charge_category(value: str | None) -> ChargeCategory:
    candidate = str(value or "unassigned").strip().lower()
    if candidate not in CHARGE_CATEGORIES:
        return "unassigned"
    return candidate  # type: ignore[return-value]


def _normalized_domain(value: str | None, *, charge_category: str | None = None) -> SelectionDomain:
    candidate = str(value or "").strip().lower()
    aliases = {
        "selection": "custom",
        "source_selection": "custom",
        "polarity": "charge",
        "leader_analysis": "custom",
        "leader": "custom",
    }
    candidate = aliases.get(candidate, candidate)
    if candidate in SELECTION_DOMAINS:
        return candidate  # type: ignore[return-value]
    # Backward-compatible migration: historical groups with an assigned polarity
    # become Charge Analysis groups; every other legacy group stays Custom Selection.
    return "charge" if _normalized_charge_category(charge_category) != "unassigned" else "custom"


@dataclass(frozen=True)
class SourceSelectionGroup:
    """One named, project-persistable group of solved-source identities."""

    name: str
    source_ids: frozenset[int] = frozenset()
    visible: bool = True
    locked: bool = False
    color: str = CHARGE_COLORS["unassigned"]
    display_style: SelectionDisplayStyle = "recolor"
    charge_category: ChargeCategory = "unassigned"
    domain: SelectionDomain = "custom"
    subtype: str = "generic_selection"
    metadata: Mapping[str, Any] | None = None
    created_utc: str = ""
    modified_utc: str = ""
    created_with_lmas_version: str = ""

    def __post_init__(self) -> None:
        now = _utc_now()
        object.__setattr__(self, "name", str(self.name).strip() or "Selection")
        object.__setattr__(
            self, "source_ids", frozenset(int(value) for value in self.source_ids)
        )
        category = _normalized_charge_category(self.charge_category)
        color = str(self.color or CHARGE_COLORS["unassigned"])
        if category == "negative" and color.casefold() == "#1f77b4":
            color = CHARGE_COLORS["negative"]
        object.__setattr__(self, "color", color)
        object.__setattr__(
            self, "display_style", _normalized_display_style(self.display_style)
        )
        object.__setattr__(self, "charge_category", category)
        domain = _normalized_domain(self.domain, charge_category=category)
        subtype = str(self.subtype or "").strip()
        if not subtype:
            subtype = {"custom": "generic_selection", "charge": "polarity_group"}[domain]
        if domain == "charge" and subtype == "generic_selection":
            subtype = "polarity_group"
        elif domain == "custom" and subtype != "generic_selection":
            subtype = "generic_selection"
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "subtype", subtype)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))
        object.__setattr__(self, "created_utc", str(self.created_utc or now))
        object.__setattr__(self, "modified_utc", str(self.modified_utc or now))
        object.__setattr__(
            self,
            "created_with_lmas_version",
            str(self.created_with_lmas_version or __version__),
        )

    def with_source_ids(self, source_ids: Iterable[int]) -> "SourceSelectionGroup":
        return replace(
            self,
            source_ids=frozenset(int(value) for value in source_ids),
            modified_utc=_utc_now(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source_ids": sorted(int(value) for value in self.source_ids),
            "visible": bool(self.visible),
            "locked": bool(self.locked),
            "color": self.color,
            "display_style": self.display_style,
            "charge_category": self.charge_category,
            "domain": self.domain,
            "subtype": self.subtype,
            "metadata": dict(self.metadata or {}),
            "created_utc": self.created_utc,
            "modified_utc": self.modified_utc,
            "created_with_lmas_version": self.created_with_lmas_version,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any] | None) -> "SourceSelectionGroup":
        payload = dict(values or {})
        name = str(payload.get("name") or "Selection")
        domain_value = payload.get("domain")
        # Pre-domain Charge Analysis projects can contain an Unassigned group,
        # whose category alone is indistinguishable from a generic selection.
        # Preserve the standard historical Charge name during migration without
        # broadly reclassifying arbitrary unassigned custom groups.
        if (
            domain_value in (None, "")
            and _normalized_charge_category(payload.get("charge_category")) == "unassigned"
            and name.casefold().startswith("unassigned ")
        ):
            domain_value = "charge"
        return cls(
            name=name,
            source_ids=frozenset(int(value) for value in payload.get("source_ids") or ()),
            visible=bool(payload.get("visible", True)),
            locked=bool(payload.get("locked", False)),
            color=str(payload.get("color") or CHARGE_COLORS["unassigned"]),
            display_style=_normalized_display_style(payload.get("display_style")),
            charge_category=_normalized_charge_category(payload.get("charge_category")),
            domain=_normalized_domain(domain_value, charge_category=payload.get("charge_category")),
            subtype=str(payload.get("subtype") or ""),
            metadata=dict(payload.get("metadata") or {}),
            created_utc=str(payload.get("created_utc") or ""),
            modified_utc=str(payload.get("modified_utc") or ""),
            created_with_lmas_version=str(
                payload.get("created_with_lmas_version") or ""
            ),
        )


@dataclass(frozen=True)
class SelectionManagerSnapshot:
    groups: tuple[SourceSelectionGroup, ...]
    active_name: str | None


class SourceSelectionManager:
    """Manage named source groups and one-step-at-a-time undo history."""

    DEFAULT_COLORS = (
        CHARGE_COLORS["unassigned"],
        "#ff4fd8",
        "#39ff14",
        "#ffb000",
        "#00d5ff",
        "#b388ff",
        "#f5e663",
        "#8be9a8",
    )

    def __init__(self, *, history_limit: int = 100) -> None:
        self._groups: "OrderedDict[str, SourceSelectionGroup]" = OrderedDict()
        self.active_name: str | None = None
        self.history_limit = max(1, int(history_limit))
        self._history: list[SelectionManagerSnapshot] = []
        self.new_group("Selection 1", record_history=False)

    @property
    def groups(self) -> tuple[SourceSelectionGroup, ...]:
        return tuple(self._groups.values())

    @property
    def active_group(self) -> SourceSelectionGroup | None:
        if self.active_name is None:
            return None
        return self._groups.get(self.active_name)

    @property
    def can_undo(self) -> bool:
        return bool(self._history)

    def snapshot(self) -> SelectionManagerSnapshot:
        return SelectionManagerSnapshot(self.groups, self.active_name)

    def restore(self, snapshot: SelectionManagerSnapshot) -> None:
        self._groups = OrderedDict((group.name, group) for group in snapshot.groups)
        self.active_name = (
            snapshot.active_name if snapshot.active_name in self._groups else None
        )
        if self.active_name is None and self._groups:
            self.active_name = next(iter(self._groups))

    def load_groups(
        self,
        groups: Iterable[SourceSelectionGroup | Mapping[str, Any]],
        *,
        active_name: str | None = None,
        clear_history: bool = True,
    ) -> None:
        rebuilt: "OrderedDict[str, SourceSelectionGroup]" = OrderedDict()
        for item in groups:
            group = (
                item
                if isinstance(item, SourceSelectionGroup)
                else SourceSelectionGroup.from_dict(item)
            )
            name = group.name
            if name in rebuilt:
                base = name
                number = 2
                while f"{base} {number}" in rebuilt:
                    number += 1
                name = f"{base} {number}"
                group = replace(group, name=name)
            rebuilt[name] = group
        if not rebuilt:
            default = SourceSelectionGroup("Selection 1")
            rebuilt[default.name] = default
        self._groups = rebuilt
        self.active_name = (
            str(active_name) if active_name is not None and str(active_name) in rebuilt else next(iter(rebuilt))
        )
        if clear_history:
            self._history.clear()

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_group": self.active_name,
            "groups": [group.to_dict() for group in self.groups],
        }

    def _push_history(self) -> None:
        self._history.append(self.snapshot())
        if len(self._history) > self.history_limit:
            del self._history[: len(self._history) - self.history_limit]

    def undo(self) -> bool:
        if not self._history:
            return False
        self.restore(self._history.pop())
        return True

    def _unique_name(self, requested: str) -> str:
        base = str(requested).strip() or "Selection"
        if base not in self._groups:
            return base
        number = 2
        while f"{base} {number}" in self._groups:
            number += 1
        return f"{base} {number}"

    def new_group(
        self,
        name: str | None = None,
        *,
        source_ids: Iterable[int] = (),
        color: str | None = None,
        display_style: SelectionDisplayStyle = "recolor",
        charge_category: ChargeCategory = "unassigned",
        domain: SelectionDomain = "custom",
        subtype: str = "",
        metadata: Mapping[str, Any] | None = None,
        record_history: bool = True,
    ) -> SourceSelectionGroup:
        if record_history:
            self._push_history()
        requested = name or f"Selection {len(self._groups) + 1}"
        unique = self._unique_name(requested)
        category = _normalized_charge_category(charge_category)
        chosen_color = color or CHARGE_COLORS[category]
        group = SourceSelectionGroup(
            name=unique,
            source_ids=frozenset(int(value) for value in source_ids),
            color=str(chosen_color),
            display_style=_normalized_display_style(display_style),
            charge_category=category,
            domain=_normalized_domain(domain, charge_category=category),
            subtype=str(subtype or ""),
            metadata=dict(metadata or {}),
        )
        self._groups[unique] = group
        self.active_name = unique
        return group

    def set_active(self, name: str) -> bool:
        value = str(name)
        if value not in self._groups:
            return False
        self.active_name = value
        return True

    def delete_group(self, name: str | None = None) -> bool:
        target = str(name or self.active_name or "")
        if target not in self._groups:
            return False
        self._push_history()
        keys = list(self._groups)
        index = keys.index(target)
        del self._groups[target]
        if not self._groups:
            self.new_group("Selection 1", record_history=False)
        else:
            remaining = list(self._groups)
            self.active_name = remaining[min(index, len(remaining) - 1)]
        return True

    def rename_group(self, new_name: str, name: str | None = None) -> str | None:
        target = str(name or self.active_name or "")
        group = self._groups.get(target)
        if group is None:
            return None
        requested = str(new_name).strip()
        if not requested:
            return None
        if requested != target and requested in self._groups:
            requested = self._unique_name(requested)
        if requested == target:
            return target
        self._push_history()
        rebuilt: "OrderedDict[str, SourceSelectionGroup]" = OrderedDict()
        for key, value in self._groups.items():
            if key == target:
                rebuilt[requested] = replace(
                    value, name=requested, modified_utc=_utc_now()
                )
            else:
                rebuilt[key] = value
        self._groups = rebuilt
        if self.active_name == target:
            self.active_name = requested
        return requested

    def _group(self, name: str | None = None) -> SourceSelectionGroup | None:
        return self._groups.get(str(name or self.active_name or ""))

    def _replace_active(self, group: SourceSelectionGroup) -> None:
        self._groups[group.name] = group

    def _set_group_value(self, group: SourceSelectionGroup, **changes: Any) -> bool:
        updated = replace(group, modified_utc=_utc_now(), **changes)
        if updated == group:
            return False
        self._push_history()
        self._replace_active(updated)
        return True

    def set_visible(self, visible: bool, name: str | None = None) -> bool:
        group = self._group(name)
        if group is None or group.visible == bool(visible):
            return False
        return self._set_group_value(group, visible=bool(visible))

    def set_locked(self, locked: bool, name: str | None = None) -> bool:
        group = self._group(name)
        if group is None or group.locked == bool(locked):
            return False
        return self._set_group_value(group, locked=bool(locked))

    def set_color(self, color: str, name: str | None = None) -> bool:
        group = self._group(name)
        value = str(color)
        if group is None or group.color == value:
            return False
        return self._set_group_value(group, color=value)

    def set_domain(
        self,
        domain: SelectionDomain | str,
        *,
        subtype: str | None = None,
        name: str | None = None,
    ) -> bool:
        group = self._groups.get(str(name or self.active_name or ""))
        if group is None:
            return False
        value = _normalized_domain(str(domain), charge_category=group.charge_category)
        changes: dict[str, Any] = {"domain": value}
        if subtype is not None:
            changes["subtype"] = str(subtype)
        elif value == "charge" and group.subtype == "generic_selection":
            changes["subtype"] = "polarity_group"
        return self._set_group_value(group, **changes)

    def groups_for_domain(self, domain: SelectionDomain | str) -> tuple[SourceSelectionGroup, ...]:
        value = _normalized_domain(str(domain))
        return tuple(group for group in self.groups if group.domain == value)

    def set_display_style(
        self, style: SelectionDisplayStyle | str, name: str | None = None
    ) -> bool:
        group = self._group(name)
        value = _normalized_display_style(str(style))
        if group is None or group.display_style == value:
            return False
        return self._set_group_value(group, display_style=value)

    def set_charge_category(
        self,
        category: ChargeCategory | str,
        name: str | None = None,
        *,
        apply_default_color: bool = True,
    ) -> bool:
        group = self._group(name)
        value = _normalized_charge_category(str(category))
        if group is None:
            return False
        changes: dict[str, Any] = {"charge_category": value}
        if apply_default_color:
            changes["color"] = CHARGE_COLORS[value]
        if all(getattr(group, key) == item for key, item in changes.items()):
            return False
        return self._set_group_value(group, **changes)

    def apply_charge_assignment(
        self,
        source_ids: Iterable[int],
        operation: SelectionOperation = "replace",
        *,
        name: str | None = None,
    ) -> bool:
        """Apply an edit to a Charge Analysis group as one exclusive assignment.

        Adding or replacing members in Positive, Negative, or Unassigned removes
        those source IDs from charge groups with a different category.  This
        matches the user-facing meaning of polarity assignment and prevents a
        newly corrected source from remaining in an opposite-polarity group and
        appearing gray as a hidden conflict.  Remove and Intersect only edit the
        active group.  The entire change occupies one undo step.
        """

        group = self._group(name)
        if group is None or group.locked:
            return False
        if group.domain != "charge":
            return self.apply(source_ids, operation, name=name)

        incoming = frozenset(int(value) for value in source_ids)
        current = group.source_ids
        if operation == "replace":
            updated = incoming
        elif operation == "add":
            updated = current | incoming
        elif operation == "subtract":
            updated = current - incoming
        elif operation == "intersect":
            updated = current & incoming
        elif operation == "toggle":
            updated = current ^ incoming
        else:
            raise ValueError(f"Unknown selection operation: {operation!r}")

        replacements: dict[str, SourceSelectionGroup] = {}
        if updated != current:
            replacements[group.name] = group.with_source_ids(updated)

        # Assignment-producing operations make the active category authoritative
        # for every member retained in the active group.  Same-category named
        # groups may overlap; different categories may not.
        if operation not in {"subtract", "intersect"} and updated:
            assigned_ids = updated
            for candidate in self.groups:
                if candidate.name == group.name or candidate.domain != "charge":
                    continue
                if candidate.charge_category == group.charge_category:
                    continue
                cleaned = candidate.source_ids - assigned_ids
                if cleaned != candidate.source_ids:
                    replacements[candidate.name] = candidate.with_source_ids(cleaned)

        if not replacements:
            return False
        self._push_history()
        for group_name, replacement in replacements.items():
            self._groups[group_name] = replacement
        return True

    def apply(
        self,
        source_ids: Iterable[int],
        operation: SelectionOperation = "replace",
        *,
        name: str | None = None,
    ) -> bool:
        group = self._group(name)
        if group is None or group.locked:
            return False
        incoming = frozenset(int(value) for value in source_ids)
        current = group.source_ids
        if operation == "replace":
            updated = incoming
        elif operation == "add":
            updated = current | incoming
        elif operation == "subtract":
            updated = current - incoming
        elif operation == "intersect":
            updated = current & incoming
        elif operation == "toggle":
            updated = current ^ incoming
        else:
            raise ValueError(f"Unknown selection operation: {operation!r}")
        if updated == current:
            return False
        self._push_history()
        self._replace_active(group.with_source_ids(updated))
        return True

    def clear(self, name: str | None = None) -> bool:
        return self.apply((), "replace", name=name)

    def invert(self, universe: Iterable[int], name: str | None = None) -> bool:
        group = self._group(name)
        if group is None or group.locked:
            return False
        available = frozenset(int(value) for value in universe)
        updated = available - group.source_ids
        if updated == group.source_ids:
            return False
        self._push_history()
        self._replace_active(group.with_source_ids(updated))
        return True

    def counts(self, universe: Iterable[int], name: str | None = None) -> tuple[int, int]:
        """Return selected count and currently available selected count."""

        group = self._group(name)
        if group is None:
            return 0, 0
        available = frozenset(int(value) for value in universe)
        return len(group.source_ids), len(group.source_ids & available)

    def overlapping_source_ids(
        self,
        name: str | None = None,
        *,
        assigned_only: bool = False,
    ) -> frozenset[int]:
        group = self._group(name)
        if group is None or not group.source_ids:
            return frozenset()
        others: set[int] = set()
        for candidate in self.groups:
            if candidate.name == group.name:
                continue
            if assigned_only and candidate.charge_category == "unassigned":
                continue
            others.update(candidate.source_ids)
        return frozenset(group.source_ids & others)

    def category_counts(self) -> dict[ChargeCategory, int]:
        result: dict[ChargeCategory, int] = {
            "unassigned": 0,
            "positive": 0,
            "negative": 0,
        }
        for group in self.groups:
            result[group.charge_category] += len(group.source_ids)
        return result



def source_mask_in_linked_limits(
    source_ids: Sequence[int] | np.ndarray,
    coordinate_names: Sequence[Sequence[str]],
    coordinate_pairs: Sequence[Sequence[Sequence[float] | np.ndarray]],
    limits_by_name: Mapping[str, Sequence[float]],
) -> np.ndarray:
    """Return sources lying inside every named linked-view limit.

    Each physical coordinate is evaluated once even when it appears in several
    projections. Arrays with incompatible shapes are ignored rather than being
    guessed, which keeps selection identity authoritative.
    """

    ids = np.asarray(source_ids, dtype=np.int64)
    mask = np.ones(ids.shape, dtype=bool)
    if ids.size == 0:
        return mask
    values_by_name: dict[str, np.ndarray] = {}
    for names, pair in zip(coordinate_names, coordinate_pairs, strict=False):
        for name, raw in zip(names, pair, strict=False):
            key = str(name)
            if key in values_by_name:
                continue
            values = np.asarray(raw, dtype=float)
            if values.shape == ids.shape:
                values_by_name[key] = values
    for raw_name, raw_bounds in limits_by_name.items():
        values = values_by_name.get(str(raw_name))
        if values is None or len(raw_bounds) != 2:
            continue
        low, high = sorted((float(raw_bounds[0]), float(raw_bounds[1])))
        mask &= np.isfinite(values) & (values >= low) & (values <= high)
    return mask

def source_ids_inside_polygon(
    x_values: Sequence[float] | np.ndarray,
    y_values: Sequence[float] | np.ndarray,
    source_ids: Sequence[int] | np.ndarray,
    vertices: Sequence[Sequence[float]],
) -> np.ndarray:
    """Return stable source IDs inside a data-coordinate lasso polygon."""

    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    ids = np.asarray(source_ids, dtype=np.int64)
    if x.shape != y.shape or x.shape != ids.shape:
        raise ValueError("Lasso coordinate and source-ID arrays must have equal shapes")
    if len(vertices) < 3 or not x.size:
        return np.array([], dtype=np.int64)
    finite = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite):
        return np.array([], dtype=np.int64)
    points = np.column_stack((x[finite], y[finite]))
    inside = MplPath(np.asarray(vertices, dtype=float), closed=True).contains_points(
        points, radius=1e-12
    )
    return np.ascontiguousarray(ids[finite][inside], dtype=np.int64)


def selection_bounds(
    source_ids: Iterable[int],
    values: Mapping[str, Sequence[float] | np.ndarray],
    all_source_ids: Sequence[int] | np.ndarray,
) -> dict[str, tuple[float, float]]:
    """Return finite min/max bounds for currently available selected sources."""

    ids = np.asarray(all_source_ids, dtype=np.int64)
    requested = np.asarray(tuple(int(value) for value in source_ids), dtype=np.int64)
    if not ids.size or not requested.size:
        return {}
    mask = np.isin(ids, requested, assume_unique=False)
    result: dict[str, tuple[float, float]] = {}
    for name, raw in values.items():
        array = np.asarray(raw)
        if array.shape != ids.shape:
            continue
        if np.issubdtype(array.dtype, np.datetime64):
            continue
        numeric = np.asarray(array, dtype=float)[mask]
        finite = numeric[np.isfinite(numeric)]
        if finite.size:
            result[str(name)] = (float(np.min(finite)), float(np.max(finite)))
    return result


@dataclass(frozen=True)
class HullGeometry:
    """Projection hull geometry ready for Matplotlib collections and lines."""

    faces: tuple[np.ndarray, ...] = ()
    boundaries: tuple[np.ndarray, ...] = ()
    input_count: int = 0
    geometry_count: int = 0
    method: str = ""

    @property
    def empty(self) -> bool:
        return not self.faces and not self.boundaries


def _finite_unique_points(points: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("Hull points must have shape (N, 2)")
    values = values[np.isfinite(values).all(axis=1)]
    if not values.size:
        return np.empty((0, 2), dtype=float)
    _, indices = np.unique(np.round(values, 12), axis=0, return_index=True)
    return np.ascontiguousarray(values[np.sort(indices)], dtype=float)


def _normalize_points(points: np.ndarray) -> np.ndarray:
    if not points.size:
        return points.copy()
    low = np.min(points, axis=0)
    span = np.ptp(points, axis=0)
    span[~np.isfinite(span) | (span <= 0.0)] = 1.0
    return (points - low) / span


def reduce_points_for_hull(
    points: Sequence[Sequence[float]] | np.ndarray,
    *,
    max_points: int = 2500,
) -> np.ndarray:
    """Deterministically reduce hull geometry work without changing membership.

    One boundary-favoring representative is retained per normalized grid cell,
    plus coordinate extrema.  Selection membership remains source-ID exact; only
    the display envelope is reduced.
    """

    values = _finite_unique_points(points)
    limit = max(64, int(max_points))
    if values.shape[0] <= limit:
        return values
    normalized = _normalize_points(values)
    side = max(8, int(np.floor(np.sqrt(limit))))
    bins = np.minimum((normalized * side).astype(np.int64), side - 1)
    keys = bins[:, 0] * side + bins[:, 1]
    center = np.array([0.5, 0.5])
    radius = np.sum((normalized - center) ** 2, axis=1)
    order = np.lexsort((-radius, keys))
    ordered_keys = keys[order]
    first = np.r_[True, ordered_keys[1:] != ordered_keys[:-1]]
    chosen = order[first]
    extrema = np.array(
        [
            np.argmin(values[:, 0]),
            np.argmax(values[:, 0]),
            np.argmin(values[:, 1]),
            np.argmax(values[:, 1]),
        ],
        dtype=np.int64,
    )
    chosen = np.unique(np.r_[chosen, extrema])
    if chosen.size > limit:
        chosen = chosen[:limit]
    return np.ascontiguousarray(values[np.sort(chosen)], dtype=float)


def _closed_polygon(points: np.ndarray) -> np.ndarray:
    if not points.size:
        return points
    if np.allclose(points[0], points[-1], equal_nan=False):
        return points
    return np.vstack((points, points[0]))


def convex_hull_geometry(
    points: Sequence[Sequence[float]] | np.ndarray,
    *,
    max_points: int = 10000,
) -> HullGeometry:
    raw = _finite_unique_points(points)
    reduced = reduce_points_for_hull(raw, max_points=max_points)
    if reduced.shape[0] < 3:
        return HullGeometry(input_count=raw.shape[0], geometry_count=reduced.shape[0], method="convex")
    try:
        hull = ConvexHull(_normalize_points(reduced))
    except QhullError:
        return HullGeometry(input_count=raw.shape[0], geometry_count=reduced.shape[0], method="convex")
    polygon = np.ascontiguousarray(reduced[hull.vertices], dtype=float)
    return HullGeometry(
        faces=(polygon,),
        boundaries=(_closed_polygon(polygon),),
        input_count=raw.shape[0],
        geometry_count=reduced.shape[0],
        method="convex",
    )


def _triangle_circumradii(points: np.ndarray, simplices: np.ndarray) -> np.ndarray:
    triangles = points[simplices]
    a = np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1)
    b = np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1)
    c = np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1)
    cross = np.abs(
        (triangles[:, 1, 0] - triangles[:, 0, 0])
        * (triangles[:, 2, 1] - triangles[:, 0, 1])
        - (triangles[:, 1, 1] - triangles[:, 0, 1])
        * (triangles[:, 2, 0] - triangles[:, 0, 0])
    )
    area = 0.5 * cross
    radii = np.full(simplices.shape[0], np.inf, dtype=float)
    valid = area > 1.0e-14
    radii[valid] = a[valid] * b[valid] * c[valid] / (4.0 * area[valid])
    return radii


def _boundary_loops(edges: Iterable[tuple[int, int]], points: np.ndarray) -> tuple[np.ndarray, ...]:
    adjacency: dict[int, set[int]] = defaultdict(set)
    unused: set[tuple[int, int]] = set()
    for first, second in edges:
        a, b = sorted((int(first), int(second)))
        if a == b:
            continue
        adjacency[a].add(b)
        adjacency[b].add(a)
        unused.add((a, b))
    loops: list[np.ndarray] = []
    while unused:
        start_edge = next(iter(unused))
        start, current = start_edge
        path = [start, current]
        unused.discard(start_edge)
        previous = start
        guard = 0
        while guard < len(adjacency) + len(unused) + 4:
            guard += 1
            candidates = [
                neighbor
                for neighbor in adjacency[current]
                if tuple(sorted((current, neighbor))) in unused
            ]
            if not candidates:
                break
            # Prefer not to reverse; then choose the smallest turning angle for a
            # stable, visually smooth boundary walk.
            viable = [candidate for candidate in candidates if candidate != previous] or candidates
            if len(viable) == 1:
                nxt = viable[0]
            else:
                incoming = points[current] - points[previous]
                angles = []
                for candidate in viable:
                    outgoing = points[candidate] - points[current]
                    cross = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
                    dot = float(np.dot(incoming, outgoing))
                    angles.append(np.arctan2(cross, dot) % (2.0 * np.pi))
                nxt = viable[int(np.argmin(angles))]
            unused.discard(tuple(sorted((current, nxt))))
            previous, current = current, nxt
            path.append(current)
            if current == start:
                break
        if len(path) >= 3:
            loop = points[np.asarray(path, dtype=np.int64)]
            loops.append(_closed_polygon(np.ascontiguousarray(loop, dtype=float)))
    return tuple(loops)


def concave_hull_geometry(
    points: Sequence[Sequence[float]] | np.ndarray,
    *,
    max_points: int = 2500,
    alpha_scale: float = 3.5,
) -> HullGeometry:
    raw = _finite_unique_points(points)
    reduced = reduce_points_for_hull(raw, max_points=max_points)
    if reduced.shape[0] < 4:
        return convex_hull_geometry(reduced, max_points=max_points)
    normalized = _normalize_points(reduced)
    try:
        triangulation = Delaunay(normalized)
    except QhullError:
        return convex_hull_geometry(reduced, max_points=max_points)
    simplices = np.asarray(triangulation.simplices, dtype=np.int64)
    radii = _triangle_circumradii(normalized, simplices)
    try:
        distances, _ = cKDTree(normalized).query(normalized, k=2)
        nearest = np.asarray(distances[:, 1], dtype=float)
        nearest = nearest[np.isfinite(nearest) & (nearest > 0.0)]
    except Exception:
        nearest = np.array([], dtype=float)
    if nearest.size:
        local_scale = float(np.percentile(nearest, 75.0))
    else:
        local_scale = 0.05
    finite_radii = radii[np.isfinite(radii)]
    if not finite_radii.size:
        return convex_hull_geometry(reduced, max_points=max_points)
    threshold = max(
        local_scale * max(1.5, float(alpha_scale)),
        float(np.percentile(finite_radii, 35.0)),
    )
    threshold = min(threshold, float(np.percentile(finite_radii, 82.0)))
    keep = np.isfinite(radii) & (radii <= threshold)
    # Avoid a misleadingly sparse shape.  Relax once, then use convex fallback.
    if np.count_nonzero(keep) < max(1, simplices.shape[0] // 12):
        threshold = float(np.percentile(finite_radii, 65.0))
        keep = np.isfinite(radii) & (radii <= threshold)
    kept = simplices[keep]
    if not kept.size:
        return convex_hull_geometry(reduced, max_points=max_points)
    edge_counts: dict[tuple[int, int], int] = defaultdict(int)
    for triangle in kept:
        for first, second in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            edge_counts[tuple(sorted((int(first), int(second))))] += 1
    boundary_edges = [edge for edge, count in edge_counts.items() if count == 1]
    faces = tuple(np.ascontiguousarray(reduced[triangle], dtype=float) for triangle in kept)
    boundaries = _boundary_loops(boundary_edges, reduced)
    if not boundaries:
        fallback = convex_hull_geometry(reduced, max_points=max_points)
        boundaries = fallback.boundaries
    return HullGeometry(
        faces=faces,
        boundaries=boundaries,
        input_count=raw.shape[0],
        geometry_count=reduced.shape[0],
        method="concave",
    )


def _neighbor_components(points: np.ndarray) -> list[np.ndarray]:
    count = points.shape[0]
    if count < 3:
        return []
    normalized = _normalize_points(points)
    tree = cKDTree(normalized)
    k = min(6, count)
    distances, neighbors = tree.query(normalized, k=k)
    if k == 1:
        return []
    nearest = np.asarray(distances[:, 1], dtype=float)
    finite_nearest = nearest[np.isfinite(nearest) & (nearest > 0.0)]
    if not finite_nearest.size:
        return [np.arange(count, dtype=np.int64)]
    radius = max(
        float(np.percentile(finite_nearest, 90.0)) * 2.75,
        float(np.median(finite_nearest)) * 4.0,
        1.0e-6,
    )
    radius = min(radius, 0.18)
    parent = np.arange(count, dtype=np.int64)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def union(first: int, second: int) -> None:
        root_a = find(first)
        root_b = find(second)
        if root_a != root_b:
            parent[root_b] = root_a

    for index in range(count):
        row_distances = np.atleast_1d(distances[index])
        row_neighbors = np.atleast_1d(neighbors[index])
        for distance, neighbor in zip(row_distances[1:], row_neighbors[1:]):
            if np.isfinite(distance) and float(distance) <= radius:
                union(index, int(neighbor))
    buckets: dict[int, list[int]] = defaultdict(list)
    for index in range(count):
        buckets[find(index)].append(index)
    components = [
        np.asarray(indices, dtype=np.int64)
        for indices in buckets.values()
        if len(indices) >= 3
    ]
    components.sort(key=lambda values: values.size, reverse=True)
    return components


def clustered_hulls_geometry(
    points: Sequence[Sequence[float]] | np.ndarray,
    *,
    max_points: int = 3000,
) -> HullGeometry:
    raw = _finite_unique_points(points)
    reduced = reduce_points_for_hull(raw, max_points=max_points)
    if reduced.shape[0] < 3:
        return HullGeometry(input_count=raw.shape[0], geometry_count=reduced.shape[0], method="clustered")
    components = _neighbor_components(reduced)
    if not components:
        return convex_hull_geometry(reduced, max_points=max_points)
    faces: list[np.ndarray] = []
    boundaries: list[np.ndarray] = []
    for indices in components:
        cluster = reduced[indices]
        geometry = concave_hull_geometry(
            cluster,
            max_points=max(128, min(max_points, cluster.shape[0])),
            alpha_scale=3.25,
        )
        faces.extend(geometry.faces)
        boundaries.extend(geometry.boundaries)
    if not faces and not boundaries:
        return convex_hull_geometry(reduced, max_points=max_points)
    return HullGeometry(
        faces=tuple(faces),
        boundaries=tuple(boundaries),
        input_count=raw.shape[0],
        geometry_count=reduced.shape[0],
        method="clustered",
    )


def projection_hull_geometry(
    points: Sequence[Sequence[float]] | np.ndarray,
    style: SelectionDisplayStyle | str,
    *,
    max_points: int | None = None,
) -> HullGeometry:
    value = _normalized_display_style(str(style))
    if value == "convex_hull":
        return convex_hull_geometry(points, max_points=max_points or 10000)
    if value == "concave_hull":
        return concave_hull_geometry(points, max_points=max_points or 2500)
    if value == "clustered_hulls":
        return clustered_hulls_geometry(points, max_points=max_points or 3000)
    return HullGeometry(method=value)


__all__ = [
    "CHARGE_CATEGORIES",
    "CHARGE_COLORS",
    "CHARGE_NUMERIC_VALUES",
    "CHARGE_REGION_LABELS",
    "DEFAULT_CHARGE_REGION_LABEL",
    "DISPLAY_STYLES",
    "ChargeCategory",
    "HullGeometry",
    "SelectionDisplayStyle",
    "SelectionManagerSnapshot",
    "SelectionOperation",
    "SourceSelectionGroup",
    "SourceSelectionManager",
    "SelectionDomain",
    "SELECTION_DOMAINS",
    "charge_region_label",
    "charge_group_overlay_visible",
    "charge_values_for_source_ids",
    "refresh_charge_source_colors",
    "group_values_for_source_ids",
    "clustered_hulls_geometry",
    "concave_hull_geometry",
    "convex_hull_geometry",
    "projection_hull_geometry",
    "reduce_points_for_hull",
    "selection_bounds",
    "source_mask_in_linked_limits",
    "source_ids_inside_polygon",
]
