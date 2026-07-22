from __future__ import annotations

from collections.abc import Callable
import copy
import time
from typing import Any

import numpy as np
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure

from .plotting.spatial_aspect import enforce_true_spatial_scale


class LinkedViewController:
    """Coordinate linked source selections across LMAS projection panels.

    Every committed axis-limit change updates a persistent set of named
    coordinate constraints. Rectangle zoom, toolbar pan/drag, wheel zoom, and
    edited limit fields therefore share one exact scientific subset. Cross-panel
    view-limit propagation is controlled by Auto-fit spatial panels.
    """

    def __init__(
        self,
        figure: Figure,
        *,
        toolbar=None,
        debounce_ms: int = 80,
        state_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.figure = figure
        self.toolbar = toolbar
        self.metadata = getattr(figure, "_lmas_metadata", None)
        self._state_callback = state_callback
        self.enabled = bool(
            isinstance(self.metadata, dict)
            and self.metadata.get("linked_view")
            and self.metadata.get("axis_order")
        )
        self._updating = False
        self._pending_axis = None
        self._pending_explicit = False
        # Saved Project Home bounds are an initial display window, not a
        # permanent cross-dimensional scientific selection.  The first
        # explicit user zoom or pan releases untouched startup constraints so
        # sources outside the opening view can appear immediately.
        self._soft_startup_view = False
        # Full saved Project bounds are retained separately from the named
        # constraints that drive ordinary linked navigation.  This lets the
        # opening frame reproduce the exact saved box while subsequent wheel,
        # zoom, and pan gestures behave exactly like a manually established
        # view instead of dropping to a full-record/Home state.
        self._startup_display_constraints: dict[str, tuple[float, float]] = {}
        self._timer = None
        self._release_timer = None
        self._release_kind = ""
        self._release_axis = None
        self._wheel_timer = None
        self._wheel_axis = None
        self._active_mask = np.array([], dtype=bool)
        self._base_membership_mask = np.array([], dtype=bool)
        self._constraints: dict[str, tuple[float, float]] = {}
        self._coordinate_values: dict[str, np.ndarray] = {}
        self._initial_limits: list[tuple[tuple[float, float], tuple[float, float]]] = []
        self._initial_positions = []
        self._initial_box_aspects: list[float | None] = []
        self._axis_data_aspects: tuple[float | None, ...] = ()
        self._locked_box_axes: set[int] = set()
        self._equal_scale_axes: set[int] = set()
        self._spatial_axes: set[int] = set()
        self._altitude_axes: set[int] = set()
        self._connections: list[int] = []
        self._axis_connections: list[tuple[object, int]] = []
        self._history_states: dict[int, dict[str, Any]] = {}
        self._history_suspended = False
        self._auto_fit_spatial = True
        self._remap_colormap = True
        self._mouse_down = False
        self._press_mode = ""
        self._press_axis = None
        self._explicit_release_event_id: int | None = None
        self._fast_pan_active = False
        self._fast_pan_proxy_active = False
        self._fast_pan_saved_displayed_count: int | None = None
        if not self.enabled:
            return

        meta = self.metadata
        axes = tuple(meta["axis_order"])
        self._auto_fit_spatial = bool(meta.get("auto_fit_spatial", True))
        self._remap_colormap = bool(
            meta.get("remap_colormap", meta.get("remap_time_colors", True))
        )
        self._initial_limits = [(axis.get_xlim(), axis.get_ylim()) for axis in axes]
        for axis in axes:
            axis.apply_aspect()
            self._initial_positions.append(axis.get_position().frozen())
            box_aspect = axis.get_box_aspect()
            self._initial_box_aspects.append(
                None if box_aspect is None else float(box_aspect)
            )
        raw_data_aspects = tuple(meta.get("axis_data_aspects", ()))
        self._axis_data_aspects = tuple(
            None if index >= len(raw_data_aspects) or raw_data_aspects[index] is None
            else float(raw_data_aspects[index])
            for index in range(len(axes))
        )
        self._locked_box_axes = {
            int(index) for index in meta.get("locked_box_axes", range(len(axes)))
        }
        self._equal_scale_axes = {
            int(index) for index in meta.get("equal_scale_axes", ())
        }
        self._spatial_axes = {
            int(index) for index in meta.get("spatial_axes", range(1, len(axes)))
        }
        self._altitude_axes = {
            int(index) for index in meta.get("altitude_axes", ())
        }
        count = np.asarray(meta["color_values"]).size
        self._active_mask = np.ones(count, dtype=bool)
        self._base_membership_mask = np.ones(count, dtype=bool)
        meta["visible_mask"] = self._active_mask.copy()

        for names, pair in zip(
            tuple(meta.get("coordinate_names", ())),
            tuple(meta.get("coordinate_pairs", ())),
        ):
            if len(names) != 2:
                continue
            for name, values in zip(names, pair):
                self._coordinate_values.setdefault(
                    str(name), np.ascontiguousarray(values, dtype=float)
                )

        self._timer = figure.canvas.new_timer(interval=int(debounce_ms))
        self._timer.single_shot = True
        self._timer.add_callback(self.update_now)
        # NavigationToolbar2 may apply a rectangle zoom in its own release
        # callback after LMAS receives the same event.  Commit on the next
        # backend event-loop turn so the final toolbar limits are authoritative.
        self._release_timer = figure.canvas.new_timer(interval=1)
        self._release_timer.single_shot = True
        self._release_timer.add_callback(self._commit_release)
        self._wheel_timer = figure.canvas.new_timer(interval=140)
        self._wheel_timer.single_shot = True
        self._wheel_timer.add_callback(self._commit_wheel_zoom)
        for axis in axes:
            self._axis_connections.append(
                (axis, axis.callbacks.connect("xlim_changed", self._schedule))
            )
            self._axis_connections.append(
                (axis, axis.callbacks.connect("ylim_changed", self._schedule))
            )
        self._connections.extend(
            [
                figure.canvas.mpl_connect("button_press_event", self._on_button_press),
                figure.canvas.mpl_connect("button_release_event", self._on_button_release),
                figure.canvas.mpl_connect("scroll_event", self._on_scroll),
            ]
        )
        if self.toolbar is not None and hasattr(self.toolbar, "set_linked_controller"):
            self.toolbar.set_linked_controller(self)
            nav_stack = getattr(self.toolbar, "_nav_stack", None)
            if nav_stack is not None and nav_stack() is None:
                self.toolbar.push_current()
        self._restore_axis_geometry()
        self._enforce_true_spatial_scale()
        self._refresh_map_underlay()
        self._notify_state()

    @property
    def visible_count(self) -> int:
        """Number of sources in the persistent linked-selection mask."""
        return int(np.count_nonzero(self._active_mask))

    @staticmethod
    def _mask_for_named_limits(
        coordinate_values: dict[str, np.ndarray],
        limits: dict[str, tuple[float, float]],
    ) -> np.ndarray:
        if not coordinate_values:
            return np.array([], dtype=bool)
        size = len(next(iter(coordinate_values.values())))
        mask = np.ones(size, dtype=bool)
        for name, bounds in limits.items():
            values = coordinate_values.get(str(name))
            if values is None:
                continue
            low, high = sorted((float(bounds[0]), float(bounds[1])))
            values = np.asarray(values, dtype=float)
            if values.size != size:
                continue
            mask &= np.isfinite(values) & (values >= low) & (values <= high)
        return mask

    def _current_view_counts(self) -> tuple[int, int]:
        """Return quality-filtered and unfiltered counts in the visible view."""
        limits = self.current_interactive_limits()
        filtered_values = dict(self.metadata.get("filtered_coordinate_values", {}))
        filtered_in_view = self._mask_for_named_limits(filtered_values, limits)
        if filtered_in_view.size == self._active_mask.size:
            filtered_in_view &= self._active_mask
            visible = int(np.count_nonzero(filtered_in_view))
        else:
            visible = self.visible_count

        unfiltered_values = dict(self.metadata.get("unfiltered_coordinate_values", {}))
        unfiltered_in_view = self._mask_for_named_limits(unfiltered_values, limits)
        in_view = (
            int(np.count_nonzero(unfiltered_in_view))
            if unfiltered_in_view.size
            else int(self.metadata.get("loaded_count", visible))
        )
        return visible, in_view

    def _history_element_key(self) -> int | None:
        nav_stack = getattr(self.toolbar, "_nav_stack", None) if self.toolbar is not None else None
        element = nav_stack() if nav_stack is not None else None
        return None if element is None else id(element)

    def _store_current_history_state(self) -> None:
        if self._history_suspended or not self.enabled:
            return
        key = self._history_element_key()
        if key is not None:
            state = self.capture_view_state()
            if state is not None:
                self._history_states[key] = state

    def on_toolbar_history_pushed(self) -> None:
        """Called by the LMAS toolbar whenever Matplotlib pushes a view."""
        self._store_current_history_state()

    def before_toolbar_history_restore(self) -> None:
        if not self.enabled:
            return
        self._updating = True
        if self._timer is not None:
            self._timer.stop()
        if self._release_timer is not None:
            self._release_timer.stop()
        if self._wheel_timer is not None:
            self._wheel_timer.stop()

    def after_toolbar_history_restore(self) -> None:
        """Restore the subset paired with Matplotlib's current nav-stack entry."""
        if not self.enabled:
            return
        try:
            key = self._history_element_key()
            state = self._history_states.get(key) if key is not None else None
            if state is not None:
                self.restore_view_state(state, exact_membership=True, record_history=False)
            else:
                callback = self.metadata.get("time_axis_callback")
                if callable(callback):
                    callback()
                self._restore_axis_geometry()
                self._enforce_true_spatial_scale()
                self._refresh_map_underlay()
                self.figure.canvas.draw_idle()
                self._notify_state()
        finally:
            self._updating = False

    def capture_session_history(self) -> dict[str, Any] | None:
        """Export the full toolbar-linked view/subset history for a redraw."""
        if not self.enabled:
            return None
        nav_stack = getattr(self.toolbar, "_nav_stack", None) if self.toolbar is not None else None
        if nav_stack is None or not getattr(nav_stack, "_elements", None):
            current = self.capture_view_state()
            return {"states": [current] if current is not None else [], "position": 0}
        states: list[dict[str, Any]] = []
        for element in nav_stack._elements:
            state = self._history_states.get(id(element))
            if state is None:
                state = self.capture_view_state()
            states.append(copy.deepcopy(state))
        return {
            "states": states,
            "position": int(getattr(nav_stack, "_pos", len(states) - 1)),
        }

    def restore_session_history(self, bundle: dict[str, Any] | None) -> bool:
        """Rebuild Matplotlib and LMAS history after an ordinary GUI redraw."""
        if not self.enabled or not bundle:
            return False
        states = [state for state in bundle.get("states", ()) if state]
        if not states:
            return False
        position = min(max(int(bundle.get("position", len(states) - 1)), 0), len(states) - 1)
        nav_stack = getattr(self.toolbar, "_nav_stack", None) if self.toolbar is not None else None
        if nav_stack is None:
            return self.restore_view_state(states[position], exact_membership=False)

        self._history_suspended = True
        try:
            nav_stack.clear()
            self._history_states.clear()
            for state in states:
                self.restore_view_state(
                    state,
                    exact_membership=False,
                    record_history=False,
                    notify=False,
                )
                self.toolbar.push_current()
                key = self._history_element_key()
                if key is not None:
                    self._history_states[key] = copy.deepcopy(state)
            nav_stack._pos = position
            self.toolbar.set_history_buttons()
            self.toolbar._update_view()
            target = states[position]
            self.restore_view_state(
                target,
                exact_membership=False,
                record_history=False,
                notify=True,
            )
        finally:
            self._history_suspended = False
            self._pending_axis = None
            self._pending_explicit = False
            if self._timer is not None:
                self._timer.stop()
            if self._release_timer is not None:
                self._release_timer.stop()
        self._store_current_history_state()
        return True

    def set_behavior(
        self,
        *,
        auto_fit_spatial: bool | None = None,
        remap_time_colors: bool | None = None,
    ) -> None:
        if not self.enabled:
            return
        if auto_fit_spatial is not None:
            self._auto_fit_spatial = bool(auto_fit_spatial)
            self.metadata["auto_fit_spatial"] = self._auto_fit_spatial
        if remap_time_colors is not None:
            self._remap_colormap = bool(remap_time_colors)
            # Keep the historical key for project/profile compatibility while
            # applying the behavior to every supported color quantity.
            self.metadata["remap_time_colors"] = self._remap_colormap
            self.metadata["remap_colormap"] = self._remap_colormap

    def refresh_display(
        self,
        *,
        preview_point_limit: int | None = None,
        update_subset: bool = False,
        notify: bool = False,
        redraw: bool = True,
    ) -> None:
        """Refresh existing artists without rebuilding the Matplotlib figure."""
        if not self.enabled:
            return
        if preview_point_limit is not None:
            self.metadata["preview_point_limit"] = max(0, int(preview_point_limit))
            update_subset = True
        if update_subset:
            norm = self.metadata.get("norm") or self._norm_for_mask(self._active_mask)
            self._set_scatter_subset(self._active_mask, norm)
        if notify:
            self._notify_state()
        if redraw:
            self.figure.canvas.draw_idle()

    def begin_fast_pan(self, *, point_limit: int = 3_500) -> bool:
        """Temporarily render a small, stable proxy population during panning."""
        if not self.enabled or self._fast_pan_active:
            return False
        self._fast_pan_active = True
        self._fast_pan_proxy_active = False
        self._fast_pan_saved_displayed_count = int(
            self.metadata.get("displayed_count", self.visible_count)
        )
        normal_limit = int(self.metadata.get("preview_point_limit", 0) or 0)
        current_count = int(self.metadata.get("displayed_count", self.visible_count))
        effective_limit = max(250, int(point_limit))
        if normal_limit > 0:
            effective_limit = min(effective_limit, normal_limit)
        if current_count <= effective_limit:
            self.metadata["fast_pan_active"] = True
            return False

        self.metadata["preview_point_limit"] = effective_limit
        try:
            norm = self.metadata.get("norm") or self._norm_for_mask(self._active_mask)
            self._set_scatter_subset(self._active_mask, norm)
        finally:
            self.metadata["preview_point_limit"] = normal_limit
        self.metadata["fast_pan_active"] = True
        self._fast_pan_proxy_active = True
        self.figure.canvas.draw_idle()
        return True

    def end_fast_pan(
        self,
        *,
        restore_artists: bool = True,
        redraw: bool = True,
    ) -> bool:
        """Restore the ordinary source population after a pan gesture."""
        if not self._fast_pan_active:
            return False
        restore = self._fast_pan_proxy_active
        self._fast_pan_active = False
        self._fast_pan_proxy_active = False
        self.metadata["fast_pan_active"] = False
        if restore and restore_artists and self.enabled:
            norm = self.metadata.get("norm") or self._norm_for_mask(self._active_mask)
            self._set_scatter_subset(self._active_mask, norm)
        elif self._fast_pan_saved_displayed_count is not None:
            self.metadata["displayed_count"] = int(
                self._fast_pan_saved_displayed_count
            )
        self._fast_pan_saved_displayed_count = None
        if restore and restore_artists and redraw:
            self.figure.canvas.draw_idle()
        return restore

    def close(self) -> None:
        """Disconnect timers and callbacks before a figure is replaced."""
        self.end_fast_pan(restore_artists=False, redraw=False)
        if not self.enabled and not self._connections and not self._axis_connections:
            return
        for timer_name in ("_timer", "_release_timer", "_wheel_timer"):
            timer = getattr(self, timer_name, None)
            if timer is None:
                continue
            try:
                timer.stop()
            except Exception:
                pass
            callbacks = getattr(timer, "callbacks", None)
            if isinstance(callbacks, list):
                callbacks.clear()
            setattr(self, timer_name, None)
        canvas = getattr(self.figure, "canvas", None)
        if canvas is not None:
            for connection in self._connections:
                try:
                    canvas.mpl_disconnect(connection)
                except Exception:
                    pass
        for axis, connection in self._axis_connections:
            try:
                axis.callbacks.disconnect(connection)
            except Exception:
                pass
        self._connections.clear()
        self._axis_connections.clear()
        self._history_states.clear()
        if self.toolbar is not None and hasattr(self.toolbar, "set_linked_controller"):
            try:
                self.toolbar.set_linked_controller(None)
            except Exception:
                pass
        self._state_callback = None
        self._fast_pan_active = False
        self._fast_pan_proxy_active = False
        self._fast_pan_saved_displayed_count = None
        self.enabled = False

    def _toolbar_mode(self) -> str:
        toolbar = self.toolbar
        if toolbar is None:
            manager = getattr(self.figure.canvas, "manager", None)
            toolbar = getattr(manager, "toolbar", None)
        mode = getattr(toolbar, "mode", "") if toolbar is not None else ""
        return str(mode or "").lower()

    @staticmethod
    def _changed(current: tuple[float, float], original: tuple[float, float]) -> bool:
        return not np.allclose(current, original, rtol=0.0, atol=1.0e-12)

    @staticmethod
    def _inside(values: np.ndarray, limits: tuple[float, float]) -> np.ndarray:
        low, high = sorted((float(limits[0]), float(limits[1])))
        values = np.asarray(values, dtype=float)
        return np.isfinite(values) & (values >= low) & (values <= high)

    @staticmethod
    def _oriented_limits(
        bounds: tuple[float, float], current: tuple[float, float]
    ) -> tuple[float, float]:
        """Return sorted bounds using the current axis direction."""
        low, high = sorted((float(bounds[0]), float(bounds[1])))
        return (high, low) if float(current[0]) > float(current[1]) else (low, high)

    @staticmethod
    def _translate_named_limits(
        limits: dict[str, tuple[float, float]],
        available: set[str],
    ) -> dict[str, tuple[float, float]]:
        """Translate viewpoint coordinate names without changing the view.

        East/West and North/South are sign-reversed representations of the same
        physical coordinates.  This lets a viewpoint redraw preserve exact
        horizontal, altitude, and time limits instead of auto-fitting them.
        """
        translated: dict[str, tuple[float, float]] = {}
        opposites = {"east": "west", "west": "east", "north": "south", "south": "north"}
        for name, bounds in limits.items():
            key = str(name)
            low, high = sorted((float(bounds[0]), float(bounds[1])))
            if key in available:
                translated[key] = (low, high)
                continue
            target = opposites.get(key)
            if target in available:
                translated[target] = (-high, -low)
        return translated

    def _normalization(self, values: np.ndarray) -> Normalize:
        """Build a selected-point normalization matching the full norm type."""
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        full_norm = self._full_norm()
        if isinstance(full_norm, LogNorm):
            finite = finite[finite > 0]
        if finite.size == 0:
            raise ValueError("Cannot normalize an empty source selection")
        low, high = float(np.min(finite)), float(np.max(finite))
        if high <= low:
            if isinstance(full_norm, LogNorm):
                low = max(low * 0.9, np.nextafter(0.0, 1.0))
                high = max(high * 1.1, np.nextafter(low, np.inf))
            else:
                pad = max(0.5, abs(low) * 1.0e-9, 1.0e-12)
                low -= pad
                high += pad
        if isinstance(full_norm, LogNorm):
            return LogNorm(vmin=low, vmax=high)
        return Normalize(vmin=low, vmax=high)

    def _norm_for_mask(self, mask: np.ndarray) -> Normalize:
        if not np.any(np.asarray(mask, dtype=bool)):
            return self._full_norm()
        if self._remap_colormap:
            return self._normalization(
                np.asarray(self.metadata["color_values"], dtype=float)[mask]
            )
        return self._full_norm()

    def _full_norm(self) -> Normalize:
        full_norm = self.metadata.get("full_norm")
        if full_norm is not None:
            return full_norm
        low, high = self.metadata["full_color_limits"]
        return Normalize(vmin=float(low), vmax=float(high))

    @staticmethod
    def _padded_limits(
        values: np.ndarray,
        original: tuple[float, float],
        *,
        fraction: float,
        minimum: float,
    ) -> tuple[float, float]:
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return original
        low, high = float(np.min(finite)), float(np.max(finite))
        span = high - low
        pad = max(float(minimum), span * float(fraction))
        if span <= 0:
            pad = max(float(minimum), abs(low) * float(fraction), 1.0e-12)
        low -= pad
        high += pad
        bound_low, bound_high = sorted((float(original[0]), float(original[1])))
        low, high = max(bound_low, low), min(bound_high, high)
        if high <= low:
            return original
        return (high, low) if original[0] > original[1] else (low, high)

    @staticmethod
    def _expanded_interval(
        limits: tuple[float, float],
        target_span: float,
        original: tuple[float, float],
    ) -> tuple[float, float]:
        # True physical aspect may legitimately require blank space outside the
        # original data envelope.  Preserve orientation and centre, but do not
        # clamp the padded display interval back to the initial source bounds.
        reverse = limits[0] > limits[1]
        low, high = sorted((float(limits[0]), float(limits[1])))
        target = max(float(target_span), high - low)
        centre = 0.5 * (low + high)
        new_low, new_high = centre - 0.5 * target, centre + 0.5 * target
        return (new_high, new_low) if reverse else (new_low, new_high)

    def _equal_scale_limits(
        self,
        index: int,
        x_limits: tuple[float, float],
        y_limits: tuple[float, float],
        original: tuple[tuple[float, float], tuple[float, float]],
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        box_aspect = self._initial_box_aspects[index]
        if box_aspect is None or not np.isfinite(box_aspect) or box_aspect <= 0:
            position = self._initial_positions[index]
            box_aspect = float(position.height / position.width)
        x_span = abs(float(x_limits[1]) - float(x_limits[0]))
        y_span = abs(float(y_limits[1]) - float(y_limits[0]))
        if x_span <= 0 or y_span <= 0:
            return x_limits, y_limits
        data_aspect = (
            self._axis_data_aspects[index]
            if index < len(self._axis_data_aspects) else None
        )
        if data_aspect is None or not np.isfinite(data_aspect) or data_aspect <= 0:
            data_aspect = 1.0
        required_y = x_span * box_aspect / data_aspect
        if y_span < required_y:
            y_limits = self._expanded_interval(y_limits, required_y, original[1])
        else:
            required_x = y_span * data_aspect / box_aspect
            x_limits = self._expanded_interval(x_limits, required_x, original[0])
        return x_limits, y_limits

    def _restore_axis_geometry(self) -> None:
        for index, axis in enumerate(self.metadata["axis_order"]):
            if index not in self._locked_box_axes:
                continue
            axis.set_position(self._initial_positions[index], which="both")
            box_aspect = self._initial_box_aspects[index]
            data_aspect = (
                self._axis_data_aspects[index]
                if index < len(self._axis_data_aspects) else None
            )
            if box_aspect is not None:
                axis.set_box_aspect(box_aspect)
            # True Aspect is represented by padded limits.  Never let a data
            # aspect setting resize the carefully designed LMAS axes boxes.
            axis.set_aspect("auto")

    def _enforce_true_spatial_scale(self, driver_axis=None) -> None:
        """Apply True Aspect after authoritative linked limits are committed.

        The legacy Portrait geometry has shallow vertical panels.  Once a user
        explicitly zooms a spatial panel, that panel must drive the common
        kilometres-per-inch scale; otherwise the untouched full-altitude range
        expands the plan view back out and makes further zooming appear blocked.
        Other layouts retain the established expand-only behavior.
        """
        if not self._equal_scale_axes:
            return
        axes = tuple(self.metadata.get("axis_order", ()))
        coordinate_names = tuple(self.metadata.get("coordinate_names", ()))
        driver_index = None
        if (
            bool(self.metadata.get("true_aspect_driver_authority", False))
            and driver_axis in axes
            and axes.index(driver_axis) in self._equal_scale_axes
        ):
            driver_index = axes.index(driver_axis)
        coordinate_centres = None
        if driver_index is not None:
            coordinate_centres = {}
            active = np.asarray(self._active_mask, dtype=bool)
            for name, values in self._coordinate_values.items():
                array = np.asarray(values, dtype=float)
                if array.shape != active.shape:
                    continue
                finite = array[active & np.isfinite(array)]
                if finite.size:
                    coordinate_centres[str(name)] = 0.5 * (
                        float(np.min(finite)) + float(np.max(finite))
                    )
        enforce_true_spatial_scale(
            self.figure,
            axes,
            coordinate_names,
            axis_indices=self._equal_scale_axes,
            reference_latitude=float(self.metadata.get("reference_latitude", 0.0)),
            driver_axis_index=driver_index,
            coordinate_centres=coordinate_centres,
            emit=False,
        )

    def _refresh_map_underlay(self) -> None:
        callback = self.metadata.get("map_update_callback")
        if callable(callback):
            callback()
        underlay = self.metadata.get("map_underlay")
        if underlay is not None:
            self.metadata["map_status"] = getattr(underlay, "status", "Map underlay enabled")

    def _schedule(self, axis=None) -> None:
        if self._updating or not self.enabled or self._mouse_down:
            return
        if axis in self.metadata["axis_order"]:
            self._pending_axis = axis
            self._pending_explicit = False
        if self._timer is not None:
            self._timer.stop()
            self._timer.start()

    def _on_button_press(self, event) -> None:
        if self._updating or not self.enabled:
            return
        self._mouse_down = True
        self._press_mode = self._toolbar_mode()
        self._press_axis = event.inaxes
        if self._timer is not None:
            self._timer.stop()

    def _on_button_release(self, event) -> None:
        if self._updating or not self.enabled:
            return
        # LMASNavigationToolbar calls ``after_toolbar_gesture`` after Matplotlib
        # has finalized the axis limits.  Ignore the matching generic canvas
        # release callback so one gesture cannot be committed twice.
        if self._explicit_release_event_id == id(event):
            self._explicit_release_event_id = None
            self._mouse_down = False
            self._press_mode = ""
            self._press_axis = None
            return
        mode = self._press_mode or self._toolbar_mode()
        driver_axis = (
            self._press_axis
            if self._press_axis in self.metadata["axis_order"]
            else event.inaxes
        )
        self._mouse_down = False
        self._press_mode = ""
        self._press_axis = None

        # Plain analysis clicks, Precision cursor placement, and lasso gestures
        # must not create linked-view history entries.  Only an active toolbar
        # zoom or pan gesture commits scientific view limits here.
        mode_text = str(mode).casefold()
        if "zoom" not in mode_text and "pan" not in mode_text:
            return
        if driver_axis not in self.metadata["axis_order"]:
            return
        if self._timer is not None:
            self._timer.stop()
        self._release_axis = driver_axis
        self._release_kind = "pan" if "pan" in mode else "selection"
        if self._release_timer is not None:
            self._release_timer.stop()
            self._release_timer.start()
        else:
            self._commit_release()

    def after_toolbar_gesture(self, kind: str, event) -> None:
        """Commit the final limits produced by the LMAS Matplotlib toolbar.

        Qt/Matplotlib callback ordering can otherwise let LMAS inspect an axis
        one event-loop turn before ``release_zoom`` has installed its final
        limits.  That race was most visible as a vertical-panel altitude zoom
        that changed only the driver panel.  The toolbar now invokes this method
        explicitly after its superclass completes, making the shared scientific
        altitude constraint authoritative in the same committed gesture.
        """
        is_pan = str(kind).casefold() == "pan"
        if self._updating or not self.enabled:
            if is_pan:
                self.end_fast_pan(redraw=False)
            return
        self._explicit_release_event_id = id(event)
        axes = tuple(self.metadata.get("axis_order", ()))
        axis = self._press_axis if self._press_axis in axes else getattr(event, "inaxes", None)
        self._mouse_down = False
        self._press_mode = ""
        self._press_axis = None
        if self._timer is not None:
            self._timer.stop()
        if self._release_timer is not None:
            self._release_timer.stop()
        self._release_axis = None
        self._release_kind = ""
        if axis not in axes:
            if is_pan:
                self.end_fast_pan(redraw=True)
            return
        if is_pan:
            # update_now() installs the exact final subset, so avoid rebuilding
            # the ordinary population once immediately beforehand.
            self.end_fast_pan(restore_artists=False, redraw=False)
        # Both rectangle zoom and toolbar pan/drag are committed scientific
        # view changes.  The final driver-axis limits become named constraints,
        # including the shared altitude range when present in the panel.
        self._pending_axis = axis
        self._pending_explicit = True
        self.update_now(axis, record_history=False)
        # NavigationToolbar2 has already pushed the finalized axis limits.
        # Replace that entry's provisional subset with the exact linked subset.
        self._store_current_history_state()

    def _commit_release(self) -> None:
        """Commit navigation after Matplotlib's toolbar has finalized limits."""
        if self._updating or not self.enabled:
            return
        axis = self._release_axis
        kind = self._release_kind
        self._release_axis = None
        self._release_kind = ""
        is_pan = str(kind).casefold() == "pan"
        if axis not in self.metadata["axis_order"]:
            if is_pan:
                self.end_fast_pan(redraw=True)
            return
        if is_pan:
            self.end_fast_pan(restore_artists=False, redraw=False)
        # Generic backends reach this delayed path rather than the explicit Qt
        # toolbar callback.  Pan/drag and rectangle zoom therefore share the
        # same scientific commit path and history semantics.
        self._pending_axis = axis
        self._pending_explicit = True
        self.update_now(axis)

    def _all_axes_at_home(self) -> bool:
        return all(
            not self._changed(axis.get_xlim(), original[0])
            and not self._changed(axis.get_ylim(), original[1])
            for axis, original in zip(self.metadata["axis_order"], self._initial_limits)
        )

    def _axis_index(self, axis) -> int:
        return tuple(self.metadata["axis_order"]).index(axis)

    def _axis_coordinates(self, axis) -> tuple[np.ndarray, np.ndarray]:
        return self.metadata["coordinate_pairs"][self._axis_index(axis)]

    def _axis_coordinate_names(self, axis) -> tuple[str, str]:
        names = self.metadata["coordinate_names"][self._axis_index(axis)]
        return str(names[0]), str(names[1])

    def _update_constraints_from_axis(self, axis) -> None:
        x_name, y_name = self._axis_coordinate_names(axis)
        self._constraints[x_name] = tuple(sorted(float(value) for value in axis.get_xlim()))
        self._constraints[y_name] = tuple(sorted(float(value) for value in axis.get_ylim()))

    def _mask_for_constraints(
        self,
        constraints: dict[str, tuple[float, float]],
    ) -> np.ndarray:
        if self._base_membership_mask.shape == self._active_mask.shape:
            mask = self._base_membership_mask.copy()
        else:
            mask = np.ones(self._active_mask.shape, dtype=bool)
        for name, bounds in constraints.items():
            values = self._coordinate_values.get(name)
            if values is not None:
                mask &= self._inside(values, bounds)
        return mask

    def _mask_from_constraints(self) -> np.ndarray:
        return self._mask_for_constraints(self._constraints)

    def _startup_navigation_constraints(
        self,
        normalized: dict[str, tuple[float, float]],
    ) -> dict[str, tuple[float, float]]:
        """Choose the named constraints that should survive Project startup.

        Saved Projects persist the displayed limits of every panel, including
        spatial ranges that were auto-fitted rather than deliberately selected.
        Reinstalling every displayed range as a permanent scientific constraint
        makes a reopened Project behave differently from an identical view
        reached manually.  Prefer the time-altitude panel as the canonical
        startup driver and treat the remaining saved ranges as display-only.
        """

        names_by_axis = tuple(self.metadata.get("coordinate_names", ()))
        preferred = None
        for names in names_by_axis:
            pair = tuple(str(name) for name in names)
            if {"time", "altitude"}.issubset(pair):
                preferred = pair
                break
        if preferred is None:
            for names in names_by_axis:
                pair = tuple(str(name) for name in names)
                if any(name in normalized for name in pair):
                    preferred = pair
                    break
        if preferred is None:
            return dict(normalized)
        selected = {
            name: normalized[name]
            for name in preferred
            if name in normalized
        }
        return selected or dict(normalized)

    def _selection_from_axis(self, axis) -> np.ndarray:
        # Public/testing helper: update only the coordinates represented by this
        # panel, retaining constraints established by earlier linked selections.
        self._update_constraints_from_axis(axis)
        return self._mask_from_constraints()

    def _preview_indices(self, mask: np.ndarray) -> np.ndarray:
        """Return deterministic display indices for the exact scientific mask."""
        scientific = np.flatnonzero(np.asarray(mask, dtype=bool))
        limit = int(self.metadata.get("preview_point_limit", 0) or 0)
        if limit <= 0 or scientific.size <= limit:
            return scientific
        preview_order = np.asarray(
            self.metadata.get("preview_order", ()), dtype=np.int64
        )
        if preview_order.shape == mask.shape:
            ordered = preview_order[np.asarray(mask, dtype=bool)[preview_order]]
        else:
            time_values = self._coordinate_values.get("time")
            if time_values is None or time_values.size != mask.size:
                ordered = scientific
            else:
                ordered = scientific[
                    np.argsort(np.asarray(time_values)[scientific], kind="stable")
                ]
        positions = np.linspace(0, ordered.size - 1, limit, dtype=np.int64)
        return np.asarray(ordered[positions], dtype=np.int64)

    @staticmethod
    def _zoomed_limits(
        current: tuple[float, float],
        centre: float,
        scale: float,
        original: tuple[float, float],
    ) -> tuple[float, float]:
        reverse = float(current[0]) > float(current[1])
        low, high = sorted((float(current[0]), float(current[1])))
        bound_low, bound_high = sorted((float(original[0]), float(original[1])))
        centre = min(max(float(centre), low), high)
        new_low = centre - (centre - low) * float(scale)
        new_high = centre + (high - centre) * float(scale)
        span = new_high - new_low
        available = bound_high - bound_low
        if span >= available:
            new_low, new_high = bound_low, bound_high
        else:
            if new_low < bound_low:
                new_high += bound_low - new_low
                new_low = bound_low
            if new_high > bound_high:
                new_low -= new_high - bound_high
                new_high = bound_high
        result = (new_low, new_high)
        return (result[1], result[0]) if reverse else result

    def _on_scroll(self, event) -> None:
        """Debounced cursor-centred wheel zoom for the hovered projection."""
        if self._updating or not self.enabled:
            return
        axis = event.inaxes
        axes = tuple(self.metadata.get("axis_order", ()))
        if axis not in axes:
            return
        step = float(getattr(event, "step", 0.0) or 0.0)
        if step == 0.0:
            button = str(getattr(event, "button", "")).lower()
            step = 1.0 if button == "up" else -1.0 if button == "down" else 0.0
        if step == 0.0:
            return
        # One notch zooms by about 18 percent.  Multiple notches compound but
        # are committed as one linked scientific-view/history update.
        scale = 0.82 ** step
        key = str(getattr(event, "key", "") or "").lower()
        tokens = {item.strip() for item in key.replace("ctrl", "control").split("+") if item.strip()}
        horizontal_only = "shift" in tokens and "control" not in tokens
        vertical_only = "control" in tokens and "shift" not in tokens
        index = axes.index(axis)
        original_x, original_y = self._initial_limits[index]
        x_limits, y_limits = axis.get_xlim(), axis.get_ylim()
        x_centre = float(event.xdata) if event.xdata is not None else 0.5 * sum(x_limits)
        y_centre = float(event.ydata) if event.ydata is not None else 0.5 * sum(y_limits)
        if not vertical_only:
            x_limits = self._zoomed_limits(x_limits, x_centre, scale, original_x)
            axis.set_xlim(x_limits, emit=False)
        if not horizontal_only:
            y_limits = self._zoomed_limits(y_limits, y_centre, scale, original_y)
            axis.set_ylim(y_limits, emit=False)
        self._wheel_axis = axis
        if self._wheel_timer is not None:
            self._wheel_timer.stop()
            self._wheel_timer.start()
        else:
            self._commit_wheel_zoom()

    def _commit_wheel_zoom(self) -> None:
        axis = self._wheel_axis
        self._wheel_axis = None
        if axis not in self.metadata.get("axis_order", ()):
            return
        self._pending_axis = axis
        self._pending_explicit = True
        self.update_now(axis, record_history=False)
        if self.toolbar is not None:
            self.toolbar.push_current()
        else:
            self._store_current_history_state()

    def _set_scatter_subset(self, mask: np.ndarray, norm: Normalize) -> None:
        started = time.perf_counter()
        color_values = np.asarray(self.metadata["color_values"], dtype=float)
        base_indices = self._preview_indices(mask)
        self.metadata["displayed_count"] = int(base_indices.size)
        depth_keys = tuple(self.metadata.get("scatter_depth_keys", ()))
        for index, (scatter, pair) in enumerate(
            zip(self.metadata["scatters"], self.metadata["coordinate_pairs"])
        ):
            x_values, y_values = (np.asarray(values, dtype=float) for values in pair)
            indices = base_indices
            if index < len(depth_keys) and depth_keys[index] is not None and base_indices.size:
                key = np.asarray(depth_keys[index], dtype=float)
                indices = base_indices[np.argsort(key[base_indices], kind="stable")]
            offsets = np.empty((indices.size, 2), dtype=float)
            offsets[:, 0] = x_values[indices]
            offsets[:, 1] = y_values[indices]
            scatter.set_offsets(offsets)
            scatter.set_array(color_values[indices])
            scatter.set_norm(norm)
        self.metadata["last_artist_update_s"] = time.perf_counter() - started

    def _apply_authoritative_constraints_to_axes(self) -> None:
        """Propagate named scientific constraints when linked auto-fit is enabled.

        The exact scientific mask always follows the driver's named limits.
        With Auto-fit spatial panels disabled, however, non-driver panels keep
        their independent view limits and only their displayed source artists
        are updated.
        """

        if not self._auto_fit_spatial:
            return

        axes = tuple(self.metadata.get("axis_order", ()))
        names_by_axis = tuple(self.metadata.get("coordinate_names", ()))
        for axis, names in zip(axes, names_by_axis):
            if len(names) != 2:
                continue
            x_name, y_name = str(names[0]), str(names[1])
            if x_name in self._constraints:
                axis.set_xlim(
                    self._oriented_limits(self._constraints[x_name], axis.get_xlim()),
                    emit=False,
                )
            if y_name in self._constraints:
                axis.set_ylim(
                    self._oriented_limits(self._constraints[y_name], axis.get_ylim()),
                    emit=False,
                )

    def _synchronize_axes(self, mask: np.ndarray, driver_axis) -> None:
        """Synchronize one committed selection across every linked panel.

        Auto-fit limits are calculated once per physical coordinate and then
        reused by every panel carrying that coordinate.  This prevents the
        altitude-vs-Northing and altitude-vs-Easting panels from independently
        choosing slightly different or stale bounds after a spatial selection.
        """

        axes = tuple(self.metadata["axis_order"])
        coordinate_names = tuple(self.metadata.get("coordinate_names", ()))
        padding_specs = tuple(self.metadata.get("axis_padding", ()))
        if not padding_specs:
            padding_specs = ((0.01, 1e-6, 0.03, 0.05),) * len(axes)
        driver_names = set(self._axis_coordinate_names(driver_axis))
        fixed_dimensions = {"time", "altitude"}

        # Gather one canonical original bound and padding rule per coordinate.
        coordinate_specs: dict[str, dict[str, object]] = {}
        for index, (axis, names, original, padding) in enumerate(
            zip(axes, coordinate_names, self._initial_limits, padding_specs)
        ):
            if len(names) != 2:
                continue
            for position, name in enumerate((str(names[0]), str(names[1]))):
                bounds = original[position]
                fraction = float(padding[0 if position == 0 else 2])
                minimum = float(padding[1 if position == 0 else 3])
                entry = coordinate_specs.setdefault(
                    name,
                    {
                        "low": min(bounds),
                        "high": max(bounds),
                        "fraction": fraction,
                        "minimum": minimum,
                        "spatial": index in self._spatial_axes,
                    },
                )
                entry["low"] = min(float(entry["low"]), min(bounds))
                entry["high"] = max(float(entry["high"]), max(bounds))
                entry["spatial"] = bool(entry["spatial"]) or index in self._spatial_axes

        shared_limits: dict[str, tuple[float, float]] = {}
        fitted_names: set[str] = set()
        for name, spec in coordinate_specs.items():
            if name in driver_names and name in self._constraints:
                if self._auto_fit_spatial:
                    shared_limits[name] = tuple(self._constraints[name])
                continue
            if name in fixed_dimensions:
                # Time/altitude remain unchanged unless they were deliberately
                # selected in the driver panel.
                continue
            if not self._auto_fit_spatial or not bool(spec["spatial"]):
                continue
            values = self._coordinate_values.get(name)
            if values is None:
                continue
            original = (float(spec["low"]), float(spec["high"]))
            fitted = self._padded_limits(
                np.asarray(values, dtype=float)[mask],
                original,
                fraction=float(spec["fraction"]),
                minimum=float(spec["minimum"]),
            )
            shared_limits[name] = tuple(sorted((float(fitted[0]), float(fitted[1]))))
            fitted_names.add(name)

        # If the plan view is auto-fitted in both dimensions, enforce equal
        # physical scale once and propagate the expanded limits back to all
        # corresponding altitude/spatial panels.
        for index in self._equal_scale_axes:
            if index >= len(axes) or index >= len(coordinate_names):
                continue
            names = coordinate_names[index]
            if len(names) != 2:
                continue
            x_name, y_name = str(names[0]), str(names[1])
            if x_name not in fitted_names or y_name not in fitted_names:
                continue
            x_limits = self._oriented_limits(shared_limits[x_name], axes[index].get_xlim())
            y_limits = self._oriented_limits(shared_limits[y_name], axes[index].get_ylim())
            x_limits, y_limits = self._equal_scale_limits(
                index, x_limits, y_limits, self._initial_limits[index]
            )
            shared_limits[x_name] = tuple(sorted((float(x_limits[0]), float(x_limits[1]))))
            shared_limits[y_name] = tuple(sorted((float(y_limits[0]), float(y_limits[1]))))

        for axis, names in zip(axes, coordinate_names):
            if len(names) != 2:
                continue
            x_name, y_name = str(names[0]), str(names[1])
            x_limits, y_limits = axis.get_xlim(), axis.get_ylim()
            if axis is not driver_axis and x_name in shared_limits:
                x_limits = self._oriented_limits(shared_limits[x_name], x_limits)
            if axis is not driver_axis and y_name in shared_limits:
                y_limits = self._oriented_limits(shared_limits[y_name], y_limits)
            axis.set_xlim(x_limits, emit=False)
            axis.set_ylim(y_limits, emit=False)

        # Reapply every active dimension constraint to every panel carrying it.
        # Altitude may be a y-axis (Landscape/time-altitude) or an x-axis
        # (Portrait altitude-latitude), so this must be name-based.
        self._apply_authoritative_constraints_to_axes()
        self._restore_axis_geometry()

    def _finish_update(
        self,
        mask: np.ndarray,
        norm: Normalize,
        *,
        driver_axis=None,
        record_history: bool = True,
        notify: bool = True,
    ) -> None:
        meta = self.metadata
        meta["norm"] = norm
        meta["visible_mask"] = mask.copy()
        self._active_mask = mask.copy()
        if meta.get("colorbar") is not None:
            meta["colorbar"].update_normal(meta["scatters"][0])
            colorbar_callback = meta.get("colorbar_update_callback")
            if callable(colorbar_callback):
                colorbar_callback()
        # Artist callbacks may clear/rebuild auxiliary axes or trigger Matplotlib
        # autoscale machinery.  Enforce the named scientific ranges both before
        # and after those callbacks so no altitude panel can drift stale.
        self._apply_authoritative_constraints_to_axes()
        callback = meta.get("subset_callback")
        if callable(callback):
            callback(mask)
        time_callback = meta.get("time_axis_callback")
        if callable(time_callback):
            time_callback()
        self._apply_authoritative_constraints_to_axes()
        self._restore_axis_geometry()
        # True Aspect is display padding, not a source-selection constraint.
        # Apply it only after the final authoritative scientific limits so a
        # preserved GUI view cannot overwrite it during redraw.
        self._enforce_true_spatial_scale(driver_axis)
        self._refresh_map_underlay()
        if notify:
            self._notify_state()
        self.figure.canvas.draw_idle()
        if record_history:
            self._store_current_history_state()

    def _notify_state(self) -> None:
        if not self.enabled:
            return
        visible, in_view = self._current_view_counts()
        title_artist = self.metadata.get("title_artist")
        title_formatter = self.metadata.get("title_formatter")
        displayed = int(self.metadata.get("displayed_count", self.visible_count))
        if title_artist is not None and callable(title_formatter):
            try:
                title_artist.set_text(title_formatter(visible, in_view, displayed))
            except TypeError:
                title_artist.set_text(title_formatter(visible, in_view))
        self.metadata["visible_in_view_count"] = visible
        self.metadata["unfiltered_in_view_count"] = in_view
        if self._state_callback is None:
            return
        self._state_callback(
            {
                "visible_count": visible,
                "displayed_count": displayed,
                "selected_count": self.visible_count,
                "in_view_count": in_view,
                "filtered_count": int(
                    self.metadata.get("filtered_count", self._active_mask.size)
                ),
                "loaded_count": int(
                    self.metadata.get("loaded_count", self._active_mask.size)
                ),
                "layout": self.metadata.get("layout"),
                "interactive_limits": self.current_interactive_limits(),
            }
        )

    def current_interactive_limits(self) -> dict[str, tuple[float, float]]:
        if not self.enabled:
            return {}
        mapping = self.metadata.get("interactive_limit_axes", {})
        axes = tuple(self.metadata["axis_order"])
        result: dict[str, tuple[float, float]] = {}
        for name, specification in mapping.items():
            if str(name) in self._constraints:
                low, high = sorted(
                    (float(self._constraints[str(name)][0]), float(self._constraints[str(name)][1]))
                )
                if np.isfinite(low) and np.isfinite(high) and high > low:
                    result[str(name)] = (low, high)
                    continue
            try:
                axis_index, dimension = specification
                axis = axes[int(axis_index)]
                raw = axis.get_xlim() if str(dimension).lower() == "x" else axis.get_ylim()
                low, high = sorted((float(raw[0]), float(raw[1])))
            except (IndexError, TypeError, ValueError):
                continue
            if np.isfinite(low) and np.isfinite(high) and high > low:
                result[str(name)] = (low, high)
        return result

    def apply_interactive_limits(
        self,
        limits: dict[str, tuple[float, float]] | None,
        *,
        initialize_all_matching_axes: bool = False,
        soft_startup_view: bool = False,
    ) -> bool:
        if not self.enabled or not limits:
            return False
        raw_normalized: dict[str, tuple[float, float]] = {}
        for name, bounds in limits.items():
            try:
                low, high = sorted((float(bounds[0]), float(bounds[1])))
            except (TypeError, ValueError, IndexError):
                continue
            if np.isfinite(low) and np.isfinite(high) and high > low:
                raw_normalized[str(name)] = (low, high)
        # Project Home and saved Projects retain canonical East/North bounds.
        # A live viewpoint may instead plot West/South, which are sign-reversed
        # representations of the same physical coordinates.  Translate before
        # constructing the scientific mask or enforcing True Aspect; otherwise
        # a missing cardinal dimension remains at the full-record range and
        # becomes the (enormous) shared kilometres-per-inch requirement.
        normalized = self._translate_named_limits(
            raw_normalized,
            set(self._coordinate_values),
        )
        if not normalized:
            return False
        candidate_constraints = dict(self._constraints)
        if soft_startup_view:
            candidate_constraints.update(
                self._startup_navigation_constraints(normalized)
            )
        else:
            candidate_constraints.update(normalized)
        old_constraints = self._constraints
        old_startup_display = dict(self._startup_display_constraints)
        self._constraints = candidate_constraints
        self._startup_display_constraints = (
            dict(normalized) if soft_startup_view else {}
        )
        # The opening frame uses every saved limit so it exactly matches the
        # Project.  Only the canonical navigation constraints persist into the
        # first user gesture, which makes outward scrolling incremental rather
        # than jumping to the full record.
        mask = self._mask_for_constraints(
            self._startup_display_constraints
            if soft_startup_view
            else self._constraints
        )
        if not np.any(mask):
            self._constraints = old_constraints
            self._startup_display_constraints = old_startup_display
            return False
        norm = self._norm_for_mask(mask)
        self._updating = True
        try:
            self._set_scatter_subset(mask, norm)
            axes = tuple(self.metadata["axis_order"])
            primary_axes = dict(self.metadata.get("interactive_limit_axes", {}))
            for axis_index, (axis, names) in enumerate(
                zip(axes, self.metadata.get("coordinate_names", ()))
            ):
                if len(names) != 2:
                    continue
                x_name, y_name = str(names[0]), str(names[1])
                x_primary = primary_axes.get(x_name)
                y_primary = primary_axes.get(y_name)
                if x_name in normalized and (
                    initialize_all_matching_axes
                    or self._auto_fit_spatial
                    or x_primary is None
                    or int(x_primary[0]) == axis_index
                ):
                    axis.set_xlim(normalized[x_name], emit=False)
                if y_name in normalized and (
                    initialize_all_matching_axes
                    or self._auto_fit_spatial
                    or y_primary is None
                    or int(y_primary[0]) == axis_index
                ):
                    axis.set_ylim(normalized[y_name], emit=False)
            self._restore_axis_geometry()
            self._pending_axis = None
            self._pending_explicit = False
            self._finish_update(mask, norm, record_history=False)
        finally:
            self._updating = False
        self._soft_startup_view = bool(soft_startup_view)
        if self.toolbar is not None:
            self.toolbar.push_current()
        else:
            self._store_current_history_state()
        return True

    def restore_full(self, *, record_history: bool = True) -> None:
        if not self.enabled:
            return
        self._constraints.clear()
        self._soft_startup_view = False
        self._startup_display_constraints.clear()
        self._base_membership_mask = np.ones(self._active_mask.shape, dtype=bool)
        mask = self._base_membership_mask.copy()
        norm = self._full_norm()
        self._updating = True
        try:
            self._set_scatter_subset(mask, norm)
            for axis, original in zip(self.metadata["axis_order"], self._initial_limits):
                axis.set_xlim(original[0], emit=False)
                axis.set_ylim(original[1], emit=False)
            self._restore_axis_geometry()
            self._pending_axis = None
            self._pending_explicit = False
            self._finish_update(mask, norm, record_history=record_history)
        finally:
            self._updating = False

    def capture_view_state(self) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        source_ids = np.asarray(
            self.metadata.get("source_ids", np.arange(self._active_mask.size)),
            dtype=np.int64,
        )
        selected_ids = (
            source_ids[self._active_mask].copy()
            if source_ids.shape == self._active_mask.shape
            else np.array([], dtype=np.int64)
        )
        visible, in_view = self._current_view_counts()
        norm = self.metadata.get("norm")
        return {
            "layout": self.metadata.get("layout"),
            "coordinate_names": tuple(self.metadata.get("coordinate_names", ())),
            "axis_limits": tuple(
                (tuple(axis.get_xlim()), tuple(axis.get_ylim()))
                for axis in self.metadata["axis_order"]
            ),
            "visible_mask": self._active_mask.copy(),
            "selected_source_ids": selected_ids,
            "constraints": dict(self._constraints),
            "soft_startup_view": bool(self._soft_startup_view),
            "startup_display_constraints": dict(self._startup_display_constraints),
            "interactive_limits": self.current_interactive_limits(),
            "filter_spec": copy.deepcopy(self.metadata.get("filter_spec")),
            "remap_colormap": bool(self._remap_colormap),
            "color_by": self.metadata.get("color_by"),
            "categorical_color": bool(self.metadata.get("categorical_color", False)),
            "norm_limits": (
                None if norm is None else (float(norm.vmin), float(norm.vmax))
            ),
            "visible_count": visible,
            "in_view_count": in_view,
        }

    def _mask_from_source_ids(self, state: dict[str, Any]) -> np.ndarray:
        current_ids = np.asarray(
            self.metadata.get("source_ids", np.arange(self._active_mask.size)),
            dtype=np.int64,
        )
        selected_ids = np.asarray(state.get("selected_source_ids", ()), dtype=np.int64)
        if selected_ids.size == 0 or current_ids.shape != self._active_mask.shape:
            return np.ones(self._active_mask.shape, dtype=bool)
        return np.isin(current_ids, selected_ids, assume_unique=False)

    def restore_view_state(
        self,
        state: dict[str, Any] | None,
        *,
        exact_membership: bool = False,
        record_history: bool = True,
        notify: bool = True,
    ) -> bool:
        """Restore a linked subset after any non-reset GUI redraw.

        Quality filters can change the filtered-array length, so an old boolean
        mask is not a durable selection.  Coordinate constraints are reapplied
        to the newly filtered sources when the projection semantics match;
        stable source identifiers provide a safe fallback across layout or
        viewpoint changes.
        """

        if not self.enabled or not state:
            return False
        axes = tuple(self.metadata["axis_order"])
        new_names = tuple(self.metadata.get("coordinate_names", ()))
        old_names = tuple(state.get("coordinate_names", ()))
        raw_old_constraints = {
            str(name): tuple(float(value) for value in bounds)
            for name, bounds in dict(state.get("constraints", {})).items()
        }
        available = set(self._coordinate_values)
        old_constraints = self._translate_named_limits(raw_old_constraints, available)
        constraints_compatible = bool(raw_old_constraints) and (
            len(old_constraints) == len(raw_old_constraints)
        )

        restore_soft_startup = bool(state.get("soft_startup_view", False))
        raw_startup_display = {
            str(name): tuple(float(value) for value in bounds)
            for name, bounds in dict(
                state.get("startup_display_constraints", {})
            ).items()
        }
        startup_display = self._translate_named_limits(
            raw_startup_display,
            available,
        )
        if restore_soft_startup and not startup_display:
            # Compatibility with navigation states captured by 1.4.0a, where
            # all saved Project bounds lived in ``constraints``.
            startup_display = dict(old_constraints)
            old_constraints = self._startup_navigation_constraints(
                startup_display
            )
            constraints_compatible = bool(old_constraints)
        # Toolbar Back/Forward may request exact membership for ordinary
        # scientific selections.  A saved startup view is intentionally soft,
        # however, so restoring its navigation entry must not freeze the
        # initially visible source IDs as a permanent base mask.
        if restore_soft_startup:
            exact_membership = False
        exact_mask = self._mask_from_source_ids(state) if exact_membership else None
        if exact_membership and exact_mask is not None:
            self._base_membership_mask = exact_mask.copy()
            self._constraints = dict(old_constraints)
            mask = exact_mask.copy()
        elif constraints_compatible:
            self._base_membership_mask = np.ones(self._active_mask.shape, dtype=bool)
            self._constraints = dict(old_constraints)
            mask = self._mask_for_constraints(
                startup_display
                if restore_soft_startup and startup_display
                else self._constraints
            )
        elif not old_constraints:
            # A full-view redraw should expose every source admitted by the new
            # quality filter, including sources newly admitted by a relaxed cut.
            self._base_membership_mask = np.ones(self._active_mask.shape, dtype=bool)
            self._constraints = {}
            mask = self._base_membership_mask.copy()
        else:
            # Projection/viewpoint changes can rename coordinates.  Preserve the
            # exact current membership by stable source ID and keep it as the
            # base mask for any subsequent linked refinement.
            self._base_membership_mask = self._mask_from_source_ids(state)
            self._constraints = {
                name: bounds for name, bounds in old_constraints.items() if name in available
            }
            mask = self._mask_from_constraints()

        norm = self._norm_for_mask(mask)
        saved_norm = state.get("norm_limits")
        same_color_mode = state.get("color_by") == self.metadata.get("color_by")
        categorical_transition = bool(state.get("categorical_color", False)) or bool(
            self.metadata.get("categorical_color", False)
        )
        if (
            same_color_mode
            and not categorical_transition
            and saved_norm is not None
            and len(saved_norm) == 2
        ):
            low, high = (float(value) for value in saved_norm)
            if np.isfinite(low) and np.isfinite(high) and high > low:
                norm = (
                    LogNorm(vmin=low, vmax=high)
                    if isinstance(self._full_norm(), LogNorm)
                    else Normalize(vmin=low, vmax=high)
                )
        same_geometry = (
            state.get("layout") == self.metadata.get("layout")
            and old_names == new_names
        )
        limits = tuple(state.get("axis_limits", ()))
        exact_limits = same_geometry and len(limits) == len(axes)

        self._updating = True
        try:
            self._set_scatter_subset(mask, norm)
            if exact_limits:
                for axis, (x_limits, y_limits) in zip(axes, limits):
                    axis.set_xlim(x_limits, emit=False)
                    axis.set_ylim(y_limits, emit=False)
            elif constraints_compatible:
                # Preserve the semantic selected window even if a projection was
                # rebuilt.  Apply each known coordinate to every matching axis.
                for axis, names in zip(axes, new_names):
                    if len(names) != 2:
                        continue
                    x_name, y_name = str(names[0]), str(names[1])
                    if x_name in self._constraints:
                        axis.set_xlim(
                            self._oriented_limits(self._constraints[x_name], axis.get_xlim()),
                            emit=False,
                        )
                    if y_name in self._constraints:
                        axis.set_ylim(
                            self._oriented_limits(self._constraints[y_name], axis.get_ylim()),
                            emit=False,
                        )
            else:
                # Coordinate names changed (for example Easting -> Westing or
                # Landscape -> Portrait).  Keep the selected source IDs, retain
                # common time/altitude limits, and auto-fit spatial projections.
                old_interactive = self._translate_named_limits(
                    {
                        str(name): tuple(float(value) for value in bounds)
                        for name, bounds in dict(state.get("interactive_limits", {})).items()
                    },
                    available,
                )
                padding_specs = tuple(self.metadata.get("axis_padding", ()))
                if not padding_specs:
                    padding_specs = ((0.03, 1e-6, 0.03, 0.05),) * len(axes)
                for index, (axis, names, pair, original, padding) in enumerate(
                    zip(
                        axes,
                        new_names,
                        self.metadata["coordinate_pairs"],
                        self._initial_limits,
                        padding_specs,
                    )
                ):
                    if len(names) != 2:
                        continue
                    x_name, y_name = str(names[0]), str(names[1])
                    x_values, y_values = (np.asarray(value, dtype=float) for value in pair)
                    x_fraction, x_minimum, y_fraction, y_minimum = padding
                    x_limits, y_limits = axis.get_xlim(), axis.get_ylim()
                    if x_name in old_interactive:
                        x_limits = self._oriented_limits(old_interactive[x_name], x_limits)
                    elif self._auto_fit_spatial and index in self._spatial_axes:
                        x_limits = self._padded_limits(
                            x_values[mask], original[0],
                            fraction=x_fraction, minimum=x_minimum,
                        )
                    if y_name in old_interactive:
                        y_limits = self._oriented_limits(old_interactive[y_name], y_limits)
                    elif self._auto_fit_spatial and index in self._spatial_axes:
                        y_limits = self._padded_limits(
                            y_values[mask], original[1],
                            fraction=y_fraction, minimum=y_minimum,
                        )
                    if (
                        index in self._equal_scale_axes
                        and self._auto_fit_spatial
                        and "altitude" not in {x_name, y_name}
                        and x_name not in old_interactive
                        and y_name not in old_interactive
                    ):
                        x_limits, y_limits = self._equal_scale_limits(
                            index, x_limits, y_limits, original
                        )
                    axis.set_xlim(x_limits, emit=False)
                    axis.set_ylim(y_limits, emit=False)
            self._apply_authoritative_constraints_to_axes()
            self._restore_axis_geometry()
            self._soft_startup_view = restore_soft_startup
            self._startup_display_constraints = (
                dict(startup_display) if restore_soft_startup else {}
            )
            self._finish_update(
                mask,
                norm,
                record_history=record_history,
                notify=notify,
            )
            self._pending_axis = None
            self._pending_explicit = False
        finally:
            self._updating = False
        return True

    def update_now(self, driver_axis=None, *, record_history: bool = True) -> None:
        if not self.enabled or self._updating:
            return
        if self._timer is not None:
            self._timer.stop()
        if self._all_axes_at_home():
            self.restore_full(record_history=record_history)
            return
        if driver_axis not in self.metadata["axis_order"]:
            driver_axis = self._pending_axis if self._pending_axis in self.metadata["axis_order"] else None
        if driver_axis is None:
            return
        if self._soft_startup_view and self._pending_explicit:
            # Project startup bounds are only the opening display window.  The
            # canonical time-altitude constraints installed at startup already
            # match an ordinary manually established view, so the first user
            # gesture only retires the temporary exact-opening mask.  Do not
            # clear the linked constraints or auto-reset to the full record.
            self._soft_startup_view = False
            self._startup_display_constraints.clear()
        mask = self._selection_from_axis(driver_axis)
        if not np.any(mask):
            self._pending_axis = None
            return
        norm = self._norm_for_mask(mask)
        self._updating = True
        try:
            self._set_scatter_subset(mask, norm)
            self._synchronize_axes(mask, driver_axis)
            self._pending_axis = None
            self._pending_explicit = False
            self._finish_update(
                mask, norm, driver_axis=driver_axis, record_history=record_history
            )
        finally:
            self._updating = False
