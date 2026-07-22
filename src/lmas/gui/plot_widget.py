from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Callable
import time

from matplotlib.backend_bases import NavigationToolbar2
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QFileDialog, QMainWindow, QSizePolicy, QVBoxLayout, QWidget

from ..interactions import LinkedViewController
from ..plotting import save_figure


class _CanvasHolder(QWidget):
    """Host a Matplotlib canvas, optionally preserving a fixed aspect ratio."""

    def __init__(self, canvas: FigureCanvasQTAgg, aspect_ratio: float | None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._canvas = canvas
        self._aspect_ratio = float(aspect_ratio) if aspect_ratio and aspect_ratio > 0 else None
        self._canvas.setParent(self)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._draw_connection = self._canvas.mpl_connect("draw_event", self._on_canvas_draw)

    def _on_canvas_draw(self, _event) -> None:
        metadata = getattr(self._canvas.figure, "_lmas_metadata", {})
        if not isinstance(metadata, dict) or metadata.get("layout") != "xlma":
            return
        size = metadata.get("export_size_inches")
        if not isinstance(size, (tuple, list)) or len(size) != 2 or float(size[1]) <= 0:
            return
        ratio = float(size[0]) / float(size[1])
        if self._aspect_ratio is None or abs(float(self._aspect_ratio) - ratio) > 1.0e-9:
            self._aspect_ratio = ratio
            self._fit_canvas()

    def _fit_canvas(self) -> None:
        width = max(1, self.width())
        height = max(1, self.height())
        if self._aspect_ratio is None:
            target_width, target_height = width, height
        else:
            target_width = width
            target_height = max(1, round(target_width / self._aspect_ratio))
            if target_height > height:
                target_height = height
                target_width = max(1, round(target_height * self._aspect_ratio))
        left = (width - target_width) // 2
        top = (height - target_height) // 2
        self._canvas.setGeometry(left, top, target_width, target_height)

    def set_aspect_ratio(self, aspect_ratio: float | None) -> None:
        self._aspect_ratio = (
            float(aspect_ratio)
            if aspect_ratio is not None and float(aspect_ratio) > 0
            else None
        )
        self._fit_canvas()

    def rebind_draw_callback(self) -> None:
        """Reconnect the holder callback after the canvas receives a new figure.

        Matplotlib stores canvas callbacks on the Figure rather than on the Qt
        widget.  Reusing the widget therefore requires reconnecting persistent
        callbacks whenever ``canvas.figure`` changes.
        """

        self._draw_connection = self._canvas.mpl_connect(
            "draw_event", self._on_canvas_draw
        )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._fit_canvas()


class LMASNavigationToolbar(NavigationToolbar2QT):
    """Navigation toolbar paired with LMAS subset-history state.

    Matplotlib's stock pan tool requests a complete canvas redraw for every
    mouse-motion event. LMAS keeps only the newest pointer position, renders at
    a bounded rate, and uses a lightweight source proxy until release.
    """

    # During drag LMAS redraws only the active scientific panel's data layer,
    # so a higher pointer refresh rate is affordable even on ordinary laptops.
    _PAN_REFRESH_HZ = 45.0
    _PAN_PROXY_LIMIT = 1_500

    def __init__(self, canvas, parent=None) -> None:
        self._linked_controller = None
        self._pending_pan_event = None
        self._last_pan_render_s = 0.0
        self._fast_pan_partial_redraw = False
        super().__init__(canvas, parent)
        self._pan_render_timer = QTimer(self)
        self._pan_render_timer.setSingleShot(True)
        self._pan_render_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._pan_render_timer.timeout.connect(self._flush_pending_pan)

    def set_linked_controller(self, controller) -> None:
        if controller is None:
            self._pan_render_timer.stop()
            self._pending_pan_event = None
            self._fast_pan_partial_redraw = False
            if self._linked_controller is not None:
                self._linked_controller.end_fast_pan(
                    restore_artists=False, redraw=False
                )
        self._linked_controller = controller

    def release_zoom(self, event) -> None:
        super().release_zoom(event)
        if self._linked_controller is not None:
            self._linked_controller.after_toolbar_gesture("selection", event)

    @staticmethod
    def _pan_event_snapshot(event):
        if getattr(event, "x", None) is None or getattr(event, "y", None) is None:
            return None
        return SimpleNamespace(
            x=float(event.x),
            y=float(event.y),
            key=getattr(event, "key", None),
            button=getattr(event, "button", None),
            inaxes=getattr(event, "inaxes", None),
        )

    def press_pan(self, event) -> None:
        super().press_pan(event)
        if getattr(self, "_pan_info", None) is None:
            return
        self._pan_render_timer.stop()
        self._pending_pan_event = None
        self._last_pan_render_s = 0.0
        proxy_changed = False
        if self._linked_controller is not None:
            proxy_changed = self._linked_controller.begin_fast_pan(
                point_limit=self._PAN_PROXY_LIMIT
            )
        # Materialize the lightweight proxy once. Subsequent motion events can
        # repaint only the active axes' interior instead of the whole figure.
        self._fast_pan_partial_redraw = False
        try:
            if bool(getattr(self.canvas, "supports_blit", False)):
                if proxy_changed:
                    self.canvas.draw()
                else:
                    self.canvas.get_renderer()
                self._fast_pan_partial_redraw = True
        except Exception:
            self._fast_pan_partial_redraw = False

    @staticmethod
    def _pan_data_artists(axis):
        """Return visible data-layer artists in stable z-order.

        Ticks, labels, titles, spines, and figure furniture remain frozen while
        the pointer is down. They are refreshed exactly once on release.
        """

        candidates = [
            *axis.images,
            *(patch for patch in axis.patches if patch is not axis.patch),
            *axis.collections,
            *axis.lines,
            *axis.artists,
            *axis.texts,
        ]
        seen: set[int] = set()
        visible = []
        for artist in candidates:
            marker = id(artist)
            if marker in seen:
                continue
            seen.add(marker)
            try:
                if not artist.get_visible():
                    continue
            except Exception:
                pass
            visible.append(artist)
        return sorted(visible, key=lambda artist: float(artist.get_zorder()))

    def _draw_fast_pan_panels(self, axes) -> bool:
        """Blit only moving panel interiors; fall back safely if unsupported."""

        if not self._fast_pan_partial_redraw:
            return False
        try:
            renderer = self.canvas.get_renderer()
            for axis in axes:
                # The opaque axes patch clears the previous data image while
                # leaving static ticks and labels untouched around the panel.
                axis.patch.draw(renderer)
                for artist in self._pan_data_artists(axis):
                    artist.draw(renderer)
                self.canvas.blit(axis.bbox)
            self.canvas.flush_events()
            return True
        except Exception:
            self._fast_pan_partial_redraw = False
            return False

    def _render_pan_event(self, event, *, redraw: bool = True) -> None:
        pan_info = getattr(self, "_pan_info", None)
        if pan_info is None or event is None:
            return
        for axis in pan_info.axes:
            axis.drag_pan(pan_info.button, event.key, event.x, event.y)
        self._last_pan_render_s = time.perf_counter()
        if redraw and not self._draw_fast_pan_panels(pan_info.axes):
            self.canvas.draw_idle()

    def _flush_pending_pan(self) -> None:
        event = self._pending_pan_event
        self._pending_pan_event = None
        self._render_pan_event(event)

    def drag_pan(self, event) -> None:
        snapshot = self._pan_event_snapshot(event)
        if snapshot is None or getattr(self, "_pan_info", None) is None:
            return
        self._pending_pan_event = snapshot
        interval = 1.0 / self._PAN_REFRESH_HZ
        elapsed = time.perf_counter() - self._last_pan_render_s
        if self._last_pan_render_s <= 0.0 or elapsed >= interval:
            self._pan_render_timer.stop()
            self._flush_pending_pan()
            return
        remaining_ms = max(1, int(round((interval - elapsed) * 1000.0)))
        if not self._pan_render_timer.isActive():
            self._pan_render_timer.start(remaining_ms)

    def release_pan(self, event) -> None:
        self._pan_render_timer.stop()
        self._fast_pan_partial_redraw = False
        final_event = self._pan_event_snapshot(event) or self._pending_pan_event
        self._pending_pan_event = None
        self._render_pan_event(final_event, redraw=False)
        super().release_pan(event)
        if self._linked_controller is not None:
            self._linked_controller.after_toolbar_gesture("pan", event)

    def push_current(self) -> None:
        super().push_current()
        if self._linked_controller is not None:
            self._linked_controller.on_toolbar_history_pushed()

    def back(self, *args) -> None:
        controller = self._linked_controller
        if controller is not None:
            controller.before_toolbar_history_restore()
        try:
            super().back(*args)
        finally:
            if controller is not None:
                controller.after_toolbar_history_restore()

    def forward(self, *args) -> None:
        controller = self._linked_controller
        if controller is not None:
            controller.before_toolbar_history_restore()
        try:
            super().forward(*args)
        finally:
            if controller is not None:
                controller.after_toolbar_history_restore()

    def home(self, *args) -> None:
        controller = self._linked_controller
        if controller is not None:
            controller.before_toolbar_history_restore()
        try:
            super().home(*args)
        finally:
            if controller is not None:
                controller.after_toolbar_history_restore()


class FigureHost(QWidget):
    view_state_changed = Signal(object)
    figure_changed = Signal(object)
    project_home_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._figure: Figure | None = None
        self._canvas: FigureCanvasQTAgg | None = None
        self._toolbar: NavigationToolbar2QT | None = None
        self._canvas_holder: _CanvasHolder | None = None
        self._linked_view: LinkedViewController | None = None
        self._precision_mode_active = False
        self._selection_mode_active = False
        self._selection_crosshair_cursor = True
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(6, 4, 6, 6)
        self._layout.setSpacing(4)

    @property
    def figure(self) -> Figure | None:
        return self._figure

    @property
    def linked_view(self) -> LinkedViewController | None:
        return self._linked_view

    @staticmethod
    def _figure_aspect_ratio(figure: Figure) -> float | None:
        metadata = getattr(figure, "_lmas_metadata", {})
        if isinstance(metadata, dict) and metadata.get("layout") == "xlma":
            size = metadata.get("export_size_inches")
            if (
                isinstance(size, (tuple, list))
                and len(size) == 2
                and float(size[1]) > 0
            ):
                return float(size[0]) / float(size[1])
        return None

    def set_figure(self, figure: Figure, *, preserve_view: bool = False) -> None:
        view_state = None
        session_history = None
        if preserve_view and self._linked_view is not None:
            view_state = self._linked_view.capture_view_state()
            session_history = self._linked_view.capture_session_history()
        if self._linked_view is not None:
            self._linked_view.close()
            self._linked_view = None

        aspect_ratio = self._figure_aspect_ratio(figure)
        if self._canvas is None:
            self._canvas = FigureCanvasQTAgg(figure)
            self._canvas.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
            )
            self._canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            self._toolbar = LMASNavigationToolbar(self._canvas, self)
            self._toolbar.addSeparator()
            project_home_action = QAction("Project Home", self._toolbar)
            project_home_action.setToolTip(
                "Return to this project's saved starting view"
            )
            project_home_action.triggered.connect(self.project_home_requested.emit)
            self._toolbar.addAction(project_home_action)
            self._canvas_holder = _CanvasHolder(self._canvas, aspect_ratio, self)
            self._layout.addWidget(self._toolbar)
            self._layout.addWidget(self._canvas_holder, 1)
        else:
            # Keep the heavyweight Qt canvas and toolbar alive. Ordinary LMAS
            # redraws replace only the Matplotlib Figure object, avoiding widget
            # destruction, layout churn, and backend reinitialization.
            old_figure = self._figure
            self._figure = figure
            self._canvas.figure = figure
            figure.set_canvas(self._canvas)
            if self._toolbar is not None:
                # NavigationToolbar2's mouse callbacks also live on the Figure.
                # Reinitialize only its backend state; the existing Qt toolbar
                # widget, actions, and Project Home action remain in place.
                NavigationToolbar2.__init__(self._toolbar, self._canvas)
            if old_figure is not None and old_figure is not figure:
                try:
                    old_figure.set_canvas(None)
                except Exception:
                    pass
            if self._canvas_holder is not None:
                self._canvas_holder.rebind_draw_callback()
                self._canvas_holder.set_aspect_ratio(aspect_ratio)

        self._figure = figure
        self._linked_view = LinkedViewController(
            figure,
            toolbar=self._toolbar,
            state_callback=self.view_state_changed.emit,
        )
        if session_history is not None:
            self._linked_view.restore_session_history(session_history)
        elif view_state is not None:
            self._linked_view.restore_view_state(view_state)
        else:
            self._linked_view._notify_state()
        self._canvas.draw_idle()
        if self._precision_mode_active:
            self._canvas.setCursor(Qt.CursorShape.CrossCursor)
        elif self._selection_mode_active:
            self._apply_selection_cursor()
        self.figure_changed.emit(figure)

    def set_linked_zoom_behavior(
        self,
        *,
        auto_fit_spatial: bool,
        remap_time_colors: bool,
    ) -> None:
        if self._linked_view is not None:
            self._linked_view.set_behavior(
                auto_fit_spatial=auto_fit_spatial,
                remap_time_colors=remap_time_colors,
            )

    def restore_full_view(self) -> None:
        if self._toolbar is not None:
            self._toolbar.home()
        elif self._linked_view is not None:
            self._linked_view.restore_full()

    def history_back(self) -> None:
        if self._toolbar is not None:
            self._toolbar.back()

    def history_forward(self) -> None:
        if self._toolbar is not None:
            self._toolbar.forward()

    @staticmethod
    def _toolbar_mode_name(toolbar) -> str:
        mode = getattr(toolbar, "mode", "")
        return str(getattr(mode, "name", mode)).casefold()

    def activate_rectangle_zoom(self) -> bool:
        """Activate Matplotlib rectangle zoom without toggling it back off."""

        self.deactivate_precision_mode()
        self.deactivate_selection_mode()
        if self._toolbar is None:
            return False
        if "zoom" not in self._toolbar_mode_name(self._toolbar):
            self._toolbar.zoom()
        if self._canvas is not None:
            self._canvas.setFocus(Qt.FocusReason.ShortcutFocusReason)
        return True

    def activate_pan_drag(self) -> bool:
        """Activate Matplotlib click-and-drag pan without toggling it back off."""

        self.deactivate_precision_mode()
        self.deactivate_selection_mode()
        if self._toolbar is None:
            return False
        if "pan" not in self._toolbar_mode_name(self._toolbar):
            self._toolbar.pan()
        if self._canvas is not None:
            self._canvas.setFocus(Qt.FocusReason.ShortcutFocusReason)
        return True


    @property
    def precision_mode_active(self) -> bool:
        return bool(self._precision_mode_active)

    def activate_precision_mode(self) -> bool:
        """Enter point-placement mode and give the canvas a crosshair cursor."""

        if self._canvas is None:
            return False
        if self._toolbar is not None:
            mode = self._toolbar_mode_name(self._toolbar)
            if "zoom" in mode:
                self._toolbar.zoom()
            elif "pan" in mode:
                self._toolbar.pan()
        self._selection_mode_active = False
        self._precision_mode_active = True
        self._canvas.setCursor(Qt.CursorShape.CrossCursor)
        self._canvas.setFocus(Qt.FocusReason.ShortcutFocusReason)
        return True

    def deactivate_precision_mode(self) -> None:
        self._precision_mode_active = False
        if self._canvas is not None and not self._selection_mode_active:
            self._canvas.unsetCursor()

    @property
    def selection_mode_active(self) -> bool:
        return bool(self._selection_mode_active)

    def _apply_selection_cursor(self) -> None:
        """Use a crosshair for lasso and the ordinary arrow for point editing."""

        if self._canvas is None:
            return
        if self._selection_crosshair_cursor:
            self._canvas.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self._canvas.setCursor(Qt.CursorShape.ArrowCursor)

    def activate_selection_mode(self, *, crosshair_cursor: bool = True) -> bool:
        """Enter linked source-selection mode and focus the plotting canvas."""

        if self._canvas is None:
            return False
        if self._toolbar is not None:
            mode = self._toolbar_mode_name(self._toolbar)
            if "zoom" in mode:
                self._toolbar.zoom()
            elif "pan" in mode:
                self._toolbar.pan()
        self._precision_mode_active = False
        self._selection_mode_active = True
        self._selection_crosshair_cursor = bool(crosshair_cursor)
        self._apply_selection_cursor()
        self._canvas.setFocus(Qt.FocusReason.ShortcutFocusReason)
        return True

    def deactivate_selection_mode(self) -> None:
        self._selection_mode_active = False
        if self._canvas is not None and not self._precision_mode_active:
            self._canvas.unsetCursor()

    def set_interactive_limits(
        self,
        limits: dict[str, tuple[float, float]],
        *,
        initialize_all_matching_axes: bool = False,
        soft_startup_view: bool = False,
    ) -> bool:
        if self._linked_view is None:
            return False
        return self._linked_view.apply_interactive_limits(
            limits,
            initialize_all_matching_axes=initialize_all_matching_axes,
            soft_startup_view=soft_startup_view,
        )

    def save_as(self, default_path: Path, *, dpi: int = 300) -> Path | None:
        if self._figure is None:
            return None
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Save LMAS figure",
            str(default_path),
            "PNG image (*.png);;PDF document (*.pdf);;SVG image (*.svg)",
        )
        if not selected:
            return None
        return save_figure(self._figure, selected, dpi=dpi)


class DetachedPlotWindow(QMainWindow):
    def __init__(
        self,
        title: str,
        figure_factory: Callable[[], Figure],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle(title)
        self.resize(1400, 900)
        host = FigureHost(self)
        host.set_figure(figure_factory())
        self.setCentralWidget(host)
