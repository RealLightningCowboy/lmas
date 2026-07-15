from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from .interactions import LinkedViewController
from .group_overlay import apply_saved_source_group_overlays
from .model import FilterSpec, LMAProject, PlotSpec
from .overlays.satellite import SatelliteOverlayManager, SatelliteOverlayRenderer
from .overlays.network import NetworkOverlayManager, NetworkOverlayRenderer
from .plotting import create_lma_figure
from .plotting.common import save_figure

SUPPORTED_EXPORT_THEMES = ("dark", "light", "space")


def apply_saved_satellite_overlays(figure, project: LMAProject) -> SatelliteOverlayRenderer | None:
    """Restore and render project GLM overlays on a static export figure.

    The export renderer ignores the interactive event cap while preserving all
    user layer visibility, color, time-rail, and z-order settings. Missing GLM
    files do not prevent the underlying LMA figure from being saved.
    """

    state = dict(getattr(project, "satellite_overlay_state", {}) or {})
    if not state.get("datasets"):
        return None
    manager = SatelliteOverlayManager()
    directory = None
    if getattr(project, "project_path", None) is not None:
        directory = project.project_path.parent
    manager.restore_project_state(state, project_directory=directory)
    if not manager.has_data:
        return None
    renderer = SatelliteOverlayRenderer(manager, for_export=True)
    renderer.bind(figure, project)
    # Keep a strong reference until the figure is saved/closed.
    figure._lmas_satellite_export_renderer = renderer
    return renderer


def apply_saved_network_overlays(figure, project: LMAProject) -> NetworkOverlayRenderer | None:
    """Restore and render ground-network overlays on a static export figure."""

    state = dict(getattr(project, "network_overlay_state", {}) or {})
    if not state.get("datasets"):
        return None
    manager = NetworkOverlayManager()
    directory = None
    if getattr(project, "project_path", None) is not None:
        directory = project.project_path.parent
    manager.restore_project_state(state, project_directory=directory)
    if not manager.has_data:
        return None
    renderer = NetworkOverlayRenderer(manager, for_export=True)
    renderer.bind(figure, project)
    figure._lmas_network_export_renderer = renderer
    return renderer


def default_custom_title(
    current_dynamic_title: str,
    time_limits: tuple[float, float] | None,
) -> str:
    """Build the default Save Figure or animation title.

    The live source-count and quality suffix is preserved verbatim. The leading
    timestamp is the UTC second containing the beginning of the committed time
    view. Flooring the lower bound avoids labeling a sub-second plot with the
    following second.
    """

    import math

    title = str(current_dynamic_title or "").strip()
    suffix = ""
    if " — " in title:
        _prefix, suffix = title.split(" — ", 1)
    elif "\n" in title:
        lines = [line.strip() for line in title.splitlines() if line.strip()]
        suffix = lines[-1] if lines else ""
    elif title:
        suffix = title

    timestamp = ""
    if time_limits is not None:
        low, _high = sorted((float(time_limits[0]), float(time_limits[1])))
        moment = mdates.num2date(low, tz=timezone.utc)
        floored = datetime.fromtimestamp(math.floor(moment.timestamp()), tz=timezone.utc)
        timestamp = floored.strftime("%Y-%m-%d %H:%M:%S UTC")

    return f"{timestamp} — {suffix}" if timestamp and suffix else (timestamp or suffix)


def theme_variant_paths(
    path: str | Path,
    themes: Iterable[str],
) -> dict[str, Path]:
    """Return one output path per selected theme without duplicate suffixes."""

    destination = Path(path).expanduser().resolve()
    stem = destination.stem
    lower = stem.lower()
    for known in SUPPORTED_EXPORT_THEMES:
        marker = f"_{known}"
        if lower.endswith(marker):
            stem = stem[: -len(marker)]
            break

    result: dict[str, Path] = {}
    for value in themes:
        theme = str(value).strip().lower()
        if theme not in SUPPORTED_EXPORT_THEMES or theme in result:
            continue
        result[theme] = destination.with_name(f"{stem}_{theme}{destination.suffix}")
    return result


def save_theme_variants(
    project: LMAProject,
    *,
    filters: FilterSpec,
    plot: PlotSpec,
    view_state: Mapping[str, Any] | None,
    path: str | Path,
    dpi: int,
    title: str,
    themes: Iterable[str],
    source_group_overlays_visible: bool = False,
) -> tuple[Path, ...]:
    """Render and save exact linked-view copies in multiple figure themes."""

    destinations = theme_variant_paths(path, themes)
    saved: list[Path] = []
    for theme, destination in destinations.items():
        figure = create_lma_figure(
            project,
            filters=filters,
            plot=replace(plot, theme=theme, preview_point_limit=0),
            for_export=True,
        )
        controller = LinkedViewController(figure)
        try:
            if view_state:
                controller.restore_view_state(
                    dict(view_state),
                    exact_membership=True,
                    record_history=False,
                )
            metadata = getattr(figure, "_lmas_metadata", {})
            title_artist = metadata.get("title_artist") if isinstance(metadata, dict) else None
            if title_artist is not None and title:
                rendered_title = str(title)
                if metadata.get("layout") == "xlma" and " — " in rendered_title:
                    rendered_title = rendered_title.replace(" — ", "\n", 1)
                title_artist.set_text(rendered_title)
            if source_group_overlays_visible:
                apply_saved_source_group_overlays(figure, project.source_selection_state)
            apply_saved_satellite_overlays(figure, project)
            apply_saved_network_overlays(figure, project)
            figure.canvas.draw()
            saved.append(save_figure(figure, destination, dpi=int(dpi)))
        finally:
            figure.clear()
            plt.close(figure)
    return tuple(saved)


def save_exact_view(
    project: LMAProject,
    *,
    filters: FilterSpec,
    plot: PlotSpec,
    view_state: Mapping[str, Any] | None,
    path: str | Path,
    dpi: int,
    title: str | None = None,
    source_group_overlays_visible: bool = False,
) -> Path:
    """Render one full-resolution static figure for the exact linked view."""

    figure = create_lma_figure(
        project,
        filters=filters,
        plot=replace(plot, preview_point_limit=0),
        for_export=True,
    )
    controller = LinkedViewController(figure)
    try:
        if view_state:
            controller.restore_view_state(
                dict(view_state),
                exact_membership=True,
                record_history=False,
            )
        metadata = getattr(figure, "_lmas_metadata", {})
        title_artist = metadata.get("title_artist") if isinstance(metadata, dict) else None
        if title_artist is not None and title:
            rendered_title = str(title)
            if metadata.get("layout") == "xlma" and " — " in rendered_title:
                rendered_title = rendered_title.replace(" — ", "\n", 1)
            title_artist.set_text(rendered_title)
        if source_group_overlays_visible:
            apply_saved_source_group_overlays(figure, project.source_selection_state)
        apply_saved_satellite_overlays(figure, project)
        apply_saved_network_overlays(figure, project)
        figure.canvas.draw()
        return save_figure(figure, path, dpi=int(dpi))
    finally:
        figure.clear()
        plt.close(figure)


__all__ = [
    "SUPPORTED_EXPORT_THEMES",
    "default_custom_title",
    "apply_saved_satellite_overlays",
    "apply_saved_network_overlays",
    "save_exact_view",
    "save_theme_variants",
    "theme_variant_paths",
]
