"""Generic satellite-overlay state with the first native GLM provider.

This is intentionally a small internal provider layer rather than a premature
external plug-in framework.  The manager owns independent observational
collections, serializable styles, and rendering state while providers remain
responsible for reading their native products.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence

import numpy as np
from matplotlib.colors import is_color_like

from .colormaps import SATELLITE_COLORMAP_NAMES
from .glm import GLMObservation, read_glm

_PLATFORM_RE = re.compile(r"_(G\d{2})_", re.IGNORECASE)


@dataclass(slots=True)
class GLMOverlayStyle:
    enabled: bool = True
    show_event_footprints: bool = True
    show_group_centroids: bool = True
    show_flash_centroids: bool = False
    show_maximum_group: bool = False
    show_colorbar: bool = True
    show_group_time_rail: bool = True
    show_time_rail_labels: bool = False
    colormap: str = "mako"
    logarithmic_energy: bool = True
    footprint_alpha: float = 0.90
    footprint_edge_width: float = 0.20
    group_marker_size: float = 15.0
    group_marker_color: str = "auto"
    group_edge_width: float = 0.55
    maximum_group_size: float = 150.0
    maximum_group_color: str = "springgreen"
    footprint_zorder: float = 0.40
    group_zorder: float = 0.60
    time_rail_marker_size: float = 18.0
    time_rail_zorder: float = 3.0
    maximum_interactive_events: int = 1500
    footprint_render_padding_fraction: float = 1.0 / 3.0

    @property
    def zorder(self) -> float:
        """Backward-compatible alias for the old single overlay z-order."""
        return float(self.footprint_zorder)

    def validated(self) -> "GLMOverlayStyle":
        cmap = str(self.colormap).strip().lower()
        if cmap not in SATELLITE_COLORMAP_NAMES:
            cmap = "mako"
        return GLMOverlayStyle(
            enabled=bool(self.enabled),
            show_event_footprints=bool(self.show_event_footprints),
            show_group_centroids=bool(self.show_group_centroids),
            show_flash_centroids=bool(self.show_flash_centroids),
            show_maximum_group=bool(self.show_maximum_group),
            show_colorbar=bool(self.show_colorbar),
            show_group_time_rail=bool(self.show_group_time_rail),
            show_time_rail_labels=bool(self.show_time_rail_labels),
            colormap=cmap,
            logarithmic_energy=bool(self.logarithmic_energy),
            footprint_alpha=float(np.clip(float(self.footprint_alpha), 0.0, 1.0)),
            footprint_edge_width=max(0.0, float(self.footprint_edge_width)),
            group_marker_size=max(1.0, float(self.group_marker_size)),
            group_marker_color=(
                str(self.group_marker_color or "auto")
                if str(self.group_marker_color or "auto").lower() == "auto"
                or is_color_like(str(self.group_marker_color))
                else "auto"
            ),
            group_edge_width=max(0.0, float(self.group_edge_width)),
            maximum_group_size=max(1.0, float(self.maximum_group_size)),
            maximum_group_color=str(self.maximum_group_color or "springgreen"),
            footprint_zorder=float(self.footprint_zorder),
            group_zorder=float(self.group_zorder),
            time_rail_marker_size=max(1.0, float(self.time_rail_marker_size)),
            time_rail_zorder=float(self.time_rail_zorder),
            maximum_interactive_events=max(0, int(self.maximum_interactive_events)),
            footprint_render_padding_fraction=float(
                np.clip(float(self.footprint_render_padding_fraction), 0.0, 1.0)
            ),
        )

    def to_dict(self) -> dict[str, object]:
        values = asdict(self.validated())
        values["format"] = "lmas-glm-overlay-style-v2"
        return values

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "GLMOverlayStyle":
        values = dict(payload or {})
        style_format = str(values.pop("format", "")).strip().lower()
        # Dev3-dev5 serialized the old default marker size (45) even when the
        # user never selected it. Migrate that legacy default to the dev6
        # default of 15. Explicit dev6 values are protected by the v2 tag.
        if style_format != "lmas-glm-overlay-style-v2":
            try:
                if float(values.get("group_marker_size", 45.0)) == 45.0:
                    values["group_marker_size"] = 15.0
            except (TypeError, ValueError):
                values["group_marker_size"] = 15.0
        # dev3/dev4 stored one z-order. Preserve the exact old visual ordering:
        # footprints used zorder and group centroids used zorder + 0.10.
        if "footprint_zorder" not in values or "group_zorder" not in values:
            try:
                legacy = float(values.get("zorder", 0.50))
                if legacy == 90.0:
                    legacy = 0.50
            except (TypeError, ValueError):
                legacy = 0.50
            values.setdefault("footprint_zorder", legacy)
            values.setdefault("group_zorder", legacy + 0.10)
        values.pop("zorder", None)
        accepted = {name for name in cls.__dataclass_fields__}
        return cls(**{key: value for key, value in values.items() if key in accepted}).validated()


@dataclass(slots=True)
class SatelliteDatasetRecord:
    key: str
    provider_id: str
    observation: GLMObservation
    style: GLMOverlayStyle = field(default_factory=GLMOverlayStyle)

    @property
    def display_name(self) -> str:
        return self.observation.identity.display_name

    @property
    def source_paths(self) -> tuple[Path, ...]:
        return tuple(item.path for item in self.observation.identity.source_files)


class SatelliteOverlayManager:
    """Own independent satellite datasets and serializable overlay settings."""

    def __init__(self) -> None:
        self._records: dict[str, SatelliteDatasetRecord] = {}
        self.shared_energy_scale = True
        self.glm_backend = "auto"
        self.last_restore_errors: list[str] = []
        self._renderer = None

    @property
    def records(self) -> tuple[SatelliteDatasetRecord, ...]:
        return tuple(self._records[key] for key in sorted(self._records))

    @property
    def has_data(self) -> bool:
        return bool(self._records)

    def record(self, key: str) -> SatelliteDatasetRecord:
        return self._records[str(key)]

    @staticmethod
    def dataset_key(observation: GLMObservation) -> str:
        identity = observation.identity
        return f"glm:{identity.platform_id}:{identity.observation_start_ns}:{identity.observation_end_ns}"

    def add_glm_paths(self, paths: Sequence[str | Path]) -> tuple[SatelliteDatasetRecord, ...]:
        grouped = group_glm_paths_by_platform(paths)
        loaded: list[SatelliteDatasetRecord] = []
        for _platform, platform_paths in sorted(grouped.items()):
            observation = read_glm(platform_paths, backend=self.glm_backend)
            key = self.dataset_key(observation)
            prior = self._records.get(key)
            if prior is not None:
                style = prior.style
            else:
                style = GLMOverlayStyle()
                if self._records:
                    global_style = self.global_layer_state()
                    style = replace(
                        style,
                        show_event_footprints=global_style.show_event_footprints,
                        show_group_centroids=global_style.show_group_centroids,
                        show_flash_centroids=global_style.show_flash_centroids,
                        show_maximum_group=global_style.show_maximum_group,
                        show_colorbar=global_style.show_colorbar,
                        show_group_time_rail=global_style.show_group_time_rail,
                        show_time_rail_labels=global_style.show_time_rail_labels,
                    ).validated()
            record = SatelliteDatasetRecord(key, "glm", observation, style)
            self._records[key] = record
            loaded.append(record)
        return tuple(loaded)

    def remove(self, key: str) -> None:
        self._records.pop(str(key), None)

    def clear(self) -> None:
        self._records.clear()
        if self._renderer is not None:
            self._renderer.clear()

    def set_renderer(self, renderer) -> None:
        if self._renderer is not None and self._renderer is not renderer:
            self._renderer.clear()
        self._renderer = renderer

    @property
    def renderer(self):
        return self._renderer


    def set_global_layer_state(
        self,
        *,
        show_event_footprints: bool | None = None,
        show_group_centroids: bool | None = None,
        show_flash_centroids: bool | None = None,
        show_maximum_group: bool | None = None,
        show_colorbar: bool | None = None,
        show_group_time_rail: bool | None = None,
        show_time_rail_labels: bool | None = None,
    ) -> None:
        """Apply GLM layer visibility to every spacecraft dataset.

        Dataset checkboxes control spacecraft visibility. Scientific layer
        checkboxes are intentionally global so disabling footprints, GLM group
        centroids, or time rails removes that layer from both East and West.
        """
        updates = {
            "show_event_footprints": show_event_footprints,
            "show_group_centroids": show_group_centroids,
            "show_flash_centroids": show_flash_centroids,
            "show_maximum_group": show_maximum_group,
            "show_colorbar": show_colorbar,
            "show_group_time_rail": show_group_time_rail,
            "show_time_rail_labels": show_time_rail_labels,
        }
        values = {key: bool(value) for key, value in updates.items() if value is not None}
        for record in self._records.values():
            record.style = replace(record.style, **values).validated()

    def global_layer_state(self) -> GLMOverlayStyle:
        """Return the common layer state represented by the first dataset."""
        if not self._records:
            return GLMOverlayStyle()
        return next(iter(self._records.values())).style.validated()

    def project_state(self) -> dict[str, object]:
        datasets = []
        for record in self.records:
            datasets.append(
                {
                    "provider": record.provider_id,
                    "key": record.key,
                    "source_files": [str(path) for path in record.source_paths],
                    "style": record.style.to_dict(),
                }
            )
        return {
            "format": "lmas-satellite-overlays-v1",
            "shared_energy_scale": bool(self.shared_energy_scale),
            "glm_backend": str(self.glm_backend),
            "datasets": datasets,
        }

    def restore_project_state(
        self,
        payload: Mapping[str, object] | None,
        *,
        project_directory: str | Path | None = None,
    ) -> None:
        self.clear()
        self.last_restore_errors.clear()
        values = dict(payload or {})
        self.shared_energy_scale = bool(values.get("shared_energy_scale", True))
        backend = str(values.get("glm_backend", "auto")).strip().lower()
        self.glm_backend = backend if backend in {"auto", "native", "glmtools"} else "auto"
        base = None if project_directory is None else Path(project_directory)
        for entry in values.get("datasets", ()) or ():
            if not isinstance(entry, Mapping) or str(entry.get("provider", "glm")) != "glm":
                continue
            raw_paths = entry.get("source_files", ()) or ()
            paths: list[Path] = []
            for raw in raw_paths:
                candidate = Path(str(raw)).expanduser()
                if not candidate.is_absolute() and base is not None:
                    candidate = base / candidate
                paths.append(candidate)
            missing = [path for path in paths if not path.is_file()]
            if missing:
                self.last_restore_errors.append(
                    "Missing GLM file(s): " + ", ".join(str(path) for path in missing)
                )
                continue
            try:
                records = self.add_glm_paths(paths)
            except Exception as exc:  # preserve the LMA project even if an overlay fails
                self.last_restore_errors.append(str(exc))
                continue
            style = GLMOverlayStyle.from_dict(entry.get("style"))
            for record in records:
                record.style = style


def group_glm_paths_by_platform(paths: Iterable[str | Path]) -> dict[str, tuple[Path, ...]]:
    """Group LCFA paths by spacecraft without opening them first."""
    grouped: dict[str, list[Path]] = {}
    for value in paths:
        path = Path(value).expanduser().resolve()
        match = _PLATFORM_RE.search(path.name)
        platform = match.group(1).upper() if match else "UNKNOWN"
        grouped.setdefault(platform, []).append(path)
    return {key: tuple(sorted(values)) for key, values in grouped.items()}


__all__ = [
    "GLMOverlayStyle",
    "SatelliteDatasetRecord",
    "SatelliteOverlayManager",
    "group_glm_paths_by_platform",
]
