"""Serializable state for ground lightning-location-network overlays."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from matplotlib.colors import is_color_like

from .model import NetworkObservation
from .readers import NetworkCSVOptions, read_network_csv


@dataclass(slots=True)
class NetworkOverlayStyle:
    enabled: bool = True
    show_events: bool = True
    show_uncertainty: bool = False
    show_time_rail: bool = True
    show_legend: bool = True
    follow_spatial_view: bool = True
    marker_size: float = 42.0
    scale_by_peak_current: bool = True
    marker_alpha: float = 0.95
    marker_edge_width: float = 0.75
    positive_color: str = "#ef5350"
    negative_color: str = "#2196f3"
    intracloud_color: str = "#ffca28"
    unknown_color: str = "#b0bec5"
    ellipse_color: str = "auto"
    ellipse_alpha: float = 0.65
    ellipse_line_width: float = 0.85
    event_zorder: float = 1.35
    ellipse_zorder: float = 1.20
    time_rail_zorder: float = 3.20
    time_rail_marker_size: float = 28.0
    minimum_absolute_peak_current_ka: float | None = None
    minimum_sensor_count: int | None = None
    show_positive: bool = True
    show_negative: bool = True
    show_unknown_polarity: bool = True
    show_cg: bool = True
    show_ic: bool = True
    show_other_types: bool = True
    maximum_interactive_events: int = 5000
    maximum_interactive_ellipses: int = 1000

    def validated(self) -> "NetworkOverlayStyle":
        def color(value: object, fallback: str) -> str:
            text = str(value or fallback)
            return text if is_color_like(text) else fallback

        minimum_current = self.minimum_absolute_peak_current_ka
        if minimum_current is not None:
            minimum_current = max(0.0, float(minimum_current))
        minimum_sensors = self.minimum_sensor_count
        if minimum_sensors is not None:
            minimum_sensors = max(0, int(minimum_sensors))
        ellipse_color = str(self.ellipse_color or "auto")
        if ellipse_color.lower() != "auto" and not is_color_like(ellipse_color):
            ellipse_color = "auto"
        return NetworkOverlayStyle(
            enabled=bool(self.enabled),
            show_events=bool(self.show_events),
            show_uncertainty=bool(self.show_uncertainty),
            show_time_rail=bool(self.show_time_rail),
            show_legend=bool(self.show_legend),
            follow_spatial_view=bool(self.follow_spatial_view),
            marker_size=max(1.0, float(self.marker_size)),
            scale_by_peak_current=bool(self.scale_by_peak_current),
            marker_alpha=float(np.clip(float(self.marker_alpha), 0.0, 1.0)),
            marker_edge_width=max(0.0, float(self.marker_edge_width)),
            positive_color=color(self.positive_color, "#ef5350"),
            negative_color=color(self.negative_color, "#2196f3"),
            intracloud_color=color(self.intracloud_color, "#ffca28"),
            unknown_color=color(self.unknown_color, "#b0bec5"),
            ellipse_color=ellipse_color,
            ellipse_alpha=float(np.clip(float(self.ellipse_alpha), 0.0, 1.0)),
            ellipse_line_width=max(0.0, float(self.ellipse_line_width)),
            event_zorder=float(self.event_zorder),
            ellipse_zorder=float(self.ellipse_zorder),
            time_rail_zorder=float(self.time_rail_zorder),
            time_rail_marker_size=max(1.0, float(self.time_rail_marker_size)),
            minimum_absolute_peak_current_ka=minimum_current,
            minimum_sensor_count=minimum_sensors,
            show_positive=bool(self.show_positive),
            show_negative=bool(self.show_negative),
            show_unknown_polarity=bool(self.show_unknown_polarity),
            show_cg=bool(self.show_cg),
            show_ic=bool(self.show_ic),
            show_other_types=bool(self.show_other_types),
            maximum_interactive_events=max(0, int(self.maximum_interactive_events)),
            maximum_interactive_ellipses=max(0, int(self.maximum_interactive_ellipses)),
        )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self.validated())
        payload["format"] = "lmas-network-overlay-style-v1"
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "NetworkOverlayStyle":
        values = dict(payload or {})
        values.pop("format", None)
        accepted = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in values.items() if key in accepted}).validated()


@dataclass(slots=True)
class NetworkDatasetRecord:
    key: str
    observation: NetworkObservation
    options: NetworkCSVOptions = field(default_factory=NetworkCSVOptions)
    style: NetworkOverlayStyle = field(default_factory=NetworkOverlayStyle)

    @property
    def display_name(self) -> str:
        return self.observation.identity.display_name

    @property
    def provider_id(self) -> str:
        return self.observation.identity.provider_id

    @property
    def source_paths(self) -> tuple[Path, ...]:
        return tuple(item.path for item in self.observation.identity.source_files)


class NetworkOverlayManager:
    """Own normalized ground-network datasets and serializable styles."""

    def __init__(self) -> None:
        self._records: dict[str, NetworkDatasetRecord] = {}
        self.last_restore_errors: list[str] = []
        self._renderer = None

    @property
    def records(self) -> tuple[NetworkDatasetRecord, ...]:
        return tuple(self._records[key] for key in sorted(self._records))

    @property
    def has_data(self) -> bool:
        return bool(self._records)

    def record(self, key: str) -> NetworkDatasetRecord:
        return self._records[str(key)]

    @staticmethod
    def dataset_key(observation: NetworkObservation) -> str:
        identity = observation.identity
        digest = hashlib.sha1(
            "\0".join(str(item.path) for item in identity.source_files).encode("utf-8")
        ).hexdigest()[:12]
        return (
            f"network:{identity.provider_id}:{identity.observation_start_ns}:"
            f"{identity.observation_end_ns}:{digest}"
        )

    def add_csv_paths(
        self,
        paths: str | Path | Sequence[str | Path],
        *,
        options: NetworkCSVOptions | None = None,
    ) -> NetworkDatasetRecord:
        options = options or NetworkCSVOptions()
        observation = read_network_csv(paths, options=options)
        key = self.dataset_key(observation)
        prior = self._records.get(key)
        style = prior.style if prior is not None else NetworkOverlayStyle()
        if prior is None and self._records:
            global_style = self.global_layer_state()
            style = replace(
                style,
                show_events=global_style.show_events,
                show_uncertainty=global_style.show_uncertainty,
                show_time_rail=global_style.show_time_rail,
                show_legend=global_style.show_legend,
            ).validated()
        record = NetworkDatasetRecord(key, observation, options, style)
        self._records[key] = record
        return record

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

    def global_layer_state(self) -> NetworkOverlayStyle:
        if not self._records:
            return NetworkOverlayStyle()
        first = next(iter(self._records.values())).style
        return first

    def set_global_layer_state(
        self,
        *,
        show_events: bool | None = None,
        show_uncertainty: bool | None = None,
        show_time_rail: bool | None = None,
        show_legend: bool | None = None,
    ) -> None:
        updates = {
            key: bool(value)
            for key, value in {
                "show_events": show_events,
                "show_uncertainty": show_uncertainty,
                "show_time_rail": show_time_rail,
                "show_legend": show_legend,
            }.items()
            if value is not None
        }
        for record in self._records.values():
            record.style = replace(record.style, **updates).validated()

    def project_state(self) -> dict[str, object]:
        datasets: list[dict[str, object]] = []
        for record in self.records:
            datasets.append(
                {
                    "provider": record.provider_id,
                    "key": record.key,
                    "source_files": [str(path) for path in record.source_paths],
                    "options": record.options.to_dict(),
                    "style": record.style.to_dict(),
                }
            )
        return {"format": "lmas-network-overlays-v1", "datasets": datasets}

    def restore_project_state(
        self,
        payload: Mapping[str, object] | None,
        *,
        project_directory: str | Path | None = None,
    ) -> None:
        self.clear()
        self.last_restore_errors.clear()
        base = None if project_directory is None else Path(project_directory)
        for entry in dict(payload or {}).get("datasets", ()) or ():
            if not isinstance(entry, Mapping):
                continue
            paths: list[Path] = []
            for raw in entry.get("source_files", ()) or ():
                candidate = Path(str(raw)).expanduser()
                if not candidate.is_absolute() and base is not None:
                    candidate = base / candidate
                paths.append(candidate)
            missing = [path for path in paths if not path.is_file()]
            if missing:
                self.last_restore_errors.append(
                    "Missing network file(s): " + ", ".join(str(path) for path in missing)
                )
                continue
            try:
                options = NetworkCSVOptions.from_dict(entry.get("options"))
                record = self.add_csv_paths(paths, options=options)
                record.style = NetworkOverlayStyle.from_dict(entry.get("style"))
            except Exception as exc:
                self.last_restore_errors.append(str(exc))


__all__ = ["NetworkDatasetRecord", "NetworkOverlayManager", "NetworkOverlayStyle"]
