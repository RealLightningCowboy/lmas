from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import yaml

from . import __version__
from .errors import ConfigurationError
from .model import FilterSpec, PlotSpec
from .paths import user_config_directory

PROFILE_FORMAT = "lmas-profile-v1.0"
V030_PROFILE_FORMAT = "lmas-profile-v0.3"
V020_PROFILE_FORMAT = "lmas-profile-v0.2"
SUPPORTED_PROFILE_FORMATS = {
    PROFILE_FORMAT,
    V030_PROFILE_FORMAT,
    V020_PROFILE_FORMAT,
}
BUILTIN_STARTUP_NAME = "Startup"


def startup_filters() -> FilterSpec:
    """Built-in first-view source-quality profile."""
    return FilterSpec(
        minimum_stations=6,
        maximum_chi2=1.0,
        minimum_altitude_km=None,
        maximum_altitude_km=None,
    ).validated()


def startup_plot() -> PlotSpec:
    """Built-in first-view plotting profile."""
    return PlotSpec(
        layout="landscape",
        theme="dark",
        color_by="time",
        cmap="turbo",
        point_size=3.0,
        show_stations=True,
        show_station_labels=False,
        auto_fit_spatial=True,
        remap_time_colors=True,
        north_south_viewpoint="south",
        east_west_viewpoint="east",
        depth_mode="spatial",
        dpi=100,
        saved_figure_dpi=300,
    ).validated()


@dataclass(frozen=True)
class AnalysisProfile:
    name: str
    filters: FilterSpec
    plot: PlotSpec
    built_in: bool = False

    def to_dict(self) -> dict:
        return {
            "format": PROFILE_FORMAT,
            "lmas_version": __version__,
            "name": self.name,
            "filters": self.filters.to_dict(),
            "plot": self.plot.to_dict(),
        }

    @classmethod
    def from_dict(cls, values: dict, *, built_in: bool = False) -> "AnalysisProfile":
        profile_format = str(values.get("format") or V020_PROFILE_FORMAT)
        if profile_format not in SUPPORTED_PROFILE_FORMATS:
            raise ConfigurationError("Unsupported LMAS profile format")
        name = str(values.get("name") or "Unnamed profile").strip()
        if not name:
            raise ConfigurationError("Profile name cannot be empty")
        plot_values = dict(values.get("plot") or {})
        if profile_format == V020_PROFILE_FORMAT:
            # Older profiles commonly serialized ``false`` from the former
            # inherited Auto-fit default. Migrate those files to the current
            # checked default; current-format profiles preserve an explicit
            # user choice to disable Auto-fit.
            if plot_values.get("auto_fit_spatial") is False:
                plot_values["auto_fit_spatial"] = True

            # v0.2 profiles often serialized the old 15 ms built-in defaults.
            # New-format profiles preserve an explicit 15 ms user choice.
            for key in ("three_d_trail_ms", "three_d_afterimage_ms"):
                try:
                    legacy_value = float(plot_values.get(key))
                except (TypeError, ValueError):
                    continue
                if legacy_value == 15.0:
                    plot_values[key] = 30.0
        return cls(
            name=name,
            filters=FilterSpec.from_dict(values.get("filters")),
            plot=PlotSpec.from_dict(plot_values),
            built_in=built_in,
        )


def startup_profile() -> AnalysisProfile:
    return AnalysisProfile(
        name=BUILTIN_STARTUP_NAME,
        filters=startup_filters(),
        plot=startup_plot(),
        built_in=True,
    )


def default_profile_directory() -> Path:
    return user_config_directory() / "profiles"


def _safe_stem(name: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip()).strip("._")
    return result or "profile"


def _normalized_profile_path(path: str | Path) -> Path:
    destination = Path(path).expanduser()
    lower = destination.name.lower()
    if not lower.endswith((".yaml", ".yml")):
        destination = destination.with_name(destination.name + ".lmas-profile.yaml")
    return destination.resolve()


class ProfileStore:
    """Reusable LMAS profiles, including user-chosen external save locations.

    The small registry file keeps profiles saved outside LMAS's default profile
    directory discoverable in the Profiles menu and by the CLI on later runs.
    """

    REGISTRY_NAME = "profile_locations.yaml"

    def __init__(self, directory: str | Path | None = None) -> None:
        self.directory = Path(directory or default_profile_directory()).expanduser().resolve()
        self.registry_path = self.directory / self.REGISTRY_NAME

    def _path_for(self, name: str) -> Path:
        return self.directory / f"{_safe_stem(name)}.lmas-profile.yaml"

    def _registered_paths(self) -> tuple[Path, ...]:
        if not self.registry_path.is_file():
            return ()
        try:
            values = yaml.safe_load(self.registry_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            return ()
        if isinstance(values, dict):
            paths = values.get("paths", [])
        elif isinstance(values, list):
            paths = values
        else:
            paths = []
        if not isinstance(paths, list):
            return ()
        result: list[Path] = []
        for value in paths:
            try:
                result.append(Path(str(value)).expanduser().resolve())
            except (OSError, RuntimeError, ValueError):
                continue
        return tuple(result)

    def _write_registered_paths(self, paths: Iterable[Path]) -> None:
        unique: list[str] = []
        seen: set[str] = set()
        for path in paths:
            resolved = str(Path(path).expanduser().resolve())
            key = resolved.casefold()
            if key not in seen:
                seen.add(key)
                unique.append(resolved)
        self.directory.mkdir(parents=True, exist_ok=True)
        temp = self.registry_path.with_suffix(self.registry_path.suffix + ".tmp")
        temp.write_text(
            yaml.safe_dump({"paths": unique}, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        temp.replace(self.registry_path)

    def register(self, path: str | Path) -> Path:
        destination = _normalized_profile_path(path)
        paths = list(self._registered_paths())
        paths = [item for item in paths if str(item).casefold() != str(destination).casefold()]
        paths.append(destination)
        self._write_registered_paths(paths)
        return destination

    def _candidate_paths(self) -> tuple[Path, ...]:
        # Explicitly chosen locations take precedence over stale/default copies
        # with the same profile name.
        candidates = list(self._registered_paths())
        if self.directory.is_dir():
            candidates.extend(sorted(self.directory.glob("*.lmas-profile.y*ml")))
        result: list[Path] = []
        seen: set[str] = set()
        for path in candidates:
            try:
                resolved = path.expanduser().resolve()
            except (OSError, RuntimeError, ValueError):
                continue
            key = str(resolved).casefold()
            if key in seen or resolved == self.registry_path:
                continue
            seen.add(key)
            result.append(resolved)
        return tuple(result)

    def list(self) -> tuple[AnalysisProfile, ...]:
        profiles: list[AnalysisProfile] = [startup_profile()]
        seen_names = {BUILTIN_STARTUP_NAME.casefold()}
        for path in self._candidate_paths():
            if not path.is_file():
                continue
            try:
                profile = self.load_file(path)
            except (OSError, ConfigurationError, yaml.YAMLError):
                continue
            key = profile.name.casefold()
            if key in seen_names:
                continue
            seen_names.add(key)
            profiles.append(profile)
        return tuple(profiles)

    def names(self) -> tuple[str, ...]:
        return tuple(profile.name for profile in self.list())

    def _find_path(self, name: str) -> Path | None:
        target = name.strip().casefold()
        for path in self._candidate_paths():
            if not path.is_file():
                continue
            try:
                if self.load_file(path).name.casefold() == target:
                    return path
            except (OSError, ConfigurationError, yaml.YAMLError):
                continue
        return None

    def get(self, name: str) -> AnalysisProfile:
        if name.strip().casefold() == BUILTIN_STARTUP_NAME.casefold():
            return startup_profile()
        path = self._find_path(name)
        if path is not None:
            return self.load_file(path)
        raise ConfigurationError(f"LMAS profile not found: {name}")

    def save(
        self,
        profile: AnalysisProfile,
        *,
        path: str | Path | None = None,
        overwrite: bool = True,
        register: bool = True,
    ) -> Path:
        if profile.name.strip().casefold() == BUILTIN_STARTUP_NAME.casefold():
            raise ConfigurationError("The built-in Startup profile cannot be overwritten")
        destination = _normalized_profile_path(path or self._path_for(profile.name))
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not overwrite:
            raise ConfigurationError(f"Profile already exists: {destination}")
        temp = destination.with_suffix(destination.suffix + ".tmp")
        temp.write_text(
            yaml.safe_dump(profile.to_dict(), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        temp.replace(destination)
        if register and destination.parent != self.directory:
            self.register(destination)
        return destination

    def load_file(self, path: str | Path) -> AnalysisProfile:
        source = Path(path).expanduser().resolve()
        try:
            values = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        except OSError as exc:
            raise ConfigurationError(f"Could not read LMAS profile {source}: {exc}") from exc
        return AnalysisProfile.from_dict(values)

    def import_file(self, path: str | Path, *, overwrite: bool = False) -> Path:
        profile = self.load_file(path)
        return self.save(profile, overwrite=overwrite)

    def export(self, name: str, path: str | Path) -> Path:
        profile = self.get(name)
        destination = _normalized_profile_path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            yaml.safe_dump(profile.to_dict(), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        return destination

    def delete(self, name: str) -> None:
        if name.strip().casefold() == BUILTIN_STARTUP_NAME.casefold():
            raise ConfigurationError("The built-in Startup profile cannot be deleted")
        path = self._find_path(name)
        if path is None:
            raise ConfigurationError(f"LMAS profile not found: {name}")
        try:
            path.unlink()
        except OSError as exc:
            raise ConfigurationError(f"Could not delete profile {name}: {exc}") from exc
        registered = [
            item
            for item in self._registered_paths()
            if str(item).casefold() != str(path).casefold()
        ]
        self._write_registered_paths(registered)


def profile_from_specs(name: str, filters: FilterSpec, plot: PlotSpec) -> AnalysisProfile:
    return AnalysisProfile(name=name.strip(), filters=filters.validated(), plot=plot.validated())


__all__ = [
    "AnalysisProfile",
    "BUILTIN_STARTUP_NAME",
    "ProfileStore",
    "profile_from_specs",
    "startup_filters",
    "startup_plot",
    "startup_profile",
]
