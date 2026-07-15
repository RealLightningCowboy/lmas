from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path, PureWindowsPath
from typing import Any, Callable, Iterable, Mapping

import yaml
import numpy as np

from .. import __version__
from ..errors import ConfigurationError, DependencyError
from ..model import FilterSpec, LMAProject, PlotSpec
from .backends import normalize_reader_backend
from .readers import load_lma_files, synthetic_dataset

PROJECT_FORMAT = "lmas-project-v1.1"
V100_PROJECT_FORMAT = "lmas-project-v1.0"
V040_PROJECT_FORMAT = "lmas-project-v0.4"
V030_PROJECT_FORMAT = "lmas-project-v0.3"
V020_PROJECT_FORMAT = "lmas-project-v0.2"
V010_PROJECT_FORMAT = "lmas-project-v0.1"
LEGACY_PROJECT_FORMAT = "lmas-project-v0"
SUPPORTED_PROJECT_FORMATS = {
    PROJECT_FORMAT,
    V100_PROJECT_FORMAT,
    V040_PROJECT_FORMAT,
    V030_PROJECT_FORMAT,
    V020_PROJECT_FORMAT,
    V010_PROJECT_FORMAT,
    LEGACY_PROJECT_FORMAT,
}
PROJECT_SUFFIX = ".lmas-project.yaml"
PROJECT_SUFFIXES = (".lmas-project.yaml", ".lmas-project.yml", ".lmas.yaml", ".lmas.yml")


SOURCE_FILE_FINGERPRINT_ALGORITHM = "sha256-sampled-v1"
_SOURCE_FINGERPRINT_SAMPLE_BYTES = 256 * 1024
_SOURCE_SEARCH_MAX_DEPTH = 5
_SOURCE_SEARCH_SKIP_NAMES = {
    ".git", ".hg", ".svn", "__pycache__", "node_modules", ".cache",
    "outputs", "figures", "animations",
}


@dataclass(frozen=True)
class SourceFileReference:
    """Portable description of one Project input file.

    ``saved_path`` is retained exactly as written in the YAML.  ``filename``
    and the optional size/fingerprint metadata allow LMAS to relocate the same
    data file after a Project tree is moved to another directory or computer.
    """

    index: int
    saved_path: str
    filename: str
    size_bytes: int | None = None
    fingerprint: str | None = None


SourceLocator = Callable[[SourceFileReference], str | Path | None]


def _source_value(item: Any) -> str:
    """Accept legacy strings and the richer mapping form proposed in dev builds."""

    if isinstance(item, str):
        return item
    if isinstance(item, Mapping):
        for key in ("path", "saved_path", "source", "value"):
            value = item.get(key)
            if value not in (None, ""):
                return str(value)
    raise ConfigurationError(f"Invalid source_files entry: {item!r}")


def _source_filename(value: str) -> str:
    raw = str(value)
    native_name = Path(raw).name
    windows_name = PureWindowsPath(raw).name
    # On POSIX, Path treats a Windows path as one literal filename.  Prefer the
    # Windows basename when separators or a drive prefix are present.
    if "\\" in raw or PureWindowsPath(raw).drive:
        return windows_name or native_name
    return native_name or windows_name


def source_file_fingerprint(path: str | Path) -> str:
    """Return a fast, content-based identity for relocation checks.

    Hashing a complete multi-gigabyte LMA archive on every Project save would
    be unnecessarily expensive.  This versioned fingerprint includes the file
    size and unique samples from the beginning, middle, and end of the file.
    The complete loaded-source fingerprint stored beside it provides the final
    scientific identity check after loading.
    """

    source = Path(path).expanduser().resolve()
    size = source.stat().st_size
    chunk = _SOURCE_FINGERPRINT_SAMPLE_BYTES
    offsets = {
        0,
        max(0, (size - chunk) // 2),
        max(0, size - chunk),
    }
    digest = hashlib.sha256()
    digest.update(SOURCE_FILE_FINGERPRINT_ALGORITHM.encode("ascii") + b"\0")
    digest.update(str(size).encode("ascii") + b"\0")
    with source.open("rb") as stream:
        for offset in sorted(offsets):
            stream.seek(offset)
            block = stream.read(chunk)
            digest.update(str(offset).encode("ascii") + b"\0")
            digest.update(block)
    return f"{SOURCE_FILE_FINGERPRINT_ALGORITHM}:{digest.hexdigest()}"


def _metadata_for_source(source: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {"filename": source.name}
    try:
        metadata["size_bytes"] = int(source.stat().st_size)
        metadata["fingerprint"] = source_file_fingerprint(source)
    except OSError:
        # A Project should still be savable if a previously loaded removable
        # drive is temporarily unavailable.  Filename-only relocation remains
        # possible and the complete dataset fingerprint is retained below.
        pass
    return metadata


def _portable_source_value(source: Path, destination: Path) -> str:
    """Prefer a useful relative path without manufacturing root-spanning paths."""

    source = source.expanduser().resolve()
    base = destination.parent.resolve()
    try:
        common = Path(os.path.commonpath((str(source), str(base))))
    except ValueError:  # Different Windows drives.
        return str(source)
    source_root = Path(source.anchor)
    base_root = Path(base.anchor)
    if common not in {source_root, base_root}:
        try:
            return Path(os.path.relpath(source, base)).as_posix()
        except ValueError:
            pass
    return str(source)


def _metadata_entry(
    payload: Mapping[str, Any], index: int, saved_value: str
) -> SourceFileReference:
    raw_metadata = payload.get("source_file_metadata") or ()
    item: Mapping[str, Any] = {}
    if isinstance(raw_metadata, list) and index < len(raw_metadata):
        candidate = raw_metadata[index]
        if isinstance(candidate, Mapping):
            item = candidate
    filename = str(item.get("filename") or _source_filename(saved_value))
    size: int | None
    try:
        size = int(item["size_bytes"]) if item.get("size_bytes") is not None else None
    except (TypeError, ValueError):
        size = None
    fingerprint = str(item.get("fingerprint") or "") or None
    return SourceFileReference(
        index=index,
        saved_path=saved_value,
        filename=filename,
        size_bytes=size,
        fingerprint=fingerprint,
    )


def _saved_path_candidate(reference: SourceFileReference, project_file: Path) -> Path | None:
    raw = os.path.expandvars(reference.saved_path)
    native = Path(raw).expanduser()
    if native.is_absolute():
        return native
    windows = PureWindowsPath(raw)
    if windows.is_absolute():
        # A Windows absolute path cannot be opened as such on POSIX.  On
        # Windows, pathlib understands it normally.
        return Path(raw) if os.name == "nt" else None
    return project_file.parent / native


def _candidate_matches(reference: SourceFileReference, candidate: Path) -> bool:
    try:
        if not candidate.is_file():
            return False
        if reference.size_bytes is not None and candidate.stat().st_size != reference.size_bytes:
            return False
        if reference.fingerprint:
            return source_file_fingerprint(candidate) == reference.fingerprint
    except OSError:
        return False
    return True


def _search_roots(
    project_file: Path, data_roots: Iterable[str | Path]
) -> tuple[Path, ...]:
    values: list[Path] = [project_file.parent, project_file.parent.parent]
    values.extend(Path(value).expanduser() for value in data_roots if str(value).strip())
    for value in os.environ.get("LMAS_DATA_ROOTS", "").split(os.pathsep):
        if value.strip():
            values.append(Path(value).expanduser())
    resolved: list[Path] = []
    seen: set[str] = set()
    for value in values:
        try:
            candidate = value.resolve()
        except OSError:
            continue
        token = os.path.normcase(str(candidate))
        if candidate == Path(candidate.anchor):
            continue
        if token in seen or not candidate.is_dir():
            continue
        seen.add(token)
        resolved.append(candidate)
    return tuple(resolved)


def _filename_candidates(root: Path, filename: str) -> Iterable[Path]:
    # Cheap common layouts first.
    for relative in (
        Path(filename),
        Path("data") / filename,
        Path("data") / "lma" / filename,
        Path("lma") / filename,
        Path("LMA") / filename,
    ):
        candidate = root / relative
        if candidate.is_file():
            yield candidate
    root_depth = len(root.parts)
    try:
        for directory, names, files in os.walk(root):
            current = Path(directory)
            depth = len(current.parts) - root_depth
            names[:] = [
                name for name in names
                if name not in _SOURCE_SEARCH_SKIP_NAMES and not name.startswith(".")
            ]
            if depth >= _SOURCE_SEARCH_MAX_DEPTH:
                names[:] = []
            if filename in files:
                yield current / filename
    except OSError:
        return


def _resolve_source_reference(
    reference: SourceFileReference,
    project_file: Path,
    *,
    data_roots: Iterable[str | Path] = (),
    source_locator: SourceLocator | None = None,
) -> tuple[Path, str]:
    direct = _saved_path_candidate(reference, project_file)
    if direct is not None and _candidate_matches(reference, direct):
        return direct.expanduser().resolve(), "saved path"

    matches: list[Path] = []
    seen: set[str] = set()
    for root in _search_roots(project_file, data_roots):
        for candidate in _filename_candidates(root, reference.filename):
            token = os.path.normcase(str(candidate.resolve()))
            if token in seen:
                continue
            seen.add(token)
            if _candidate_matches(reference, candidate):
                matches.append(candidate.resolve())
    if matches and (reference.fingerprint or reference.size_bytes is not None or len(matches) == 1):
        # Multiple metadata-matched copies are scientifically equivalent.
        # Filename-only legacy Projects are auto-relocated only when the match
        # is unambiguous.
        matches.sort(key=lambda value: (len(value.parts), len(str(value))))
        return matches[0], "automatic relocation"

    if source_locator is not None:
        located = source_locator(reference)
        if located not in (None, ""):
            candidate = Path(located).expanduser()
            if candidate.is_file():
                # Explicit user relocation may point to a repackaged but
                # scientifically identical input.  The complete dataset
                # fingerprint is verified after loading.
                return candidate.resolve(), "user relocation"
            raise ConfigurationError(
                f"The selected replacement for {reference.filename} is not a file: {candidate}"
            )

    expected = reference.filename or reference.saved_path
    roots = ", ".join(str(value) for value in _search_roots(project_file, data_roots))
    hint = (
        " Set LMAS_DATA_ROOTS, place the data beside the Project, or open the "
        "Project in the GUI to locate the file."
    )
    raise ConfigurationError(
        f"Could not locate Project source file {expected!r}. Searched: {roots or 'Project directory'}."
        + hint
    )


def _portable_satellite_overlay_state(
    payload: Mapping[str, Any] | None, destination: Path
) -> dict[str, Any]:
    """Return overlay state with source paths relative to the Project when useful."""
    state = dict(payload or {})
    datasets: list[dict[str, Any]] = []
    for entry in state.get("datasets", ()) or ():
        if not isinstance(entry, Mapping):
            continue
        item = dict(entry)
        portable: list[str] = []
        for raw in item.get("source_files", ()) or ():
            source = Path(str(raw)).expanduser()
            try:
                source = source.resolve()
                portable.append(_portable_source_value(source, destination))
            except OSError:
                portable.append(str(raw))
        item["source_files"] = portable
        datasets.append(item)
    state["datasets"] = datasets
    return state


def _portable_network_overlay_state(
    payload: Mapping[str, Any] | None, destination: Path
) -> dict[str, Any]:
    """Return ground-network state with useful Project-relative source paths."""
    state = dict(payload or {})
    datasets: list[dict[str, Any]] = []
    for entry in state.get("datasets", ()) or ():
        if not isinstance(entry, Mapping):
            continue
        item = dict(entry)
        portable: list[str] = []
        for raw in item.get("source_files", ()) or ():
            source = Path(str(raw)).expanduser()
            try:
                source = source.resolve()
                portable.append(_portable_source_value(source, destination))
            except OSError:
                portable.append(str(raw))
        item["source_files"] = portable
        datasets.append(item)
    state["datasets"] = datasets
    return state


def save_project(project: LMAProject, path: str | Path) -> Path:
    destination = Path(path).expanduser()
    if not destination.name.lower().endswith((".yaml", ".yml")):
        destination = destination.with_name(destination.name + PROJECT_SUFFIX)
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_values: list[str] = []
    source_metadata: list[dict[str, Any]] = []
    for source in project.source_files:
        source = source.expanduser().resolve()
        source_values.append(_portable_source_value(source, destination))
        source_metadata.append(_metadata_for_source(source))
    dataset_identity: str | None = None
    if project.source_files:
        try:
            from ..polarity_product import dataset_fingerprint

            dataset_identity = dataset_fingerprint(project)
        except Exception:
            # Some third-party/adapted datasets may not expose every canonical
            # source-identity field.  File metadata still improves portability.
            dataset_identity = None
    payload = {
        "format": PROJECT_FORMAT,
        "lmas_version": __version__,
        "name": project.name,
        "source_files": source_values,
        "source_file_metadata": source_metadata,
        "dataset_fingerprint": dataset_identity,
        "reference": {
            "latitude": float(project.reference_latitude),
            "longitude": float(project.reference_longitude),
        },
        "reader": {
            "backend": project.reader_backend,
            "version": project.reader_backend_version,
            "details": dict(project.reader_details),
        },
        "filters": project.filters.to_dict(),
        "view": project.view_filters.to_dict(),
        "project_home": {
            str(name): [float(bounds[0]), float(bounds[1])]
            for name, bounds in dict(project.project_home_limits or {}).items()
        },
        "plot": project.plot.to_dict(),
        "selection": {
            "source_ids": (
                None
                if project.selected_source_ids is None
                else [int(value) for value in project.selected_source_ids]
            ),
            "color_norm_limits": (
                None
                if project.color_norm_limits is None
                else [float(value) for value in project.color_norm_limits]
            ),
        },
        "analysis": {
            "source_selection": dict(project.source_selection_state or {}),
            "satellite_overlays": _portable_satellite_overlay_state(
                project.satellite_overlay_state, destination
            ),
            "network_overlays": _portable_network_overlay_state(
                project.network_overlay_state, destination
            ),
        },
        "notes": project.notes,
        "demo": bool(
            not project.source_files
            and project.dataset.attrs.get("source") == "lmas.synthetic_dataset"
        ),
        "demo_event_count": int(project.event_count),
    }
    with destination.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(payload, stream, sort_keys=False, allow_unicode=True)
    project.project_path = destination
    return destination


def _migrate_plot_values(payload: dict[str, Any], project_format: str) -> dict[str, Any]:
    values = dict(payload.get("plot") or {})
    if project_format == LEGACY_PROJECT_FORMAT:
        # v0.1.0 exposed the restrained gray theme as "dark" and the black
        # theme as "space". Preserve the visual appearance of old projects
        # while adopting the corrected v0.1.1 names.
        legacy_theme = str(values.get("theme", "dark")).lower()
        values["theme"] = {
            "dark": "space",
            "space": "dark",
        }.get(legacy_theme, legacy_theme)
    if project_format not in {PROJECT_FORMAT, V100_PROJECT_FORMAT, V040_PROJECT_FORMAT}:
        # Auto-fit spatial panels became the standard first-view behavior for
        # the release-candidate line. Older project formats commonly persisted
        # ``false`` from the former inherited default, so migrate those files
        # to the new checked default. Current-format projects preserve an
        # explicit user choice to disable Auto-fit.
        if values.get("auto_fit_spatial") is False:
            values["auto_fit_spatial"] = True

        # LMAS releases through v0.3.8 used 15 ms as the built-in trail and
        # afterimage defaults.  Those values were commonly persisted into
        # projects even when the user never selected them. Files written in
        # v0.4 or later retain an explicitly chosen 15 ms value, while older
        # formats migrate the legacy default to the current 30 ms.
        for key in ("three_d_trail_ms", "three_d_afterimage_ms"):
            try:
                legacy_value = float(values.get(key))
            except (TypeError, ValueError):
                continue
            if legacy_value == 15.0:
                values[key] = 30.0
    return values


def load_project(
    path: str | Path,
    *,
    reader_backend: str = "auto",
    data_roots: Iterable[str | Path] = (),
    source_locator: SourceLocator | None = None,
) -> LMAProject:
    source = Path(path).expanduser().resolve()
    try:
        with source.open("r", encoding="utf-8") as stream:
            payload = yaml.safe_load(stream) or {}
    except OSError as exc:
        raise ConfigurationError(f"Could not read LMAS project {source}: {exc}") from exc
    project_format = str(payload.get("format") or "")
    if project_format not in SUPPORTED_PROJECT_FORMATS:
        raise ConfigurationError(f"Unsupported LMAS project format in {source}")
    raw_sources = payload.get("source_files", []) or []
    if not isinstance(raw_sources, list):
        raise ConfigurationError(f"source_files must be a list in {source}")
    source_references = [
        _metadata_entry(payload, index, _source_value(item))
        for index, item in enumerate(raw_sources)
    ]
    resolution_records: list[dict[str, str]] = []
    files: list[Path] = []
    for item in source_references:
        resolved, method = _resolve_source_reference(
            item,
            source,
            data_roots=data_roots,
            source_locator=source_locator,
        )
        files.append(resolved)
        resolution_records.append({
            "filename": item.filename,
            "method": method,
        })
    reference = payload.get("reference") or {}
    reader = payload.get("reader") or {}
    requested_backend = normalize_reader_backend(reader_backend)
    saved_backend = normalize_reader_backend(reader.get("backend") or "auto")
    effective_backend = saved_backend if requested_backend == "auto" else requested_backend
    if payload.get("demo"):
        project = LMAProject(
            dataset=synthetic_dataset(count=int(payload.get("demo_event_count", 1200))),
            name=payload.get("name") or source.stem,
            reference_latitude=reference.get("latitude"),
            reference_longitude=reference.get("longitude"),
            reader_backend="synthetic",
            reader_backend_version="1",
            reader_details={"source": "project-demo"},
        )
    else:
        try:
            project = load_lma_files(
                files,
                name=payload.get("name") or source.stem,
                reference_latitude=reference.get("latitude"),
                reference_longitude=reference.get("longitude"),
                reader_backend=effective_backend,
            )
        except DependencyError:
            if requested_backend != "auto" or saved_backend != "pyxlma":
                raise
            project = load_lma_files(
                files,
                name=payload.get("name") or source.stem,
                reference_latitude=reference.get("latitude"),
                reference_longitude=reference.get("longitude"),
                reader_backend="native",
            )
            project.reader_details["fallback_from"] = "pyxlma"
            project.reader_details["fallback_reason"] = "pyxlma unavailable"
    saved_dataset_identity = str(payload.get("dataset_fingerprint") or "")
    if saved_dataset_identity:
        try:
            from ..polarity_product import dataset_fingerprint

            actual_dataset_identity = dataset_fingerprint(project)
        except Exception as exc:
            raise ConfigurationError(
                "The Project contains a dataset fingerprint, but LMAS could not "
                "verify the located source data."
            ) from exc
        if actual_dataset_identity != saved_dataset_identity:
            raise ConfigurationError(
                "The located source data do not match this Project's saved "
                "dataset fingerprint; the Project was not opened."
            )
    if resolution_records:
        project.reader_details["project_source_resolution"] = resolution_records
    if reader:
        project.reader_details = {**project.reader_details, **{
            "project_saved_backend": saved_backend,
            "project_saved_version": str(reader.get("version") or "unknown"),
        }}
    saved_filters = FilterSpec.from_dict(payload.get("filters"))
    if project_format in {PROJECT_FORMAT, V100_PROJECT_FORMAT, V040_PROJECT_FORMAT, V030_PROJECT_FORMAT, V020_PROJECT_FORMAT}:
        project.filters = saved_filters
        project.view_filters = FilterSpec.from_dict(payload.get("view"))
    else:
        # v0/v0.1 stored quality and linked-view constraints in one FilterSpec.
        # Split them so the complete dataset remains loaded and the saved view is
        # restored non-destructively after the first GUI figure is constructed.
        project.filters = FilterSpec(
            minimum_stations=saved_filters.minimum_stations,
            maximum_chi2=saved_filters.maximum_chi2,
            minimum_power=saved_filters.minimum_power,
            maximum_power=saved_filters.maximum_power,
        ).validated()
        project.view_filters = FilterSpec(
            start_time=saved_filters.start_time,
            end_time=saved_filters.end_time,
            minimum_stations=None,
            maximum_chi2=None,
            minimum_altitude_km=saved_filters.minimum_altitude_km,
            maximum_altitude_km=saved_filters.maximum_altitude_km,
            minimum_x_km=saved_filters.minimum_x_km,
            maximum_x_km=saved_filters.maximum_x_km,
            minimum_y_km=saved_filters.minimum_y_km,
            maximum_y_km=saved_filters.maximum_y_km,
        ).validated()
    project_home = payload.get("project_home") or {}
    if isinstance(project_home, dict):
        normalized_home: dict[str, tuple[float, float]] = {}
        for name, bounds in project_home.items():
            try:
                low, high = sorted((float(bounds[0]), float(bounds[1])))
            except (TypeError, ValueError, IndexError):
                continue
            if np.isfinite(low) and np.isfinite(high) and high > low:
                normalized_home[str(name)] = (low, high)
        project.project_home_limits = normalized_home
    project.plot = PlotSpec.from_dict(_migrate_plot_values(payload, project_format))
    if project_format in {PROJECT_FORMAT, V100_PROJECT_FORMAT, V040_PROJECT_FORMAT, V030_PROJECT_FORMAT}:
        selection = payload.get("selection") or {}
        # Historical projects sometimes stored the sources visible inside the
        # saved view as an exact membership subset.  That made surrounding
        # sources impossible to recover by zooming out after a fresh load.
        # Saved view bounds now define Project Home while the full dataset stays
        # available; intentional subsets belong in named source groups instead.
        project.selected_source_ids = None
        color_limits = selection.get("color_norm_limits")
        project.color_norm_limits = (
            None
            if color_limits in (None, [])
            else (float(color_limits[0]), float(color_limits[1]))
        )
    analysis = payload.get("analysis") or {}
    source_selection = analysis.get("source_selection") or {}
    if isinstance(source_selection, dict):
        project.source_selection_state = dict(source_selection)
    satellite_overlays = analysis.get("satellite_overlays") or {}
    if isinstance(satellite_overlays, dict):
        project.satellite_overlay_state = dict(satellite_overlays)
    network_overlays = analysis.get("network_overlays") or {}
    if isinstance(network_overlays, dict):
        project.network_overlay_state = dict(network_overlays)
    project.project_path = source
    project.notes = str(payload.get("notes") or "")
    return project
