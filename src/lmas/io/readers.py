from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tarfile
from typing import Iterable, Mapping, Sequence

import numpy as np
import xarray as xr

from ..errors import ArchiveMemberSelectionRequired, DatasetError, DependencyError
from ..model import EVENT_DIM, LMAProject, validate_dataset
from ..source_store import LmaSourceStore
from .backends import read_with_backend
from ..profiles import startup_filters, startup_plot
from ..paths import user_cache_directory

NETCDF_SUFFIXES = {".nc", ".netcdf"}
ARCHIVE_SUFFIXES = (".tar.gz", ".tgz", ".tar")
LMA_MEMBER_SUFFIXES = (".dat", ".dat.gz", ".nc", ".netcdf")


def _resolved_files(paths: Iterable[str | Path]) -> tuple[Path, ...]:
    files = tuple(Path(path).expanduser().resolve() for path in paths)
    if not files:
        raise DatasetError("No LMA input files were provided")
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise DatasetError("LMA input file(s) not found: " + ", ".join(missing))
    return files


def is_lma_archive(path: str | Path) -> bool:
    name = Path(path).name.lower()
    return name.endswith(ARCHIVE_SUFFIXES)


def archive_cache_root() -> Path:
    return user_cache_directory() / "archives"


def _archive_cache_key(path: Path) -> str:
    stat = path.stat()
    token = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8")
    return hashlib.sha256(token).hexdigest()[:20]


def list_archive_lma_members(path: str | Path) -> tuple[str, ...]:
    archive = Path(path).expanduser().resolve()
    try:
        with tarfile.open(archive, "r:*") as stream:
            members = [
                item.name
                for item in stream.getmembers()
                if item.isfile() and item.name.lower().endswith(LMA_MEMBER_SUFFIXES)
            ]
    except (OSError, tarfile.TarError) as exc:
        raise DatasetError(f"Could not inspect LMA archive {archive}: {exc}") from exc
    members = sorted(dict.fromkeys(members))
    if not members:
        raise DatasetError(
            f"Archive {archive} contains no supported .dat, .dat.gz, or NetCDF LMA files"
        )
    return tuple(members)


def _safe_member_destination(root: Path, member_name: str) -> Path:
    destination = (root / member_name).resolve()
    try:
        destination.relative_to(root.resolve())
    except ValueError as exc:
        raise DatasetError(f"Unsafe archive member path: {member_name}") from exc
    return destination


def extract_archive_members(
    path: str | Path,
    members: Sequence[str],
    *,
    cache_root: str | Path | None = None,
) -> tuple[Path, ...]:
    archive = Path(path).expanduser().resolve()
    root = Path(cache_root or archive_cache_root()).expanduser() / _archive_cache_key(archive)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    requested = tuple(dict.fromkeys(str(value) for value in members))
    available = set(list_archive_lma_members(archive))
    missing = [value for value in requested if value not in available]
    if missing:
        raise DatasetError(
            f"Archive member(s) not found in {archive}: " + ", ".join(missing)
        )

    outputs = tuple(_safe_member_destination(root, name) for name in requested)
    if all(path.is_file() for path in outputs):
        return outputs

    try:
        with tarfile.open(archive, "r:*") as stream:
            by_name = {item.name: item for item in stream.getmembers()}
            for name, destination in zip(requested, outputs, strict=True):
                if destination.is_file():
                    continue
                item = by_name[name]
                source = stream.extractfile(item)
                if source is None:
                    raise DatasetError(f"Could not read archive member {name}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(destination.suffix + ".partial")
                with temporary.open("wb") as target:
                    while True:
                        block = source.read(1024 * 1024)
                        if not block:
                            break
                        target.write(block)
                temporary.replace(destination)
    except (OSError, tarfile.TarError, KeyError) as exc:
        raise DatasetError(f"Could not extract LMA archive {archive}: {exc}") from exc

    manifest = {
        "archive": str(archive),
        "archive_size": archive.stat().st_size,
        "archive_mtime_ns": archive.stat().st_mtime_ns,
        "members": list(requested),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return outputs


def resolve_lma_inputs(
    paths: Iterable[str | Path],
    *,
    archive_members: Mapping[str | Path, Sequence[str] | str] | None = None,
) -> tuple[tuple[Path, ...], dict[str, tuple[str, ...]]]:
    """Resolve direct and archive inputs to reader-ready files.

    The returned provenance mapping records the archive members used from each
    original bundle. Archives with more than one plausible LMA member require a
    caller choice; pass ``"__all__"`` to load every member.
    """

    original = _resolved_files(paths)
    choices = {
        str(Path(key).expanduser().resolve()): value
        for key, value in (archive_members or {}).items()
    }
    actual: list[Path] = []
    provenance: dict[str, tuple[str, ...]] = {}
    for source in original:
        if not is_lma_archive(source):
            actual.append(source)
            continue
        candidates = list_archive_lma_members(source)
        selected = choices.get(str(source))
        if selected is None:
            if len(candidates) > 1:
                raise ArchiveMemberSelectionRequired(source, candidates)
            members = candidates
        elif isinstance(selected, str):
            members = candidates if selected == "__all__" else (selected,)
        else:
            members = tuple(selected)
        actual.extend(extract_archive_members(source, members))
        provenance[str(source)] = tuple(members)
    return tuple(actual), provenance


def load_netcdf(path: str | Path) -> xr.Dataset:
    source = Path(path).expanduser().resolve()
    try:
        dataset = xr.load_dataset(source)
    except Exception as exc:
        raise DatasetError(f"Could not read NetCDF LMA dataset {source}: {exc}") from exc
    return validate_dataset(dataset)


def load_lma_files(
    paths: Iterable[str | Path],
    *,
    name: str | None = None,
    reference_latitude: float | None = None,
    reference_longitude: float | None = None,
    archive_members: Mapping[str | Path, Sequence[str] | str] | None = None,
    reader_backend: str = "auto",
) -> LMAProject:
    """Load solved LMA source files or compressed LMA bundles.

    The default ``auto`` backend selects the LMAS-native reader.  The optional
    ``pyxlma`` compatibility backend is available when pyxlma is installed.
    Archive members are resolved before the selected backend is invoked.
    """

    source_files = _resolved_files(paths)
    reader_files, archive_provenance = resolve_lma_inputs(
        source_files, archive_members=archive_members
    )
    result = read_with_backend(reader_files, backend=reader_backend)
    provenance_attrs = {
        "lmas_original_inputs": json.dumps([str(path) for path in source_files])
    }
    if archive_provenance:
        provenance_attrs["lmas_archive_members"] = json.dumps(archive_provenance)
    store = result.store.with_attrs(**provenance_attrs)
    dataset = store.to_xarray()
    project_name = name or source_files[0].name
    for suffix in (".tar.gz", ".dat.gz", ".netcdf", ".dat", ".tgz", ".tar", ".nc"):
        if project_name.lower().endswith(suffix):
            project_name = project_name[: -len(suffix)]
            break
    return LMAProject(
        dataset=dataset,
        source_files=source_files,
        name=project_name,
        reference_latitude=reference_latitude,
        reference_longitude=reference_longitude,
        filters=startup_filters(),
        plot=startup_plot(),
        reader_backend=result.backend,
        reader_backend_version=result.backend_version,
        reader_details=result.details,
    )


def project_from_xarray(
    dataset: xr.Dataset,
    *,
    name: str = "xarray LMA dataset",
    source_files: Iterable[str | Path] = (),
    reference_latitude: float | None = None,
    reference_longitude: float | None = None,
    reader_backend: str = "xarray",
    reader_backend_version: str = "1",
) -> LMAProject:
    """Adapt a pyxlma-style or other compatible xarray Dataset into LMAS."""

    validated = validate_dataset(dataset)
    store = LmaSourceStore.from_xarray(validated).with_attrs(
        lmas_reader_backend=str(reader_backend),
        lmas_reader_backend_version=str(reader_backend_version),
    )
    return LMAProject(
        dataset=store.to_xarray(),
        source_files=tuple(Path(path).expanduser() for path in source_files),
        name=name,
        reference_latitude=reference_latitude,
        reference_longitude=reference_longitude,
        filters=startup_filters(),
        plot=startup_plot(),
        reader_backend=str(reader_backend),
        reader_backend_version=str(reader_backend_version),
        reader_details={"adapter": "xarray"},
    )


def project_from_source_store(
    store: LmaSourceStore,
    *,
    name: str = "LMAS source store",
    source_files: Iterable[str | Path] = (),
    reference_latitude: float | None = None,
    reference_longitude: float | None = None,
    reader_backend: str = "source-store",
    reader_backend_version: str = "1",
) -> LMAProject:
    """Create an LMAS project from an already normalized source store."""

    prepared = store.with_attrs(
        lmas_reader_backend=str(reader_backend),
        lmas_reader_backend_version=str(reader_backend_version),
    )
    return LMAProject(
        dataset=prepared.to_xarray(),
        source_files=tuple(Path(path).expanduser() for path in source_files),
        name=name,
        reference_latitude=reference_latitude,
        reference_longitude=reference_longitude,
        filters=startup_filters(),
        plot=startup_plot(),
        reader_backend=str(reader_backend),
        reader_backend_version=str(reader_backend_version),
        reader_details={"adapter": "source-store"},
    )


def synthetic_dataset(count: int = 1200, seed: int = 7) -> xr.Dataset:
    """Return a deterministic LMA-like dataset for demos and tests."""

    rng = np.random.default_rng(seed)
    count = int(count)
    if count <= 0:
        raise ValueError("Synthetic event count must be positive")
    start = np.datetime64("2026-07-06T21:18:38.900000000", "ns")
    seconds = np.sort(rng.uniform(0.0, 1.25, count))
    time = start + np.rint(seconds * 1e9).astype("timedelta64[ns]")
    branch = rng.integers(0, 4, count)
    progress = seconds / seconds.max()
    x = 2.0 * progress + (branch - 1.5) * 0.35 * progress + rng.normal(0, 0.05, count)
    y = -1.5 + 2.5 * progress + (branch - 1.5) * 0.18 + rng.normal(0, 0.05, count)
    altitude = 1.0 + 8.5 * progress + (branch % 2) * 0.5 + rng.normal(0, 0.12, count)
    ref_lat, ref_lon = 33.978, -107.181
    radius = 6371.0088
    latitude = ref_lat + np.rad2deg(y / radius)
    longitude = ref_lon + np.rad2deg(x / (radius * np.cos(np.deg2rad(ref_lat))))
    stations = rng.integers(4, 12, count)
    chi2 = np.clip(rng.lognormal(mean=-0.25, sigma=0.55, size=count), 0.05, 8.0)
    power = rng.normal(8.0, 5.0, count)
    station_angles = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    station_r = np.array([8, 14, 18, 23, 27, 30, 34, 38, 42, 47], dtype=float)
    station_x = station_r * np.cos(station_angles)
    station_y = station_r * np.sin(station_angles)
    station_lat = ref_lat + np.rad2deg(station_y / radius)
    station_lon = ref_lon + np.rad2deg(station_x / (radius * np.cos(np.deg2rad(ref_lat))))
    return xr.Dataset(
        data_vars={
            "event_time": (EVENT_DIM, time, {"long_name": "LMA event time"}),
            "event_latitude": (EVENT_DIM, latitude, {"units": "degrees_north"}),
            "event_longitude": (EVENT_DIM, longitude, {"units": "degrees_east"}),
            "event_altitude": (EVENT_DIM, altitude * 1000.0, {"units": "m"}),
            "event_power": (EVENT_DIM, power, {"units": "dBW"}),
            "event_stations": (EVENT_DIM, stations),
            "event_chi2": (EVENT_DIM, chi2),
            "event_id": (EVENT_DIM, np.arange(count, dtype=np.int64)),
            "network_center_latitude": ((), ref_lat),
            "network_center_longitude": ((), ref_lon),
            "network_center_altitude": ((), 0.0, {"units": "m"}),
            "station_latitude": ("number_of_stations", station_lat),
            "station_longitude": ("number_of_stations", station_lon),
            "station_altitude": ("number_of_stations", np.full(10, 2.0e3), {"units": "m"}),
            "station_code": ("number_of_stations", np.asarray([f"S{i:02d}" for i in range(10)], dtype="U3")),
        },
        attrs={"network_name": "Synthetic LMAS demonstration", "source": "lmas.synthetic_dataset"},
    )
