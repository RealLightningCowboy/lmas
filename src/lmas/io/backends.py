from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Iterable, Protocol, Sequence

import xarray as xr

from ..errors import DatasetError, DependencyError
from ..model import validate_dataset
from ..source_store import LmaSourceStore
from .native_dat import NATIVE_READER_VERSION, native_dat_dataset

READER_ENTRY_POINT_GROUP = "lmas.reader_backends"
BUILTIN_BACKENDS = ("native", "pyxlma")
READER_CHOICES = ("auto", *BUILTIN_BACKENDS)
NETCDF_SUFFIXES = {".nc", ".netcdf"}


def _is_netcdf(path: Path) -> bool:
    return path.suffix.lower() in NETCDF_SUFFIXES


def _is_dat(path: Path) -> bool:
    lower = path.name.lower()
    return lower.endswith(".dat") or lower.endswith(".dat.gz")


def normalize_reader_backend(value: str | None) -> str:
    backend = str(value or "auto").strip().lower().replace("_", "-")
    aliases = {
        "lmas": "native",
        "lmas-native": "native",
        "native": "native",
        "pyxlma": "pyxlma",
        "pyxlma-compatibility": "pyxlma",
        "auto": "auto",
    }
    return aliases.get(backend, backend)


@dataclass(frozen=True)
class ReaderResult:
    store: LmaSourceStore
    backend: str
    backend_version: str
    details: dict[str, str]

    @property
    def dataset(self) -> xr.Dataset:
        """Return an xarray compatibility view of the immutable source store."""

        return self.store.to_xarray()


@dataclass(frozen=True)
class ReaderBackendStatus:
    name: str
    label: str
    available: bool
    version: str | None
    description: str


class ReaderBackend(Protocol):
    name: str
    label: str

    def available(self) -> bool: ...

    def version(self) -> str | None: ...

    def can_read(self, paths: Sequence[Path]) -> bool: ...

    def read(self, paths: Sequence[Path]) -> xr.Dataset | LmaSourceStore: ...


class NativeReaderBackend:
    name = "native"
    label = "LMAS native"

    def available(self) -> bool:
        return True

    def version(self) -> str:
        return NATIVE_READER_VERSION

    def can_read(self, paths: Sequence[Path]) -> bool:
        return bool(paths) and (all(_is_netcdf(path) for path in paths) or all(_is_dat(path) for path in paths))

    def read(self, paths: Sequence[Path]) -> xr.Dataset:
        if all(_is_netcdf(path) for path in paths):
            if len(paths) != 1:
                raise DatasetError("The native NetCDF reader currently accepts one dataset at a time")
            try:
                return validate_dataset(xr.load_dataset(paths[0]))
            except Exception as exc:
                raise DatasetError(f"Could not read NetCDF LMA dataset {paths[0]}: {exc}") from exc
        if all(_is_dat(path) for path in paths):
            return validate_dataset(native_dat_dataset(paths))
        raise DatasetError("The LMAS native reader cannot mix DAT and NetCDF inputs in one load")


class PyxlmaReaderBackend:
    name = "pyxlma"
    label = "pyxlma compatibility"

    def available(self) -> bool:
        try:
            from pyxlma.lmalib.io import read as _read  # noqa: F401
        except (ImportError, OSError):
            return False
        return True

    def version(self) -> str | None:
        if not self.available():
            return None
        try:
            return metadata.version("pyxlma")
        except metadata.PackageNotFoundError:
            try:
                import pyxlma

                return str(getattr(pyxlma, "__version__", "unknown"))
            except Exception:
                return "unknown"

    def can_read(self, paths: Sequence[Path]) -> bool:
        return bool(paths) and all(_is_dat(path) for path in paths)

    def read(self, paths: Sequence[Path]) -> xr.Dataset:
        if not self.available():
            raise DependencyError(
                "The optional pyxlma compatibility backend was requested, but pyxlma is not installed. "
                "Install pyxlma separately or use --reader native."
            )
        if not self.can_read(paths):
            raise DatasetError("The pyxlma compatibility backend accepts only .dat and .dat.gz files")
        try:
            from pyxlma.lmalib.io import read

            dataset, _start_time = read.dataset([str(path) for path in paths])
        except Exception as exc:
            raise DatasetError(f"pyxlma could not read the selected LMA files: {exc}") from exc
        return validate_dataset(dataset)


_BUILTINS: dict[str, ReaderBackend] = {
    "native": NativeReaderBackend(),
    "pyxlma": PyxlmaReaderBackend(),
}


def _plugin_backends() -> dict[str, ReaderBackend]:
    result: dict[str, ReaderBackend] = {}
    try:
        entry_points = metadata.entry_points()
        selected = entry_points.select(group=READER_ENTRY_POINT_GROUP)
    except (AttributeError, TypeError):
        selected = metadata.entry_points().get(READER_ENTRY_POINT_GROUP, ())  # type: ignore[assignment]
    for entry_point in selected:
        try:
            candidate = entry_point.load()
            backend = candidate() if isinstance(candidate, type) else candidate
            name = normalize_reader_backend(getattr(backend, "name", entry_point.name))
            if name not in _BUILTINS:
                result[name] = backend
        except Exception:
            continue
    return result


def reader_backends() -> dict[str, ReaderBackend]:
    return {**_BUILTINS, **_plugin_backends()}


def reader_backend_statuses() -> tuple[ReaderBackendStatus, ...]:
    statuses: list[ReaderBackendStatus] = []
    for name, backend in reader_backends().items():
        available = bool(backend.available())
        statuses.append(
            ReaderBackendStatus(
                name=name,
                label=str(getattr(backend, "label", name)),
                available=available,
                version=backend.version() if available else None,
                description=(
                    "Built-in solved-LMA reader"
                    if name == "native"
                    else "Optional compatibility backend"
                ),
            )
        )
    return tuple(statuses)


def read_with_backend(
    paths: Iterable[str | Path],
    *,
    backend: str = "auto",
) -> ReaderResult:
    resolved = tuple(Path(path).expanduser().resolve() for path in paths)
    if not resolved:
        raise DatasetError("No reader-ready LMA files were provided")
    requested = normalize_reader_backend(backend)
    backends = reader_backends()
    if requested == "auto":
        selected_name = "native"
    else:
        selected_name = requested
    selected = backends.get(selected_name)
    if selected is None:
        raise DatasetError(
            f"Unknown LMA reader backend {backend!r}. Available backends: "
            + ", ".join(sorted(backends))
        )
    if not selected.available():
        raise DependencyError(
            f"The {getattr(selected, 'label', selected_name)} reader backend is unavailable"
        )
    if not selected.can_read(resolved):
        raise DatasetError(
            f"The {getattr(selected, 'label', selected_name)} reader backend cannot read the selected inputs"
        )
    loaded = selected.read(resolved)
    store = (
        loaded
        if isinstance(loaded, LmaSourceStore)
        else LmaSourceStore.from_xarray(validate_dataset(loaded))
    )
    version = str(selected.version() or "unknown")
    store = store.with_attrs(
        lmas_reader_backend=selected_name,
        lmas_reader_backend_version=version,
    )
    return ReaderResult(
        store=store,
        backend=selected_name,
        backend_version=version,
        details={"requested_backend": requested},
    )


__all__ = [
    "BUILTIN_BACKENDS",
    "READER_CHOICES",
    "READER_ENTRY_POINT_GROUP",
    "ReaderBackendStatus",
    "ReaderResult",
    "normalize_reader_backend",
    "read_with_backend",
    "reader_backend_statuses",
    "reader_backends",
]
