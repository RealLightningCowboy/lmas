"""Read original solved-LMA headers and build metadata summaries for the GUI."""

from __future__ import annotations

from dataclasses import dataclass
import gzip
from io import BytesIO, TextIOWrapper
import json
from pathlib import Path
import tarfile
from typing import Any, BinaryIO, Iterable, TextIO

import numpy as np

_HEADER_END = "*** data ***"
_DAT_SUFFIXES = (".dat", ".dat.gz")
_ARCHIVE_SUFFIXES = (".tar", ".tar.gz", ".tgz")


@dataclass(frozen=True)
class DataHeaderDocument:
    """One read-only header or metadata document shown by LMAS."""

    title: str
    text: str
    source: str
    kind: str


def _is_dat_name(name: str) -> bool:
    lower = str(name).casefold()
    return lower.endswith(_DAT_SUFFIXES)


def _is_archive_name(name: str) -> bool:
    lower = str(name).casefold()
    return lower.endswith(_ARCHIVE_SUFFIXES)


def _read_header_lines(stream: TextIO) -> str:
    lines: list[str] = []
    for raw in stream:
        line = raw.rstrip("\r\n")
        if line.strip() == _HEADER_END:
            break
        lines.append(line)
    return "\n".join(lines).rstrip() + ("\n" if lines else "")


def read_dat_header_text(path: str | Path) -> str:
    """Return the literal text before ``*** data ***`` in a DAT/DAT.GZ file."""

    source = Path(path).expanduser().resolve()
    if source.name.casefold().endswith(".gz"):
        with gzip.open(source, "rt", encoding="utf-8", errors="replace") as stream:
            return _read_header_lines(stream)
    with source.open("rt", encoding="utf-8", errors="replace") as stream:
        return _read_header_lines(stream)


def _header_from_binary_member(member: BinaryIO, name: str) -> str:
    if str(name).casefold().endswith(".gz"):
        with gzip.GzipFile(fileobj=member, mode="rb") as compressed:
            with TextIOWrapper(compressed, encoding="utf-8", errors="replace") as text:
                return _read_header_lines(text)
    with TextIOWrapper(member, encoding="utf-8", errors="replace") as text:
        return _read_header_lines(text)


def read_archive_header_documents(path: str | Path) -> tuple[DataHeaderDocument, ...]:
    """Return DAT headers stored inside a tar-compatible LMA archive."""

    source = Path(path).expanduser().resolve()
    documents: list[DataHeaderDocument] = []
    with tarfile.open(source, "r:*") as archive:
        for member_info in archive.getmembers():
            if not member_info.isfile() or not _is_dat_name(member_info.name):
                continue
            member = archive.extractfile(member_info)
            if member is None:
                continue
            text = _header_from_binary_member(member, member_info.name)
            documents.append(
                DataHeaderDocument(
                    title=f"{source.name} :: {member_info.name}",
                    text=text,
                    source=f"{source}::{member_info.name}",
                    kind="dat-header",
                )
            )
    return tuple(documents)


def _safe_text(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (dict, list, tuple)):
        try:
            return json.dumps(value, indent=2, sort_keys=True, default=str)
        except TypeError:
            return str(value)
    return str(value)


def project_metadata_summary(project: Any) -> DataHeaderDocument:
    """Build an equivalent metadata summary when no literal DAT header exists."""

    dataset = project.dataset
    lines = [
        "LMAS dataset metadata summary",
        "=============================",
        "",
        f"Project: {getattr(project, 'name', 'LMA dataset')}",
        f"Reader: {getattr(project, 'reader_backend', 'unknown')} "
        f"{getattr(project, 'reader_backend_version', '')}".rstrip(),
    ]
    source_files = tuple(getattr(project, "source_files", ()) or ())
    lines.append("Source files:")
    if source_files:
        lines.extend(f"  - {Path(path)}" for path in source_files)
    else:
        lines.append("  - none recorded")

    reader_details = dict(getattr(project, "reader_details", {}) or {})
    if reader_details:
        lines.extend(("", "Reader details:"))
        for key in sorted(reader_details):
            lines.append(f"  {key}: {_safe_text(reader_details[key])}")

    lines.extend(("", "Dimensions:"))
    for name, size in dataset.sizes.items():
        lines.append(f"  {name}: {int(size):,}")

    lines.extend(("", "Global attributes:"))
    if dataset.attrs:
        for key in sorted(dataset.attrs):
            value = _safe_text(dataset.attrs[key]).replace("\n", "\n    ")
            lines.append(f"  {key}: {value}")
    else:
        lines.append("  none")

    lines.extend(("", "Variables:"))
    for name in sorted(dataset.variables):
        variable = dataset[name]
        dimensions = ", ".join(variable.dims) or "scalar"
        units = variable.attrs.get("units")
        suffix = f"; units={units}" if units not in (None, "") else ""
        lines.append(f"  {name}: {variable.dtype}; dims=({dimensions}){suffix}")

    return DataHeaderDocument(
        title="Dataset metadata summary",
        text="\n".join(lines).rstrip() + "\n",
        source="dataset",
        kind="metadata-summary",
    )


def data_header_documents(project: Any) -> tuple[DataHeaderDocument, ...]:
    """Return literal DAT headers, followed by a metadata-summary fallback.

    Literal headers are read from direct DAT/DAT.GZ inputs and from DAT members
    inside tar-compatible LMA archives. Missing/unreadable inputs are reported in
    the metadata summary rather than making the viewer unavailable.
    """

    documents: list[DataHeaderDocument] = []
    source_files: Iterable[Any] = getattr(project, "source_files", ()) or ()
    errors: list[str] = []
    for raw_path in source_files:
        source = Path(raw_path).expanduser()
        if not source.exists():
            errors.append(f"Source is no longer available: {source}")
            continue
        try:
            if _is_dat_name(source.name):
                documents.append(
                    DataHeaderDocument(
                        title=source.name,
                        text=read_dat_header_text(source),
                        source=str(source.resolve()),
                        kind="dat-header",
                    )
                )
            elif _is_archive_name(source.name):
                documents.extend(read_archive_header_documents(source))
        except (OSError, EOFError, gzip.BadGzipFile, tarfile.TarError) as exc:
            errors.append(f"Could not read {source}: {exc}")

    summary = project_metadata_summary(project)
    if errors:
        summary = DataHeaderDocument(
            title=summary.title,
            text=(
                "Header access notes\n"
                "===================\n"
                + "\n".join(f"- {message}" for message in errors)
                + "\n\n"
                + summary.text
            ),
            source=summary.source,
            kind=summary.kind,
        )
    documents.append(summary)
    return tuple(documents)


__all__ = [
    "DataHeaderDocument",
    "data_header_documents",
    "project_metadata_summary",
    "read_archive_header_documents",
    "read_dat_header_text",
]
