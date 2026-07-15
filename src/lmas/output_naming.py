from __future__ import annotations

from pathlib import Path
import re

from .model import LMAProject


def safe_output_stem(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip()).strip("._")
    return text or "lmas_output"


def project_output_directory(
    project: LMAProject,
    fallback: str | Path,
    *,
    output_directory: str | Path | None = None,
) -> Path:
    if output_directory is not None:
        return Path(output_directory).expanduser().resolve()
    return (project.output_directory or Path(fallback).expanduser()).resolve()


def default_output_path(
    project: LMAProject,
    fallback_directory: str | Path,
    *parts: str,
    extension: str,
    output_directory: str | Path | None = None,
) -> Path:
    stem = safe_output_stem(project.output_stem)
    qualifiers = [safe_output_stem(part).lower() for part in parts if str(part).strip()]
    suffix = extension if str(extension).startswith(".") else f".{extension}"
    filename = "_".join([stem, *qualifiers]) + suffix.lower()
    return project_output_directory(
        project, fallback_directory, output_directory=output_directory
    ) / filename


def display_mode_label(value: str) -> str:
    mode = str(value).strip().lower().replace("_", "-")
    return {
        "cumulative-afterimage": "cumulative",
        "trail-afterimage": "trail_afterimage",
        "develop-orbit": "development_orbit",
        "develop": "development",
    }.get(mode, mode.replace("-", "_"))


__all__ = [
    "default_output_path",
    "display_mode_label",
    "project_output_directory",
    "safe_output_stem",
]
