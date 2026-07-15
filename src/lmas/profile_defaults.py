from __future__ import annotations

from pathlib import Path

from .model import LMAProject


def profile_save_directory(project: LMAProject, fallback: str | Path) -> Path:
    """Choose a visible, project-relevant initial directory for profile saves."""

    for source in project.source_files:
        path = Path(source).expanduser()
        if path.name:
            return path.resolve().parent
    if project.project_path is not None:
        return Path(project.project_path).expanduser().resolve().parent
    return Path(fallback).expanduser().resolve()


__all__ = ["profile_save_directory"]
