from __future__ import annotations

import os
from pathlib import Path
import sys


def _expanded_env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    if not value:
        return None
    value = value.strip().strip('"').replace("$HOME", str(Path.home()))
    if not value:
        return None
    return Path(value).expanduser()


def is_lmas_development_path(value: object) -> bool:
    """Return whether *value* points inside an LMAS Development tree.

    The check is lexical and separator-agnostic so Windows paths can be
    migrated safely even when inspected on another platform.  It deliberately
    does not encode a workstation name, drive letter, or laboratory root.
    """

    raw = str(value or "").strip().replace("\\", "/")
    if not raw:
        return False
    parts = [part.casefold() for part in raw.split("/") if part not in {"", "."}]
    return any(
        parts[index] == "lmas" and parts[index + 1] == "development"
        for index in range(len(parts) - 1)
    )


def user_documents_directory() -> Path:
    """Return a portable first-run directory for open/save dialogs."""
    override = _expanded_env_path("LMAS_DATA_ROOT")
    if override is not None and override.exists():
        return override.resolve()

    if sys.platform.startswith("win"):
        home = Path(os.environ.get("USERPROFILE", str(Path.home()))).expanduser()
        candidates = (home / "Documents", home)
    else:
        xdg_documents = _expanded_env_path("XDG_DOCUMENTS_DIR")
        candidates = tuple(
            candidate
            for candidate in (xdg_documents, Path.home() / "Documents", Path.home())
            if candidate is not None
        )
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate.resolve()
    return Path.home().resolve()


def user_config_directory() -> Path:
    """Return LMAS's platform-appropriate per-user configuration directory."""
    if sys.platform.startswith("win"):
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        return base.expanduser() / "LMAS"
    base = _expanded_env_path("XDG_CONFIG_HOME") or (Path.home() / ".config")
    return base / "lmas"


def user_cache_directory() -> Path:
    """Return LMAS's platform-appropriate per-user cache directory."""
    if sys.platform.startswith("win"):
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base.expanduser() / "LMAS" / "Cache"
    base = _expanded_env_path("XDG_CACHE_HOME") or (Path.home() / ".cache")
    return base / "lmas"


__all__ = [
    "is_lmas_development_path",
    "user_cache_directory",
    "user_config_directory",
    "user_documents_directory",
]
