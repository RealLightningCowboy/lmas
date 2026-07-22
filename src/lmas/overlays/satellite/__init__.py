"""Generic satellite-lightning overlay support.

Public names are loaded on first access to keep the desktop startup path small.
"""
from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "GLMOverlayStyle": (".manager", "GLMOverlayStyle"),
    "SatelliteDatasetRecord": (".manager", "SatelliteDatasetRecord"),
    "SatelliteOverlayManager": (".manager", "SatelliteOverlayManager"),
    "group_glm_paths_by_platform": (".manager", "group_glm_paths_by_platform"),
    "RenderSummary": (".rendering", "RenderSummary"),
    "SatelliteOverlayRenderer": (".rendering", "SatelliteOverlayRenderer"),
    "configure_group_energy_time_axis": (".rendering", "configure_group_energy_time_axis"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
