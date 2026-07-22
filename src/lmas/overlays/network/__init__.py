"""Ground lightning-location-network overlay support.

Public names are loaded on first access so opening the LMAS main window does
not import the network export/readers stack unless that feature is used.
"""
from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "export_network_csv": (".export", "export_network_csv"),
    "export_network_netcdf": (".export", "export_network_netcdf"),
    "network_dataframe": (".export", "network_dataframe"),
    "network_dataset": (".export", "network_dataset"),
    "NetworkDatasetRecord": (".manager", "NetworkDatasetRecord"),
    "NetworkOverlayManager": (".manager", "NetworkOverlayManager"),
    "NetworkOverlayStyle": (".manager", "NetworkOverlayStyle"),
    "NAT_NS": (".model", "NAT_NS"),
    "NetworkEvents": (".model", "NetworkEvents"),
    "NetworkIdentity": (".model", "NetworkIdentity"),
    "NetworkObservation": (".model", "NetworkObservation"),
    "NetworkSelection": (".model", "NetworkSelection"),
    "NetworkSourceFile": (".model", "NetworkSourceFile"),
    "ALIASES": (".readers", "ALIASES"),
    "NetworkCSVOptions": (".readers", "NetworkCSVOptions"),
    "read_network_csv": (".readers", "read_network_csv"),
    "write_generic_network_example": (".readers", "write_generic_network_example"),
    "NetworkOverlayRenderer": (".rendering", "NetworkOverlayRenderer"),
    "NetworkRenderSummary": (".rendering", "NetworkRenderSummary"),
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
