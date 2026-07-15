"""Satellite-overlay colormaps.

LMAS packages the established research-code ``mako`` and ``crest`` lookup
palettes directly, then exposes a compact set of Matplotlib-native alternatives
without adding Seaborn as a runtime dependency.
"""
from __future__ import annotations

from functools import lru_cache

from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap

MAKO_COLORS = (
    "#0f0609", "#13090f", "#170c15", "#1b0f1a", "#1f1220", "#231526",
    "#27182d", "#2a1b33", "#2e1e39", "#312140", "#342447", "#36274d",
    "#382a54", "#3b2d5b", "#3c3162", "#3e3469", "#3f366f", "#403a76",
    "#413e7d", "#414184", "#40468a", "#3f4a8f", "#3e4f94", "#3c5397",
    "#3b589a", "#395d9c", "#38629d", "#37669e", "#366b9f", "#3670a0",
    "#3574a1", "#3579a2", "#357ca3", "#3480a4", "#3485a5", "#3489a6",
    "#348ea7", "#3492a8", "#3497a9", "#359baa", "#35a0ab", "#36a4ab",
    "#38a9ac", "#3aadac", "#3cb2ad", "#3fb6ad", "#43baad", "#47bfad",
    "#4bc2ad", "#50c6ad", "#57cbad", "#60ceac", "#6ad2ad", "#76d5ae",
    "#82d8b0", "#8edbb3", "#99ddb6", "#a4e0bb", "#aee3c0", "#b7e6c5",
    "#c0e9cc", "#c8ecd2", "#d0efd9", "#d8f2e0",
)

CREST_COLORS = (
    "#a2cb91", "#9dc991", "#99c791", "#94c591", "#90c391", "#8bc191",
    "#87be91", "#82bc91", "#7eba91", "#79b891", "#75b690", "#71b490",
    "#6db290", "#69b090", "#65ad90", "#62ab90", "#5fa990", "#5ba790",
    "#58a590", "#55a290", "#52a090", "#4f9e90", "#4c9b90", "#49998f",
    "#47978f", "#44948f", "#41928f", "#3f8f8e", "#3c8d8e", "#3a8b8e",
    "#37888e", "#35868d", "#33848d", "#30828d", "#2d808c", "#2b7d8c",
    "#287b8c", "#26788c", "#24768b", "#22748b", "#20718b", "#1e6f8a",
    "#1d6c8a", "#1d6a8a", "#1c6789", "#1c6488", "#1d6288", "#1d5f87",
    "#1e5d86", "#1f5b86", "#205885", "#215584", "#225283", "#235081",
    "#244d80", "#254a7f", "#27477d", "#28457c", "#29427a", "#2a3f78",
    "#2a3c77", "#2b3975", "#2b3674", "#2c3373",
)

SATELLITE_COLORMAP_NAMES = (
    "mako",
    "crest",
    "inferno",
    "magma",
    "plasma",
    "viridis",
    "cividis",
    "turbo",
)


@lru_cache(maxsize=None)
def satellite_colormap(name: str):
    normalized = str(name).strip().lower()
    if normalized == "mako":
        return LinearSegmentedColormap.from_list("lmas_mako", MAKO_COLORS, N=256)
    if normalized == "crest":
        return LinearSegmentedColormap.from_list("lmas_crest", CREST_COLORS, N=256)
    if normalized in SATELLITE_COLORMAP_NAMES:
        return colormaps[normalized]
    raise KeyError(f"Unknown LMAS satellite colormap: {name!r}")


__all__ = [
    "MAKO_COLORS",
    "CREST_COLORS",
    "SATELLITE_COLORMAP_NAMES",
    "satellite_colormap",
]
