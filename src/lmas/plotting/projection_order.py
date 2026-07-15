from __future__ import annotations

import numpy as np


def spatial_depth_keys(
    east_km: np.ndarray,
    north_km: np.ndarray,
    altitude_km: np.ndarray,
    *,
    source_time_s: np.ndarray,
    east_west_viewpoint: str,
    north_south_viewpoint: str,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return stable far-to-near painter keys for the Landscape panels.

    The time-altitude panel is always time ordered. In ``spatial`` mode, each
    orthographic spatial projection uses its hidden coordinate and the selected
    observer side so nearer sources are drawn last. The plan view is drawn from
    below to above so higher-altitude sources appear on top.

    This follows the depth-order behavior developed for the INTFS projection
    viewer and adapted here for LMA Cartesian coordinates.
    """

    east = np.asarray(east_km, dtype=float)
    north = np.asarray(north_km, dtype=float)
    altitude = np.asarray(altitude_km, dtype=float)
    source_time = np.asarray(source_time_s, dtype=float)
    if not (east.shape == north.shape == altitude.shape == source_time.shape):
        raise ValueError("Spatial coordinates and source time must have matching shapes")

    resolved_mode = str(mode).strip().lower()
    if resolved_mode not in {"time", "spatial"}:
        raise ValueError(f"Unsupported depth mode: {mode!r}")
    if resolved_mode == "time":
        return source_time, source_time, source_time, source_time

    ew = str(east_west_viewpoint).strip().lower()
    ns = str(north_south_viewpoint).strip().lower()
    if ew not in {"east", "west"}:
        raise ValueError(f"Unsupported east/west viewpoint: {east_west_viewpoint!r}")
    if ns not in {"north", "south"}:
        raise ValueError(f"Unsupported north/south viewpoint: {north_south_viewpoint!r}")

    # Altitude versus Northing hides Easting.
    north_panel_key = east if ew == "east" else -east
    # Altitude versus Easting hides Northing.
    east_panel_key = north if ns == "north" else -north
    # Plan view is viewed from above: low altitude is far, high altitude near.
    plan_panel_key = altitude
    return source_time, north_panel_key, east_panel_key, plan_panel_key


def painter_indices(depth_key: np.ndarray | None, size: int) -> np.ndarray:
    indices = np.arange(int(size), dtype=int)
    if depth_key is None:
        return indices
    key = np.asarray(depth_key, dtype=float)
    if key.shape != (int(size),):
        raise ValueError("Depth key must be one-dimensional and match source count")
    return np.argsort(key, kind="stable")
