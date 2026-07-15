"""Generic satellite-lightning overlay support."""
from .manager import (
    GLMOverlayStyle,
    SatelliteDatasetRecord,
    SatelliteOverlayManager,
    group_glm_paths_by_platform,
)
from .rendering import RenderSummary, SatelliteOverlayRenderer, configure_group_energy_time_axis

__all__ = [
    "GLMOverlayStyle",
    "RenderSummary",
    "SatelliteDatasetRecord",
    "SatelliteOverlayManager",
    "SatelliteOverlayRenderer",
    "group_glm_paths_by_platform",
    "configure_group_energy_time_axis",
]
