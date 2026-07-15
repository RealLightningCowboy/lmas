"""Native, NumPy-first support for GOES-R Series GLM L2 LCFA data."""

from .model import (
    GLMDataError,
    GLMDatasetIdentity,
    GLMEventTable,
    GLMFlashTable,
    GLMGroupTable,
    GLMHierarchyReport,
    GLMObservation,
    GLMProjectionMetadata,
    GLMSelection,
    GLMSourceFile,
)
from .geometry import (
    GLMEventGeometry,
    fixed_grid_to_lon_lat,
    lightning_ellipse_radii,
    lightning_ellipse_revision,
    lon_lat_to_fixed_grid,
)
from .reader import read_glm, read_glm_l2_lcfa, read_glm_with_glmtools

__all__ = [
    "GLMDataError",
    "GLMEventGeometry",
    "GLMDatasetIdentity",
    "GLMEventTable",
    "GLMFlashTable",
    "GLMGroupTable",
    "GLMHierarchyReport",
    "GLMObservation",
    "GLMProjectionMetadata",
    "GLMSelection",
    "GLMSourceFile",
    "fixed_grid_to_lon_lat",
    "lightning_ellipse_radii",
    "lightning_ellipse_revision",
    "lon_lat_to_fixed_grid",
    "read_glm",
    "read_glm_l2_lcfa",
    "read_glm_with_glmtools",
]
