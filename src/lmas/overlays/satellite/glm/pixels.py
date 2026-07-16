"""GLM event accumulation on the instrument fixed grid.

GLM Level-2 events are repeated measurements of illuminated detector pixels.
For map rendering, events within the selected time window are accumulated by
fixed-grid pixel before one footprint polygon is drawn per pixel.  This matches
the established glmtools ``get_lutevents(..., scale_factor=56e-6)`` behavior
used by the legacy LMAS plotting workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

GLM_PIXEL_SCALE_RAD = 56.0e-6
GLM_FIXED_GRID_RANGE_RAD = (-0.31, 0.31)


@dataclass(frozen=True, slots=True)
class GLMAccumulatedPixels:
    """Per-pixel event-energy accumulation for one GLM observation."""

    pixel_id: np.ndarray
    center_x_rad: np.ndarray
    center_y_rad: np.ndarray
    energy_j: np.ndarray
    event_count: np.ndarray

    def __len__(self) -> int:
        return int(self.pixel_id.size)

    @property
    def centers_fixed_grid(self) -> np.ndarray:
        return np.column_stack((self.center_x_rad, self.center_y_rad))

    def take(self, indices: Sequence[int] | np.ndarray) -> "GLMAccumulatedPixels":
        idx = np.asarray(indices, dtype=np.int64).reshape(-1)
        return GLMAccumulatedPixels(
            pixel_id=self.pixel_id[idx],
            center_x_rad=self.center_x_rad[idx],
            center_y_rad=self.center_y_rad[idx],
            energy_j=self.energy_j[idx],
            event_count=self.event_count[idx],
        )


def discretize_fixed_grid_pixels(
    x_rad: np.ndarray,
    y_rad: np.ndarray,
    *,
    scale_factor: float = GLM_PIXEL_SCALE_RAD,
    x_range: tuple[float, float] = GLM_FIXED_GRID_RANGE_RAD,
    y_range: tuple[float, float] = GLM_FIXED_GRID_RANGE_RAD,
) -> np.ndarray:
    """Return glmtools-compatible integer IDs for fixed-grid locations."""

    x = np.asarray(x_rad, dtype=np.float64)
    y = np.asarray(y_rad, dtype=np.float64)
    scale = float(scale_factor)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("GLM pixel scale_factor must be finite and positive")

    x_offset = float(x_range[0])
    y_offset = float(y_range[0])
    x_count = np.asarray((float(x_range[1]) - x_offset) / scale, dtype=np.uint64)
    x_discrete = ((x - x_offset) / scale).astype(np.uint64)
    y_discrete = ((y - y_offset) / scale).astype(np.uint64)
    return x_discrete + y_discrete * x_count


def accumulate_event_pixels(
    observation,
    event_indices: Sequence[int] | np.ndarray,
    *,
    scale_factor: float = GLM_PIXEL_SCALE_RAD,
) -> GLMAccumulatedPixels:
    """Sum selected event energy by GLM fixed-grid pixel.

    Pixel centers are the mean fixed-grid locations of their constituent events,
    and energies are summed across the entire supplied selection.  Events with
    non-finite fixed-grid centers are omitted because no footprint can be drawn.
    """

    indices = np.asarray(event_indices, dtype=np.int64).reshape(-1)
    if indices.size == 0:
        return _empty_pixels()

    centers = np.asarray(
        observation.geometry.event_centers_fixed_grid(indices), dtype=np.float64
    )
    energies = np.asarray(observation.events.energy_j[indices], dtype=np.float64)
    finite_center = np.all(np.isfinite(centers), axis=1)
    if not np.any(finite_center):
        return _empty_pixels()

    centers = centers[finite_center]
    energies = energies[finite_center]
    pixel_ids = discretize_fixed_grid_pixels(
        centers[:, 0], centers[:, 1], scale_factor=scale_factor
    )

    order = np.argsort(pixel_ids, kind="stable")
    sorted_ids = pixel_ids[order]
    sorted_centers = centers[order]
    sorted_energies = energies[order]
    starts = np.concatenate(
        (np.array([0], dtype=np.int64), np.flatnonzero(np.diff(sorted_ids)) + 1)
    )
    counts = np.diff(np.append(starts, sorted_ids.size)).astype(np.int64)

    center_x = np.add.reduceat(sorted_centers[:, 0], starts) / counts
    center_y = np.add.reduceat(sorted_centers[:, 1], starts) / counts
    energy = np.add.reduceat(sorted_energies, starts)

    return GLMAccumulatedPixels(
        pixel_id=sorted_ids[starts],
        center_x_rad=center_x,
        center_y_rad=center_y,
        energy_j=energy,
        event_count=counts,
    )


def _empty_pixels() -> GLMAccumulatedPixels:
    return GLMAccumulatedPixels(
        pixel_id=np.empty(0, dtype=np.uint64),
        center_x_rad=np.empty(0, dtype=np.float64),
        center_y_rad=np.empty(0, dtype=np.float64),
        energy_j=np.empty(0, dtype=np.float64),
        event_count=np.empty(0, dtype=np.int64),
    )
