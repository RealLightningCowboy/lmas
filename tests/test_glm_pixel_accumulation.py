from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from lmas.overlays.satellite.glm.pixels import accumulate_event_pixels
from lmas.overlays.satellite.glm.reader import read_glm_l2_lcfa


class _GeometryStub:
    def __init__(self, centers):
        self._centers = np.asarray(centers, dtype=np.float64)

    def event_centers_fixed_grid(self, indices):
        return self._centers[np.asarray(indices, dtype=np.int64)]


def test_event_energy_is_summed_once_per_fixed_grid_pixel():
    observation = SimpleNamespace(
        events=SimpleNamespace(
            energy_j=np.asarray([1.0, 2.0, 3.0, 5.0]) * 1.0e-15
        ),
        geometry=_GeometryStub(
            [
                [0.010000, 0.020000],
                [0.010000, 0.020000],
                [0.010000, 0.020000],
                [0.011000, 0.021000],
            ]
        ),
    )

    pixels = accumulate_event_pixels(observation, np.arange(4))

    assert len(pixels) == 2
    assert sorted(pixels.event_count.tolist()) == [1, 3]
    assert np.isclose(np.sum(pixels.energy_j), 11.0e-15)
    accumulated = pixels.energy_j[pixels.event_count == 3]
    assert accumulated.size == 1
    assert np.isclose(accumulated[0], 6.0e-15)


def test_packaged_demo_matches_glmtools_pixel_accumulation_reference():
    root = Path(
        "src/lmas/resources/demo/hybrid_20190430_144914/data/glm/goes16_east"
    )
    observation = read_glm_l2_lcfa(sorted(root.glob("*.nc")))
    selection = observation.select(
        time_range_ns=(
            np.datetime64("2019-04-30T14:49:14.142212000", "ns"),
            np.datetime64("2019-04-30T14:49:15.079513000", "ns"),
        )
    )

    pixels = accumulate_event_pixels(observation, selection.event_indices)
    raw_energy = observation.events.energy_j[selection.event_indices]

    assert selection.event_indices.size == 495
    assert len(pixels) == 74
    assert int(np.max(pixels.event_count)) == 68
    assert np.isclose(np.sum(pixels.energy_j), np.sum(raw_energy), rtol=1e-13)
    assert np.isclose(np.max(pixels.energy_j) * 1.0e15, 1960.8714277, rtol=1e-7)
