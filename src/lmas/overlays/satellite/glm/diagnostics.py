"""Text diagnostics for native GLM observations."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np

from .reader import read_glm_l2_lcfa


def summarize_glm(paths: Sequence[str | Path]) -> str:
    observation = read_glm_l2_lcfa(paths)
    report = observation.validate_hierarchy()
    identity = observation.identity
    lines = [
        "LMAS native GLM diagnostic",
        f"Dataset: {identity.display_name}",
        f"Product: {identity.product_level}",
        f"Coverage: {identity.observation_start} to {identity.observation_end}",
        f"Files: {len(identity.source_files)}",
        f"Events: {len(observation.events):,}",
        f"Groups: {len(observation.groups):,}",
        f"Flashes: {len(observation.flashes):,}",
        f"Hierarchy valid: {'yes' if report.valid else 'no'}",
        f"Event energy range (J): {_range_text(observation.events.energy_j)}",
        f"Group energy range (J): {_range_text(observation.groups.energy_j)}",
        f"Flash energy range (J): {_range_text(observation.flashes.energy_j)}",
    ]
    return "\n".join(lines)


def _range_text(values: np.ndarray) -> str:
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return "n/a"
    return f"{finite.min():.6g} to {finite.max():.6g}"
