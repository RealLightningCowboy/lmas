"""Historical GOES operational-role resolution.

The spacecraft platform and the operational East/West assignment are distinct.
File metadata remains authoritative; this registry is only a dated fallback.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class OperationalAssignment:
    platform_id: str
    role: str
    start: np.datetime64
    end: np.datetime64 | None

    def contains(self, time: np.datetime64) -> bool:
        t = np.datetime64(time, "ns")
        return t >= self.start and (self.end is None or t < self.end)


# Versioned internal fallback. Product metadata and projection metadata take
# precedence. End times are exclusive.
_ASSIGNMENTS = (
    OperationalAssignment("G16", "east", np.datetime64("2017-12-18", "ns"), np.datetime64("2025-04-07", "ns")),
    OperationalAssignment("G17", "west", np.datetime64("2019-02-12", "ns"), np.datetime64("2023-01-04", "ns")),
    OperationalAssignment("G18", "west", np.datetime64("2023-01-04", "ns"), None),
    OperationalAssignment("G19", "east", np.datetime64("2025-04-07", "ns"), None),
)


def resolve_operational_role(platform_id: str, observation_time_ns: int) -> tuple[str, str]:
    """Resolve East/West from spacecraft and date, or return unknown."""
    time = np.datetime64(int(observation_time_ns), "ns")
    platform = str(platform_id).upper()
    for assignment in _ASSIGNMENTS:
        if assignment.platform_id == platform and assignment.contains(time):
            return assignment.role, "registry:goes_operational_assignments_v1"
    return "unknown", "unknown"


def role_consistent_with_longitude(role: str, subpoint_lon_deg: float | None) -> bool | None:
    """Return whether a role agrees with the broad orbital longitude region."""
    if subpoint_lon_deg is None or not np.isfinite(subpoint_lon_deg):
        return None
    normalized = str(role).strip().lower()
    if normalized == "east":
        return float(subpoint_lon_deg) > -100.0
    if normalized == "west":
        return float(subpoint_lon_deg) <= -100.0
    return None
