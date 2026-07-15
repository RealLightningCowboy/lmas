from __future__ import annotations

import numpy as np

from .coordinates import altitude_km
from .model import LMAProject


def project_summary(project: LMAProject) -> dict[str, object]:
    dataset = project.dataset
    times = np.asarray(dataset["event_time"].values).astype("datetime64[ns]")
    finite_times = times[~np.isnat(times)]
    alt = altitude_km(dataset)
    finite_alt = alt[np.isfinite(alt)]
    result: dict[str, object] = {
        "name": project.name,
        "source_files": [str(path) for path in project.source_files],
        "event_count": project.event_count,
        "reference_latitude": float(project.reference_latitude),
        "reference_longitude": float(project.reference_longitude),
        "available_color_fields": list(project.available_color_fields),
        "reader_backend": project.reader_backend,
        "reader_backend_version": project.reader_backend_version,
        "reader_details": dict(project.reader_details),
    }
    if finite_times.size:
        result["start_time"] = str(finite_times.min())
        result["end_time"] = str(finite_times.max())
    if finite_alt.size:
        result["minimum_altitude_km"] = float(finite_alt.min())
        result["maximum_altitude_km"] = float(finite_alt.max())
    for mode, field in (("stations", "event_stations"), ("chi2", "event_chi2"), ("power", "event_power")):
        if field in dataset:
            values = np.asarray(dataset[field].values, dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size:
                result[f"minimum_{mode}"] = float(finite.min())
                result[f"maximum_{mode}"] = float(finite.max())
    return result
