"""Portable tabular products for normalized network observations."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .model import NetworkObservation


def network_dataframe(observation: NetworkObservation, indices=None) -> pd.DataFrame:
    if indices is None:
        indices = np.arange(len(observation.events), dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int64)
    events = observation.events
    return pd.DataFrame(
        {
            "time_utc": events.time_ns[indices].astype("datetime64[ns]"),
            "longitude_deg": events.longitude_deg[indices],
            "latitude_deg": events.latitude_deg[indices],
            "altitude_m": events.altitude_m[indices],
            "event_type": events.event_type[indices],
            "original_event_type": events.original_event_type[indices],
            "polarity": events.polarity[indices],
            "peak_current_ka": events.peak_current_ka[indices],
            "amplitude": events.amplitude[indices],
            "sensor_count": events.sensor_count[indices],
            "quality": events.quality[indices],
            "ellipse_major_km": events.ellipse_major_km[indices],
            "ellipse_minor_km": events.ellipse_minor_km[indices],
            "ellipse_angle_deg": events.ellipse_angle_deg[indices],
            "original_id": events.original_id[indices],
            "provider": observation.identity.provider_id,
            "dataset": observation.identity.display_name,
        }
    )


def network_dataset(observation: NetworkObservation, indices=None) -> xr.Dataset:
    frame = network_dataframe(observation, indices)
    dimension = "number_of_network_events"
    data_vars = {}
    for column in frame.columns:
        if column in {"provider", "dataset"}:
            continue
        values = frame[column].to_numpy()
        data_vars[column] = ((dimension,), values)
    dataset = xr.Dataset(data_vars)
    dataset.attrs.update(
        {
            "title": f"LMAS normalized network observations — {observation.identity.display_name}",
            "provider": observation.identity.provider_id,
            "product_name": observation.identity.product_name,
            "reader": observation.identity.reader_name,
            "reader_version": observation.identity.reader_version,
            "source_files": ";".join(str(item.path) for item in observation.identity.source_files),
        }
    )
    return dataset


def export_network_csv(observation: NetworkObservation, path: str | Path, indices=None) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    network_dataframe(observation, indices).to_csv(destination, index=False)
    return destination


def export_network_netcdf(observation: NetworkObservation, path: str | Path, indices=None) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    network_dataset(observation, indices).to_netcdf(destination)
    return destination


__all__ = ["export_network_csv", "export_network_netcdf", "network_dataframe", "network_dataset"]
