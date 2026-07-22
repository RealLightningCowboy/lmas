"""Readers for ENTLN-style and generic ground-network CSV products."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence

import numpy as np

from .model import NAT_NS, NetworkEvents, NetworkIdentity, NetworkObservation, NetworkSourceFile


def _key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


ALIASES: dict[str, tuple[str, ...]] = {
    "time": (
        "time", "timestamp", "datetime", "eventtime", "eventdatetime", "utctime",
        "utc", "eventtimestamp", "timestamputc", "utctimestamp", "dateandtime", "eventdateandtime",
    ),
    "date": ("date", "eventdate", "utcdate"),
    "clock": ("clock", "timeofday", "utctimeofday", "eventclock", "timeutc", "utctime", "time"),
    "longitude": ("longitude", "lon", "eventlongitude", "eventlon", "longitudedeg"),
    "latitude": ("latitude", "lat", "eventlatitude", "eventlat", "latitudedeg"),
    "altitude": (
        "altitude", "height", "alt", "eventaltitude", "heightm", "altitudem", "heightkm",
        "icheight", "icheightm", "cloudheight", "cloudheightm",
    ),
    "event_type": ("type", "eventtype", "classification", "kind", "dischargetype", "pulsetype"),
    "polarity": ("polarity", "pol", "sign", "eventpolarity"),
    "peak_current": (
        "peakcurrent", "peakcurrentka", "current", "currentka", "amplitudeka",
        "peakamplitude", "estimatedpeakcurrent", "peakcurrenta",
    ),
    "amplitude": ("amplitude", "signalamplitude", "strength", "magnitude"),
    "sensor_count": (
        "sensorcount", "numsensors", "numberofsensors", "stations", "stationcount",
        "numstations", "sensors", "numberofstations", "nsta", "numbersensors",
    ),
    "quality": ("quality", "qualityflag", "status", "solutionquality", "qualitycode"),
    "ellipse_major": (
        "ellipsemajor", "ellipsemajorkm", "majoraxis", "majoraxiskm", "semimajor", "semimajoraxis",
        "locationmajoraxis", "major",
    ),
    "ellipse_minor": (
        "ellipseminor", "ellipseminorkm", "minoraxis", "minoraxiskm", "semiminor", "semiminoraxis",
        "locationminoraxis", "minor",
    ),
    "ellipse_angle": (
        "ellipseangle", "ellipseangledeg", "bearing", "azimuth", "orientation", "ellipseazimuth",
        "majoraxisazimuth", "angle",
    ),
    "id": ("id", "eventid", "recordid", "networkid", "solutionid", "strokeid", "pulseid"),
}


@dataclass(frozen=True, slots=True)
class NetworkCSVOptions:
    provider: str = "auto"
    display_name: str | None = None
    product_name: str | None = None
    column_mapping: Mapping[str, str] | None = None
    time_format: str | None = None
    peak_current_unit: str = "auto"
    altitude_unit: str = "auto"
    ellipse_unit: str = "auto"

    def to_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "display_name": self.display_name,
            "product_name": self.product_name,
            "column_mapping": dict(self.column_mapping or {}),
            "time_format": self.time_format,
            "peak_current_unit": self.peak_current_unit,
            "altitude_unit": self.altitude_unit,
            "ellipse_unit": self.ellipse_unit,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "NetworkCSVOptions":
        values = dict(payload or {})
        return cls(
            provider=str(values.get("provider") or "auto"),
            display_name=None if values.get("display_name") in (None, "") else str(values.get("display_name")),
            product_name=None if values.get("product_name") in (None, "") else str(values.get("product_name")),
            column_mapping=(
                dict(values.get("column_mapping") or {})
                if isinstance(values.get("column_mapping") or {}, Mapping)
                else {}
            ),
            time_format=None if values.get("time_format") in (None, "") else str(values.get("time_format")),
            peak_current_unit=str(values.get("peak_current_unit") or "auto"),
            altitude_unit=str(values.get("altitude_unit") or "auto"),
            ellipse_unit=str(values.get("ellipse_unit") or "auto"),
        )


def _resolve_columns(columns: Sequence[object], explicit: Mapping[str, str] | None = None) -> dict[str, str]:
    lookup = {_key(column): str(column) for column in columns}
    resolved: dict[str, str] = {}
    for canonical, raw in dict(explicit or {}).items():
        if str(raw) in columns:
            resolved[str(canonical)] = str(raw)
        elif _key(raw) in lookup:
            resolved[str(canonical)] = lookup[_key(raw)]
    for canonical, aliases in ALIASES.items():
        if canonical in resolved:
            continue
        for alias in aliases:
            if alias in lookup:
                resolved[canonical] = lookup[alias]
                break
    return resolved


def _numeric(frame: pd.DataFrame, column: str | None, *, default: float = np.nan) -> np.ndarray:
    import pandas as pd

    if column is None:
        return np.full(len(frame), default, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def _string(frame: pd.DataFrame, column: str | None, *, default: str = "") -> np.ndarray:
    if column is None:
        return np.full(len(frame), default, dtype="U32")
    values = frame[column].fillna("").astype(str).str.strip().to_numpy(dtype="U64")
    return values


def _infer_provider(paths: Sequence[Path], columns: Sequence[object], requested: str) -> str:
    requested = str(requested or "auto").strip().lower()
    if requested == "generic":
        return "generic"
    if requested not in {"", "auto"}:
        return requested
    haystack = " ".join(path.name.lower() for path in paths) + " " + " ".join(_key(c) for c in columns)
    if any(token in haystack for token in ("lxarchive", "entln", "earthnetworks", "earthnetwork")):
        return "entln"
    if "gld360" in haystack:
        return "gld360"
    if "nldn" in haystack:
        return "nldn"
    return "generic"


def _display_name(provider: str, requested: str | None) -> str:
    if requested:
        return str(requested)
    return {
        "entln": "ENTLN",
        "nldn": "NLDN",
        "gld360": "GLD360",
        "generic": "Network CSV",
    }.get(provider.lower(), provider.upper())


def _parse_times(frame: pd.DataFrame, columns: Mapping[str, str], fmt: str | None) -> np.ndarray:
    import pandas as pd

    # A surprising number of network exports call the clock-only column simply
    # ``time``. Prefer the explicit date+clock pair when both are present, then
    # fall back to one complete timestamp column.
    date_column = columns.get("date")
    clock_column = columns.get("clock")
    time_column = columns.get("time")
    if date_column is not None and clock_column is not None and date_column != clock_column:
        raw = frame[date_column].astype(str).str.strip() + " " + frame[clock_column].astype(str).str.strip()
    elif date_column is not None and time_column is not None and date_column != time_column:
        raw = frame[date_column].astype(str).str.strip() + " " + frame[time_column].astype(str).str.strip()
    elif time_column is not None:
        raw = frame[time_column]
    else:
        raise ValueError(
            "Could not identify a UTC timestamp column. Provide a column mapping for 'time', "
            "or mappings for both 'date' and 'clock'."
        )
    parsed = pd.to_datetime(raw, errors="coerce", utc=True, format=fmt)
    values = parsed.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    values[np.asarray(pd.isna(parsed), dtype=bool)] = NAT_NS
    return values


def _normalize_event_type(
    raw: np.ndarray, polarity: np.ndarray, *, provider: str = "generic"
) -> np.ndarray:
    """Normalize provider event classes without applying one provider's codes globally."""
    output = np.full(raw.size, "UNKNOWN", dtype="U16")
    provider = str(provider or "generic").strip().lower()
    for i, value in enumerate(raw.astype(str)):
        original = value.strip()
        numeric = original
        if numeric.endswith(".0"):
            numeric = numeric[:-2]
        # Established ENTLN archive convention used by the historical LMAS
        # research workflow: 0 = cloud-to-ground, 1 = intracloud.  Keep this
        # provider-scoped so generic/NLDN/GLD360 numeric codes are not guessed.
        if provider == "entln" and numeric in {"0", "1"}:
            output[i] = "CG" if numeric == "0" else "IC"
        else:
            token = original.upper().replace("_", "").replace("-", "")
            if token in {"CG", "CLOUDGROUND", "GROUNDFLASH", "STROKE", "RETURNSTROKE"} or token.endswith("CG"):
                output[i] = "CG"
            elif token in {"IC", "INTRACLOUD", "CLOUD", "CLOUDPULSE", "PULSE"} or token.endswith("IC"):
                output[i] = "IC"
            elif token in {"CA", "CLOUDATMOSPHERE"}:
                output[i] = "CA"
            elif token:
                output[i] = token[:16]
        if original.startswith("+"):
            polarity[i] = 1
        elif original.startswith("-"):
            polarity[i] = -1
    return output


def _normalize_polarity(raw: np.ndarray, current: np.ndarray) -> np.ndarray:
    result = np.zeros(raw.size, dtype=np.int8)
    for i, value in enumerate(raw.astype(str)):
        token = value.strip().lower()
        if token in {"+", "+1", "1", "positive", "pos", "p"}:
            result[i] = 1
        elif token in {"-", "-1", "negative", "neg", "n"}:
            result[i] = -1
    missing = result == 0
    result[missing & np.isfinite(current) & (current > 0)] = 1
    result[missing & np.isfinite(current) & (current < 0)] = -1
    return result


def _scale_peak_current(values: np.ndarray, column: str | None, unit: str) -> np.ndarray:
    unit = str(unit or "auto").strip().lower()
    key = _key(column or "")
    if unit in {"a", "amp", "amps", "ampere", "amperes"} or (unit == "auto" and key.endswith("a") and not key.endswith("ka")):
        return values / 1000.0
    return values


def _scale_distance(values: np.ndarray, column: str | None, unit: str, *, altitude: bool) -> np.ndarray:
    unit = str(unit or "auto").strip().lower()
    key = _key(column or "")
    finite = np.abs(values[np.isfinite(values)])
    if altitude:
        # Return metres.
        if unit in {"km", "kilometer", "kilometers"} or "km" in key:
            return values * 1000.0
        if unit == "auto" and finite.size and np.nanmedian(finite) < 100.0:
            return values * 1000.0
        return values
    # Return kilometres for ellipses.
    if unit in {"m", "meter", "meters"} or (unit == "auto" and key.endswith("m") and "km" not in key):
        return values / 1000.0
    if unit == "auto" and finite.size and np.nanmedian(finite) > 100.0:
        return values / 1000.0
    return values


def read_network_csv(
    paths: str | Path | Sequence[str | Path],
    *,
    options: NetworkCSVOptions | None = None,
) -> NetworkObservation:
    import pandas as pd

    options = options or NetworkCSVOptions()
    if isinstance(paths, (str, Path)):
        source_paths = [Path(paths).expanduser().resolve()]
    else:
        source_paths = [Path(path).expanduser().resolve() for path in paths]
    if not source_paths:
        raise ValueError("At least one network CSV file is required")

    frames: list[pd.DataFrame] = []
    file_records: list[NetworkSourceFile] = []
    for path in source_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            frame = pd.read_csv(path, comment="#", low_memory=False)
        except UnicodeDecodeError:
            frame = pd.read_csv(path, comment="#", low_memory=False, encoding="latin-1")
        if frame.empty:
            continue
        frame["__lmas_source_file"] = path.name
        frame["__lmas_source_row"] = np.arange(len(frame), dtype=np.int64)
        frames.append(frame)
        file_records.append(NetworkSourceFile(path=path, row_count=len(frame)))
    if not frames:
        raise ValueError("The selected network CSV file(s) contain no event rows")

    # Require compatible headers across a multi-file dataset. This avoids silent
    # column drift while still allowing ordinary consecutive archive files.
    canonical_columns = tuple(str(value) for value in frames[0].columns if not str(value).startswith("__lmas_"))
    for frame in frames[1:]:
        columns = tuple(str(value) for value in frame.columns if not str(value).startswith("__lmas_"))
        if set(columns) != set(canonical_columns):
            raise ValueError("Network CSV files selected together must use compatible columns")
    frame = pd.concat(frames, ignore_index=True, sort=False)
    resolved = _resolve_columns(frame.columns, options.column_mapping)
    if "longitude" not in resolved or "latitude" not in resolved:
        raise ValueError("Could not identify latitude and longitude columns in the network CSV")

    provider = _infer_provider(source_paths, canonical_columns, options.provider)
    time_ns = _parse_times(frame, resolved, options.time_format)
    lon = _numeric(frame, resolved.get("longitude"))
    lat = _numeric(frame, resolved.get("latitude"))
    current = _scale_peak_current(
        _numeric(frame, resolved.get("peak_current")), resolved.get("peak_current"), options.peak_current_unit
    )
    raw_polarity = _string(frame, resolved.get("polarity"))
    polarity = _normalize_polarity(raw_polarity, current)
    original_event_type = _string(frame, resolved.get("event_type"), default="")
    event_type = _normalize_event_type(original_event_type, polarity, provider=provider)
    altitude = _scale_distance(
        _numeric(frame, resolved.get("altitude")), resolved.get("altitude"), options.altitude_unit, altitude=True
    )
    major = _scale_distance(
        _numeric(frame, resolved.get("ellipse_major")), resolved.get("ellipse_major"), options.ellipse_unit, altitude=False
    )
    minor = _scale_distance(
        _numeric(frame, resolved.get("ellipse_minor")), resolved.get("ellipse_minor"), options.ellipse_unit, altitude=False
    )
    angle_column = resolved.get("ellipse_angle")
    angle = _numeric(frame, angle_column)
    angle_key = _key(angle_column or "")
    if "bearing" in angle_key or "azimuth" in angle_key:
        angle = np.mod(90.0 - angle, 180.0)
    sensors_float = _numeric(frame, resolved.get("sensor_count"), default=-1.0)
    sensors = np.where(np.isfinite(sensors_float), np.rint(sensors_float), -1).astype(np.int32)
    quality = _string(frame, resolved.get("quality"), default="")
    amplitude = _numeric(frame, resolved.get("amplitude"))

    if "id" in resolved:
        original_id = _string(frame, resolved.get("id"), default="")
    else:
        original_id = np.asarray(
            [f"{filename}:{row}" for filename, row in zip(frame["__lmas_source_file"], frame["__lmas_source_row"], strict=True)],
            dtype="U128",
        )

    valid = (
        (time_ns != NAT_NS)
        & np.isfinite(lon) & np.isfinite(lat)
        & (lon >= -180.0) & (lon <= 180.0)
        & (lat >= -90.0) & (lat <= 90.0)
    )
    if not np.any(valid):
        raise ValueError("No rows contain a valid UTC time, latitude, and longitude")

    arrays = {
        "time_ns": time_ns[valid].astype(np.int64, copy=False),
        "longitude_deg": lon[valid].astype(float, copy=False),
        "latitude_deg": lat[valid].astype(float, copy=False),
        "altitude_m": altitude[valid].astype(float, copy=False),
        "event_type": event_type[valid].astype("U16", copy=False),
        "original_event_type": original_event_type[valid].astype("U64", copy=False),
        "polarity": polarity[valid].astype(np.int8, copy=False),
        "peak_current_ka": current[valid].astype(float, copy=False),
        "amplitude": amplitude[valid].astype(float, copy=False),
        "sensor_count": sensors[valid].astype(np.int32, copy=False),
        "quality": quality[valid].astype("U32", copy=False),
        "ellipse_major_km": major[valid].astype(float, copy=False),
        "ellipse_minor_km": minor[valid].astype(float, copy=False),
        "ellipse_angle_deg": angle[valid].astype(float, copy=False),
        "original_id": original_id[valid].astype("U128", copy=False),
    }
    order = np.argsort(arrays["time_ns"], kind="stable")
    events = NetworkEvents(**{name: values[order] for name, values in arrays.items()})

    display = _display_name(provider, options.display_name)
    product = options.product_name or ("Earth Networks lightning events" if provider == "entln" else "Ground network events")
    identity = NetworkIdentity(
        provider_id=provider,
        display_name=display,
        product_name=product,
        observation_start_ns=int(events.time_ns[0]),
        observation_end_ns=int(events.time_ns[-1]),
        source_files=tuple(file_records),
        schema={
            **{canonical: column for canonical, column in resolved.items()},
            **({"event_type_decoding": "ENTLN numeric: 0 = CG, 1 = IC"} if provider == "entln" else {}),
        },
    )
    return NetworkObservation(identity, events)


def write_generic_network_example(path: str | Path, *, center_time: str = "2019-04-30T14:49:14.265") -> Path:
    """Write a small redistribution-safe CSV used by docs and tests."""
    import pandas as pd

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    center = np.datetime64(center_time, "ns")
    offsets_ms = np.array([-180, -92, -41, 7, 64, 145, 231], dtype=np.int64)
    timestamps = center + offsets_ms.astype("timedelta64[ms]")
    frame = pd.DataFrame(
        {
            "timestamp_utc": timestamps.astype("datetime64[us]").astype(str) + "Z",
            "latitude": [35.42, 35.46, 35.48, 35.50, 35.52, 35.55, 35.57],
            "longitude": [-97.64, -97.61, -97.59, -97.56, -97.53, -97.50, -97.47],
            "event_type": ["IC", "-CG", "IC", "+CG", "IC", "-CG", "IC"],
            "polarity": [0, -1, 0, 1, 0, -1, 0],
            "peak_current_kA": [np.nan, -31.4, np.nan, 22.8, np.nan, -18.7, np.nan],
            "sensor_count": [12, 18, 14, 21, 13, 17, 11],
            "ellipse_major_km": [0.8, 0.5, 0.7, 0.4, 0.9, 0.6, 1.0],
            "ellipse_minor_km": [0.4, 0.2, 0.3, 0.2, 0.5, 0.3, 0.5],
            "ellipse_angle_deg": [20, 55, 95, 130, 160, 35, 75],
            "event_id": [f"EXAMPLE-{index:03d}" for index in range(7)],
        }
    )
    frame.to_csv(destination, index=False)
    return destination


__all__ = ["ALIASES", "NetworkCSVOptions", "read_network_csv", "write_generic_network_example"]
