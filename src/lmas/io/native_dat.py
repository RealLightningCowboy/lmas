from __future__ import annotations

"""LMAS-native reader for solved LMA ``.dat`` and ``.dat.gz`` files.

The file-format interpretation in this module is derived from the MIT-licensed
``xlma-python`` reader by Eric Bruning.  LMAS keeps the upstream notice in
``licenses/xlma-python-LICENSE.txt`` and ``THIRD_PARTY_NOTICES.md``.
"""

from dataclasses import dataclass
from datetime import datetime
import gzip
from pathlib import Path
from typing import Iterable, TextIO

import numpy as np
import pandas as pd
import xarray as xr

from ..errors import DatasetError

EVENT_DIM = "number_of_events"
STATION_DIM = "number_of_stations"
NATIVE_READER_VERSION = "1"


def _open_text(path: Path) -> TextIO:
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, mode="rt", encoding="utf-8", errors="replace")
    return path.open(mode="rt", encoding="utf-8", errors="replace")


def _parse_number(value: object, *, default: float = np.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _mask_to_int(values: Iterable[object]) -> np.ndarray:
    result: list[int] = []
    for value in values:
        text = str(value).strip()
        try:
            result.append(int(text, 0))
        except ValueError:
            try:
                result.append(int(text, 16))
            except ValueError as exc:
                raise DatasetError(f"Invalid LMA station mask value: {text!r}") from exc
    if not result:
        return np.asarray([], dtype=np.uint64)
    maximum = max(result)
    dtype = np.uint32 if maximum <= np.iinfo(np.uint32).max else np.uint64
    return np.asarray(result, dtype=dtype)


@dataclass(frozen=True)
class NativeLMAFile:
    path: Path
    start_day: np.datetime64
    start_time: np.datetime64
    network_name: str
    center_latitude: float
    center_longitude: float
    center_altitude_m: float
    minimum_stations: int | None
    maximum_chi2: float | None
    analysis_program: str
    analysis_version: str
    file_created: str
    station_codes: np.ndarray
    station_names: np.ndarray
    station_latitudes: np.ndarray
    station_longitudes: np.ndarray
    station_altitudes_m: np.ndarray
    station_event_fraction: np.ndarray
    station_power_ratio: np.ndarray
    event_time: np.ndarray
    event_latitude: np.ndarray
    event_longitude: np.ndarray
    event_altitude_m: np.ndarray
    event_chi2: np.ndarray
    event_power: np.ndarray
    event_mask: np.ndarray
    event_stations: np.ndarray
    event_contributing_stations: np.ndarray


def _header_value(line: str) -> str:
    return line.split(":", 1)[1].strip() if ":" in line else ""


def read_native_lma_file(path: str | Path) -> NativeLMAFile:
    source = Path(path).expanduser().resolve()
    header: dict[str, object] = {}
    station_info: list[tuple[str, str, float, float, float, float]] = []
    station_data: dict[str, tuple[float, float]] = {}
    data_names: list[str] = []
    data_start_line: int | None = None

    try:
        with _open_text(source) as stream:
            for line_number, raw in enumerate(stream):
                line = raw.rstrip("\r\n")
                if line.startswith("Analysis program:"):
                    header["analysis_program"] = _header_value(line)
                elif line.startswith("Analysis program version:"):
                    header["analysis_version"] = _header_value(line)
                elif line.startswith("File created:") or line.startswith("Analysis finished:"):
                    header["file_created"] = _header_value(line)
                elif line.startswith("Location:"):
                    header["network_name"] = _header_value(line)
                elif line.startswith("Data start time:"):
                    value = _header_value(line)
                    try:
                        parsed = datetime.strptime(value, "%m/%d/%y %H:%M:%S")
                    except ValueError as exc:
                        raise DatasetError(
                            f"Unsupported LMA data-start timestamp in {source}: {value!r}"
                        ) from exc
                    header["start_time"] = np.datetime64(parsed, "ns")
                    header["start_day"] = np.datetime64(parsed.date(), "ns")
                elif line.startswith("Coordinate center"):
                    parts = line.split()
                    try:
                        header["center"] = tuple(float(value) for value in parts[-3:])
                    except ValueError as exc:
                        raise DatasetError(f"Invalid coordinate center in {source}") from exc
                elif line.startswith("Minimum number of stations per solution:"):
                    try:
                        header["minimum_stations"] = int(_header_value(line))
                    except ValueError:
                        header["minimum_stations"] = None
                elif line.startswith("Maximum reduced chi-squared:"):
                    try:
                        header["maximum_chi2"] = float(_header_value(line))
                    except ValueError:
                        header["maximum_chi2"] = None
                elif line.startswith("Station mask order:"):
                    header["mask_order"] = _header_value(line).split()[-1]
                elif line.startswith("Data:"):
                    data_names = [value.strip() for value in line[5:].split(",")]
                elif line.startswith("Sta_info:"):
                    parts = line.split()
                    if len(parts) >= 9:
                        code = parts[1]
                        name = " ".join(parts[2:-6]).strip()
                        try:
                            latitude, longitude, altitude, delay = (
                                float(value) for value in parts[-6:-2]
                            )
                        except ValueError as exc:
                            raise DatasetError(
                                f"Invalid station-information row in {source}: {line}"
                            ) from exc
                        station_info.append(
                            (code, name, latitude, longitude, altitude, delay)
                        )
                elif line.startswith("Sta_data:"):
                    parts = line.split()
                    if len(parts) >= 9:
                        code = parts[1]
                        tail = parts[-7:]
                        station_data[code] = (
                            _parse_number(tail[3]),
                            _parse_number(tail[5]),
                        )
                elif line.strip() == "*** data ***":
                    data_start_line = line_number
                    break
    except OSError as exc:
        raise DatasetError(f"Could not read LMA file {source}: {exc}") from exc

    required = ("start_time", "start_day", "center", "mask_order")
    missing = [name for name in required if name not in header]
    if missing:
        raise DatasetError(
            f"LMA header in {source} is missing: " + ", ".join(missing)
        )
    if data_start_line is None or not data_names:
        raise DatasetError(f"LMA file {source} has no readable data section")

    compression = "gzip" if source.name.lower().endswith(".gz") else None
    try:
        frame = pd.read_csv(
            source,
            compression=compression,
            sep=r"\s+",
            header=None,
            skiprows=data_start_line + 1,
            names=data_names,
            on_bad_lines="skip",
        )
    except pd.errors.EmptyDataError:
        frame = pd.DataFrame(columns=data_names)
    except Exception as exc:
        raise DatasetError(f"Could not parse LMA event rows in {source}: {exc}") from exc

    aliases = {
        "time": ("time (UT sec of day)", "time", "seconds"),
        "latitude": ("lat", "latitude"),
        "longitude": ("lon", "longitude", "long"),
        "altitude": ("alt(m)", "alt", "altitude"),
        "chi2": ("reduced chi^2", "chi2", "reduced_chi2"),
        "power": ("P(dBW)", "power", "p(dbw)"),
        "mask": ("mask",),
    }

    def column(kind: str, *, required_field: bool = True) -> pd.Series:
        by_lower = {str(name).strip().lower(): name for name in frame.columns}
        for candidate in aliases[kind]:
            name = by_lower.get(candidate.lower())
            if name is not None:
                return frame[name]
        if required_field:
            raise DatasetError(
                f"LMA file {source} is missing the {kind} data column; found {list(frame.columns)!r}"
            )
        return pd.Series(np.full(len(frame), np.nan))

    seconds_series = pd.to_numeric(column("time"), errors="coerce")
    event_time = (
        pd.to_timedelta(seconds_series, unit="s").to_numpy(dtype="timedelta64[ns]")
        + np.asarray(header["start_day"], dtype="datetime64[ns]")
    )
    mask = _mask_to_int(column("mask").to_numpy())

    codes = np.asarray([item[0] for item in station_info], dtype="U32")
    names = np.asarray([item[1] for item in station_info], dtype="U128")
    latitudes = np.asarray([item[2] for item in station_info], dtype=np.float64)
    longitudes = np.asarray([item[3] for item in station_info], dtype=np.float64)
    altitudes = np.asarray([item[4] for item in station_info], dtype=np.float64)
    event_fraction = np.asarray(
        [station_data.get(code, (np.nan, np.nan))[0] for code in codes], dtype=np.float64
    )
    power_ratio = np.asarray(
        [station_data.get(code, (np.nan, np.nan))[1] for code in codes], dtype=np.float64
    )

    mask_order = str(header["mask_order"])
    bit_by_code = {code: index for index, code in enumerate(mask_order[::-1])}
    contributions = np.zeros((len(frame), len(codes)), dtype=np.uint8)
    for station_index, code in enumerate(codes):
        bit = bit_by_code.get(str(code))
        if bit is not None:
            contributions[:, station_index] = ((mask >> bit) & 1).astype(np.uint8)
    station_count = contributions.sum(axis=1).astype(np.uint8)

    center_lat, center_lon, center_alt = header["center"]  # type: ignore[misc]
    return NativeLMAFile(
        path=source,
        start_day=np.asarray(header["start_day"], dtype="datetime64[ns]"),
        start_time=np.asarray(header["start_time"], dtype="datetime64[ns]"),
        network_name=str(header.get("network_name") or "LMA"),
        center_latitude=float(center_lat),
        center_longitude=float(center_lon),
        center_altitude_m=float(center_alt),
        minimum_stations=(
            None if header.get("minimum_stations") is None else int(header["minimum_stations"])
        ),
        maximum_chi2=(
            None if header.get("maximum_chi2") is None else float(header["maximum_chi2"])
        ),
        analysis_program=str(header.get("analysis_program") or "unknown"),
        analysis_version=str(header.get("analysis_version") or "unknown"),
        file_created=str(header.get("file_created") or "unknown"),
        station_codes=codes,
        station_names=names,
        station_latitudes=latitudes,
        station_longitudes=longitudes,
        station_altitudes_m=altitudes,
        station_event_fraction=event_fraction,
        station_power_ratio=power_ratio,
        event_time=event_time.astype("datetime64[ns]"),
        event_latitude=pd.to_numeric(column("latitude"), errors="coerce").to_numpy(dtype=np.float64),
        event_longitude=pd.to_numeric(column("longitude"), errors="coerce").to_numpy(dtype=np.float64),
        event_altitude_m=pd.to_numeric(column("altitude"), errors="coerce").to_numpy(dtype=np.float64),
        event_chi2=pd.to_numeric(column("chi2"), errors="coerce").to_numpy(dtype=np.float64),
        event_power=pd.to_numeric(column("power"), errors="coerce").to_numpy(dtype=np.float64),
        event_mask=mask,
        event_stations=station_count,
        event_contributing_stations=contributions,
    )


def _first_finite(values: Iterable[float]) -> float:
    array = np.asarray(tuple(values), dtype=float)
    finite = array[np.isfinite(array)]
    return float(finite[0]) if finite.size else float("nan")


def _merge_attributes(files: list[NativeLMAFile], attribute: str) -> str:
    values = [str(getattr(item, attribute)) for item in files]
    return values[0] if len(set(values)) == 1 else "; ".join(dict.fromkeys(values))


def native_dat_dataset(paths: Iterable[str | Path]) -> xr.Dataset:
    files = [read_native_lma_file(path) for path in paths]
    if not files:
        raise DatasetError("No native LMA files were provided")

    station_codes = list(dict.fromkeys(code for item in files for code in item.station_codes.tolist()))
    station_index = {code: index for index, code in enumerate(station_codes)}
    station_count = len(station_codes)

    station_names = np.full(station_count, "", dtype="U128")
    station_latitude = np.full(station_count, np.nan, dtype=np.float64)
    station_longitude = np.full(station_count, np.nan, dtype=np.float64)
    station_altitude = np.full(station_count, np.nan, dtype=np.float64)
    station_event_fraction_values: list[list[float]] = [[] for _ in station_codes]
    station_power_ratio_values: list[list[float]] = [[] for _ in station_codes]

    event_time: list[np.ndarray] = []
    event_latitude: list[np.ndarray] = []
    event_longitude: list[np.ndarray] = []
    event_altitude: list[np.ndarray] = []
    event_chi2: list[np.ndarray] = []
    event_power: list[np.ndarray] = []
    event_mask: list[np.ndarray] = []
    event_stations: list[np.ndarray] = []
    event_id: list[np.ndarray] = []
    event_contributions: list[np.ndarray] = []
    next_event_id = 0

    for item in files:
        for local_index, code in enumerate(item.station_codes.tolist()):
            global_index = station_index[code]
            if not station_names[global_index]:
                station_names[global_index] = item.station_names[local_index]
                station_latitude[global_index] = item.station_latitudes[local_index]
                station_longitude[global_index] = item.station_longitudes[local_index]
                station_altitude[global_index] = item.station_altitudes_m[local_index]
            station_event_fraction_values[global_index].append(
                item.station_event_fraction[local_index]
            )
            station_power_ratio_values[global_index].append(item.station_power_ratio[local_index])

        aligned = np.zeros((item.event_time.size, station_count), dtype=np.uint8)
        for local_index, code in enumerate(item.station_codes.tolist()):
            aligned[:, station_index[code]] = item.event_contributing_stations[:, local_index]
        count = item.event_time.size
        event_time.append(item.event_time)
        event_latitude.append(item.event_latitude)
        event_longitude.append(item.event_longitude)
        event_altitude.append(item.event_altitude_m)
        event_chi2.append(item.event_chi2)
        event_power.append(item.event_power)
        event_mask.append(item.event_mask)
        event_stations.append(aligned.sum(axis=1).astype(np.uint8))
        event_id.append(np.arange(next_event_id, next_event_id + count, dtype=np.uint64))
        event_contributions.append(aligned)
        next_event_id += count

    def concatenate(chunks: list[np.ndarray], dtype=None) -> np.ndarray:
        if not chunks:
            return np.asarray([], dtype=dtype)
        result = np.concatenate(chunks)
        return result.astype(dtype, copy=False) if dtype is not None else result

    times = concatenate(event_time, "datetime64[ns]")
    order = np.argsort(times, kind="stable")
    center_latitude = float(np.nanmean([item.center_latitude for item in files]))
    center_longitude = float(np.nanmean([item.center_longitude for item in files]))
    center_altitude = float(np.nanmean([item.center_altitude_m for item in files]))

    station_event_fraction = np.asarray(
        [np.nanmean(values) if np.any(np.isfinite(values)) else np.nan for values in station_event_fraction_values],
        dtype=np.float32,
    )
    station_power_ratio = np.asarray(
        [np.nanmean(values) if np.any(np.isfinite(values)) else np.nan for values in station_power_ratio_values],
        dtype=np.float32,
    )

    minimum_stations = [item.minimum_stations for item in files if item.minimum_stations is not None]
    maximum_chi2 = [item.maximum_chi2 for item in files if item.maximum_chi2 is not None]

    dataset = xr.Dataset(
        data_vars={
            "network_center_latitude": xr.DataArray(
                center_latitude,
                attrs={"units": "degrees_north", "standard_name": "latitude"},
            ),
            "network_center_longitude": xr.DataArray(
                center_longitude,
                attrs={"units": "degrees_east", "standard_name": "longitude"},
            ),
            "network_center_altitude": xr.DataArray(
                center_altitude,
                attrs={"units": "meters", "standard_name": "altitude"},
            ),
            "station_code": xr.DataArray(np.asarray(station_codes, dtype="U32"), dims=(STATION_DIM,)),
            "station_name": xr.DataArray(station_names, dims=(STATION_DIM,)),
            "station_network": xr.DataArray(
                np.asarray([files[0].network_name] * station_count, dtype="U128"),
                dims=(STATION_DIM,),
            ),
            "station_latitude": xr.DataArray(
                station_latitude.astype(np.float32),
                dims=(STATION_DIM,),
                attrs={"units": "degrees_north", "standard_name": "latitude"},
            ),
            "station_longitude": xr.DataArray(
                station_longitude.astype(np.float32),
                dims=(STATION_DIM,),
                attrs={"units": "degrees_east", "standard_name": "longitude"},
            ),
            "station_altitude": xr.DataArray(
                station_altitude.astype(np.float32),
                dims=(STATION_DIM,),
                attrs={"units": "meters", "standard_name": "altitude", "positive": "up"},
            ),
            "station_event_fraction": xr.DataArray(
                station_event_fraction,
                dims=(STATION_DIM,),
                attrs={"units": "percent"},
            ),
            "station_power_ratio": xr.DataArray(station_power_ratio, dims=(STATION_DIM,)),
            "event_time": xr.DataArray(times[order], dims=(EVENT_DIM,), attrs={"standard_name": "time"}),
            "event_latitude": xr.DataArray(
                concatenate(event_latitude, np.float32)[order],
                dims=(EVENT_DIM,),
                attrs={"units": "degrees_north", "standard_name": "latitude"},
            ),
            "event_longitude": xr.DataArray(
                concatenate(event_longitude, np.float32)[order],
                dims=(EVENT_DIM,),
                attrs={"units": "degrees_east", "standard_name": "longitude"},
            ),
            "event_altitude": xr.DataArray(
                concatenate(event_altitude, np.float32)[order],
                dims=(EVENT_DIM,),
                attrs={"units": "meters", "standard_name": "altitude", "positive": "up"},
            ),
            "event_power": xr.DataArray(
                concatenate(event_power, np.float32)[order],
                dims=(EVENT_DIM,),
                attrs={"units": "dBW", "long_name": "VHF source power"},
            ),
            "event_chi2": xr.DataArray(
                concatenate(event_chi2, np.float32)[order],
                dims=(EVENT_DIM,),
                attrs={
                    "long_name": "Reduced chi-square goodness of fit",
                    "valid_range": [0.0, max(maximum_chi2, default=np.nan)],
                },
            ),
            "event_mask": xr.DataArray(concatenate(event_mask)[order], dims=(EVENT_DIM,)),
            "event_stations": xr.DataArray(
                concatenate(event_stations, np.uint8)[order],
                dims=(EVENT_DIM,),
                attrs={"valid_range": [min(minimum_stations, default=0), 255]},
            ),
            "event_id": xr.DataArray(concatenate(event_id, np.uint64)[order], dims=(EVENT_DIM,)),
            "event_contributing_stations": xr.DataArray(
                np.concatenate(event_contributions, axis=0)[order],
                dims=(EVENT_DIM, STATION_DIM),
                attrs={"valid_range": [0, 1]},
            ),
        },
        attrs={
            "title": "Lightning Mapping Array solved-source dataset",
            "source": "VHF Lightning Mapping Array",
            "network_name": _merge_attributes(files, "network_name"),
            "history": "; ".join(f"LMA source file created {item.file_created}" for item in files),
            "event_algorithm_name": _merge_attributes(files, "analysis_program"),
            "event_algorithm_version": _merge_attributes(files, "analysis_version"),
            "lmas_native_reader_version": NATIVE_READER_VERSION,
        },
    )
    return dataset


__all__ = ["NATIVE_READER_VERSION", "native_dat_dataset", "read_native_lma_file"]
