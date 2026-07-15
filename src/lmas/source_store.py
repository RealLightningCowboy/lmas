from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping
import copy

import numpy as np
import xarray as xr

from .errors import DatasetError

EVENT_DIM = "number_of_events"


def _immutable_array(values: Any, *, copy_values: bool) -> np.ndarray:
    array = np.array(values, copy=copy_values, order="C", subok=False)
    if array.ndim and not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    array.setflags(write=False)
    return array


def _frozen_mapping(values: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(values))


@dataclass(frozen=True, slots=True)
class LmaSourceStore:
    """Immutable NumPy-backed representation of one solved LMA dataset.

    The store preserves every xarray variable, its dimensions and attributes,
    but exposes the numerical arrays directly for filtering and rendering.
    ``to_xarray`` remains the compatibility boundary for notebooks, exports,
    plugins, and existing LMAS plotting code.
    """

    _arrays: Mapping[str, np.ndarray]
    _variable_dims: Mapping[str, tuple[str, ...]]
    _variable_attrs: Mapping[str, Mapping[str, Any]]
    _dataset_attrs: Mapping[str, Any]
    _coord_names: frozenset[str]
    _sizes: Mapping[str, int]
    event_dimension: str = EVENT_DIM

    @classmethod
    def from_xarray(
        cls,
        dataset: xr.Dataset,
        *,
        event_dimension: str = EVENT_DIM,
    ) -> "LmaSourceStore":
        if not isinstance(dataset, xr.Dataset):
            raise DatasetError("LMA source-store input must be an xarray.Dataset")
        arrays: dict[str, np.ndarray] = {}
        dims: dict[str, tuple[str, ...]] = {}
        attrs: dict[str, Mapping[str, Any]] = {}
        for name, variable in dataset.variables.items():
            arrays[name] = _immutable_array(variable.values, copy_values=True)
            dims[name] = tuple(str(value) for value in variable.dims)
            attrs[name] = _frozen_mapping(copy.deepcopy(dict(variable.attrs)))
        event_count = int(dataset.sizes.get(event_dimension, 0))
        if event_dimension in dataset.sizes and "event_source_index" not in arrays:
            arrays["event_source_index"] = _immutable_array(
                np.arange(event_count, dtype=np.int64), copy_values=False
            )
            dims["event_source_index"] = (event_dimension,)
            attrs["event_source_index"] = _frozen_mapping(
                {"long_name": "stable LMAS source index"}
            )
        return cls(
            _arrays=_frozen_mapping(arrays),
            _variable_dims=_frozen_mapping(dims),
            _variable_attrs=_frozen_mapping(attrs),
            _dataset_attrs=_frozen_mapping(copy.deepcopy(dict(dataset.attrs))),
            _coord_names=frozenset(str(name) for name in dataset.coords),
            _sizes=_frozen_mapping({str(name): int(size) for name, size in dataset.sizes.items()}),
            event_dimension=str(event_dimension),
        )

    @classmethod
    def _from_components(
        cls,
        *,
        arrays: Mapping[str, np.ndarray],
        variable_dims: Mapping[str, tuple[str, ...]],
        variable_attrs: Mapping[str, Mapping[str, Any]],
        dataset_attrs: Mapping[str, Any],
        coord_names: Iterable[str],
        sizes: Mapping[str, int],
        event_dimension: str,
    ) -> "LmaSourceStore":
        return cls(
            _arrays=_frozen_mapping(arrays),
            _variable_dims=_frozen_mapping(variable_dims),
            _variable_attrs=_frozen_mapping(variable_attrs),
            _dataset_attrs=_frozen_mapping(dataset_attrs),
            _coord_names=frozenset(coord_names),
            _sizes=_frozen_mapping(sizes),
            event_dimension=event_dimension,
        )

    def __contains__(self, name: object) -> bool:
        return name in self._arrays

    def __getitem__(self, name: str) -> np.ndarray:
        try:
            return self._arrays[name]
        except KeyError as exc:
            raise DatasetError(f"LMA source store has no {name} field") from exc

    def get(self, name: str, default: Any = None) -> np.ndarray | Any:
        return self._arrays.get(name, default)

    @property
    def fields(self) -> tuple[str, ...]:
        return tuple(self._arrays)

    @property
    def attrs(self) -> Mapping[str, Any]:
        return self._dataset_attrs

    @property
    def sizes(self) -> Mapping[str, int]:
        return self._sizes

    @property
    def event_count(self) -> int:
        return int(self._sizes.get(self.event_dimension, 0))

    def dimensions(self, name: str) -> tuple[str, ...]:
        try:
            return self._variable_dims[name]
        except KeyError as exc:
            raise DatasetError(f"LMA source store has no {name} field") from exc

    def field_attrs(self, name: str) -> Mapping[str, Any]:
        try:
            return self._variable_attrs[name]
        except KeyError as exc:
            raise DatasetError(f"LMA source store has no {name} field") from exc

    def event_array(self, name: str, *, dtype: Any | None = None) -> np.ndarray:
        values = self[name]
        if self.event_dimension not in self.dimensions(name):
            raise DatasetError(f"{name} is not aligned with {self.event_dimension}")
        return np.asarray(values, dtype=dtype) if dtype is not None else values

    def select_events(self, selector: np.ndarray | Iterable[int]) -> "LmaSourceStore":
        values = np.asarray(selector)
        if values.dtype == bool:
            if values.ndim != 1 or values.size != self.event_count:
                raise DatasetError("Boolean source-store selection has the wrong length")
            indices = np.flatnonzero(values)
        else:
            try:
                indices = np.asarray(values, dtype=np.int64).reshape(-1)
            except (TypeError, ValueError) as exc:
                raise DatasetError("Source-store event selection is not integer-like") from exc
            if indices.size and (indices.min() < 0 or indices.max() >= self.event_count):
                raise DatasetError("Source-store event selection is out of range")

        arrays: dict[str, np.ndarray] = {}
        for name, array in self._arrays.items():
            variable_dims = self._variable_dims[name]
            if self.event_dimension in variable_dims:
                axis = variable_dims.index(self.event_dimension)
                arrays[name] = _immutable_array(
                    np.take(array, indices, axis=axis), copy_values=False
                )
            else:
                arrays[name] = array
        sizes = dict(self._sizes)
        sizes[self.event_dimension] = int(indices.size)
        return self._from_components(
            arrays=arrays,
            variable_dims=self._variable_dims,
            variable_attrs=self._variable_attrs,
            dataset_attrs=self._dataset_attrs,
            coord_names=self._coord_names,
            sizes=sizes,
            event_dimension=self.event_dimension,
        )

    def with_attrs(self, **updates: Any) -> "LmaSourceStore":
        attrs = dict(self._dataset_attrs)
        attrs.update(copy.deepcopy(updates))
        return self._from_components(
            arrays=self._arrays,
            variable_dims=self._variable_dims,
            variable_attrs=self._variable_attrs,
            dataset_attrs=attrs,
            coord_names=self._coord_names,
            sizes=self._sizes,
            event_dimension=self.event_dimension,
        )

    def to_xarray(self, *, copy_arrays: bool = False) -> xr.Dataset:
        data_vars: dict[str, xr.DataArray] = {}
        coords: dict[str, xr.DataArray] = {}
        for name, array in self._arrays.items():
            values = np.array(array, copy=True) if copy_arrays else array
            variable = xr.DataArray(
                values,
                dims=self._variable_dims[name],
                attrs=copy.deepcopy(dict(self._variable_attrs[name])),
            )
            (coords if name in self._coord_names else data_vars)[name] = variable
        return xr.Dataset(
            data_vars=data_vars,
            coords=coords,
            attrs=copy.deepcopy(dict(self._dataset_attrs)),
        )


__all__ = ["EVENT_DIM", "LmaSourceStore"]
