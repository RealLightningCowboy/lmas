from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from lmas.glm import GLMDataError, GLMObservation, read_glm_l2_lcfa


def _packed_dataset(handle, name, values, *, dtype, scale=None, offset=None, units=None, fill=None):
    target = np.dtype(dtype)
    if target.kind == "i":
        unsigned = np.asarray(values, dtype=np.dtype(target.str.replace("i", "u")))
        array = unsigned.view(target)
    else:
        array = np.asarray(values, dtype=target)
    dataset = handle.create_dataset(name, data=array)
    if np.dtype(dtype).kind == "i":
        dataset.attrs["_Unsigned"] = np.bytes_(b"true")
    if scale is not None:
        dataset.attrs["scale_factor"] = np.asarray([scale], dtype=np.float32)
    if offset is not None:
        dataset.attrs["add_offset"] = np.asarray([offset], dtype=np.float32)
    if units is not None:
        dataset.attrs["units"] = np.bytes_(units.encode())
    if fill is not None:
        dataset.attrs["_FillValue"] = np.asarray([fill], dtype=dtype)
    return dataset


def _write_lcfa(path: Path, *, platform: str, slot: str, second: int, base_id: int) -> None:
    origin = f"2020-01-01 00:00:{second:02d}.000"
    with h5py.File(path, "w") as f:
        f.attrs["dataset_name"] = np.bytes_(path.name.encode())
        f.attrs["platform_ID"] = np.bytes_(platform.encode())
        f.attrs["orbital_slot"] = np.bytes_(slot.encode())
        f.attrs["time_coverage_start"] = np.bytes_((origin + "Z").encode())
        f.attrs["time_coverage_end"] = np.bytes_(f"2020-01-01T00:00:{second + 2:02d}.000Z".encode())
        f.attrs["processing_level"] = np.bytes_(b"L2")

        projection = f.create_dataset("goes_lat_lon_projection", data=np.int32(0))
        projection.attrs["semi_major_axis"] = np.asarray([6378137.0])
        projection.attrs["semi_minor_axis"] = np.asarray([6356752.31414])
        projection.attrs["inverse_flattening"] = np.asarray([298.2572221])
        projection.attrs["longitude_of_prime_meridian"] = np.asarray([0.0])
        f.create_dataset("nominal_satellite_subpoint_lat", data=np.float32(0.0))
        f.create_dataset("nominal_satellite_subpoint_lon", data=np.float32(-75.2 if "East" in slot else -137.2))
        f.create_dataset("nominal_satellite_height", data=np.float32(35786.0))

        _packed_dataset(f, "event_id", [base_id + 1, base_id + 2, base_id + 3], dtype=np.int32)
        _packed_dataset(
            f,
            "event_time_offset",
            [100, 200, 300],
            dtype=np.int16,
            scale=0.001,
            offset=0.0,
            units=f"seconds since {origin}",
        )
        # 40000 exercises signed-storage/unsigned-decoding behavior.
        _packed_dataset(f, "event_lat", [40000, 40010, 40020], dtype=np.int16, scale=0.002, offset=-66.0)
        _packed_dataset(f, "event_lon", [20000, 20010, 20020], dtype=np.int16, scale=0.002, offset=-140.0)
        _packed_dataset(f, "event_energy", [10, 20, 30], dtype=np.int16, scale=1e-15, offset=0.0, fill=-1)
        _packed_dataset(f, "event_parent_group_id", [base_id + 10, base_id + 10, base_id + 11], dtype=np.int32)

        _packed_dataset(f, "group_id", [base_id + 10, base_id + 11], dtype=np.int32)
        _packed_dataset(f, "group_time_offset", [150, 300], dtype=np.int16, scale=0.001, offset=0.0, units=f"seconds since {origin}")
        _packed_dataset(f, "group_frame_time_offset", [150, 300], dtype=np.int16, scale=0.001, offset=0.0, units=f"seconds since {origin}")
        f.create_dataset("group_lat", data=np.asarray([14.01, 14.03], dtype=np.float32))
        f.create_dataset("group_lon", data=np.asarray([-99.99, -99.97], dtype=np.float32))
        _packed_dataset(f, "group_area", [10, 20], dtype=np.int16, scale=1000.0, offset=0.0, fill=-1)
        _packed_dataset(f, "group_energy", [40, 50], dtype=np.int16, scale=1e-15, offset=0.0, fill=-1)
        _packed_dataset(f, "group_parent_flash_id", [base_id + 100, base_id + 100], dtype=np.int16)
        _packed_dataset(f, "group_quality_flag", [0, 0], dtype=np.int16, fill=-1)

        _packed_dataset(f, "flash_id", [base_id + 100], dtype=np.int16)
        for name, values in (
            ("flash_time_offset_of_first_event", [100]),
            ("flash_time_offset_of_last_event", [300]),
            ("flash_frame_time_offset_of_first_event", [100]),
            ("flash_frame_time_offset_of_last_event", [300]),
        ):
            _packed_dataset(f, name, values, dtype=np.int16, scale=0.001, offset=0.0, units=f"seconds since {origin}")
        f.create_dataset("flash_lat", data=np.asarray([14.02], dtype=np.float32))
        f.create_dataset("flash_lon", data=np.asarray([-99.98], dtype=np.float32))
        _packed_dataset(f, "flash_area", [30], dtype=np.int16, scale=1000.0, offset=0.0, fill=-1)
        _packed_dataset(f, "flash_energy", [90], dtype=np.int16, scale=1e-15, offset=0.0, fill=-1)
        _packed_dataset(f, "flash_quality_flag", [0], dtype=np.int16, fill=-1)


def test_native_reader_decodes_packed_data_and_hierarchy(tmp_path: Path) -> None:
    path = tmp_path / "OR_GLM-L2-LCFA_G16_test.nc"
    _write_lcfa(path, platform="G16", slot="GOES-East", second=0, base_id=1000)
    glm = read_glm_l2_lcfa(path)

    assert glm.identity.platform_id == "G16"
    assert glm.identity.operational_role == "east"
    assert glm.identity.display_name == "GLM East — G16"
    assert len(glm.events) == 3
    assert len(glm.groups) == 2
    assert len(glm.flashes) == 1
    assert glm.validate_hierarchy().valid
    assert np.isclose(glm.events.latitude_deg[0], 40000 * 0.002 - 66.0, atol=1e-5)
    assert np.array_equal(glm.events.parent_group_index, [0, 0, 1])
    assert np.array_equal(glm.events.parent_flash_index, [0, 0, 0])
    assert np.array_equal(glm.groups.child_event_count, [2, 1])
    assert np.array_equal(glm.flashes.child_event_count, [3])


def test_multifile_concatenation_selection_and_xarray_roundtrip(tmp_path: Path) -> None:
    first = tmp_path / "OR_GLM-L2-LCFA_G16_a.nc"
    second = tmp_path / "OR_GLM-L2-LCFA_G16_b.nc"
    _write_lcfa(first, platform="G16", slot="GOES-East", second=0, base_id=1000)
    _write_lcfa(second, platform="G16", slot="GOES-East", second=2, base_id=2000)

    glm = read_glm_l2_lcfa([second, first])
    assert len(glm.identity.source_files) == 2
    assert len(glm.events) == 6
    assert np.array_equal(glm.events.source_file_index, [0, 0, 0, 1, 1, 1])

    start = np.datetime64("2020-01-01T00:00:02.000", "ns")
    end = np.datetime64("2020-01-01T00:00:03.000", "ns")
    selection = glm.select(
        time_range_ns=(start, end),
        geographic_bounds=(-101.0, -99.0, 13.0, 15.0),
    )
    assert selection.event_indices.size == 3
    compact = selection.materialize()
    assert compact.validate_hierarchy().valid
    assert len(compact.groups) == 2
    assert len(compact.flashes) == 1

    dataset = glm.to_glmtools_compatible_xarray()
    assert dataset.sizes == {
        "number_of_events": 6,
        "number_of_groups": 4,
        "number_of_flashes": 2,
        "number_of_time_bounds": 2,
    }
    assert dataset["event_energy"].attrs["units"] == "nJ"
    restored = GLMObservation.from_xarray(dataset)
    assert restored.validate_hierarchy().valid
    assert np.allclose(restored.events.energy_j, glm.events.energy_j)
    assert np.array_equal(restored.events.parent_group_index, glm.events.parent_group_index)


def test_mixed_platform_collection_is_rejected(tmp_path: Path) -> None:
    east = tmp_path / "east.nc"
    west = tmp_path / "west.nc"
    _write_lcfa(east, platform="G16", slot="GOES-East", second=0, base_id=1000)
    _write_lcfa(west, platform="G17", slot="GOES-West", second=2, base_id=2000)
    try:
        read_glm_l2_lcfa([east, west])
    except GLMDataError as exc:
        assert "different spacecraft" in str(exc)
    else:
        raise AssertionError("mixed-platform collection should fail")


def test_event_geometry_roundtrip_and_polygons(tmp_path: Path) -> None:
    path = tmp_path / "OR_GLM-L2-LCFA_G16_geometry.nc"
    _write_lcfa(path, platform="G16", slot="GOES-East", second=0, base_id=1000)
    glm = read_glm_l2_lcfa(path)

    centers = glm.geometry.event_centers_fixed_grid()
    polygons = glm.geometry.event_corners_lonlat()
    assert centers.shape == (3, 2)
    assert polygons.shape == (3, 4, 2)
    assert np.isfinite(polygons).all()
    polygon_centers = polygons.mean(axis=1)
    assert np.allclose(polygon_centers[:, 0], glm.events.longitude_deg, atol=5e-4)
    assert np.allclose(polygon_centers[:, 1], glm.events.latitude_deg, atol=5e-4)
    status = glm.geometry.cache_status()
    assert status["centers_cached"]
    assert status["geographic_corner_scales"] == [1.0]
