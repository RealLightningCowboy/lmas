from pathlib import Path

import numpy as np
import pytest

from lmas.demo import synthetic_project
from lmas.errors import ConfigurationError
from lmas.polarity_product import (
    POLARITY_PRODUCT_SCHEMA,
    dataset_fingerprint,
    export_polarity_csv,
    export_polarity_netcdf,
    import_polarity_netcdf,
    load_polarity_dataset,
    polarity_dataframe,
    polarity_dataset,
)
from lmas.source_selection import SourceSelectionManager


def _project_with_groups(count: int = 48):
    project = synthetic_project(count=count)
    manager = SourceSelectionManager()
    manager.apply([1, 2, 3, 4], "replace")
    manager.set_domain("charge", subtype="polarity_group")
    manager.set_charge_category("positive")
    manager.new_group("Negative branch", source_ids=[4, 5, 6, 7], charge_category="negative", domain="charge", subtype="polarity_group")
    manager.new_group("Unassigned review", source_ids=[8, 9], charge_category="unassigned", domain="charge", subtype="polarity_group")
    project.source_selection_state = {
        **manager.to_dict(),
        "category_visibility": {"unassigned": True, "positive": True, "negative": False},
        "selection_scope": "filtered",
        "member_display_scope": "all",
        "charge_region_label": "charge_region_polarity",
    }
    return project


def test_dataframe_is_full_source_table_with_conflict_and_group_references():
    project = _project_with_groups()
    frame = polarity_dataframe(project)
    assert len(frame) == project.event_count
    assert frame.attrs["schema"] == "lmas-polarity-table-v1"
    assert {"event_time", "event_latitude", "event_longitude", "event_altitude"}.issubset(frame.columns)
    assert {"polarity_code", "polarity", "polarity_conflict", "group_ids_json"}.issubset(frame.columns)
    by_id = frame.set_index("source_id")
    assert by_id.loc[1, "polarity_code"] == 1
    assert by_id.loc[5, "polarity_code"] == -1
    assert by_id.loc[4, "polarity_code"] == -1
    assert bool(by_id.loc[4, "polarity_conflict"])
    assert by_id.loc[10, "polarity"] == "Unassigned"


def test_complete_xarray_product_preserves_original_data_and_sparse_groups():
    project = _project_with_groups()
    dataset = polarity_dataset(project)
    assert dataset.attrs["lmas_polarity_schema"] == POLARITY_PRODUCT_SCHEMA
    assert dataset.attrs["dataset_fingerprint"] == dataset_fingerprint(project)
    assert dataset.sizes["source"] == project.event_count
    assert "station_code" in dataset
    assert "polarity_membership_source_id" in dataset
    assert dataset.sizes["polarity_group"] == 3
    assert dataset.sizes["polarity_membership"] == 10
    assert int(dataset["polarity_conflict"].sel(source=4)) == 1


def test_csv_and_netcdf_exports_and_exact_round_trip(tmp_path: Path):
    project = _project_with_groups()
    csv_path = export_polarity_csv(project, tmp_path / "storm")
    nc_path = export_polarity_netcdf(project, tmp_path / "storm-polarity", engine="scipy")
    assert csv_path.suffix == ".csv"
    assert nc_path.suffix == ".nc"
    assert csv_path.read_text(encoding="utf-8").splitlines()[0].startswith("lmas_polarity_schema,")

    loaded = load_polarity_dataset(nc_path)
    assert loaded.attrs["lmas_polarity_schema"] == POLARITY_PRODUCT_SCHEMA
    state = import_polarity_netcdf(project, nc_path)
    assert state["active_group"] == project.source_selection_state["active_group"]
    assert state["category_visibility"]["negative"] is False
    assert state["member_display_scope"] == "all"
    assert state["charge_region_label"] == "charge_region_polarity"
    groups = {group["name"]: group for group in state["groups"]}
    assert groups["Selection 1"]["source_ids"] == [1, 2, 3, 4]
    assert groups["Negative branch"]["source_ids"] == [4, 5, 6, 7]
    assert groups["Selection 1"]["charge_category"] == "positive"


def test_import_rejects_mismatched_dataset(tmp_path: Path):
    source = _project_with_groups(48)
    destination = synthetic_project(count=49)
    path = export_polarity_netcdf(source, tmp_path / "polarity.nc", engine="scipy")
    with pytest.raises(ConfigurationError, match="fingerprint"):
        import_polarity_netcdf(destination, path)


def test_partial_product_requires_explicit_permission(tmp_path: Path):
    project = _project_with_groups()
    path = export_polarity_netcdf(project, tmp_path / "assigned.nc", scope="assigned", engine="scipy")
    with pytest.raises(ConfigurationError, match="allow_partial"):
        import_polarity_netcdf(project, path)
    state = import_polarity_netcdf(project, path, allow_partial=True)
    assert state["import_provenance"]["export_scope"] == "assigned"
