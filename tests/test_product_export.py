from lmas.product_export import (
    EXPORT_PRODUCTS,
    EXPORT_SCOPES,
    export_product_by_key,
    export_scope_by_key,
)


def test_general_export_registry_contains_standard_polarity_products():
    assert [item.key for item in EXPORT_PRODUCTS] == [
        "polarity_csv",
        "polarity_netcdf",
    ]
    assert export_product_by_key("polarity_csv").format_name == "csv"
    assert export_product_by_key("polarity_netcdf").format_name == "netcdf"
    assert "complete" in export_product_by_key("polarity_netcdf").description.lower()


def test_general_export_registry_defaults_to_all_source_scope():
    assert EXPORT_SCOPES[0].key == "all"
    assert export_scope_by_key("active_group").label == "Active group only"
