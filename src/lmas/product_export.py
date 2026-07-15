"""Extensible registry for products exported from the LMAS GUI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ProductFormat = Literal["csv", "netcdf"]
ProductScope = Literal["all", "filtered", "assigned", "active_group"]


@dataclass(frozen=True)
class ExportProductDefinition:
    """One product choice shown by the general export interface."""

    key: str
    label: str
    format_name: ProductFormat
    description: str


@dataclass(frozen=True)
class ExportScopeDefinition:
    key: ProductScope
    label: str
    description: str


EXPORT_PRODUCTS: tuple[ExportProductDefinition, ...] = (
    ExportProductDefinition(
        key="polarity_csv",
        label="Polarity source table (CSV)",
        format_name="csv",
        description=(
            "One row per LMA source with coordinates, quality fields, polarity "
            "assignments, conflicts, group references, and export provenance."
        ),
    ),
    ExportProductDefinition(
        key="polarity_netcdf",
        label="Complete polarity dataset (NetCDF/xarray)",
        format_name="netcdf",
        description=(
            "The complete loaded LMA dataset plus polarity assignments, named "
            "groups, sparse memberships, conflicts, and provenance."
        ),
    ),
)

EXPORT_SCOPES: tuple[ExportScopeDefinition, ...] = (
    ExportScopeDefinition(
        key="all",
        label="All loaded sources",
        description="Authoritative complete product and the default for round-trip import.",
    ),
    ExportScopeDefinition(
        key="filtered",
        label="Passing saved filters/view",
        description="Only sources passing the Project's saved filter and view state.",
    ),
    ExportScopeDefinition(
        key="assigned",
        label="Assigned sources only",
        description="Only sources belonging to Positive or Negative groups.",
    ),
    ExportScopeDefinition(
        key="active_group",
        label="Active group only",
        description="Only sources in the currently active named source group.",
    ),
)


def export_product_by_key(key: str) -> ExportProductDefinition:
    for product in EXPORT_PRODUCTS:
        if product.key == str(key):
            return product
    raise KeyError(f"Unknown export product: {key!r}")


def export_scope_by_key(key: str) -> ExportScopeDefinition:
    for scope in EXPORT_SCOPES:
        if scope.key == str(key):
            return scope
    raise KeyError(f"Unknown export scope: {key!r}")


__all__ = [
    "EXPORT_PRODUCTS",
    "EXPORT_SCOPES",
    "ExportProductDefinition",
    "ExportScopeDefinition",
    "ProductFormat",
    "ProductScope",
    "export_product_by_key",
    "export_scope_by_key",
]
