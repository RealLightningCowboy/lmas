# LMAS polarity product format

## Status

Schema identifier: `lmas-polarity-v1`

LMAS manual charge analysis has one canonical scientific representation. The
complete representation is an xarray `Dataset` normally written as NetCDF. A
flat CSV/pandas table is derived from the same source-level classification
logic for interoperability.

Polarity codes are fixed:

| Code | Label |
|---:|---|
| `-1` | Negative |
| `0` | Unassigned |
| `+1` | Positive |

A source assigned to both Positive and Negative groups is exported as
`polarity_code = 0` with `polarity_conflict = 1`. Conflict therefore never
silently becomes a physical neutral classification.

## Complete NetCDF/xarray product

The default export scope is **all loaded sources**. This is the authoritative
round-trip product. It contains the complete loaded LMA dataset, including
non-source dimensions and variables such as station metadata, plus the
following LMAS variables.

### Source-level variables

The solved-source dimension is named `source` in the exported product.

- `source_id(source)` — stable LMAS source identity.
- `polarity_code(source)` — `-1`, `0`, or `+1`.
- `polarity_label(source)` — human-readable category.
- `polarity_conflict(source)` — one when Positive and Negative assignments overlap.
- `polarity_group_count(source)` — number of named groups containing the source.

Original source variables such as event time, latitude, longitude, altitude,
power, reduced chi-square, contributing-station count, and reader-specific
fields remain in the dataset.

### Group-level variables

The `polarity_group` dimension describes named source groups.

- `polarity_group_id`
- `polarity_group_name`
- `polarity_group_code`
- `polarity_group_category`
- `polarity_group_color`
- `polarity_group_display_style`
- `polarity_group_visible`
- `polarity_group_locked`
- creation/modification UTC values
- creating LMAS version

Group IDs are deterministic identifiers derived from persistent group
provenance. Group names remain user-editable labels.

### Sparse membership

Membership is stored sparsely rather than as a potentially enormous
source-by-group matrix:

- `polarity_membership_source_index(polarity_membership)`
- `polarity_membership_group_index(polarity_membership)`
- `polarity_membership_source_id(polarity_membership)`

This preserves overlapping groups exactly.

### Global provenance

Important global attributes include:

- `lmas_polarity_schema`
- `lmas_version`
- product creation UTC
- project name
- export scope
- full and exported source counts
- dataset fingerprint
- source-identity fields
- polarity encoding
- active group
- category visibility
- selection/member-display preferences
- reference latitude/longitude
- source basenames
- quality and linked-view filters
- reader backend information

Absolute workstation paths are not required for product identity.

## Dataset fingerprint and import safety

The fingerprint is a SHA-256 digest of the full loaded source count and the
canonical source identity arrays:

- stable source ID;
- UTC event time;
- event latitude;
- event longitude;
- event altitude.

LMAS verifies this fingerprint before restoring named groups. A mismatched
product is rejected rather than silently remapped. Full-scope products restore
exact group membership. Scoped products require explicit partial-import
permission because omitted sources cannot be reconstructed.

## CSV/DataFrame representation

The CSV/DataFrame export contains one row per exported source. It includes all
one-dimensional source-aligned variables from the loaded dataset plus stable
aliases and charge-analysis fields, including:

- source ID and UTC time;
- latitude, longitude, altitude MSL, local east/north position;
- power, reduced chi-square, and station count where available;
- polarity code and label;
- conflict flag;
- group count;
- JSON arrays of group IDs, names, and categories;
- schema, fingerprint, scope, creation UTC, and LMAS version.

`LMAProject.polarity_dataframe()` returns the same table in memory.

## Export scopes

- `all` — every loaded source; authoritative round-trip default.
- `filtered` — sources passing the Project quality filters, linked-view filters,
  and saved exact membership when present.
- `assigned` — sources belonging to at least one named group.
- `active_group` — sources in the active group only.

Scoped NetCDF products remain complete scientific subsets, but they are not
full group-membership archives unless every group member is included.

## Python API

```python
from lmas.polarity_product import (
    export_polarity_csv,
    export_polarity_netcdf,
    import_polarity_netcdf,
)

frame = project.polarity_dataframe(scope="all")
dataset = project.polarity_dataset(scope="all")

export_polarity_csv(project, "storm_polarity.csv")
export_polarity_netcdf(project, "storm_polarity.nc")
state = import_polarity_netcdf(project, "storm_polarity.nc")
```

## Command line

```bash
lma export-polarity \
  --project storm.lmas-project.yaml \
  --format netcdf \
  --scope all \
  --output storm_polarity.nc

lma export-polarity \
  --project storm.lmas-project.yaml \
  --format csv \
  --output storm_polarity.csv

lma import-polarity \
  --project storm.lmas-project.yaml \
  --polarity storm_polarity.nc \
  --output storm_with_polarity.lmas-project.yaml
```
