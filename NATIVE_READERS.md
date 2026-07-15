# LMAS native readers

LMAS includes native readers for solved Lightning Mapping Array products and GOES-R Geostationary Lightning Mapper observations. The native readers allow the standard LMAS workflow to operate without requiring the upstream reader packages at runtime, while preserving provenance and optional compatibility paths.

## Solved LMA products

The native LMA reader supports solved `.dat` and `.dat.gz` products and the supported archive and NetCDF workflows described in the User Manual. It reads the source table, network metadata, station information, reference location, and quality fields used by LMAS.

The implementation is heavily based on the reader conventions and parsing logic developed in the [`deeplycloudy/xlma-python`](https://github.com/deeplycloudy/xlma-python) project, whose Python package is named `pyxlma`. When `pyxlma` is installed separately, users may select it as an optional compatibility backend for supported ASCII LMA products.

## GOES-R GLM products

The native GLM reader supports GOES-R GLM Level 2 LCFA NetCDF files. It preserves event, group, and flash hierarchy; spacecraft identity; product times; projection metadata; energy fields; and source-file provenance. Each spacecraft remains an independent dataset in Satellite Overlays.

The GLM implementation is heavily based on conventions and geometry developed in [`deeplycloudy/glmtools`](https://github.com/deeplycloudy/glmtools). LMAS preserves the established fixed-grid pixel-corner lookup approach, includes an optional `glmtools` compatibility backend, and can exchange glmtools-style xarray datasets.

GLM event footprints include the operational lightning-ellipsoid correction represented by the product geometry. LMAS does not automatically apply an additional cloud-top parallax correction.

## Reader selection and provenance

The default **Auto** mode prefers the LMAS native reader. Users may explicitly choose a supported compatibility backend when the corresponding external package is installed. Projects record the selected reader and relevant provenance so that reopened analyses remain interpretable.

The upstream notices for `xlma-python` and `glmtools` are preserved in the `licenses/` directory. See `LINEAGE_AND_ATTRIBUTION.md` and `THIRD_PARTY_NOTICES.md` for details.
