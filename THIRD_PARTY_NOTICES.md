# Third-party notices

The top-level `LICENSE` applies to original LMAS code and documentation. Files and portions derived from third-party projects remain subject to their preserved upstream notices in `licenses/`.

## xlma-python / pyxlma lineage

LMAS native solved-LMA reading is heavily based on reader conventions and parsing logic from the MIT-licensed [`deeplycloudy/xlma-python`](https://github.com/deeplycloudy/xlma-python) project developed by Eric Bruning and collaborators. The repository provides the Python package `pyxlma`, which remains available as an optional LMAS compatibility backend.

The exact upstream notice is included at:

- `licenses/xlma-python-LICENSE.txt`
- Copyright (c) 2019 Eric Bruning
- License: MIT

## glmtools lineage

LMAS native GLM reading, compatibility conventions, and event-footprint geometry are heavily based on the BSD-3-Clause-licensed [`deeplycloudy/glmtools`](https://github.com/deeplycloudy/glmtools) project. LMAS preserves the established fixed-grid pixel-corner lookup approach and packages the corresponding lookup resource. `glmtools` remains available as an optional compatibility backend.

The exact upstream notice is included at:

- `licenses/glmtools-BSD-3-Clause-LICENSE.txt`
- Copyright (c) 2016-2021, Eric Bruning and contributors
- License: BSD-3-Clause

## Bundled map vectors

LMAS includes compact vector derivatives generated from [`basemap-data` 2.0.0](https://github.com/matplotlib/basemap) so basic boundary maps work offline. The GSHHG-derived coastline and political-boundary data are distributed under LGPL-3.0-or-later. Other source assets, including the United States county data used by LMAS, carry the packaged MIT notice. LMAS preserves the applicable upstream texts:

- `licenses/basemap-data-LGPL-3.0-or-later.txt`
- `licenses/GPL-3.0.txt` — included because LGPL version 3 incorporates the GPL version 3 terms
- `licenses/basemap-data-MIT.txt`

The bundled map vectors remain under the applicable upstream terms and are not relicensed by the LMAS MIT license. Provenance is also recorded in `src/lmas/resources/maps/README.md`.

## Runtime dependencies

NumPy, xarray, Matplotlib, SciPy, PyYAML, pandas, h5py, PySide6, Cartopy, PyVista, VTK, imageio, and imageio-ffmpeg are separately distributed projects with their own licenses. LMAS does not relicense those dependencies.
