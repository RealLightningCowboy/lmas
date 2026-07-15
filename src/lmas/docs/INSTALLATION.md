# Installing LMAS 1.6.0

LMAS requires Python 3.11 or newer.

## Recommended release-kit installation

Activate the Python or Conda environment in which LMAS should be installed, open a terminal in the extracted `LMAS_1_6_0` folder, and run:

```bash
python install.py
```

The installer installs the bundled wheel and standard GUI dependencies, then creates the platform launcher when supported. Use `python install.py --help` for core-only, no-3D, no-launcher, and user-install options.

## Direct wheel installation

```bash
python -m pip install ./wheels/lmas-1.6.0-py3-none-any.whl
```

The GLM native reader requires `h5py`, which is declared as a required dependency. A Conda or Mamba environment can install it from conda-forge with either solver:

```bash
mamba install -n lma -c conda-forge h5py
# or
conda install -n lma -c conda-forge h5py
```

## Verify and start

```bash
lma --version
lma gui
lma gui --demo
```

Expected version: `LMAS 1.6.0`.

## Cartography

LMAS includes compact offline boundary vectors for map underlays. Coast, country, state/province, and United States county lines require no separate map package, internet connection, or first-use download. Cartopy is optional and is not required for the boundary maps included in LMAS 1.6.0.
