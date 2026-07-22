# LMAS 1.6.2 command-line reference

Run `lma --help` for the installed command tree. The aliases `lmas`, `lma-gui`, and `lmas-gui` are also installed.

## Version

```bash
lma --version
```

## Start the GUI

```bash
lma gui
lma gui file.dat.gz
lma gui project.lmas-project.yaml
```

Use `--reader auto|native|pyxlma` to select a reader backend where supported.

## Inspect data

```bash
lma info file.dat.gz
```

## Create figures

```bash
lma plot file.dat.gz --output figure.png
```

Use `lma plot --help` for filter, layout, color, title, theme, and output controls.

## Batch figures

```bash
lma batch-figures figure_batch_manifest.json
```

## Projection and 3D animation

```bash
lma animate-projection file.dat.gz --output projection.mp4
lma animate-3d file.dat.gz --output sources.mp4
```

Optional visualization dependencies are required for 3D output. Interactive `view-projections` and `view-3d` commands accept `--point-limit`; zero disables the interactive cap. Saved animations remain uncapped.

## Launcher installation

```bash
lmas-install-launcher
```

This installs the platform-appropriate visible launcher where supported. The user-facing launcher name is **Lightning Mapping Array Suite** while the command/package identity remains `lma`/`lmas`.

Source Selection, Charge Analysis, and polarity-product review are primarily interactive GUI workflows. Their saved state is preserved in LMAS Projects, and polarity tables/products are exported through the GUI's Export Product commands.

## Packaged demonstration

```bash
lma gui --demo
```

Loads the included one-minute Oklahoma LMA + GOES-16/GOES-17 GLM demonstration.
