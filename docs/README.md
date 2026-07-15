# Lightning Mapping Array Suite (LMAS) 1.6.0

LMAS is an open-source Python application for interactive, reproducible analysis and presentation of already solved Lightning Mapping Array data. It combines linked LMA views with GOES-R GLM overlays, ground-network overlays, Projects and Profiles, publication figures, animations, True Aspect, and offline boundary maps.

[![LMAS main window in Landscape mode with the dark theme](docs/images/main_window_landscape_dark.png)](docs/images/main_window_landscape_dark.png)

## Highlights

- Native solved-LMA and GOES-R GLM Level 2 LCFA readers.
- Linked local and geographic projections with filtering and coordinated zoom/pan.
- Precision Mode, Source Selection, Charge Analysis, and exportable scientific products.
- Separate Satellite Overlays and Network Overlays workspaces.
- GLM event footprints, official group centroids, optical-energy diagnostics, time rails, legends, and colorbars.
- Network event categories, peak-current styling, uncertainty ellipses, time rails, and signed peak-current diagnostics.
- True Aspect across all Landscape spatial panels and the Portrait plan view, without allowing shallow Portrait altitude panels to limit zoom.
- Offline coast, country, state/province, and United States county boundaries.
- Portable Projects, reusable Profiles, figures, animations, and CLI workflows.

## Interface examples

### Portrait layout and file browser

[![LMAS main window in Portrait mode with the light theme and file browser](docs/images/main_window_portrait_light.png)](docs/images/main_window_portrait_light.png)

### Satellite and network overlays

[![LMAS Satellite Overlays and Network Overlays workspaces](docs/images/network_and_satellite_overlays.png)](docs/images/network_and_satellite_overlays.png)

## Install the release kit

Activate the intended Python or Conda environment, extract the release kit, open a terminal in `LMAS_1_6_0`, and run:

```bash
python install.py
```

This installs the bundled LMAS wheel and creates the platform launcher when supported. Direct wheel installation is also available; see `INSTALLATION.md`.

Start LMAS with:

```bash
lma gui
```

## Try the demonstration

Run:

```bash
lma gui --demo
```

The included one-minute Oklahoma case contains LMA data and GOES-16/GOES-17 GLM files. Overlay datasets are kept disabled by default so users can enable and explore them deliberately.

## Documentation

- `USER_MANUAL.md` — complete GUI and workflow guide
- `INSTALLATION.md` — installation and launcher options
- `CLI_REFERENCE.md` — command-line reference
- `NATIVE_READERS.md` — concise LMA and GLM reader overview
- `NETWORK_OVERLAYS.md` — supported network-overlay workflow
- `KNOWN_LIMITATIONS.md` — current scientific and platform limitations

## License and attribution

Original LMAS code is distributed under the MIT License in `LICENSE`. Upstream and bundled materials retain their own notices in `licenses/`; `licenses/README.md` explains which file applies to each component. See also `CREDITS.md`, `LINEAGE_AND_ATTRIBUTION.md`, and `THIRD_PARTY_NOTICES.md`.