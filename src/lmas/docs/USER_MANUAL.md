# LMAS User Manual

## Packaged real-data demonstration

Choose **File → Open demonstration** or run `lma gui --demo`. LMAS loads a one-minute Oklahoma LMA subset and matching GOES-16 East and GOES-17 West GLM L2 LCFA files, then opens directly on the 2019-04-30 14:49:14.265 UTC flash. The installed demo is a template; use Save As before editing it permanently.


## 1. Purpose

Lightning Mapping Array Suite (LMAS) is a desktop and command-line environment for viewing, filtering, analyzing, animating, and presenting **already solved Lightning Mapping Array source data**. LMAS is intended to provide a modern, open, Python-based alternative to the legacy IDL program XLMA.

LMAS includes its own native solved-LMA reader. It does not require pyxlma. An optional pyxlma compatibility backend can be selected when pyxlma is installed separately.

## 2. Starting LMAS

Launch the desktop application with:

```bash
lma gui
```

The `lmas`, `lma-gui`, and `lmas-gui` commands remain available as aliases. Open data with **File → Open LMA data**, or reopen an exact saved session with **Projects → Open Project**.

The release-kit installer also creates a clickable LMAS launcher by default: a Start Menu/Desktop shortcut on Windows, `LMAS.app` in the user Applications folder on macOS, or an application-menu/Desktop entry on Linux. These launchers use the same installed Python environment as the command-line entry points.

## 3. Supported inputs

LMAS accepts solved LMA `.dat`, `.dat.gz`, NetCDF, and supported `.tar`, `.tar.gz`, and `.tgz` bundles. When an archive contains multiple plausible LMA datasets, LMAS asks which dataset or datasets to load.

LMAS does not ingest raw waveforms or solve time-of-arrival locations. GOES-R GLM L2 LCFA overlays and ground-network CSV overlays are available through the Satellite Overlays and Network Overlays workspaces.

Use **View → Data File Header…** to inspect the literal `.dat`/`.dat.gz` header in a searchable read-only window. Archive members are supported. Formats without a literal DAT header receive a clearly labeled metadata summary.

Use **File → Export Product…** for the general product-export interface; analysis tabs may provide shortcuts to the same backends.

## 4. Reader backends and provenance

**Auto — prefer LMAS native** is the default reader mode. The built-in native backend reads solved-LMA products without pyxlma. The optional **pyxlma compatibility** backend is available only when pyxlma is installed separately.

Projects record the selected reader, reader version, and relevant fallback details. Older Projects saved with pyxlma provenance can fall back to the LMAS native reader when the saved input is supported and pyxlma is unavailable.

Reader outputs are normalized into an immutable NumPy-backed `LmaSourceStore`. LMAS also exposes an xarray Dataset compatibility view for analysis, export, and plugin workflows.

## 5. Main linked projection view

LMAS provides two figure layouts:

- **Landscape**: time–altitude, plan view, and two vertical projections; an altitude histogram is optional.
- **Portrait**: the same linked scientific projections in a print-oriented layout with a dedicated altitude histogram.

Both layouts default to **Local (km)** coordinates. Local axes use cardinal labels such as **W ← (km) → E** and **S ← (km) → N**. Viewpoint reversals swap the letters automatically. **Geodetic (lat/lon)** mode uses longitude and latitude.

The Matplotlib toolbar provides zoom, pan, Home, Back, and Forward. Typed limits commit when Enter is pressed or the field loses focus. Keyboard shortcuts work while the main window or embedded plotting canvas has focus, but remain inactive while text, numeric, combo-box, or shortcut-entry fields are being edited.

Useful navigation keybinds include:

- **W** — activate rectangle zoom and drag a zoom-to-box region;
- **D** — activate click-and-drag pan;
- **Home** — restore the full linked view;
- **Alt+Left / Alt+Right** — move through linked scientific-view history.

Use **Options → Keyboard Shortcuts** to inspect or customize bindings, and **Help → Keybinds** for the complete reference.


### Maps, True Aspect, and relative time

The main **View options** include **Map underlay**, **True spatial aspect (1 km = 1 km)**, and **Relative time from window start**. Map underlays appear in the plan panel in both Local and Geodetic coordinates. In Local mode LMAS transforms the geographic line geometry into the same kilometer coordinates used by the observations.

Maps require undistorted geometry, so enabling **Map underlay** automatically enables **True spatial aspect**. The True Aspect control remains interactive: turning it off disables the map underlay in the same action. With maps off, True Aspect is optional and off by default. Strict aspect is applied after every preserved redraw and linked navigation commit. One common physical scale is used across all spatial panels by expanding displayed coordinate limits while preserving the normal panel sizes; the time-altitude panel is not forced into a distance ratio.

Relative time changes only tick labels. Absolute UTC timestamps remain authoritative internally. The origin is the fixed beginning of the windowed record and does not reset when the visible plot is zoomed. Units adapt among microseconds, milliseconds, seconds, minutes, and hours.

### Satellite Overlays (GLM development workflow)

Open **Satellite Overlays** from the analysis toolbar or **View** menu. It is an independent top-level workspace and may move behind the LMAS main window. Add GOES-R GLM L2 LCFA files with the browser controls, or paste/type a file or directory into the editable **File or directory** field; LMAS keeps each spacecraft independent and identifies its dated East/West operational role.

The GLM tab controls event footprints, outlined group centroids, optional flash centroids, optional highest-energy-group emphasis, the **GLM Total Optical Energy (fJ)** colorbar, and East/West group-time rails at the top of the altitude-versus-time panel. The TOE colorbar is horizontal below the scientific panels in Landscape and vertical in a dedicated left gutter in Portrait. Separate z-order controls place event footprints and group centroids independently below or above LMA sources. The time rail uses true group times and axes-relative tracks, not an invented GLM altitude.

The embedded group-energy plot uses narrow group-frame bars, logarithmic energy, concise UTC labels, and major/minor ticks. **Save figure** carries the current GLM overlay state into the full-resolution saved figure.

Choose **Auto (prefer LMAS native)**, **LMAS native**, or **glmtools** from the GLM reader control. **Reload** rereads listed files with the selected backend while preserving platform styles where possible. The native NumPy object remains the interactive representation; glmtools is optional and must be importable when explicitly selected.

Scientific layer checkboxes apply globally across all enabled spacecraft. Select an individual spacecraft to adjust its colormap, footprint opacity, GLM group-centroid size/color, z-orders, and interactive event limit. Official GLM groups remain distinct from user-defined LMA source groups.


### Network Overlays

Open **Network Overlays** from the analysis toolbar or **View** menu. Add one or more lightning-location-network CSV files with the browser buttons, or paste/type a CSV file or directory into the editable path field. Select Auto detect or a provider identity, and LMAS normalizes recognized UTC, location, type, polarity, peak-current, sensor-count, quality, altitude, uncertainty, and identifier columns. ENTLN numeric type values use the provider-specific mapping `0 = CG`, `1 = IC`; the original value is preserved for export. Multiple compatible files may be combined into one chronological dataset.

The workspace uses a compact two-column layout. A signed peak-current-versus-time plot occupies the lower left and follows the selected dataset and linked time view. Loaded datasets and current-view diagnostics occupy the left; global layers, selected-dataset filters, colors, marker sizes, z-orders, and interactive caps occupy a scrollable pane on the right. Event markers appear in the plan view, optional uncertainty ellipses use the supplied major/minor axes, and a true-time rail appears just above the time–altitude axes, below the figure title and separate from the GLM East/West rails. LMAS never invents an altitude for a network event.

Type, polarity, minimum absolute peak current, minimum sensor count, time, and current-map-view filters are available. Marker scaling by absolute peak current is enabled by default for newly loaded datasets. Project saves preserve local file references and styles; missing network files do not block the LMA Project. The selected current-view subset can be exported as normalized CSV or NetCDF. See **Help → Network Overlays Guide**.

### Precision Mode (scope mode)

**Precision Mode** is the official name for the linked two-cursor measurement workflow; **scope mode** is the informal shorthand. Open it from the crosshairs button in the main GUI, from **View → Precision Mode**, or with **P**. The window is non-modal, so the projection figure remains fully interactive.

Choose cursor A or B and click any scientific panel. Plain click places the active cursor, while **Shift+click** places cursor B directly. Snap choices are:

- **Off — Free** — place arbitrary coordinates and refine the same cursor from multiple linked projections;
- **Nearest visible source** — the exact source subset represented by the current linked view;
- **Nearest full filtered source** — every source passing the active quality filters, even outside the current linked subset.

Crosshair labels are optional and default off. Each source-snapped cursor reports UTC, source ID, local E/N position, latitude/longitude, altitude, source power, reduced χ², and station count when available. Free cursors report only dimensions that have actually been defined. With both cursors set, LMAS reports available B-minus-A time and displacement, horizontal and three-dimensional distance, bearing clockwise from north, and apparent cursor-derived horizontal/3D speeds. These are cursor measurements, not fitted propagation estimates.

The **Axis cursors** section provides linked oscilloscope-style line pairs. **Ctrl+click** places line 1 for both dimensions represented by the selected panel; **Ctrl+Shift+click** places line 2. Free A/B crosshairs and individual visible axis lines can also be dragged directly. Enable or disable time, horizontal-coordinate, and altitude pairs independently. Compatible panels share the same physical line values and the window reports the direct difference for each pair.

Axis values can be typed directly. Time accepts either UTC or milliseconds from the first source, selected with the **Time entry** control; spatial values use kilometres in Local mode and degrees in Geodetic mode. **Set lines from A/B** copies compatible cursor coordinates into line 1 and line 2. **Apply A–B time interval** applies the cursor times to the linked scientific view. Cursor A/line 1 is green and cursor B/line 2 is cyan. Apparent speeds are shown in metres per second, using scientific notation for large values.

Configurable main-window shortcuts include:

- **P** — open or focus Precision Mode;
- **Ctrl+1 / Ctrl+2** — select cursor A or B while Precision Mode is open;
- **Ctrl+Left / Ctrl+Right** — previous or next eligible source by time;
- **Ctrl+Shift+Left / Ctrl+Shift+Right** — step 10 sources;
- **Ctrl+Shift+X** — swap A and B;
- **Ctrl+Z** — undo the last Precision Mode cursor action;
- **Delete / Shift+Delete** — clear the active cursor or both cursors;
- **Ctrl+C** — copy the current measurement report.

When the Precision Mode window itself has focus, the concise equivalents are **A/B**, **Left/Right**, **Shift+Left/Shift+Right**, **X**, **Ctrl+Z**, **Delete**, **Shift+Delete**, **Ctrl+C**, and **Esc**.

### Auto-fit spatial panels

**Auto-fit spatial panels** is enabled by default. With Auto-fit enabled, matching spatial and altitude limits propagate across linked panels from one authoritative subset. With Auto-fit disabled, scientific membership still updates while non-driver panels preserve their independent view limits.

## 6. Source-count wording

The dynamic title reports the population represented by the current view.

For example:

> 1,218 visible of 3,905 sources in view (χ² < 1.00)

- **Sources in view** is the number of finite sources inside the current time, horizontal, and altitude bounds **before** the active quality filters are applied.
- **Visible** is the number inside those same view bounds that also passes the active station-count, χ², and power filters.
- **Displayed** appears only when the interactive preview cap samples the visible population. It does not change filtering, counts, saved figures, exports, or animations.

The default interactive preview cap is **12,000 sources**. Saved outputs use the full exact selected population.

## 7. Filtering and display controls

Quality controls include:

- minimum station count;
- maximum reduced χ²;
- optional minimum and maximum source power.

Power limits retain a true **Auto/full-range** state until the user deliberately enters explicit limits.

View controls include time, west–east position, south–north position, and altitude limits. Display controls include layout, coordinate system, color quantity, colormap, theme, point size, grid, station overlays, viewpoints, depth ordering, histogram, legend, panel labels, and time-color remapping.

Text-size presets are **Normal**, **Publication**, and **Poster**.

## 8. Titles, legends, panel labels, and colorbars

Portrait titles are stacked at the semantic em dash for **all** text-size presets. This keeps long multi-minute summaries inside the locked Portrait page in both preview and saved output.

**Show legend** places a figure-level legend in the space below the bottom axes. The legend represents labeled source and categorical overlay elements and does not replace the continuous colorbar.

**Show panel labels** adds publication-style labels in visual reading order:

- `(a)` through `(d)` for the four primary scientific panels;
- `(e)` for the altitude histogram in Portrait;
- `(e)` for the Landscape altitude histogram when **Hist** is enabled.

The ordinary Portrait source colorbar occupies its own right-side gutter and matches the combined height of the scientific-axis stack. When GLM event footprints are visible, the GLM TOE colorbar occupies a separate vertical gutter on the left. Neither colorbar compresses the scientific panels.

The LMA power colorbar is labeled **VHF Source Power (dBW)**. Because dBW is already logarithmic, the Log normalization control is disabled for Source Power. Logarithmic χ² colorbars are labeled **log₁₀(χ²)**.

## 9. Profiles

A **Profile** is a reusable settings recipe. It stores quality filters and plotting choices but does not bind itself to one dataset or exact linked subset.

Use **Profiles → Save Profile**. Profiles use the `.lmas-profile.yaml` suffix. Applying a Profile to an open dataset preserves the current linked subset while updating reusable settings.

LMAS 1.0 writes `lmas-profile-v1.0`. Older supported profile formats migrate on load. Legacy serialized 15 ms trail/afterimage defaults migrate to 30 ms where the older format cannot distinguish a default from an explicit choice.

## 10. Projects

A **Project** is an exact data-bound working session. Use:

- **Projects → Open Project**
- **Projects → Save Project**
- **Projects → Save Project As**

Projects use the `.lmas-project.yaml` suffix and preserve source references, reader provenance, quality filters, exact linked-view limits, selected-source membership, color normalization, title, viewpoints, and plotting state. Source paths are stored relative to the Project file where practical, including sibling `projects/` and `data/` directories inside the same workspace tree.

Project source references remain backward compatible with the original string-based `source_files` list. New saves also record filename, file size, a fast sampled-content fingerprint, and a complete loaded-dataset fingerprint. If a saved path is unavailable after moving a Project to another computer, LMAS searches the Project/workspace and configured data roots, then asks the user to locate the file. A relocated dataset must pass the saved identity check before the Project opens. Command-line users may provide search roots with the `LMAS_DATA_ROOTS` environment variable.

Quality filters remain separate from non-destructive view limits, so reopening a narrowed Project does not discard sources outside the saved view.

LMAS 1.6.1 writes `lmas-project-v1.1` and continues to read v1.0 and supported v0 project formats.

## 11. Portable directories and output behavior

LMAS does not assume a laboratory workstation layout.

Directory precedence is:

1. an opened data file or archive uses its parent directory;
2. an opened Project uses the resolved source-data directory;
3. otherwise LMAS uses the remembered valid data directory;
4. if no valid directory is remembered, LMAS uses Documents or the user home directory.

Figures, animations, and related outputs default beside the input data. **Options → Preferences** can select a custom output directory. Stale or missing remembered directories fall back portably. Legacy development-directory preferences migrate back to **Same directory as input data**.

## 12. Saving figures

Use **Save figure** to choose the output path, resolution, and an optional custom title. The custom title applies to the saved figure without permanently replacing the live dynamic title.

The **Single** tab provides ordinary export and optional multiple-theme output. The **Batch** tab queues combinations of themes, color quantities, and maximum χ² thresholds. The exact committed view is preserved while each job receives honest source counts and provenance.

Portrait figures use a fixed **8.55 × 11 inch** page when the ordinary source colorbar is hidden and a **10.55 × 11 inch** page when it is shown. The additional width is a dedicated right-side colorbar gutter; the scientific axes retain their established physical dimensions. GLM event footprints use a separate vertical TOE colorbar in the reserved left margin.

## 13. Projection animation

**View proj.** and **Save proj.** operate on the exact current linked projection layout. Available display modes are:

- Cumulative;
- Trail;
- Trail + afterimage.

Interactive playback supports pause, restart, timeline scrubbing, looping, and live display-mode changes. Saved animations include FPS, duration, final hold, resolution, and video-quality controls. Batch projection export can combine themes and display modes while preserving one committed subset.

## 14. Three-dimensional visualization

**View 3D** and **Save 3D** use the exact current linked source subset. PyVista and VTK are optional dependencies.

The 3D base-grid control draws only the horizontal base grid and N/S/E/W labels. Vertical walls, cube edges, and altitude ladders are intentionally omitted so they cannot obscure the flash during an orbit.

The compatible project fields for trail and afterimage timing remain available, while the GUI presents one shared **Transition** value. Fresh data use 30 ms.

## 15. Array information

Use the top-level **Array Info** action to inspect:

- network/reference information;
- station codes, latitude, longitude, altitude, and local coordinates;
- pairwise horizontal and three-dimensional baseline lengths;
- baseline azimuth measured clockwise from north;
- baseline count, minimum, quartiles, median, mean, maximum, and standard deviation.

The station and baseline tables are sortable and resizable.

## 16. Command line

Run `lma --help` for current commands. Principal workflows include GUI launch, static plotting, Project creation and inspection, Profile management, linked-projection animation, 3D snapshot/viewing, and 3D animation. Batch workers use manifests written by the GUI or supplied directly.

Useful figure flags include:

```text
--show-legend
--show-panel-labels
```

## 17. Optional dependencies

The core package requires NumPy, xarray, Matplotlib, SciPy, PyYAML, and pandas. The GUI requires PySide6. Interactive 3D visualization requires PyVista and VTK. MP4 output requires imageio/imageio-ffmpeg and an FFmpeg executable.

Python 3.13 is the primary tested development environment. The package metadata currently permits Python 3.11 and newer with Python 3.13 used for primary release qualification.

## 18. Scientific boundaries

LMAS visualizes already solved LMA source products. It does not replace the upstream LMA location solver, calibrate source power, infer channel identity automatically, or repair a dataset whose timing or geometry is wrong. Altitude is displayed in kilometers MSL according to the loaded product convention; LMAS does not subtract terrain.

See also:

- [What LMAS can do](lmas-doc:WHAT_LMAS_CAN_DO.md)
- [Lineage and attribution](lmas-doc:LINEAGE_AND_ATTRIBUTION.md)
- [Development provenance](lmas-doc:DEVELOPMENT_PROVENANCE.md)
- [Release notes](lmas-doc:RELEASE_NOTES.md)
- [Known limitations](lmas-doc:KNOWN_LIMITATIONS.md)
- [Changelog](lmas-doc:CHANGELOG.md)


### Source Distributions

Open **View → Source Distributions** to inspect source **χ²**, source power, or contributing-station distributions. The default filter-diagnostic scope overlays the population after all other active filters with the population accepted after the selected variable’s own filter, and marks the active threshold. The window provides adjustable bins, linear or logarithmic axes, source counts and robust summary percentiles, figure saving, and CSV export of histogram-bin counts.

### Source Selection

Open **Source Selection** from the Analysis toolbar, **View → Source Selection**, or press **L**. The non-modal workspace creates named groups using stable source identities.

- **Default tool action** applies the selected Add, Replace, Remove, or Intersect operation to both **Lasso** and **Point edit**. **Shift** temporarily removes sources from the active group, **Alt** also removes, and **Ctrl** intersects.
- Removal edits can target existing active-group members even when those sources are visible only through the group overlay because they no longer pass the current filters.
- Each group has its own color, visibility, lock state, and display style. **+ New Group** creates and activates a group immediately; double-click a group to rename it. Clear and invert operate on the active group.
- Display choices are **Recolor**, **Halo**, **Outline**, **Convex Hull**, **Concave Hull**, **Clustered Hulls**, and **Hidden**. Recolor is the default for new groups. Hulls summarize the projection envelope; source IDs remain authoritative.
- **Display members** controls whether overlays show sources passing the current filters, all group members, or filtered-out members only. Sources outside the normal filtered linked view appear as dim hollow markers. Large overlays may be display-thinned for responsiveness, but group membership and counts remain exact.
- **Ctrl+Z** undoes the last selection, group-management, or charge-assignment action.
- Selected sources are represented across every linked projection, including sources omitted by the interactive preview cap.
- Group membership survives redraws, layout and coordinate changes, theme changes, and filter changes. A selected source excluded by later filtering remains a member and is counted separately.

### Charge Analysis

Open **Charge Analysis** from the Analysis toolbar or **View → Charge Analysis**. It shares the same source groups and lasso/point tools but gives **Polarity Assignment** its own tab.

- Categories are exactly **Unassigned**, **Positive**, and **Negative**.
- Use **+ Positive**, **+ Negative**, or **+ Unassigned** to create an already classified active group. Changing the **Polarity** dropdown applies immediately; **Reset color** restores the standard category color.
- Unassigned defaults to neutral gray, Positive to red, and Negative to vibrant blue. Custom group colors remain available.
- Category visibility and group-member display can be changed without deleting assignments.
- **Show charge overlays with other Color by modes** is off by default. Enable it to retain Positive/Negative halos or outlines when the main figure is colored by time, altitude, power, stations, or reduced χ².
- The active group reports assignment totals, source overlap with other groups, filtered-out membership, and creation/modification provenance.
- Named groups, assignments, and member-display preference are stored in Projects, not Profiles. The original solved-LMA source data are never modified.
### Polarity products

Charge Analysis and **File → Export Product…** provide two standard representations:

- **CSV** is a one-row-per-source table containing ordinary source fields, polarity code/label, conflicts, and named-group references.
- **NetCDF** is the authoritative complete xarray product. It retains the loaded LMA dataset, named groups, sparse membership, display metadata, filters, dataset identity, and provenance.

**All loaded sources** is the default export scope and is required for an exact round trip. Filtered, assigned-only, and active-group exports are available as analysis subsets. NetCDF import verifies a SHA-256 fingerprint of source IDs, UTC times, latitude, longitude, and altitude before replacing the Project's current groups. A mismatch is refused.

The same interfaces are available from Python as `project.polarity_dataframe()` and `project.polarity_dataset()`, and from the CLI through `lma export-polarity` and `lma import-polarity`. See [Polarity product format](lmas-doc:POLARITY_PRODUCT_FORMAT.md).


## 19. Source Selection application tabs

The **Source Selection** window contains three application tabs backed by one stable source-ID membership engine:

- **Custom Selection** — arbitrary reusable scientific or quality-control groups;
- **Charge Analysis** — Unassigned, Positive, and Negative polarity-oriented groups;

Groups remain in separate lists but may contain the same source IDs. Copy/promotion tools preserve the original group and create an exact cross-domain membership copy. Charge-overlap summaries make polarity correspondence visible for an active leader group.

The **Default tool action** applies to both lasso and point editing. Add, Replace, Remove, and Intersect use the active group in the active tab. Removal edits can target assigned sources that no longer pass the current filters but remain visible through a group overlay.

Saved figures reproduce visible group overlay styles, including Recolor, Halo, Outline, Convex Hull, Concave Hull, and Clustered Hulls.


## 20. Release scope

LMAS 1.6.1 builds on the stable Network Overlays release built on the public Satellite Overlays architecture. The packaged real-data demonstration opens with GLM disabled so users can first inspect the LMA flash and then enable either spacecraft deliberately.
