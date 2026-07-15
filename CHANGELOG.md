# LMAS changelog

## 1.6.0

- Prevented repeated Portrait overlay-layer changes from progressively resizing and cropping the embedded figure canvas.
- Restored the Landscape GLM TOE colorbar and figure legend to the compact caption position near the lower scientific axes, raised the TOE bar slightly, and matched its text to the surrounding axis typography.
- Established a fixed legacy XLMA-style Portrait canvas with a dedicated left GLM TOE-colorbar gutter and a separate right source-colorbar gutter, preventing overlay changes from altering or cropping the scientific layout.
- Corrected legacy XLMA-style Portrait legends so multi-row LMA/GLM/network entries remain fully inside the saved page.
- Kept True Aspect in the Portrait plan view while relaxing altitude-panel aspect coupling so shallow vertical projections no longer prevent tight plan-view zooms.
- Project Home now translates saved East/West and North/South bounds across live viewpoint changes before applying True Aspect, making repeated Home restores idempotent.
- Added strict True Aspect with one shared physical kilometres-per-inch scale across every Landscape spatial panel and the Portrait plan view in Local and Geodetic coordinates.
- Preserved full-size panel layouts by padding displayed spatial limits instead of shrinking or stretching axes boxes.
- Enforced True Aspect after Project Home, view restoration, linked zoom/pan, coordinate changes, and toolbar history navigation.
- Added bundled offline coast, country, state/province, and United States county boundary underlays in local and geographic coordinates.
- Added adaptive relative-time labels from the fixed start of the active windowed record.
- Added signed peak-current-versus-time diagnostics to Network Overlays.
- Added detailed file-visible browsers and direct path entry for Satellite Overlays and Network Overlays.
- Added restore, raise, and focus behavior for minimized or obscured top-level analysis windows.
- Kept Satellite Overlays and Network Overlays as separate top-level workspaces.
- Hardened Project/Profile persistence, saved-figure parity, map status reporting, launcher identity, and public release packaging.

## 1.5.0

- Corrected ENTLN numeric event classification (`0 = CG`, `1 = IC`) with provider-scoped decoding and raw-code export.
- Relocated Network Overlay time rails above the time-altitude axes.
- Added direct file/directory path entry to the Network Overlays window.
- Added direct file/directory path entry to the Satellite Overlays window.
- Enabled marker scaling by absolute peak current by default for newly loaded network datasets.
- Added the Network Overlays workspace for ground-based lightning-location-network observations.
- Added ENTLN-oriented and generic CSV normalization, linked local/geodetic event rendering, uncertainty ellipses, event-time rails, category/current/sensor filters, and normalized CSV/NetCDF export.
- Added portable Project state, missing-file-tolerant restoration, saved-figure parity, combined legends, retained-artist rendering, documentation, and an included generic CSV example.

## 1.4.0

- Added the NumPy-native GOES-R GLM L2 LCFA reader and glmtools-compatible interchange.
- Added the Satellite Overlays workspace with independent spacecraft/position identity, event footprints, official GLM group centroids, time rails, energy diagnostics, legends, and TOE colorbars.
- Added responsive retained-artist rendering, lazy/cached footprint geometry, configurable render padding, colormaps, marker styles, and independent z-orders.
- Added portable GLM Project state and the real-data Oklahoma LMA + GOES-16/GOES-17 demonstration.
- Corrected generated saved-figure and animation titles to floor the view-start time and removed event-specific default wording.
- Corrected saved Project startup views so incremental zoom and pan behave like an identical manually established view, revealing surrounding sources smoothly without jumping to Home or the full-record state.

## 1.2.0

- Streamlined Custom Selection and Charge Analysis around one stable source-ID engine.
- Added Source Distributions, exact Project Home bounds, portable Project paths, and saved-figure DPI in the Save Figure dialog.
- Hardened charge colors, source-group overlays, selection recovery, project persistence, and help organization.

## 1.1.1

- Applied stability fixes to Source Selection, Charge Analysis, Projects, and packaged documentation.

## 1.1.0

- Added manual Charge Analysis, polarity products, linked source selection, `.dat` header viewing, and the extensible Export Product interface.

## 1.0.0

- First stable LMAS build with native solved-LMA reading, linked 2D projections, Projects, Profiles, figures, animations, Precision Mode, CLI workflows, and publication-oriented output.
