# LMAS 1.6.0 release notes

LMAS 1.6.0 is the first broad public release of the integrated Lightning Mapping Array Suite. It combines the established LMA-centered analysis environment with GOES-R GLM overlays, ground-based lightning-location-network overlays, physical-distance-preserving spatial views, and offline cartographic context.

## Highlights

- Native readers for solved LMA products and GOES-R GLM Level 2 LCFA files, with optional pyxlma and glmtools compatibility backends.
- Separate Satellite Overlays and Network Overlays workspaces with portable Project persistence.
- GLM event footprints, official group centroids, time rails, optical-energy diagnostics, legends, and TOE colorbars.
- ENTLN-oriented and generic network CSV import with event categories, polarity/current styling, uncertainty ellipses, time rails, peak-current diagnostics, and CSV/NetCDF export.
- True Aspect across all Landscape spatial panels and the Portrait plan view in local or geodetic coordinates; shallow Portrait altitude panels remain linked without constraining plan zoom.
- Bundled offline coast, country, state/province, and United States county boundaries.
- Adaptive relative-time labels from the fixed start of the active windowed record.
- Detailed file-visible overlay browsers, direct path entry, and restore/focus behavior for top-level analysis windows.

## Figure and project behavior

Project Home is viewpoint-independent and repeatable under True Aspect. Linked zoom/pan, coordinate changes, saved-view restoration, and toolbar history retain the intended spatial scale and Project state.

Legacy XLMA-style Portrait output uses a fixed print-oriented canvas. GLM event-footprint TOE colorbars occupy a dedicated vertical gutter on the left, the ordinary LMA source colorbar occupies a separate gutter on the right, and one- or two-spacecraft legends remain inside the page without changing the scientific-panel dimensions. Landscape caption elements remain compact near the lower scientific axes, with the TOE colorbar raised slightly for full-size labeling.

Saved figures use the same LMA, GLM, network, legend, colorbar, map, and aspect state as the interactive view.

## Scope

LMAS visualizes already solved LMA products; it does not solve raw station data. The included maps provide boundary underlays rather than terrain, roads, imagery, or web tiles.
