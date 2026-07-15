# What LMAS can do

LMAS 1.6.0 is an end-to-end environment for already solved Lightning Mapping Array source products and linked observational overlays.

## Data and projects

- Open solved LMA `.dat`, `.dat.gz`, supported archives, NetCDF files, Projects, and Profiles.
- Read solved LMA data with the LMAS native backend and preserve reader provenance.
- Save portable Projects with non-destructive starting views, source fingerprints, relative paths, and relocation support.
- Open a packaged real-data one-minute LMA + dual-GLM demonstration.

## Visualization and output

- Render linked local-coordinate and geodetic projections in multiple layouts and themes.
- Color by time, altitude, power, reduced χ², station count, charge, or source group.
- Save publication-quality figures, theme variants, projection animations, 3D snapshots, and animations.
- Use exact Project Home bounds and linked zoom/pan behavior across panels.

## Interactive analysis

- Use Precision Mode for source-snapped or free A/B measurements and apparent speeds.
- Create stable named LMA source groups with linked lasso and point editing.
- Assign Unassigned, Positive, and Negative charge-region polarity and export CSV/NetCDF products.
- Inspect source distributions for reduced χ², source power, and station count.

## Satellite overlays

- Read GOES-R GLM L2 LCFA files with the fast LMAS native reader or glmtools.
- Keep spacecraft identity and East/West operational position separate.
- Overlay energy-colored event footprints and official GLM group centroids.
- Show group-time rails, group-energy bars, TOE colorbars, explicit legends, and independent spacecraft controls.
- Save and restore GLM files and styles in portable Projects and export glmtools-compatible xarray datasets.

## Network overlays

- Read ENTLN-oriented and generic lightning-location-network CSV files into a normalized NumPy event model.
- Overlay CG, IC, polarity, and unknown categories in local or geodetic plan views.
- Show optional location-uncertainty ellipses and true-time rails without inventing an altitude.
- Filter by event type, polarity, minimum absolute peak current, sensor count, time, and current map view.
- Save network paths/styles in portable Projects and export normalized CSV or NetCDF products.

## Geographic context and physical scale

- Draw coastlines, national borders, state/province boundaries, and United States county boundaries in either longitude/latitude or LMAS local kilometers.
- Enforce strict 1 km = 1 km scale across Landscape spatial panels and the Portrait plan view, while keeping Portrait altitude projections linked without constraining zoom.
- Require True Aspect automatically whenever map underlays are enabled.
- Label the time axis in UTC or adaptive elapsed units from the fixed windowed-record start.
