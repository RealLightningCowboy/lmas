# LMAS 1.6.2 known limitations

## Data and analysis

- LMAS visualizes already solved LMA source products; it does not solve raw station data.
- Precision Mode speeds are two-point apparent measurements, not fitted physical propagation models.
- Manual source selections and polarity assignments require scientific review.

## Satellite overlays

- The native reader targets GOES-R GLM Level 2 LCFA products. Lower-level L1/L0 processing is not included.
- Event footprints use product projection metadata and the operational GLM lightning-ellipsoid correction; no additional cloud-top parallax correction is applied automatically.
- GLM geometry caching is currently in memory, so very long records can require additional memory and slower initial footprint generation.

## Network overlays

- ENTLN-oriented and generic CSV import uses automatic column aliases. Provider or archive variants that do not match the supported aliases may require column renaming or preprocessing.
- Dedicated NLDN and GLD360 schema adapters have not been qualified against representative files.
- Uncertainty geometry is rendered from supplied major/minor axes and orientation; interpretation depends on the provider's documented convention and confidence level.

## Map underlays

- The bundled maps include coastlines, national borders, state/province boundaries, and United States county boundaries. Terrain, roads, imagery, web tiles, and detailed municipal features are not included.
- A close view entirely inside one county or state may contain few visible boundary lines; the status bar reports the active map source and feature counts.

## Platform and optional dependencies

- Final GUI behavior depends on Qt, Matplotlib, graphics drivers, and desktop scaling.
- Optional 3D features require PyVista/VTK and suitable graphics support.
