# LMAS 1.6.1 validation summary

LMAS 1.6.1 is a focused corrective release for GLM event-footprint Total Optical Energy rendering.

## Required checks

- Package identity reports `1.6.1`.
- GLM events are consolidated by 56 microradian fixed-grid pixel before drawing.
- Event energy is summed across the selected time window.
- One polygon is generated per accumulated pixel.
- Shared and per-dataset color normalization use accumulated energies.
- Packaged Oklahoma GOES-16 reference: 495 selected events, 74 accumulated pixels, maximum 68 events in one pixel, maximum accumulated energy approximately 1960.8714 fJ.
- Accumulated energy equals selected raw-event energy within floating-point tolerance.
- Corrected interactive rendering visually matches the established glmtools reference pattern.

The build report records the exact automated test and artifact checks completed for the packaged release.
