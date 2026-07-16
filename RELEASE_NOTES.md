# LMAS 1.6.1 release notes

LMAS 1.6.1 is a focused corrective release for GOES-R GLM event-footprint rendering. GLM events in the selected time window are now accumulated by fixed-grid detector pixel before map drawing, matching the established glmtools behavior used by the legacy LMAS workflow.

## Corrected GLM Total Optical Energy

- All selected events occupying the same GLM detector pixel are summed across the complete visible time window.
- LMAS draws one footprint polygon per accumulated pixel rather than overlapping successive raw event layers.
- Each footprint is colored by its accumulated Total Optical Energy.
- Footprint geometry remains based on the operational GLM fixed grid and lightning-ellipsoid correction.
- Shared East/West and per-dataset color normalization now use accumulated pixel energies.
- The interactive footprint cap is applied after accumulation.

## Validation

The packaged Oklahoma demonstration selects 495 GOES-16 events in the saved flash window and consolidates them into 74 detector pixels. Event energy is conserved exactly within floating-point tolerance, the busiest pixel contains 68 events, and the brightest accumulated pixel is approximately 1960.8714 fJ. The corrected rendering was visually compared with the established glmtools reference.

## Scope

This release changes GLM pixel accumulation and footprint coloring only. It does not alter GLM event/group/flash parent relationships, LMA processing, network overlays, project formats, or the established 1.6.0 interface behavior.
