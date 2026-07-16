# LMAS 1.6.1 build report

## Scope

LMAS 1.6.1 is a focused corrective release for GOES-R GLM event-footprint Total Optical Energy rendering. It consolidates selected events by fixed-grid detector pixel, sums event energy across the active time window, and draws one polygon per accumulated pixel.

No event/group/flash parent-hierarchy behavior, LMA analysis behavior, network-overlay behavior, project format, or unrelated GUI feature was changed for this release.

## Source validation

- Package version: `1.6.1`
- Python used for build validation: 3.13.5
- Full source test suite: 123 tests passed
- Focused release tests: 9 tests passed
- Packaged demonstration loads 42,513 LMA sources
- GOES-16 reference window: 495 raw events consolidated to 74 pixels
- Maximum events in one accumulated pixel: 68
- Maximum accumulated pixel energy: approximately 1960.8714 fJ
- Accumulated energy conserves selected raw-event energy within floating-point tolerance
- Corrected interactive rendering visually matches the established glmtools reference pattern

## Release artifacts

The release build contains:

- stable pure-Python wheel;
- source ZIP and source tar.gz archives;
- packaged release ZIP with installer, dependency files, documentation, licenses, examples, wheel, source archives, and demonstration;
- standalone demonstration ZIP;
- SHA-256 manifests for the packaged kit and public upload artifacts.

Artifact integrity and install/import checks are performed after construction from this source tree.
