# LMAS 1.6.2 build report

## Scope

LMAS 1.6.2 backports the proven responsiveness changes to the stable 1.6.1 source baseline. The implementation was applied selectively from the exact development-wheel deltas rather than by downgrading a later feature branch.

The public 1.6 analysis feature set, scientific filters, Project/Profile formats, overlay workflows, saved products, and data readers are retained.

## Source validation

- Package version: `1.6.2`
- Python used for build validation: 3.13.5
- Full source test suite: 124 tests passed
- All packaged Python sources compiled successfully
- Focused animation-window tests verify empty opening frames before the first source
- Projection Space-bar and 3D keyboard-control contracts are present
- Selected-dataset caching, fast-pan restoration, source-selection, Project startup, and existing 1.6 feature contracts passed
- The published README differs from the supplied 1.6.1 README only by the expected `1.6.2` and `LMAS_1_6_2` substitutions
- Source and release artifacts are scanned to exclude removed private-workspace code and terminology

## Release artifacts

The release build contains:

- pure-Python wheel;
- source distribution (`tar.gz`);
- clean source ZIP;
- documentation ZIP;
- packaged release ZIP with installer, dependency files, documentation, licenses, examples, wheel, and source distributions;
- SHA-256 manifests for the packaged kit and public upload artifacts.

Wheel installation, package identity, entry points, archive integrity, and wheel `RECORD` hashes are checked after construction.
