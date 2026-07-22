# LMAS 1.6.3 build report

## Scope

LMAS 1.6.3 is a corrective patch built from the clean LMAS 1.6.2 source baseline. Only the recent projection-animation header correction and replacement-figure canvas synchronization are added. The stable 1.6 analysis feature set and 1.6.2 responsiveness architecture are retained.

## Source validation

- Package version: `1.6.3`
- Complete source test suite: 128 tests passed.
- All packaged Python sources compile successfully.
- Focused tests verify that live projection source time is not inserted into the figure title.
- Focused tests verify that saved projection source time uses a separate header clear of the top axes.
- Focused tests verify that replacement figures replay the Qt/Matplotlib resize path before the new linked-view controller is attached.
- The root README differs from the 1.6.2 README only by the expected `1.6.3` and `LMAS_1_6_3` substitutions.
- Source and release artifacts are scanned to exclude unrelated development-branch functionality and terminology.

## Release artifacts

The release build contains a pure-Python wheel, source distribution, clean source ZIP, documentation ZIP, complete packaged release ZIP, installer, dependency files, licenses, examples, demonstration data, and SHA-256 manifests.
