# LMAS 1.6.3 validation summary

LMAS 1.6.3 is a stable-line corrective release based on LMAS 1.6.2.

## Required checks

- Package identity reports `1.6.3`.
- The complete 128-test source suite passes.
- Replacement figures synchronize with the existing canvas size before display.
- Repeated Project and source-file switches retain the canvas and do not progressively shrink the plot.
- Interactive projection source time appears in the control row rather than the figure title.
- Saved projection source time remains embedded in a dedicated compact header clear of the top axes.
- Existing 1.6.2 responsiveness behavior and uncapped saved products remain intact.
- The root README passes the exact version-only transformation guard.
- Built artifacts pass the excluded-feature text and filename audit.
- Wheel metadata, installation, `RECORD` hashes, internal checksums, and archive integrity are verified.

Live repeated file switching and Qt/VTK interaction remain the final acceptance tests on supported user hardware.
