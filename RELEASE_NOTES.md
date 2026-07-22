# LMAS 1.6.3 release notes

LMAS 1.6.3 is a small corrective update for the stable 1.6 line. It begins with the published 1.6.2 source and preserves that release's responsiveness work and public feature set.

## Corrected Project and source switching

LMAS retains the Qt canvas and Matplotlib toolbar for responsiveness. When a different Project or source file was opened in the same session, the newly attached Matplotlib figure could keep its smaller constructor dimensions because Qt had no reason to issue another widget resize event. LMAS now explicitly synchronizes the replacement figure, renderer, DPI, and resize callbacks to the existing canvas before drawing. The new plot therefore fills the current canvas immediately.

## Projection-animation source time

- The live **View proj.** source time is displayed in the control row below the figure rather than as a second figure-title line.
- Saved projection animations retain source time in a compact dedicated header above the top science axes.
- The main title remains one line, preventing overlap with the altitude-versus-time panel on shorter laptop displays.

## Scope

The release retains all LMAS 1.6.2 startup, panning, redraw, interactive animation, and 3D responsiveness improvements. It adds no new scientific workspace and does not change Project or product formats.
