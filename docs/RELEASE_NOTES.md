# LMAS 1.6.2 release notes

LMAS 1.6.2 is a responsiveness and interaction release for the stable 1.6 line. It carries the proven performance work from the later development builds into the published 1.6 feature set, without adding new scientific workspaces or changing established Project and product formats.

## Main-window responsiveness

- The Qt canvas and Matplotlib toolbar are retained across ordinary redraws instead of being destroyed and recreated.
- Cosmetic controls update existing artists in place when the figure structure does not need to change.
- Plot-ready NumPy arrays and repeated scientific selections are cached against the loaded source store.
- Linked-view controllers and callbacks are disconnected cleanly when a structural redraw is required.

## Faster panning

While the mouse is held down, LMAS temporarily displays a stable 1,500-source proxy and renders only the moving scientific panel at a bounded update rate. Releasing the mouse restores the configured preview population, applies the exact final limits, updates linked scientific membership once, and performs one complete redraw.

## Interactive animation

- **View proj.** opens inside the running LMAS application and reuses the already-loaded Project and source data.
- Interactive projection and 3D viewers use time-stratified point limits while saved animations remain uncapped.
- Projection frames use pre-sorted source times, binary-search slices, reusable buffers, elapsed-time playback, missed-frame skipping, throttled timeline scrubbing, and Matplotlib blitting.
- The Space bar controls projection Play/Pause immediately after the viewer opens.
- Projection and 3D animations begin at the exact start of the selected time window, even when the first frames contain no sources.
- 3D shutdown handling is more defensive when a native VTK window closes or the graphics backend fails.

## Startup

LMAS now shows a small startup window before importing the full GUI stack. Initial Project or data loading begins after the Qt event loop starts, and optional readers, overlays, dialogs, product exporters, plotting helpers, and analysis tools are imported only when needed.

## Scope

The release preserves 1.6 scientific filtering, source selection, charge analysis, overlays, saved figures, saved animations, data products, Project/Profile compatibility, and command-line workflows. The performance changes affect interactive preparation and rendering; exact exports remain based on the complete selected source population.
