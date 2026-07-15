# Network Overlays

**Network Overlays** displays ground-based lightning-location-network observations alongside LMAS. Network events remain separate from LMA source groups and GLM groups.

## Load and view data

Open **Network Overlays** from the analysis toolbar or **View** menu. Load ENTLN-oriented or generic CSV files with the browser or by entering a file/directory path directly.

LMAS recognizes common columns for time, latitude, longitude, event type, polarity, peak current, sensor count, optional altitude, quality, uncertainty geometry, and event identifiers. The included `examples/generic_network_events.csv` shows the normalized format. Supported ENTLN exports use `0 = CG` and `1 = IC`; LMAS preserves the original provider value in exported products.

Available displays include plan-view event markers, optional uncertainty ellipses, event-time rails, figure-legend categories, and signed peak current versus time. **No artificial altitude is assigned:** events without a reported altitude remain plan-view and time-rail observations. Marker size scales with absolute peak current by default.

## Filters, Projects, and export

Filter by event type, polarity, minimum absolute peak current, minimum sensor count, linked time range, and current plan-view bounds. Projects preserve file references, provider identity, visibility, filters, and appearance; missing overlay files do not prevent the LMA Project from opening.

Exported CSV or NetCDF products retain the available normalized fields and original provider code.

## Limitations

Unfamiliar provider schemas may require column renaming before import. Dedicated NLDN and GLD360 adapters have not been qualified against representative files. Provider-supplied uncertainty geometry should be interpreted according to that provider's product documentation.
