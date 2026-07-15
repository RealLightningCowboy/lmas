# LMAS real-data hybrid demonstration

This packaged demonstration opens the Oklahoma LMA flash centered near
**2019-04-30 14:49:14.265 UTC** with simultaneous GOES-R GLM observations.

## Included data

- `data/lma/LYLOUT_190430_144844_0060.dat.gz`
  - one-minute subset from 14:48:44 through 14:49:44 UTC;
  - 42,513 unchanged solved-source rows;
  - header start time, duration, event count, and subset provenance updated;
  - station summary lines retained from the ten-minute parent file.
- Four GOES-16 East GLM L2 LCFA files spanning 14:48:40–14:50:00 UTC.
- Four GOES-17 West GLM L2 LCFA files spanning 14:48:40–14:50:00 UTC.
- `Hybrid_20190430_144914.lmas-project.yaml`, which opens directly on the
  approximately 0.94-second flash interval. Both GLM spacecraft are loaded but
  disabled by default; enable them in Satellite Overlays to show event footprints,
  official GLM group centroids, East/West time rails, shared optical-energy scaling,
  legend, and the bottom GLM total-optical-energy colorbar.


## Opening

Use **File → Open demonstration** in the GUI, or run:

```bash
lma gui --demo
```

The installed demo Project is a template. Save it to a user-writable location
before making permanent changes.
