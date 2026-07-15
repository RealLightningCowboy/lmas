# LMAS lineage and attribution

**LMAS is intended to provide a modern, open, Python-based alternative to the legacy IDL program XLMA for viewing, filtering, analyzing, and presenting solved Lightning Mapping Array source data.** It is not a direct port of XLMA.

LMAS includes its own native solved-LMA reader and does not require pyxlma. Its development draws on established open-source LMA and GOES-R GLM software lineages, while the LMAS application architecture, workflows, native readers, interactive tools, project system, and user-facing implementation were designed and directed by R. Stetson Reger.

## Authorship and maintenance

- Author and maintainer: **R. Stetson Reger**
- LMAS license: MIT
- Copyright © 2026 R. Stetson Reger

## Lightning Mapping Array lineage

LMAS draws on the xlma-python / pyxlma software lineage developed by Eric Bruning and collaborators. LMAS uses its own native solved-LMA reader and functions independently of pyxlma, but preserves applicable upstream attribution and license text in `licenses/xlma-python-LICENSE.txt`.

## GOES-R GLM lineage

The LMAS native GLM reader, overlay manager, Project integration, GUI, diagnostics, and rendering architecture are LMAS implementations. GLM event-footprint geometry preserves the established glmtools fixed-grid pixel-corner lookup approach and packages the corresponding upstream lookup resource. glmtools also serves as the compatibility reference for LMAS observational xarray interchange.

The applicable glmtools BSD 3-Clause notice is preserved in `licenses/glmtools-BSD-3-Clause-LICENSE.txt`.

## Third-party notices

Additional dependency and bundled-resource notices are listed in `THIRD_PARTY_NOTICES.md` and the `licenses/` directory.
