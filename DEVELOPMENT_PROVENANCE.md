# Development provenance

LMAS was designed and directed by **R. Stetson Reger** as a modern application for the analysis, visualization, and presentation of solved Lightning Mapping Array data.

The software consolidates and extends several years of research code, analysis methods, plotting tools, and scientific workflows developed by R. Stetson Reger. Those materials provided the technical and scientific foundation for LMAS, including its data-handling requirements, filtering behavior, visualization conventions, user-interface design, validation cases, and release criteria.

Native LMAS data readers are heavily based on the open-source [`xlma-python`](https://github.com/deeplycloudy/xlma-python) and [`glmtools`](https://github.com/deeplycloudy/glmtools) packages developed by Eric Bruning and collaborators. The `xlma-python` repository provides the Python package `pyxlma`. LMAS uses `pyxlma` and `glmtools` only as optional compatibility backends; it does not directly use `lmatools` as a reader backend or runtime dependency. Applicable upstream attribution and license texts are preserved with the release.

LMAS was developed with substantial assistance from **ChatGPT Plus**, which was used as an AI-assisted software-engineering tool for code generation, refactoring, debugging, testing, documentation, packaging, and iterative design work. The scientific objectives, architecture, feature requirements, interpretations, acceptance decisions, and overall direction of the project were supplied and reviewed by R. Stetson Reger.

AI-generated or AI-modified code was incorporated through an iterative process of inspection, testing, comparison against existing research workflows, and user-directed revision. Responsibility for the released software, its scientific behavior, and its maintenance remains with the author and maintainer.
