from __future__ import annotations

from importlib.resources import files
from pathlib import Path

from .io.project import load_project
from .io.readers import synthetic_dataset
from .model import LMAProject

_DEMO_RELATIVE_PROJECT = (
    "demo/hybrid_20190430_144914/"
    "Hybrid_20190430_144914.lmas-project.yaml"
)


def packaged_demo_project_path() -> Path:
    """Return the installed real-data hybrid demonstration Project path."""

    resource = files("lmas.resources").joinpath(_DEMO_RELATIVE_PROJECT)
    return Path(str(resource)).resolve()


def demo_project() -> LMAProject:
    """Load the packaged one-minute LMA + dual-GLM hybrid demonstration."""

    return load_project(packaged_demo_project_path(), reader_backend="native")


def synthetic_project(count: int = 1200) -> LMAProject:
    """Retain the deterministic synthetic project for tests and small examples."""

    return LMAProject(
        dataset=synthetic_dataset(count=count),
        name="LMAS synthetic LMA demonstration",
        reader_backend="synthetic",
        reader_backend_version="1",
        reader_details={"source": "built-in demo"},
    )


__all__ = ["demo_project", "packaged_demo_project_path", "synthetic_project"]
