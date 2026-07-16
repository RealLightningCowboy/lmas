"""Lightning Mapping Array Suite (LMAS)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["FilterSpec", "LMAProject", "LmaSourceStore", "PlotSpec"]
__version__ = "1.6.1"

if TYPE_CHECKING:
    from .model import FilterSpec, LMAProject, PlotSpec
    from .source_store import LmaSourceStore


def __getattr__(name: str) -> Any:
    """Load public scientific classes lazily.

    Keeping the package root lightweight allows version checks and the
    operating-system launcher installer to run before optional scientific and
    GUI dependencies are imported.
    """
    if name in {"FilterSpec", "LMAProject", "PlotSpec"}:
        from . import model

        return getattr(model, name)
    if name == "LmaSourceStore":
        from .source_store import LmaSourceStore

        return LmaSourceStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
