from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import gc
import json
from pathlib import Path
from typing import Iterable, Literal

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from .errors import ConfigurationError
from .interactions import LinkedViewController
from .io.project import load_project
from .output_naming import safe_output_stem
from .plotting import create_lma_figure
from .plotting.common import save_figure
from .figure_export import apply_saved_satellite_overlays

OverwritePolicy = Literal["skip", "replace", "fail"]
THEMES = ("dark", "light", "space")
COLOR_BY_OPTIONS = ("time", "altitude", "power", "stations", "chi2", "charge", "group", "log-chi2")


@dataclass(frozen=True)
class FigureBatchJob:
    output_path: str
    theme: str
    color_by: str
    maximum_chi2: float


@dataclass(frozen=True)
class FigureBatchManifest:
    project_path: str
    jobs: tuple[FigureBatchJob, ...]
    dpi: int = 300
    dynamic_titles: bool = True
    custom_title: str = ""
    overwrite_policy: OverwritePolicy = "replace"
    continue_on_error: bool = True
    cancel_file: str | None = None

    def validated(self) -> "FigureBatchManifest":
        if not self.project_path:
            raise ConfigurationError("Figure batches require a project path")
        if not self.jobs:
            raise ConfigurationError("The figure batch contains no jobs")
        if self.overwrite_policy not in {"skip", "replace", "fail"}:
            raise ConfigurationError(f"Unknown overwrite policy: {self.overwrite_policy}")
        if int(self.dpi) < 72 or int(self.dpi) > 1200:
            raise ConfigurationError("Figure DPI must be between 72 and 1200")
        for job in self.jobs:
            if job.theme not in THEMES:
                raise ConfigurationError(f"Unsupported figure theme: {job.theme}")
            if job.color_by not in COLOR_BY_OPTIONS:
                raise ConfigurationError(f"Unsupported color quantity: {job.color_by}")
            if not np.isfinite(job.maximum_chi2) or job.maximum_chi2 <= 0:
                raise ConfigurationError("Maximum chi-squared values must be positive")
        return self

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["format"] = "lmas-figure-batch-v0.1"
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "FigureBatchManifest":
        values = dict(payload)
        values.pop("format", None)
        values["jobs"] = tuple(FigureBatchJob(**job) for job in values.get("jobs", ()))
        return cls(**values).validated()


def _normalise(values: Iterable[str], allowed: tuple[str, ...], label: str) -> tuple[str, ...]:
    selected: list[str] = []
    for value in values:
        item = str(value).strip().lower().replace("_", "-")
        if item not in allowed:
            raise ConfigurationError(f"Unsupported {label}: {value}")
        if item not in selected:
            selected.append(item)
    if not selected:
        raise ConfigurationError(f"Choose at least one {label}")
    return tuple(selected)


def _chi2_token(value: float) -> str:
    text = f"{float(value):.3f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def figure_batch_jobs(
    *,
    output_directory: str | Path,
    base_stem: str,
    extension: str,
    themes: Iterable[str],
    color_by_options: Iterable[str],
    maximum_chi2_values: Iterable[float],
) -> tuple[FigureBatchJob, ...]:
    root = Path(output_directory).expanduser().resolve()
    suffix = extension if str(extension).startswith(".") else f".{extension}"
    theme_values = _normalise(themes, THEMES, "theme")
    color_values = _normalise(color_by_options, COLOR_BY_OPTIONS, "color quantity")
    chi2_values: list[float] = []
    for raw in maximum_chi2_values:
        value = float(raw)
        if not np.isfinite(value) or value <= 0:
            raise ConfigurationError("Maximum chi-squared values must be positive")
        if value not in chi2_values:
            chi2_values.append(value)
    if not chi2_values:
        raise ConfigurationError("Choose at least one maximum chi-squared value")

    jobs: list[FigureBatchJob] = []
    base = safe_output_stem(base_stem)
    for theme in theme_values:
        for color_by in color_values:
            for maximum_chi2 in chi2_values:
                filename = (
                    f"{base}_projection_color-{color_by}_chi2-{_chi2_token(maximum_chi2)}_"
                    f"{theme}{suffix.lower()}"
                )
                jobs.append(
                    FigureBatchJob(
                        output_path=str(root / filename),
                        theme=theme,
                        color_by=color_by,
                        maximum_chi2=float(maximum_chi2),
                    )
                )
    return tuple(jobs)


def write_figure_batch_manifest(manifest: FigureBatchManifest, path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest.validated().to_dict(), indent=2), encoding="utf-8"
    )
    return destination


def load_figure_batch_manifest(path: str | Path) -> FigureBatchManifest:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ConfigurationError("Figure batch manifest must contain a JSON object")
    return FigureBatchManifest.from_dict(payload)


def _project_view_limits(project) -> dict[str, tuple[float, float]]:
    view = project.view_filters.validated()
    plot = project.plot.validated()
    limits: dict[str, tuple[float, float]] = {}
    if view.start_time and view.end_time:
        values = mdates.date2num(
            np.asarray(
                [np.datetime64(view.start_time), np.datetime64(view.end_time)]
            ).astype("datetime64[us]").astype(object)
        )
        limits["time"] = tuple(sorted((float(values[0]), float(values[1]))))
    if view.minimum_altitude_km is not None and view.maximum_altitude_km is not None:
        limits["altitude"] = (
            float(view.minimum_altitude_km),
            float(view.maximum_altitude_km),
        )
    if plot.layout == "intfs":
        if view.minimum_x_km is not None and view.maximum_x_km is not None:
            low, high = float(view.minimum_x_km), float(view.maximum_x_km)
            if plot.north_south_viewpoint == "north":
                limits["west"] = (-high, -low)
            else:
                limits["east"] = (low, high)
        if view.minimum_y_km is not None and view.maximum_y_km is not None:
            low, high = float(view.minimum_y_km), float(view.maximum_y_km)
            if plot.east_west_viewpoint == "west":
                limits["south"] = (-high, -low)
            else:
                limits["north"] = (low, high)
    return limits


def run_figure_batch_manifest(path: str | Path) -> int:
    manifest = load_figure_batch_manifest(path)
    project = load_project(manifest.project_path)
    view_limits = _project_view_limits(project)
    cancel = Path(manifest.cancel_file).expanduser() if manifest.cancel_file else None
    total = len(manifest.jobs)
    completed = skipped = failures = 0
    print(f"LMAS figure batch: {total} job(s)", flush=True)

    for index, job in enumerate(manifest.jobs, start=1):
        if cancel is not None and cancel.exists():
            print(f"Batch cancellation requested; stopping before job {index}/{total}.", flush=True)
            break
        output = Path(job.output_path).expanduser().resolve()
        print(f"\n[batch {index}/{total}] {output.name}", flush=True)
        if output.exists():
            if manifest.overwrite_policy == "skip":
                print("  skipped: output already exists", flush=True)
                skipped += 1
                continue
            if manifest.overwrite_policy == "fail":
                print("  failed: output already exists", flush=True)
                failures += 1
                if not manifest.continue_on_error:
                    break
                continue
            output.unlink()

        figure = None
        try:
            filters = replace(project.filters, maximum_chi2=job.maximum_chi2).validated()
            logarithmic = job.color_by == "log-chi2"
            effective_color_by = "chi2" if logarithmic else job.color_by
            plot = replace(
                project.plot,
                theme=job.theme,
                color_by=effective_color_by,
                log_color_scale=logarithmic,
                preview_point_limit=0,
            ).validated()
            figure = create_lma_figure(project, filters=filters, plot=plot, for_export=True)
            controller = LinkedViewController(figure)
            if view_limits:
                controller.apply_interactive_limits(view_limits)
            # A pure color/theme batch should preserve an exact non-rectangular
            # linked subset.  A chi-squared sweep must instead preserve the
            # committed geometric view while allowing membership to change.
            same_quality = (
                project.filters.maximum_chi2 is None
                or abs(float(project.filters.maximum_chi2) - float(job.maximum_chi2)) < 1.0e-12
            )
            if same_quality and project.selected_source_ids is not None:
                state = controller.capture_view_state() or {}
                state["selected_source_ids"] = project.selected_source_ids
                same_color_scale = (
                    effective_color_by == project.plot.color_by
                    and logarithmic == bool(project.plot.log_color_scale)
                )
                if same_color_scale:
                    state["norm_limits"] = project.color_norm_limits
                else:
                    state["norm_limits"] = None
                controller.restore_view_state(
                    state,
                    exact_membership=True,
                    record_history=False,
                    notify=False,
                )
            metadata = getattr(figure, "_lmas_metadata", {})
            title_artist = metadata.get("title_artist") if isinstance(metadata, dict) else None
            if title_artist is not None and not manifest.dynamic_titles and manifest.custom_title:
                title_artist.set_text(manifest.custom_title)
            apply_saved_satellite_overlays(figure, project)
            output.parent.mkdir(parents=True, exist_ok=True)
            figure.canvas.draw()
            save_figure(figure, output, dpi=int(manifest.dpi))
            completed += 1
            print(f"  completed ({completed + skipped}/{total} handled)", flush=True)
        except KeyboardInterrupt:
            print("Batch interrupted.", flush=True)
            return 130
        except Exception as exc:
            failures += 1
            print(f"  failed: {exc}", flush=True)
            if not manifest.continue_on_error:
                break
        finally:
            if figure is not None:
                figure.clear()
                plt.close(figure)
            gc.collect()

    print(
        f"\nBatch finished: {completed} completed, {skipped} skipped, {failures} failed.",
        flush=True,
    )
    return 1 if failures else 0


__all__ = [
    "COLOR_BY_OPTIONS",
    "FigureBatchJob",
    "FigureBatchManifest",
    "THEMES",
    "figure_batch_jobs",
    "load_figure_batch_manifest",
    "run_figure_batch_manifest",
    "write_figure_batch_manifest",
]
