from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import gc
import json
from pathlib import Path
from typing import Iterable, Literal

from .errors import ConfigurationError
from .output_naming import display_mode_label, safe_output_stem

BatchKind = Literal["projection", "3d"]
OverwritePolicy = Literal["skip", "replace", "fail"]

THEMES = ("dark", "light", "space")
DISPLAY_MODES = ("cumulative", "trail", "trail-afterimage")
ANIMATION_MODES_3D = ("develop", "orbit", "develop-orbit")

# Canonical defaults for the GUI's 3D Batch tab.  Keeping these values in a
# Qt-free module makes them testable and keeps future CLI/manifest workflows
# from quietly drifting away from the dialog.
THREE_D_BATCH_DEFAULT_THEMES = ("dark", "light")
THREE_D_BATCH_DEFAULT_ANIMATION_MODES = ("develop-orbit",)
THREE_D_BATCH_DEFAULT_DISPLAY_MODES = ("cumulative", "trail-afterimage")
THREE_D_BATCH_DEFAULT_EXTENSION = ".mp4"
THREE_D_BATCH_DEFAULT_OVERWRITE = "replace"
THREE_D_BATCH_DEFAULT_CONTINUE_ON_ERROR = True
THREE_D_BATCH_DEFAULT_TRAIL_MS = 30.0
THREE_D_BATCH_DEFAULT_AFTERIMAGE_MS = 30.0
THREE_D_BATCH_DEFAULT_FPS = 30
THREE_D_BATCH_DEFAULT_DURATION_S = 15.0
THREE_D_BATCH_DEFAULT_HOLD_END_S = 5.0
THREE_D_BATCH_DEFAULT_ORBIT_SPEED_DEG_S = 14.0
THREE_D_BATCH_DEFAULT_POINT_SIZE = 5.0
THREE_D_BATCH_DEFAULT_WINDOW = (1400, 900)
THREE_D_BATCH_DEFAULT_RENDER_PROFILE = "compatible"
THREE_D_BATCH_DEFAULT_VIDEO_QUALITY = 7


@dataclass(frozen=True)
class AnimationBatchJob:
    kind: BatchKind
    output_path: str
    theme: str
    display_mode: str
    animation_mode: str = "develop"


@dataclass(frozen=True)
class AnimationBatchManifest:
    jobs: tuple[AnimationBatchJob, ...]
    overwrite_policy: OverwritePolicy = "replace"
    continue_on_error: bool = True
    project_path: str | None = None
    snapshot_path: str | None = None
    cancel_file: str | None = None
    trail_ms: float = 30.0
    afterimage_ms: float = 30.0
    fps: int = 30
    duration_s: float = 15.0
    hold_end_s: float = 5.0
    width: int = 1600
    height: int = 900
    video_quality: int = 8
    orbit_speed_deg_s: float = 14.0
    point_size: float = THREE_D_BATCH_DEFAULT_POINT_SIZE
    cmap: str = "turbo"
    reverse_cmap: bool = False
    render_profile: str = "compatible"
    camera_path: str | None = None
    custom_title: str = ""
    show_grid_and_labels: bool = True

    def validated(self) -> "AnimationBatchManifest":
        if not self.jobs:
            raise ConfigurationError("The animation batch contains no jobs")
        if self.overwrite_policy not in {"skip", "replace", "fail"}:
            raise ConfigurationError(f"Unknown overwrite policy: {self.overwrite_policy}")
        kinds = {job.kind for job in self.jobs}
        if kinds == {"projection"} and not self.project_path:
            raise ConfigurationError("Projection animation batches require a project path")
        if kinds == {"3d"} and not self.snapshot_path:
            raise ConfigurationError("3D animation batches require a snapshot path")
        if len(kinds) != 1:
            raise ConfigurationError("A batch manifest must contain only projection or only 3D jobs")
        return self

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["format"] = "lmas-animation-batch-v0.1"
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "AnimationBatchManifest":
        values = dict(payload)
        values.pop("format", None)
        values["jobs"] = tuple(AnimationBatchJob(**job) for job in values.get("jobs", ()))
        return cls(**values).validated()


def _normalise_selection(values: Iterable[str], allowed: tuple[str, ...], label: str) -> tuple[str, ...]:
    selected: list[str] = []
    for value in values:
        item = str(value).strip().lower().replace("_", "-")
        if item == "cumulative-afterimage":
            item = "cumulative"
        if item not in allowed:
            raise ConfigurationError(f"Unsupported {label}: {value}")
        if item not in selected:
            selected.append(item)
    if not selected:
        raise ConfigurationError(f"Choose at least one {label}")
    return tuple(selected)


def projection_batch_jobs(
    *,
    output_directory: str | Path,
    base_stem: str,
    extension: str,
    themes: Iterable[str],
    display_modes: Iterable[str],
) -> tuple[AnimationBatchJob, ...]:
    root = Path(output_directory).expanduser().resolve()
    suffix = extension if str(extension).startswith(".") else f".{extension}"
    theme_values = _normalise_selection(themes, THEMES, "theme")
    display_values = _normalise_selection(display_modes, DISPLAY_MODES, "display mode")
    jobs: list[AnimationBatchJob] = []
    for theme in theme_values:
        for display in display_values:
            filename = "_".join(
                [safe_output_stem(base_stem), "projection", "development", display_mode_label(display), theme]
            ) + suffix.lower()
            jobs.append(AnimationBatchJob("projection", str(root / filename), theme, display, "develop"))
    return tuple(jobs)


def three_d_batch_jobs(
    *,
    output_directory: str | Path,
    base_stem: str,
    extension: str,
    themes: Iterable[str],
    display_modes: Iterable[str],
    animation_modes: Iterable[str],
) -> tuple[AnimationBatchJob, ...]:
    root = Path(output_directory).expanduser().resolve()
    suffix = extension if str(extension).startswith(".") else f".{extension}"
    theme_values = _normalise_selection(themes, THEMES, "theme")
    display_values = _normalise_selection(display_modes, DISPLAY_MODES, "display mode")
    mode_values = _normalise_selection(animation_modes, ANIMATION_MODES_3D, "animation kind")
    jobs: list[AnimationBatchJob] = []
    for theme in theme_values:
        for mode in mode_values:
            # A pure orbit always shows the completed source cloud. Repeating it
            # for every display mode would create visually identical outputs.
            effective_displays = ("cumulative",) if mode == "orbit" else display_values
            for display in effective_displays:
                parts = [safe_output_stem(base_stem), "3d", display_mode_label(mode)]
                if mode != "orbit":
                    parts.append(display_mode_label(display))
                parts.append(theme)
                filename = "_".join(parts) + suffix.lower()
                jobs.append(AnimationBatchJob("3d", str(root / filename), theme, display, mode))
    return tuple(jobs)


def write_batch_manifest(manifest: AnimationBatchManifest, path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest.validated().to_dict(), indent=2), encoding="utf-8")
    return destination


def load_batch_manifest(path: str | Path) -> AnimationBatchManifest:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ConfigurationError("Animation batch manifest must contain a JSON object")
    return AnimationBatchManifest.from_dict(payload)


def run_batch_manifest(path: str | Path) -> int:
    manifest = load_batch_manifest(path)
    total = len(manifest.jobs)
    cancel = Path(manifest.cancel_file).expanduser() if manifest.cancel_file else None
    failures = 0
    completed = 0
    skipped = 0
    print(f"LMAS animation batch: {total} job(s)", flush=True)
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
        try:
            if job.kind == "projection":
                from .io.project import load_project
                from .visualization.projection_animation import animate_projection_project

                project = load_project(manifest.project_path)
                project.plot = replace(project.plot, theme=job.theme).validated()
                animate_projection_project(
                    project,
                    output_path=output,
                    display_mode=job.display_mode,
                    trail_ms=manifest.trail_ms,
                    afterimage_ms=manifest.afterimage_ms,
                    fps=manifest.fps,
                    duration_s=manifest.duration_s,
                    hold_end_s=manifest.hold_end_s,
                    width=manifest.width,
                    height=manifest.height,
                    video_quality=manifest.video_quality,
                    custom_title=manifest.custom_title,
                )
            else:
                from .visualization.animation_3d import animate_3d_snapshot

                animate_3d_snapshot(
                    manifest.snapshot_path,
                    output_path=output,
                    mode=job.animation_mode,
                    display_mode=job.display_mode,
                    trail_ms=manifest.trail_ms,
                    afterimage_ms=manifest.afterimage_ms,
                    point_size=manifest.point_size,
                    cmap=manifest.cmap,
                    reverse_cmap=manifest.reverse_cmap,
                    theme=job.theme,
                    render_profile=manifest.render_profile,
                    camera_path=manifest.camera_path,
                    fps=manifest.fps,
                    duration_s=manifest.duration_s,
                    hold_end_s=manifest.hold_end_s,
                    orbit_speed_deg_s=manifest.orbit_speed_deg_s,
                    video_quality=manifest.video_quality,
                    show_grid_and_labels=manifest.show_grid_and_labels,
                    window_size=(manifest.width, manifest.height),
                )
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
            gc.collect()
    print(
        f"\nBatch finished: {completed} completed, {skipped} skipped, {failures} failed.",
        flush=True,
    )
    return 1 if failures else 0


__all__ = [
    "ANIMATION_MODES_3D",
    "AnimationBatchJob",
    "AnimationBatchManifest",
    "DISPLAY_MODES",
    "THEMES",
    "THREE_D_BATCH_DEFAULT_AFTERIMAGE_MS",
    "THREE_D_BATCH_DEFAULT_ANIMATION_MODES",
    "THREE_D_BATCH_DEFAULT_CONTINUE_ON_ERROR",
    "THREE_D_BATCH_DEFAULT_DISPLAY_MODES",
    "THREE_D_BATCH_DEFAULT_DURATION_S",
    "THREE_D_BATCH_DEFAULT_EXTENSION",
    "THREE_D_BATCH_DEFAULT_FPS",
    "THREE_D_BATCH_DEFAULT_HOLD_END_S",
    "THREE_D_BATCH_DEFAULT_ORBIT_SPEED_DEG_S",
    "THREE_D_BATCH_DEFAULT_OVERWRITE",
    "THREE_D_BATCH_DEFAULT_POINT_SIZE",
    "THREE_D_BATCH_DEFAULT_RENDER_PROFILE",
    "THREE_D_BATCH_DEFAULT_THEMES",
    "THREE_D_BATCH_DEFAULT_TRAIL_MS",
    "THREE_D_BATCH_DEFAULT_VIDEO_QUALITY",
    "THREE_D_BATCH_DEFAULT_WINDOW",
    "load_batch_manifest",
    "projection_batch_jobs",
    "run_batch_manifest",
    "three_d_batch_jobs",
    "write_batch_manifest",
]
