from __future__ import annotations

from pathlib import Path
import os
import tempfile
from typing import Iterable

import matplotlib as mpl
import numpy as np
from matplotlib.ticker import LogLocator, NullFormatter

from ..coordinates import altitude_km
from ..errors import DatasetError

# Preserve the v0.1.0 visual styles while correcting their user-facing names:
#   Space mode = the original restrained dark-gray viewer appearance.
#   Dark mode  = the original true-black appearance.
THEMES = {
    "light": {
        "figure": "white",
        "axes": "white",
        "text": "black",
        "grid": "0.82",
        "station": "black",
    },
    "space": {
        "figure": "#202020",
        "axes": "black",
        "text": "#eeeeee",
        "grid": "#555555",
        "station": "white",
    },
    "dark": {
        "figure": "black",
        "axes": "black",
        "text": "white",
        "grid": "#444444",
        "station": "white",
    },
}


def theme_values(name: str) -> dict[str, str]:
    return THEMES.get(str(name).lower(), THEMES["space"])


def apply_figure_theme(
    figure,
    axes: Iterable,
    theme: str,
    *,
    show_grid: bool = True,
) -> None:
    values = theme_values(theme)
    figure.patch.set_facecolor(values["figure"])
    for text in figure.texts:
        text.set_color(values["text"])
    for axis in axes:
        axis.set_facecolor(values["axes"])
        axis.tick_params(colors=values["text"], which="both")
        axis.xaxis.label.set_color(values["text"])
        axis.yaxis.label.set_color(values["text"])
        axis.title.set_color(values["text"])
        for text in axis.texts:
            text.set_color(values["text"])
        for spine in axis.spines.values():
            spine.set_color(values["text"])
        if show_grid:
            axis.grid(
                True,
                which="major",
                linewidth=0.35,
                alpha=0.24,
                color=values["grid"],
            )
        else:
            axis.grid(False, which="major")
    figure._lmas_theme = values


def _field_values(dataset, normalized: str) -> tuple[np.ndarray, str]:
    if normalized == "altitude":
        return altitude_km(dataset), "Altitude (km MSL)"
    mapping = {
        "power": ("event_power", "VHF Source Power (dBW)"),
        "stations": ("event_stations", "Contributing stations"),
        "chi2": ("event_chi2", "Reduced χ²"),
    }
    field, label = mapping[normalized]
    if field not in dataset:
        raise DatasetError(f"Cannot color by {normalized}: dataset has no {field}")
    return np.asarray(dataset[field].values, dtype=float), label


def color_values(
    dataset,
    mode: str,
    *,
    normalization_dataset=None,
    logarithmic: bool = False,
) -> tuple[np.ndarray, str, mpl.colors.Normalize]:
    """Return display values and a normalization shared across all panels.

    ``normalization_dataset`` is primarily used by the GUI's INTFS-style
    "Remap colormap to selected points" preference.  When it is provided, the selected
    points retain their own values while the color scale is computed from the
    larger reference dataset.
    """

    normalized = str(mode).lower().replace("_", "-")
    reference = dataset if normalization_dataset is None else normalization_dataset
    if normalized == "time":
        times = np.asarray(dataset["event_time"].values).astype("datetime64[ns]")
        reference_times = np.asarray(reference["event_time"].values).astype("datetime64[ns]")
        valid_reference = reference_times[~np.isnat(reference_times)]
        if valid_reference.size == 0:
            raise DatasetError("No valid times are available for time coloring")
        origin = valid_reference.min()
        values = np.asarray((times - origin) / np.timedelta64(1, "s"), dtype=float)
        reference_values = np.asarray(
            (reference_times - origin) / np.timedelta64(1, "s"), dtype=float
        )
        label = f"Seconds after {str(origin).replace('T', ' ')[:23]} UTC"
    else:
        values, label = _field_values(dataset, normalized)
        reference_values, _ = _field_values(reference, normalized)

    finite = reference_values[np.isfinite(reference_values)]
    if logarithmic:
        finite = finite[finite > 0]
    if finite.size == 0:
        qualifier = "positive finite" if logarithmic else "finite"
        raise DatasetError(f"No {qualifier} values are available for {label}")
    low, high = float(np.min(finite)), float(np.max(finite))
    if high <= low:
        if logarithmic:
            low = max(low * 0.9, np.nextafter(0.0, 1.0))
            high = high * 1.1
        else:
            low -= 0.5
            high += 0.5
    norm = (
        mpl.colors.LogNorm(vmin=low, vmax=high)
        if logarithmic
        else mpl.colors.Normalize(vmin=low, vmax=high)
    )
    if logarithmic and normalized == "chi2":
        label = "log₁₀(χ²)"
    return values, label, norm


def resolved_cmap(name: str, *, reverse: bool = False) -> str:
    normalized = str(name).strip()
    if reverse and not normalized.endswith("_r"):
        return f"{normalized}_r"
    if not reverse and normalized.endswith("_r"):
        return normalized[:-2]
    return normalized


def automatic_point_size(count: int) -> float:
    if count <= 250:
        return 16.0
    if count <= 1000:
        return 8.0
    if count <= 5000:
        return 4.0
    if count <= 20000:
        return 1.8
    return 0.8


def finite_limits(
    values: np.ndarray,
    *,
    pad_fraction: float = 0.04,
    minimum_pad: float = 0.05,
) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return -1.0, 1.0
    low, high = float(np.min(finite)), float(np.max(finite))
    pad = max(minimum_pad, (high - low) * pad_fraction)
    if high <= low:
        pad = max(pad, abs(low) * pad_fraction, 1e-6)
    return low - pad, high + pad


def centered_span_limits(limits: tuple[float, float], span: float) -> tuple[float, float]:
    center = 0.5 * (float(limits[0]) + float(limits[1]))
    half = 0.5 * float(span)
    return center - half, center + half


def add_aligned_vertical_colorbar(
    figure,
    mappable,
    *,
    axes: Iterable,
    label: str,
    theme: str,
    pad: float = 0.018,
    width: float = 0.014,
):
    axes = tuple(axes)
    if not axes:
        raise ValueError("At least one axis is required for an aligned colorbar")
    figure.canvas.draw()
    positions = [axis.get_position() for axis in axes]
    bottom = min(position.y0 for position in positions)
    top = max(position.y1 for position in positions)
    right = max(position.x1 for position in positions)
    color_axis = figure.add_axes([right + pad, bottom, width, top - bottom])
    colorbar = figure.colorbar(mappable, cax=color_axis)
    colorbar.set_label(label)
    style_colorbar(colorbar, theme)
    return colorbar


def style_colorbar(colorbar, theme: str) -> None:
    values = theme_values(theme)
    colorbar.ax.set_facecolor(values["axes"])
    colorbar.ax.tick_params(colors=values["text"], which="both")
    if isinstance(getattr(colorbar.mappable, "norm", None), mpl.colors.LogNorm):
        # Matplotlib can omit logarithmic minor ticks for narrow ranges.  LMAS
        # makes the base-10 subdivisions explicit so log χ² colorbars remain
        # readable at a glance.
        axis = colorbar.ax.yaxis if colorbar.orientation == "vertical" else colorbar.ax.xaxis
        axis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2.0, 10.0), numticks=100))
        axis.set_minor_formatter(NullFormatter())
        colorbar.ax.tick_params(which="minor", length=3.0, width=0.6, colors=values["text"])
    colorbar.ax.xaxis.label.set_color(values["text"])
    colorbar.ax.yaxis.label.set_color(values["text"])
    colorbar.outline.set_edgecolor(values["text"])
    for spine in colorbar.ax.spines.values():
        spine.set_color(values["text"])


def save_figure(figure, path: str | Path, *, dpi: int = 300) -> Path:
    """Save a figure through a verified same-directory temporary file.

    Writing beside the destination and then using :func:`os.replace` makes new
    saves and explicit overwrites follow one reliable path on Windows and Linux.
    LMAS never reports success unless the rendered temporary file exists and is
    non-empty.
    """

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = destination.suffix.lower()
    if not suffix:
        raise ValueError("Figure output filename must include an extension")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.lmas-",
        suffix=suffix,
        dir=str(destination.parent),
    )
    os.close(descriptor)
    temporary = Path(temporary_name)

    try:
        metadata = getattr(figure, "_lmas_metadata", {})
        layout = metadata.get("layout") if isinstance(metadata, dict) else None
        save_kwargs = {"dpi": int(dpi), "format": suffix.lstrip(".")}
        restore_size = None
        if layout == "xlma":
            # Save at the canonical Portrait page size without resizing the
            # embedded Qt widget.  Restoring the transient live-canvas inches
            # after the render prevents Save Figure from changing the preview.
            export_size = metadata.get("export_size_inches", figure.get_size_inches())
            width, height = (float(export_size[0]), float(export_size[1]))
            restore_size = tuple(float(value) for value in figure.get_size_inches())
            figure.set_size_inches(width, height, forward=False)
            save_kwargs["bbox_inches"] = None
        else:
            save_kwargs.update({"bbox_inches": "tight", "pad_inches": 0.24})

        try:
            figure.savefig(temporary, **save_kwargs)
        finally:
            if restore_size is not None:
                figure.set_size_inches(*restore_size, forward=False)
        if not temporary.is_file() or temporary.stat().st_size <= 0:
            raise OSError(f"Figure renderer did not produce a valid file: {temporary}")

        # Flush the completed render before replacing the user-visible path.
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)

        if not destination.is_file() or destination.stat().st_size <= 0:
            raise OSError(f"Saved figure could not be verified: {destination}")
        return destination
    finally:
        temporary.unlink(missing_ok=True)
