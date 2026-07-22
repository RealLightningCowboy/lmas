from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from . import __version__
from .errors import LMASError

# Parser-only constants stay local so launching ``lma gui`` does not import the
# scientific, plotting, product-export, and profile stacks before dispatch.
EXPORT_SCOPES = ("all", "filtered", "assigned", "active_group")



def _add_filter_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--start-time", help="Inclusive UTC start time")
    parser.add_argument("--end-time", help="Inclusive UTC end time")
    parser.add_argument("--min-stations", type=int, default=6)
    parser.add_argument("--max-chi2", type=float, default=1.0)
    parser.add_argument("--min-altitude-km", type=float)
    parser.add_argument("--max-altitude-km", type=float)
    parser.add_argument("--min-power", type=float)
    parser.add_argument("--max-power", type=float)
    parser.add_argument("--min-x-km", type=float)
    parser.add_argument("--max-x-km", type=float)
    parser.add_argument("--min-y-km", type=float)
    parser.add_argument("--max-y-km", type=float)


def _filters(args):
    from .model import FilterSpec

    return FilterSpec(
        start_time=getattr(args, "start_time", None),
        end_time=getattr(args, "end_time", None),
        minimum_stations=getattr(args, "min_stations", 6),
        maximum_chi2=getattr(args, "max_chi2", 1.0),
        minimum_altitude_km=getattr(args, "min_altitude_km", None),
        maximum_altitude_km=getattr(args, "max_altitude_km", None),
        minimum_power=getattr(args, "min_power", None),
        maximum_power=getattr(args, "max_power", None),
        minimum_x_km=getattr(args, "min_x_km", None),
        maximum_x_km=getattr(args, "max_x_km", None),
        minimum_y_km=getattr(args, "min_y_km", None),
        maximum_y_km=getattr(args, "max_y_km", None),
    ).validated()


def _project_from_args(args):
    from .demo import demo_project
    from .io.project import load_project
    from .io.readers import load_lma_files

    if getattr(args, "project", None):
        return load_project(args.project, reader_backend=getattr(args, "reader", "auto"))
    files = getattr(args, "files", None) or []
    if getattr(args, "demo", False):
        return demo_project()
    return load_lma_files(
        files,
        reference_latitude=getattr(args, "reference_latitude", None),
        reference_longitude=getattr(args, "reference_longitude", None),
        reader_backend=getattr(args, "reader", "auto"),
    )


def command_inspect(args) -> int:
    from .summary import project_summary

    print(json.dumps(project_summary(_project_from_args(args)), indent=2))
    return 0


def command_plot(args) -> int:
    from .model import PlotSpec
    from .plotting import create_lma_figure, save_figure

    project = _project_from_args(args)
    filters = _filters(args)
    plot = PlotSpec(
        layout=args.layout,
        coordinate_system=args.coordinates,
        show_histogram=bool(getattr(args, "hist", False)),
        text_size_preset=args.text_size,
        color_by=args.color_by,
        cmap=args.cmap,
        theme=args.theme,
        point_size=args.point_size,
        show_stations=not args.hide_stations,
        show_colorbar=not args.hide_colorbar,
        show_grid=not args.hide_grid,
        show_legend=bool(getattr(args, "show_legend", False)),
        show_panel_labels=bool(getattr(args, "show_panel_labels", False)),
        reverse_cmap=args.reverse_cmap,
        auto_fit_spatial=args.auto_fit_spatial,
        remap_time_colors=not args.full_record_time_colors,
        north_south_viewpoint=args.north_south_viewpoint,
        east_west_viewpoint=args.east_west_viewpoint,
        depth_mode=args.depth,
        title=args.title,
        dpi=args.preview_dpi,
        saved_figure_dpi=args.dpi,
        preview_point_limit=0,
    ).validated()
    figure = create_lma_figure(project, filters=filters, plot=plot, for_export=True)
    print(save_figure(figure, args.output, dpi=args.dpi))
    return 0


def command_project(args) -> int:
    from .io.project import save_project
    from .io.readers import load_lma_files
    from .model import PlotSpec

    project = load_lma_files(
        args.files,
        name=args.name,
        reference_latitude=args.reference_latitude,
        reference_longitude=args.reference_longitude,
        reader_backend=args.reader,
    )
    project.filters = _filters(args)
    project.plot = PlotSpec(
        layout=args.layout,
        coordinate_system=args.coordinates,
        show_histogram=bool(getattr(args, "hist", False)),
        text_size_preset=args.text_size,
        color_by=args.color_by,
        cmap=args.cmap,
        theme=args.theme,
        point_size=args.point_size,
        reverse_cmap=args.reverse_cmap,
        show_colorbar=not args.hide_colorbar,
        show_grid=not args.hide_grid,
        show_legend=bool(getattr(args, "show_legend", False)),
        show_panel_labels=bool(getattr(args, "show_panel_labels", False)),
        auto_fit_spatial=args.auto_fit_spatial,
        remap_time_colors=not args.full_record_time_colors,
        north_south_viewpoint=args.north_south_viewpoint,
        east_west_viewpoint=args.east_west_viewpoint,
        depth_mode=args.depth,
    ).validated()
    print(save_project(project, args.output))
    return 0


def command_gui(args) -> int:
    from .gui.app import run_application

    files = [Path(value) for value in args.files]
    project_path = args.project
    if project_path is None and len(files) == 1 and files[0].name.lower().endswith(
        (".lmas-project.yaml", ".lmas-project.yml", ".lmas.yaml", ".lmas.yml")
    ):
        project_path, files = files[0], []
    return run_application(
        files=files,
        project_path=project_path,
        demo=args.demo,
        profile_name=args.profile,
        reader_backend=args.reader,
    )


def command_export_polarity(args) -> int:
    from .io.project import load_project
    from .polarity_product import export_polarity_csv, export_polarity_netcdf

    project = load_project(args.project, reader_backend=getattr(args, "reader", "auto"))
    if args.format == "csv":
        destination = export_polarity_csv(
            project, args.output, scope=args.scope
        )
    else:
        destination = export_polarity_netcdf(
            project, args.output, scope=args.scope
        )
    print(destination)
    return 0


def command_import_polarity(args) -> int:
    from .io.project import load_project, save_project
    from .polarity_product import import_polarity_netcdf

    project = load_project(args.project, reader_backend=getattr(args, "reader", "auto"))
    project.source_selection_state = import_polarity_netcdf(
        project, args.polarity, allow_partial=bool(args.allow_partial)
    )
    print(save_project(project, args.output))
    return 0


def command_readers(args) -> int:
    from .io.backends import reader_backend_statuses

    payload = [
        {
            "name": status.name,
            "label": status.label,
            "available": status.available,
            "version": status.version,
            "description": status.description,
        }
        for status in reader_backend_statuses()
    ]
    print(json.dumps(payload, indent=2))
    return 0


def command_profiles(args) -> int:
    from .profiles import ProfileStore

    store = ProfileStore()
    if args.profile_command == "list":
        for profile in store.list():
            marker = " (built-in)" if profile.built_in else ""
            print(f"{profile.name}{marker}")
        return 0
    if args.profile_command == "show":
        print(json.dumps(store.get(args.name).to_dict(), indent=2))
        return 0
    if args.profile_command == "import":
        print(store.import_file(args.path, overwrite=args.overwrite))
        return 0
    if args.profile_command == "export":
        print(store.export(args.name, args.path))
        return 0
    if args.profile_command == "delete":
        store.delete(args.name)
        return 0
    raise LMASError("Unknown profile command")


def command_snapshot_3d(args) -> int:
    from .model import PlotSpec
    from .visualization.snapshot import build_visualization_snapshot

    project = _project_from_args(args)
    plot = PlotSpec(
        color_by=args.color_by,
        cmap=args.cmap,
        theme=args.theme,
        point_size=args.point_size,
        reverse_cmap=args.reverse_cmap,
        log_color_scale=args.log_color_scale,
        title=args.title,
    ).validated()
    snapshot = build_visualization_snapshot(
        project,
        filters=_filters(args),
        plot=plot,
        output_path=args.output,
    )
    print(snapshot.path)
    return 0


def command_view_3d(args) -> int:
    from .visualization.pyvista_3d import view_3d_snapshot

    view_3d_snapshot(
        args.snapshot,
        display_mode=args.display_mode,
        trail_ms=args.trail_ms,
        afterimage_ms=args.afterimage_ms,
        point_size=args.point_size,
        cmap=args.cmap,
        reverse_cmap=args.reverse_cmap,
        theme=args.theme,
        render_profile=args.render_profile,
        interaction_mode=args.interaction_mode,
        camera_path=args.camera,
        camera_output=args.camera_output,
        playback_fps=args.fps,
        playback_duration_s=args.duration_s,
        point_limit=args.point_limit,
        start_playing=args.play,
        show_grid_and_labels=not args.hide_axes,
        window_size=(args.width, args.height),
    )
    return 0


def command_animate_3d(args) -> int:
    from .visualization.animation_3d import animate_3d_snapshot

    destination = animate_3d_snapshot(
        args.snapshot,
        output_path=args.output,
        mode=args.mode,
        display_mode=args.display_mode,
        trail_ms=args.trail_ms,
        afterimage_ms=args.afterimage_ms,
        point_size=args.point_size,
        cmap=args.cmap,
        reverse_cmap=args.reverse_cmap,
        theme=args.theme,
        render_profile=args.render_profile,
        camera_path=args.camera,
        camera_output=args.camera_output,
        fps=args.fps,
        duration_s=args.duration_s,
        hold_end_s=args.hold_end_s,
        orbit_speed_deg_s=args.orbit_speed_deg_s,
        video_quality=args.video_quality,
        show_grid_and_labels=not args.hide_axes,
        window_size=(args.width, args.height),
    )
    print(destination)
    return 0



def command_batch_animations(args) -> int:
    from .animation_batch import run_batch_manifest

    return run_batch_manifest(args.manifest)


def command_batch_figures(args) -> int:
    from .figure_batch import run_figure_batch_manifest

    return run_figure_batch_manifest(args.manifest)


def command_view_projections(args) -> int:
    from .gui.projection_animation_viewer import run_projection_animation_viewer
    from .io.project import load_project

    project = load_project(args.project)
    return run_projection_animation_viewer(
        project,
        display_mode=args.display_mode,
        trail_ms=args.trail_ms,
        afterimage_ms=args.afterimage_ms,
        fps=args.fps,
        duration_s=args.duration_s,
        point_limit=args.point_limit,
    )


def command_animate_projections(args) -> int:
    from .io.project import load_project
    from .visualization.projection_animation import animate_projection_project

    project = load_project(args.project)
    destination = animate_projection_project(
        project,
        output_path=args.output,
        display_mode=args.display_mode,
        trail_ms=args.trail_ms,
        afterimage_ms=args.afterimage_ms,
        fps=args.fps,
        duration_s=args.duration_s,
        hold_end_s=args.hold_end_s,
        width=args.width,
        height=args.height,
        video_quality=args.video_quality,
        custom_title=args.title,
    )
    print(destination)
    return 0


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("files", nargs="*")
    parser.add_argument("--project", type=Path)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--reference-latitude", type=float)
    parser.add_argument("--reference-longitude", type=float)
    parser.add_argument(
        "--reader",
        default="auto",
        help="Reader backend: auto prefers the LMAS-native reader",
    )


def _add_3d_display_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--display-mode",
        choices=("full", "cumulative", "trail", "trail-afterimage"),
        default="cumulative",
    )
    parser.add_argument("--trail-ms", type=float, default=30.0)
    parser.add_argument("--afterimage-ms", type=float, default=30.0)
    parser.add_argument("--point-size", type=float, default=3.0)
    parser.add_argument("--cmap", default="turbo")
    parser.add_argument("--reverse-cmap", action="store_true")
    parser.add_argument("--theme", choices=("dark", "space", "light"), default="dark")
    parser.add_argument("--render-profile", choices=("compatible", "quality"), default="compatible")
    parser.add_argument("--camera", type=Path)
    parser.add_argument("--camera-output", type=Path)
    parser.add_argument(
        "--hide-axes",
        action="store_true",
        help="Hide the 3D base grid and coordinate/cardinal labels",
    )
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=900)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lma", description="Lightning Mapping Array Suite")
    parser.add_argument("--version", action="version", version=f"LMAS {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect", help="Summarize solved LMA source data")
    _add_source_arguments(inspect)
    inspect.set_defaults(func=command_inspect)

    plot = sub.add_parser("plot", help="Create an LMAS projection figure")
    _add_source_arguments(plot)
    plot.add_argument("--layout", choices=("landscape", "portrait", "intfs", "xlma"), default="landscape")
    plot.add_argument("--coordinates", choices=("local", "geodetic"), default="local")
    plot.add_argument("--hist", action="store_true", help="Show the altitude source-fraction histogram in Landscape mode")
    plot.add_argument("--text-size", choices=("normal", "publication", "poster"), default="normal")
    plot.add_argument("--color-by", choices=("time", "altitude", "power", "stations", "chi2", "charge", "group"), default="time")
    plot.add_argument("--cmap", default="turbo")
    plot.add_argument("--theme", choices=("dark", "light", "space"), default="dark")
    plot.add_argument("--point-size", type=float, default=3.0)
    plot.add_argument("--hide-stations", action="store_true")
    plot.add_argument("--hide-colorbar", action="store_true")
    plot.add_argument("--hide-grid", action="store_true")
    plot.add_argument("--show-legend", action="store_true", help="Show a legend for categorical overlays")
    plot.add_argument("--show-panel-labels", action="store_true", help="Add publication-style panel labels")
    plot.add_argument("--reverse-cmap", action="store_true")
    plot.add_argument("--auto-fit-spatial", action=argparse.BooleanOptionalAction, default=True)
    plot.add_argument("--north-south-viewpoint", choices=("south", "north"), default="south")
    plot.add_argument("--east-west-viewpoint", choices=("east", "west"), default="east")
    plot.add_argument("--depth", choices=("spatial", "time"), default="spatial")
    plot.add_argument("--full-record-time-colors", action="store_true")
    plot.add_argument("--title")
    plot.add_argument("--preview-dpi", type=int, default=100)
    plot.add_argument("--dpi", type=int, default=300)
    plot.add_argument("--output", type=Path, required=True)
    _add_filter_arguments(plot)
    plot.set_defaults(func=command_plot)

    project = sub.add_parser("create-project", help="Save an exact data-bound .lmas-project.yaml project")
    project.add_argument("files", nargs="+")
    project.add_argument("--name")
    project.add_argument("--reference-latitude", type=float)
    project.add_argument("--reference-longitude", type=float)
    project.add_argument("--reader", default="auto")
    project.add_argument("--layout", choices=("landscape", "portrait", "intfs", "xlma"), default="landscape")
    project.add_argument("--coordinates", choices=("local", "geodetic"), default="local")
    project.add_argument("--hist", action="store_true")
    project.add_argument("--text-size", choices=("normal", "publication", "poster"), default="normal")
    project.add_argument("--color-by", choices=("time", "altitude", "power", "stations", "chi2", "charge", "group"), default="time")
    project.add_argument("--cmap", default="turbo")
    project.add_argument("--theme", choices=("dark", "light", "space"), default="dark")
    project.add_argument("--point-size", type=float, default=3.0)
    project.add_argument("--hide-colorbar", action="store_true")
    project.add_argument("--hide-grid", action="store_true")
    project.add_argument("--show-legend", action="store_true")
    project.add_argument("--show-panel-labels", action="store_true")
    project.add_argument("--reverse-cmap", action="store_true")
    project.add_argument("--auto-fit-spatial", action=argparse.BooleanOptionalAction, default=True)
    project.add_argument("--north-south-viewpoint", choices=("south", "north"), default="south")
    project.add_argument("--east-west-viewpoint", choices=("east", "west"), default="east")
    project.add_argument("--depth", choices=("spatial", "time"), default="spatial")
    project.add_argument("--full-record-time-colors", action="store_true")
    project.add_argument("--output", type=Path, required=True)
    _add_filter_arguments(project)
    project.set_defaults(func=command_project)

    gui = sub.add_parser("gui", help="Launch the LMAS desktop viewer")
    gui.add_argument("files", nargs="*")
    gui.add_argument("--project", type=Path)
    gui.add_argument("--demo", action="store_true")
    gui.add_argument("--profile", help="Named LMAS analysis profile")
    gui.add_argument("--reader", default="auto")
    gui.set_defaults(func=command_gui)

    export_polarity = sub.add_parser(
        "export-polarity",
        help="Export manual charge assignments as CSV or a complete NetCDF/xarray product",
    )
    export_polarity.add_argument("--project", type=Path, required=True)
    export_polarity.add_argument("--output", type=Path, required=True)
    export_polarity.add_argument("--format", choices=("csv", "netcdf"), default="netcdf")
    export_polarity.add_argument("--scope", choices=EXPORT_SCOPES, default="all")
    export_polarity.add_argument("--reader", default="auto")
    export_polarity.set_defaults(func=command_export_polarity)

    import_polarity = sub.add_parser(
        "import-polarity",
        help="Verify and import an LMAS NetCDF polarity product into a Project",
    )
    import_polarity.add_argument("--project", type=Path, required=True)
    import_polarity.add_argument("--polarity", type=Path, required=True)
    import_polarity.add_argument("--output", type=Path, required=True)
    import_polarity.add_argument("--reader", default="auto")
    import_polarity.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow a scoped polarity product to restore partial group membership",
    )
    import_polarity.set_defaults(func=command_import_polarity)

    readers = sub.add_parser("readers", help="List LMAS reader backends and availability")
    readers.set_defaults(func=command_readers)

    profiles = sub.add_parser("profiles", help="Manage reusable LMAS analysis profiles")
    profile_sub = profiles.add_subparsers(dest="profile_command", required=True)
    profile_sub.add_parser("list", help="List profiles")
    show_profile = profile_sub.add_parser("show", help="Show a profile as JSON")
    show_profile.add_argument("name")
    import_profile = profile_sub.add_parser("import", help="Import a profile")
    import_profile.add_argument("path", type=Path)
    import_profile.add_argument("--overwrite", action="store_true")
    export_profile = profile_sub.add_parser("export", help="Export a profile")
    export_profile.add_argument("name")
    export_profile.add_argument("path", type=Path)
    delete_profile = profile_sub.add_parser("delete", help="Delete a custom profile")
    delete_profile.add_argument("name")
    profiles.set_defaults(func=command_profiles)

    snapshot = sub.add_parser("snapshot-3d", help="Write the current LMA selection as a portable 3D snapshot")
    _add_source_arguments(snapshot)
    snapshot.add_argument("--color-by", choices=("time", "altitude", "power", "stations", "chi2", "charge", "group"), default="time")
    snapshot.add_argument("--cmap", default="turbo")
    snapshot.add_argument("--theme", choices=("dark", "space", "light"), default="dark")
    snapshot.add_argument("--point-size", type=float, default=3.0)
    snapshot.add_argument("--reverse-cmap", action="store_true")
    snapshot.add_argument("--log-color-scale", action="store_true")
    snapshot.add_argument("--title")
    snapshot.add_argument("--output", type=Path, required=True)
    _add_filter_arguments(snapshot)
    snapshot.set_defaults(func=command_snapshot_3d)

    view3d = sub.add_parser("view-3d", help="Open an interactive PyVista viewer for an LMAS 3D snapshot")
    view3d.add_argument("snapshot", type=Path)
    _add_3d_display_arguments(view3d)
    view3d.add_argument("--fps", type=float, default=30.0, help="Interactive playback refresh rate")
    view3d.add_argument("--duration-s", type=float, default=15.0, help="Development playback duration")
    view3d.add_argument(
        "--point-limit",
        type=int,
        default=50_000,
        help="Maximum sources rendered interactively; zero disables the cap",
    )
    view3d.add_argument("--interaction-mode", choices=("z-orbit", "full-3d"), default="z-orbit")
    view3d.add_argument("--play", action="store_true")
    view3d.set_defaults(func=command_view_3d)

    animate3d = sub.add_parser("animate-3d", help="Render an LMAS 3D MP4 or GIF")
    animate3d.add_argument("snapshot", type=Path)
    _add_3d_display_arguments(animate3d)
    animate3d.add_argument("--mode", choices=("orbit", "develop", "develop-orbit"), default="develop")
    animate3d.add_argument("--fps", type=int, default=30)
    animate3d.add_argument("--duration-s", type=float, default=15.0)
    animate3d.add_argument("--hold-end-s", type=float, default=5.0)
    animate3d.add_argument("--orbit-speed-deg-s", type=float, default=14.0)
    animate3d.add_argument("--video-quality", type=int, default=7)
    animate3d.add_argument("--output", type=Path, required=True)
    animate3d.set_defaults(func=command_animate_3d)

    view2d = sub.add_parser(
        "view-projections",
        help="Open interactive 3D source development in linked 2D projections",
    )
    view2d.add_argument("--project", type=Path, required=True)
    view2d.add_argument(
        "--display-mode",
        choices=("cumulative", "trail", "trail-afterimage"),
        default="cumulative",
    )
    view2d.add_argument("--trail-ms", type=float, default=30.0)
    view2d.add_argument("--afterimage-ms", type=float, default=30.0)
    view2d.add_argument("--fps", type=int, default=30)
    view2d.add_argument("--duration-s", type=float, default=15.0)
    view2d.add_argument(
        "--point-limit",
        type=int,
        default=50_000,
        help="Maximum sources rendered interactively; zero disables the cap",
    )
    view2d.set_defaults(func=command_view_projections)

    animate2d = sub.add_parser(
        "animate-projections",
        help="Render 3D source development in the linked 2D projection layout",
    )
    animate2d.add_argument("--project", type=Path, required=True)
    animate2d.add_argument("--output", type=Path, required=True)
    animate2d.add_argument(
        "--display-mode",
        choices=("cumulative", "trail", "trail-afterimage"),
        default="cumulative",
    )
    animate2d.add_argument("--trail-ms", type=float, default=30.0)
    animate2d.add_argument("--afterimage-ms", type=float, default=30.0)
    animate2d.add_argument("--fps", type=int, default=30)
    animate2d.add_argument("--duration-s", type=float, default=15.0)
    animate2d.add_argument("--hold-end-s", type=float, default=5.0)
    animate2d.add_argument("--width", type=int, default=1600)
    animate2d.add_argument("--height", type=int, default=900)
    animate2d.add_argument("--video-quality", type=int, default=8)
    animate2d.add_argument(
        "--title",
        help="Leading title text; live source counts and source time are appended",
    )
    animate2d.set_defaults(func=command_animate_projections)

    batch = sub.add_parser(
        "batch-animations",
        help="Run a queued LMAS projection or 3D animation manifest",
    )
    batch.add_argument("--manifest", type=Path, required=True)
    batch.set_defaults(func=command_batch_animations)

    figure_batch = sub.add_parser(
        "batch-figures",
        help="Run a queued LMAS figure-export manifest",
    )
    figure_batch.add_argument("--manifest", type=Path, required=True)
    figure_batch.set_defaults(func=command_batch_figures)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except LMASError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
