from __future__ import annotations

import argparse
from importlib.resources import as_file, files
import os
from pathlib import Path
import plistlib
import shlex
import shutil
import subprocess
import sys

from . import __version__


class LauncherError(RuntimeError):
    pass


def _resource(name: str) -> Path:
    resource = files("lmas.resources").joinpath(name)
    context = as_file(resource)
    path = context.__enter__()
    # Package resources are ordinary files for wheel installs. Keep the
    # context alive for the duration of this process.
    _RESOURCE_CONTEXTS.append(context)
    return Path(path)


_RESOURCE_CONTEXTS: list[object] = []


def _quoted_desktop_exec(path: Path) -> str:
    value = str(path).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{value}"'


def install_linux_launcher(*, desktop: bool = True) -> list[Path]:
    home = Path.home()
    icon_target = home / ".local/share/icons/hicolor/256x256/apps/lmas.png"
    app_target = home / ".local/share/applications/lmas.desktop"
    icon_target.parent.mkdir(parents=True, exist_ok=True)
    app_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(_resource("lmas_bolt_256.png"), icon_target)
    executable = Path(sys.executable).resolve()
    content = (
        "[Desktop Entry]\n"
        "Version=1.0\n"
        "Type=Application\n"
        "Name=Lightning Mapping Array Suite\n"
        "GenericName=LMAS\n"
        "Comment=View and analyze solved LMA source data\n"
        f"Exec={_quoted_desktop_exec(executable)} -m lmas.gui\n"
        "Icon=lmas\n"
        "Terminal=false\n"
        "Categories=Science;Education;DataVisualization;\n"
        "StartupNotify=true\n"
        "StartupWMClass=lmas\n"
    )
    app_target.write_text(content, encoding="utf-8")
    app_target.chmod(0o755)
    outputs = [app_target, icon_target]
    if desktop:
        desktop_dir = home / "Desktop"
        if desktop_dir.exists():
            desktop_target = desktop_dir / "Lightning Mapping Array Suite.desktop"
            shutil.copy2(app_target, desktop_target)
            desktop_target.chmod(0o755)
            outputs.append(desktop_target)
    return outputs


def install_windows_launcher(*, desktop: bool = True) -> list[Path]:
    home = Path.home()
    python = Path(sys.executable).resolve()
    pythonw = python.with_name("pythonw.exe")
    target = pythonw if pythonw.exists() else python
    icon = _resource("lmas_bolt.ico").resolve()
    appdata = Path(os.environ.get("APPDATA", home / "AppData/Roaming"))
    start_menu = appdata / "Microsoft/Windows/Start Menu/Programs"
    start_menu.mkdir(parents=True, exist_ok=True)
    destinations = [start_menu / "Lightning Mapping Array Suite.lnk"]
    if desktop:
        desktop_dir = Path(os.environ.get("USERPROFILE", str(home))) / "Desktop"
        desktop_dir.mkdir(parents=True, exist_ok=True)
        destinations.append(desktop_dir / "Lightning Mapping Array Suite.lnk")
    for destination in destinations:
        escaped_destination = str(destination).replace("'", "''")
        escaped_target = str(target).replace("'", "''")
        escaped_home = str(home).replace("'", "''")
        escaped_icon = str(icon).replace("'", "''")
        script = (
            "$ws = New-Object -ComObject WScript.Shell; "
            f"$s = $ws.CreateShortcut('{escaped_destination}'); "
            f"$s.TargetPath = '{escaped_target}'; "
            "$s.Arguments = '-m lmas.gui'; "
            f"$s.WorkingDirectory = '{escaped_home}'; "
            f"$s.IconLocation = '{escaped_icon}'; "
            "$s.Description = 'Lightning Mapping Array Suite'; "
            "$s.Save()"
        )
        completed = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise LauncherError(completed.stderr.strip() or "PowerShell could not create the shortcut")
    return destinations


def install_macos_launcher(*, desktop: bool = True) -> list[Path]:
    home = Path.home()
    app_target = home / "Applications" / "Lightning Mapping Array Suite.app"
    contents = app_target / "Contents"
    macos_dir = contents / "MacOS"
    resources_dir = contents / "Resources"

    if app_target.is_symlink() or app_target.is_file():
        app_target.unlink()
    elif app_target.exists():
        shutil.rmtree(app_target)

    macos_dir.mkdir(parents=True, exist_ok=True)
    resources_dir.mkdir(parents=True, exist_ok=True)

    executable = Path(sys.executable).resolve()
    launcher_target = macos_dir / "LMAS"
    log_dir = home / "Library" / "Logs" / "LMAS"
    launcher_target.write_text(
        "#!/bin/sh\n"
        f"mkdir -p {shlex.quote(str(log_dir))}\n"
        f"exec {shlex.quote(str(executable))} -m lmas.gui "
        f">>{shlex.quote(str(log_dir / 'launcher.log'))} 2>&1\n",
        encoding="utf-8",
    )
    launcher_target.chmod(0o755)

    icon_target = resources_dir / "LMAS.icns"
    shutil.copy2(_resource("lmas_bolt.icns"), icon_target)

    info = {
        "CFBundleDevelopmentRegion": "en",
        "CFBundleDisplayName": "Lightning Mapping Array Suite",
        "CFBundleExecutable": "LMAS",
        "CFBundleIconFile": "LMAS.icns",
        "CFBundleIdentifier": "org.langmuir.lmas",
        "CFBundleInfoDictionaryVersion": "6.0",
        "CFBundleName": "Lightning Mapping Array Suite",
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": __version__,
        "CFBundleVersion": __version__,
        "LSApplicationCategoryType": "public.app-category.education",
        "NSHighResolutionCapable": True,
    }
    with (contents / "Info.plist").open("wb") as stream:
        plistlib.dump(info, stream, sort_keys=True)

    outputs = [app_target, icon_target]
    if desktop:
        desktop_dir = home / "Desktop"
        if desktop_dir.exists():
            desktop_target = desktop_dir / "Lightning Mapping Array Suite.app"
            if desktop_target.is_symlink():
                desktop_target.unlink()
            if not desktop_target.exists():
                desktop_target.symlink_to(app_target, target_is_directory=True)
                outputs.append(desktop_target)
    return outputs


def install_launcher(*, desktop: bool = True) -> list[Path]:
    if sys.platform.startswith("win"):
        return install_windows_launcher(desktop=desktop)
    if sys.platform == "darwin":
        return install_macos_launcher(desktop=desktop)
    if sys.platform.startswith("linux"):
        return install_linux_launcher(desktop=desktop)
    raise LauncherError("LMAS launcher installation supports Windows, macOS, and Linux")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lmas-install-launcher",
        description="Install a clickable LMAS application launcher",
    )
    parser.add_argument(
        "--no-desktop",
        action="store_true",
        help="Install only the Start menu, Applications folder, or application-menu launcher",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        outputs = install_launcher(desktop=not args.no_desktop)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    for path in outputs:
        print(path)
    if sys.platform.startswith("win"):
        print("You can pin the LMAS Start-menu shortcut to the taskbar normally.")
    elif sys.platform == "darwin":
        print("LMAS.app was installed in your user Applications folder and can be added to the Dock.")
    elif sys.platform.startswith("linux"):
        print("LMAS should now appear in your desktop environment's application menu.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
