#!/usr/bin/env python3
"""Install the bundled LMAS wheel from a release kit."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import sys

_WHEEL_PATTERN = re.compile(r"^lmas-(?P<version>[^-]+)-py3-none-any\.whl$")


def run(command: list[str], *, dry_run: bool = False) -> None:
    print("+", " ".join(f'"{item}"' if " " in item else item for item in command))
    if not dry_run:
        subprocess.run(command, check=True)


def pip_install(arguments: list[str], *, user: bool, dry_run: bool) -> None:
    command = [sys.executable, "-m", "pip", "install"]
    if user:
        command.append("--user")
    command.extend(arguments)
    run(command, dry_run=dry_run)


def find_bundled_wheel(root: Path) -> tuple[Path, str]:
    wheel_dir = root / "wheels"
    candidates = sorted(wheel_dir.glob("lmas-*-py3-none-any.whl"))
    if not candidates:
        raise FileNotFoundError(
            f"No bundled LMAS wheel was found in: {wheel_dir}"
        )
    if len(candidates) != 1:
        names = ", ".join(path.name for path in candidates)
        raise RuntimeError(
            "Expected exactly one bundled LMAS wheel in "
            f"{wheel_dir}, but found {len(candidates)}: {names}"
        )

    wheel = candidates[0]
    match = _WHEEL_PATTERN.match(wheel.name)
    if match is None:
        raise RuntimeError(f"Could not determine the LMAS version from: {wheel.name}")
    return wheel, match.group("version")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Install the bundled LMAS release into the active Python environment."
    )
    parser.add_argument("--core-only", action="store_true", help="Install core CLI/static plotting dependencies only")
    parser.add_argument("--no-3d", action="store_true", help="Install the GUI but omit PyVista/VTK and animation dependencies")
    parser.add_argument("--no-deps", action="store_true", help="Install only the bundled LMAS wheel")
    parser.add_argument("--no-launcher", action="store_true", help="Do not create the Windows, macOS, or Linux launcher")
    parser.add_argument("--no-desktop", action="store_true", help="Skip the Desktop copy/link and install only to Start, Applications, or the application menu")
    parser.add_argument("--user", action="store_true", help="Pass --user to pip")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    return parser


def main() -> int:
    args = make_parser().parse_args()
    if sys.version_info < (3, 11):
        print("ERROR: This LMAS release requires Python 3.11 or newer.", file=sys.stderr)
        return 2

    root = Path(__file__).resolve().parent
    try:
        wheel, version = find_bundled_wheel(root)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"Installing LMAS {version} with: {sys.executable}")
    if not args.no_deps:
        requirements = root / "requirements" / (
            "core.txt" if args.core_only else "gui.txt" if args.no_3d else "full.txt"
        )
        pip_install(["-r", str(requirements)], user=args.user, dry_run=args.dry_run)

    pip_install(
        ["--force-reinstall", "--no-deps", str(wheel)],
        user=args.user,
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        run([sys.executable, "-c", "from importlib.metadata import version; print('LMAS', version('lmas'))"])

    if not args.no_launcher and not args.core_only:
        launcher = [sys.executable, "-m", "lmas.launcher"]
        if args.no_desktop:
            launcher.append("--no-desktop")
        try:
            run(launcher, dry_run=args.dry_run)
        except subprocess.CalledProcessError as exc:
            print(f"WARNING: LMAS installed, but launcher creation failed: {exc}", file=sys.stderr)

    print("\nInstallation complete.\nRun: lma --version")
    if not args.core_only:
        print("Launch separately with: lma gui")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
