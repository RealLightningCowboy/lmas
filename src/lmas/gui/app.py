from __future__ import annotations

import argparse
from pathlib import Path
import sys

from ..errors import DependencyError


def _stylesheet() -> str:
    return """
    QMainWindow { background: palette(window); }
    QGroupBox { font-weight: 600; margin-top: 10px; }
    QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
    QPushButton { padding: 6px 10px; }
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { padding: 3px; }
    QDockWidget::title { padding: 6px; font-weight: 600; }
    QWidget#lmaFileBrowserCollapsedStrip { border-right: 1px solid palette(mid); }
    """


def run_application(*, files: list[Path] | None = None, project_path: Path | None = None, demo: bool = False, profile_name: str | None = None, reader_backend: str = "auto") -> int:
    try:
        from PySide6.QtWidgets import QApplication
    except (ImportError, OSError) as exc:
        raise DependencyError(
            "The LMAS desktop viewer requires PySide6. Install it with "
            "mamba install -c conda-forge pyside6."
        ) from exc
    from .main_window import MainWindow
    from .panel_theme import apply_dark_palette
    from .icon import application_icon

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Lightning Mapping Array Suite")
    app.setOrganizationName("Langmuir Laboratory")
    app.setDesktopFileName("lmas")
    app.setWindowIcon(application_icon())
    app.setStyle("Fusion")
    apply_dark_palette(app)
    app.setStyleSheet(_stylesheet())
    window = MainWindow(profile_name=profile_name, reader_backend=reader_backend)
    window.show()
    if project_path is not None:
        window.open_project(Path(project_path))
    elif files:
        window.open_files([Path(path) for path in files])
    elif demo:
        window.open_demo()
    return app.exec()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lmas-gui", description="Launch the LMAS desktop viewer")
    parser.add_argument("files", nargs="*", type=Path)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--profile")
    parser.add_argument("--reader", default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run_application(files=args.files, project_path=args.project, demo=args.demo, profile_name=args.profile, reader_backend=args.reader)
    except DependencyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
