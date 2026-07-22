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


def run_application(
    *,
    files: list[Path] | None = None,
    project_path: Path | None = None,
    demo: bool = False,
    profile_name: str | None = None,
    reader_backend: str = "auto",
) -> int:
    try:
        from PySide6.QtCore import QTimer, Qt
        from PySide6.QtWidgets import QApplication, QLabel
    except (ImportError, OSError) as exc:
        raise DependencyError(
            "The LMAS desktop viewer requires PySide6. Install it with "
            "mamba install -c conda-forge pyside6."
        ) from exc

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Lightning Mapping Array Suite")
    app.setOrganizationName("Langmuir Laboratory")
    app.setDesktopFileName("lmas")
    app.setStyle("Fusion")

    # Give immediate feedback before importing and constructing the full GUI.
    # This is deliberately a tiny Qt-only shell; the scientific/plotting stack
    # remains lazy until MainWindow or the first figure actually needs it.
    startup = QLabel("Starting Lightning Mapping Array Suite…")
    startup.setAlignment(Qt.AlignmentFlag.AlignCenter)
    startup.setMinimumSize(420, 120)
    startup.setWindowFlag(Qt.WindowType.SplashScreen, True)
    startup.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
    startup.show()
    startup.raise_()
    startup.activateWindow()
    app.setActiveWindow(startup)
    app.processEvents()

    from .icon import application_icon
    from .main_window import MainWindow
    from .panel_theme import apply_dark_palette

    app.setWindowIcon(application_icon())
    apply_dark_palette(app)
    app.setStyleSheet(_stylesheet())
    window = MainWindow(profile_name=profile_name, reader_backend=reader_backend)
    window.show()
    startup.close()

    # Let Qt paint and activate the application shell before synchronous reader
    # and first-figure work begins. This does not pretend the data are already
    # loaded, but it removes the long blank interval before the window appears.
    def load_initial_content() -> None:
        if project_path is not None:
            window.open_project(Path(project_path))
        elif files:
            window.open_files([Path(path) for path in files])
        elif demo:
            window.open_demo()

    if project_path is not None or files or demo:
        QTimer.singleShot(25, load_initial_content)
    return int(app.exec())


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
