from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "src" / "lmas" / "gui"


def test_file_menu_has_general_product_export():
    source = (ROOT / "main_window.py").read_text(encoding="utf-8")
    assert 'QAction("Export Product…", self)' in source
    assert "ExportProductDialog" in source
    assert "open_export_product" in source


def test_view_menu_has_searchable_header_viewer():
    main_source = (ROOT / "main_window.py").read_text(encoding="utf-8")
    header_source = (ROOT / "data_header_window.py").read_text(encoding="utf-8")
    assert 'QAction("Data File Header…", self)' in main_source
    assert "self.text.setReadOnly(True)" in header_source
    assert "QKeySequence.StandardKey.Find" in header_source
    assert "Copy all" in header_source
    assert "Save as text" in header_source


def test_charge_window_requests_full_main_height_once():
    source = (ROOT / "main_window.py").read_text(encoding="utf-8")
    assert "full_height_tool_geometry" in source
    assert 'window.property("lmasChargeHeightInitialized")' in source
    assert "QTimer.singleShot" in source


def test_satellite_window_has_direct_file_or_directory_path_entry():
    source = (ROOT / "satellite_overlay_window.py").read_text(encoding="utf-8")
    assert 'QLabel("File or directory")' in source
    assert 'self.path_edit = QLineEdit()' in source
    assert 'self.path_edit.returnPressed.connect(self._load_entered_path)' in source
    assert 'def _glm_paths_from_entry' in source
    assert 'def _load_entered_path' in source


def test_network_peak_current_scaling_defaults_on():
    source = (ROOT.parent / "overlays" / "network" / "manager.py").read_text(encoding="utf-8")
    assert 'scale_by_peak_current: bool = True' in source
