from __future__ import annotations

from pathlib import Path
import plistlib
import sys

from lmas.launcher import install_linux_launcher, install_macos_launcher


def test_macos_launcher_bundle(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    fake_python = tmp_path / "env" / "bin" / "python"
    fake_python.parent.mkdir(parents=True)
    fake_python.write_text("", encoding="utf-8")
    monkeypatch.setattr(sys, "executable", str(fake_python))
    (tmp_path / "Desktop").mkdir()

    outputs = install_macos_launcher(desktop=True)
    app = tmp_path / "Applications" / "Lightning Mapping Array Suite.app"
    executable = app / "Contents" / "MacOS" / "LMAS"
    plist_path = app / "Contents" / "Info.plist"
    icon = app / "Contents" / "Resources" / "LMAS.icns"

    assert app in outputs
    assert executable.is_file()
    assert executable.stat().st_mode & 0o111
    assert str(fake_python.resolve()) in executable.read_text(encoding="utf-8")
    assert icon.stat().st_size > 1000
    with plist_path.open("rb") as stream:
        info = plistlib.load(stream)
    assert info["CFBundleIdentifier"] == "org.langmuir.lmas"
    assert info["CFBundleExecutable"] == "LMAS"
    assert info["CFBundleDisplayName"] == "Lightning Mapping Array Suite"
    assert info["CFBundleShortVersionString"] == "1.6.2"
    assert (tmp_path / "Desktop" / "Lightning Mapping Array Suite.app").is_symlink()


def test_linux_launcher_files(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    fake_python = tmp_path / "env with space" / "bin" / "python"
    fake_python.parent.mkdir(parents=True)
    fake_python.write_text("", encoding="utf-8")
    monkeypatch.setattr(sys, "executable", str(fake_python))
    (tmp_path / "Desktop").mkdir()

    outputs = install_linux_launcher(desktop=True)
    desktop_entry = tmp_path / ".local/share/applications/lmas.desktop"
    text = desktop_entry.read_text(encoding="utf-8")
    assert desktop_entry in outputs
    assert 'Exec="' in text
    assert " -m lmas.gui" in text
    assert "Icon=lmas" in text
    assert "Name=Lightning Mapping Array Suite" in text
    assert (tmp_path / "Desktop" / "Lightning Mapping Array Suite.desktop").is_file()


def test_windows_launcher_shortcuts(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import lmas.launcher as launcher

    fake_python = tmp_path / "env" / "python.exe"
    fake_python.parent.mkdir(parents=True)
    fake_python.write_text("", encoding="utf-8")
    fake_pythonw = fake_python.with_name("pythonw.exe")
    fake_pythonw.write_text("", encoding="utf-8")
    monkeypatch.setattr(sys, "executable", str(fake_python))
    monkeypatch.setenv("APPDATA", str(tmp_path / "AppData" / "Roaming"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "User"))

    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    outputs = launcher.install_windows_launcher(desktop=True)

    assert len(outputs) == 2
    assert len(commands) == 2
    assert all(command[0] == "powershell.exe" for command in commands)
    assert all("-m lmas.gui" in command[-1] for command in commands)
    assert all(str(fake_pythonw.resolve()) in command[-1] for command in commands)
