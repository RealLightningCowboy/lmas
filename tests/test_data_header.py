from __future__ import annotations

from io import BytesIO
import gzip
from pathlib import Path
from types import SimpleNamespace
import tarfile

import xarray as xr

from lmas.data_header import (
    data_header_documents,
    read_archive_header_documents,
    read_dat_header_text,
)


HEADER = """Analysis program: test-lma\nLocation: Test Network\nData start time: 07/06/26 21:18:38\nData: time, lat, lon\n"""
DATA = "*** data ***\n1.0 33.0 -107.0\n"


def test_read_literal_dat_and_gzip_headers(tmp_path: Path):
    dat = tmp_path / "sample.dat"
    dat.write_text(HEADER + DATA, encoding="utf-8")
    compressed = tmp_path / "sample.dat.gz"
    with gzip.open(compressed, "wt", encoding="utf-8") as stream:
        stream.write(HEADER + DATA)

    assert read_dat_header_text(dat) == HEADER
    assert read_dat_header_text(compressed) == HEADER
    assert "*** data ***" not in read_dat_header_text(dat)


def test_read_dat_gzip_member_from_archive(tmp_path: Path):
    archive_path = tmp_path / "bundle.tar.gz"
    member_bytes = gzip.compress((HEADER + DATA).encode("utf-8"))
    with tarfile.open(archive_path, "w:gz") as archive:
        info = tarfile.TarInfo("nested/sample.dat.gz")
        info.size = len(member_bytes)
        archive.addfile(info, BytesIO(member_bytes))

    documents = read_archive_header_documents(archive_path)
    assert len(documents) == 1
    assert documents[0].kind == "dat-header"
    assert documents[0].text == HEADER
    assert "nested/sample.dat.gz" in documents[0].title


def test_header_documents_always_include_metadata_summary(tmp_path: Path):
    dat = tmp_path / "sample.dat"
    dat.write_text(HEADER + DATA, encoding="utf-8")
    project = SimpleNamespace(
        name="Header test",
        source_files=(dat,),
        reader_backend="native",
        reader_backend_version="1",
        reader_details={"mode": "test"},
        dataset=xr.Dataset(
            {"event_time": ("number_of_events", [0, 1])},
            attrs={"network_name": "Test Network"},
        ),
    )

    documents = data_header_documents(project)
    assert documents[0].kind == "dat-header"
    assert documents[-1].kind == "metadata-summary"
    assert "Test Network" in documents[-1].text
    assert "number_of_events: 2" in documents[-1].text
