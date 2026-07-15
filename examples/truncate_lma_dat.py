#!/usr/bin/env python3
"""Create a time-limited .dat or .dat.gz LMA source file without altering rows."""
from __future__ import annotations

import argparse
import gzip
from pathlib import Path


def _open_text(path: Path, mode: str):
    return gzip.open(path, mode, encoding="utf-8", newline="") if path.suffix.lower() == ".gz" else path.open(mode, encoding="utf-8", newline="")


def truncate(source: Path, destination: Path, start_sod: float, end_sod: float, *, start_header: str, duration_s: float) -> int:
    if end_sod <= start_sod:
        raise ValueError("end_sod must be later than start_sod")
    with _open_text(source, "rt") as stream:
        lines = stream.readlines()
    try:
        marker = next(i for i, line in enumerate(lines) if line.strip() == "*** data ***")
    except StopIteration as exc:
        raise RuntimeError(f"{source} has no '*** data ***' marker") from exc
    retained: list[str] = []
    for line in lines[marker + 1 :]:
        if not line.strip():
            continue
        try:
            time_sod = float(line.split(None, 1)[0])
        except (ValueError, IndexError):
            continue
        if start_sod <= time_sod < end_sod:
            retained.append(line if line.endswith("\n") else line + "\n")
    header: list[str] = []
    for line in lines[:marker]:
        if line.startswith("Data start time:"):
            header.append(f"Data start time: {start_header}\n")
        elif line.startswith("Number of seconds analyzed:"):
            header.append(f"Number of seconds analyzed: {duration_s:g}\n")
        elif line.startswith("Number of events:"):
            header.append(f"Number of events: {len(retained)}\n")
            header.append(f"LMAS subset parent: {source.name}\n")
            header.append("LMAS subset note: source rows are unchanged; station summary lines are retained from the parent file.\n")
        else:
            header.append(line)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _open_text(destination, "wt") as stream:
        stream.writelines(header)
        stream.write("*** data ***\n")
        stream.writelines(retained)
    return len(retained)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--start-sod", type=float, required=True, help="UTC seconds of day, inclusive")
    parser.add_argument("--end-sod", type=float, required=True, help="UTC seconds of day, exclusive")
    parser.add_argument("--start-header", required=True, help="Header text, e.g. '04/30/19 14:48:44'")
    args = parser.parse_args()
    count = truncate(args.source, args.destination, args.start_sod, args.end_sod, start_header=args.start_header, duration_s=args.end_sod-args.start_sod)
    print(f"Wrote {count:,} unchanged source rows to {args.destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
