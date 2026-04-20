#!/usr/bin/env python3
"""Merge per-protein FANTASIA raw result CSVs into a single table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge all CSV files from a FANTASIA raw_results model/layer directory "
            "into a single CSV."
        )
    )
    parser.add_argument(
        "input_dir",
        help="Directory containing per-protein CSV files, e.g. raw_results/prot-t5/layer_0",
    )
    parser.add_argument(
        "-o",
        "--output",
        help=(
            "Output CSV path. Defaults to <input_dir>/../<layer_name>_merged.csv "
            "(for example raw_results/prot-t5/layer_0_merged.csv)."
        ),
    )
    parser.add_argument(
        "--add-source-file",
        action="store_true",
        help="Append a source_file column with the original per-protein CSV filename.",
    )
    return parser.parse_args()


def resolve_output_path(input_dir: Path, output_arg: str | None) -> Path:
    if output_arg:
        return Path(output_arg).expanduser().resolve()
    return input_dir.parent / f"{input_dir.name}_merged.csv"


def merge_csvs(input_dir: Path, output_path: Path, add_source_file: bool) -> tuple[int, int]:
    csv_files = sorted(path for path in input_dir.glob("*.csv") if path.is_file())
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {input_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    expected_header: list[str] | None = None
    rows_written = 0

    with output_path.open("w", newline="", encoding="utf-8") as out_handle:
        writer: csv.writer | None = None

        for csv_path in csv_files:
            with csv_path.open("r", newline="", encoding="utf-8") as in_handle:
                reader = csv.reader(in_handle)
                try:
                    header = next(reader)
                except StopIteration:
                    continue

                if expected_header is None:
                    expected_header = header[:]
                    output_header = expected_header + (["source_file"] if add_source_file else [])
                    writer = csv.writer(out_handle)
                    writer.writerow(output_header)
                elif header != expected_header:
                    raise ValueError(
                        f"Header mismatch in {csv_path}.\n"
                        f"Expected: {expected_header}\n"
                        f"Found:    {header}"
                    )

                assert writer is not None
                for row in reader:
                    if add_source_file:
                        row = row + [csv_path.name]
                    writer.writerow(row)
                    rows_written += 1

    return len(csv_files), rows_written


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.is_dir():
        print(f"Input directory does not exist or is not a directory: {input_dir}", file=sys.stderr)
        return 1

    output_path = resolve_output_path(input_dir, args.output)

    try:
        file_count, row_count = merge_csvs(input_dir, output_path, args.add_source_file)
    except Exception as exc:
        print(f"Merge failed: {exc}", file=sys.stderr)
        return 1

    print(f"Merged {file_count} files into {output_path}")
    print(f"Rows written: {row_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
