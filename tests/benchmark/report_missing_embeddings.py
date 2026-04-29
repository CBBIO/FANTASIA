#!/usr/bin/env python
"""Report input FASTA proteins that are absent from a FANTASIA embeddings.h5."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py


def fasta_ids(path: Path) -> list[str]:
    ids: list[str] = []
    with path.open() as handle:
        for line in handle:
            if line.startswith(">"):
                ids.append(line[1:].strip().split()[0].replace("|", "_"))
    return ids


def embedded_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with h5py.File(path, "r") as h5:
        return {
            key.removeprefix("accession_")
            for key in h5.keys()
            if key.startswith("accession_")
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare input FASTA accessions with accessions present in embeddings.h5."
    )
    parser.add_argument("--input", required=True, help="Input FASTA used for the run.")
    parser.add_argument("--embeddings", required=True, help="Path to embeddings.h5.")
    parser.add_argument("--output", required=True, help="TSV report path.")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    embeddings_path = Path(args.embeddings).expanduser()
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    query_ids = fasta_ids(input_path)
    present = embedded_ids(embeddings_path)

    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["accession", "embedded"])
        for accession in query_ids:
            writer.writerow([accession, "yes" if accession in present else "no"])

    missing = sum(1 for accession in query_ids if accession not in present)
    print(
        f"input={len(query_ids)} embedded={len(present)} missing={missing} report={output_path}"
    )


if __name__ == "__main__":
    main()
