#!/usr/bin/env python3
"""Filter FANTASIA raw lookup outputs by global identity and keep the best donor per query."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read per-accession raw result CSVs, discard donor hits with global identity above "
            "one or more cutoffs, and keep the best remaining donor per query by reliability_index."
        )
    )
    parser.add_argument(
        "input_dir",
        help="Directory containing per-accession raw CSVs, e.g. raw_results/prot-t5/layer_0",
    )
    parser.add_argument(
        "--threshold",
        dest="thresholds",
        type=float,
        action="append",
        required=True,
        help="Global identity cutoff in [0,1]. May be provided multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where filtered reports will be written.",
    )
    return parser.parse_args()


def donor_key(row: dict[str, str]) -> tuple[str, ...]:
    return (
        row.get("protein_id", ""),
        row.get("organism", ""),
        row.get("gene_name", ""),
        row.get("distance", ""),
        row.get("reliability_index", ""),
        row.get("identity", ""),
        row.get("query_len", ""),
        row.get("ref_len", ""),
    )


def load_rows(input_dir: Path) -> tuple[list[dict[str, str]], list[str]]:
    rows: list[dict[str, str]] = []
    header: list[str] | None = None
    for csv_path in sorted(input_dir.glob("*.csv")):
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                continue
            if header is None:
                header = list(reader.fieldnames)
            elif list(reader.fieldnames) != header:
                raise ValueError(f"Header mismatch in {csv_path}")
            for row in reader:
                row = dict(row)
                row["source_file"] = csv_path.name
                rows.append(row)
    if header is None:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")
    return rows, header + ["source_file"]


def parse_float(value: str | None, default: float) -> float:
    if value in (None, ""):
        return default
    return float(value)


def select_best_donors(rows: list[dict[str, str]], threshold: float) -> dict[str, object]:
    donors_by_accession: dict[str, dict[tuple[str, ...], list[dict[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        donors_by_accession[row.get("accession", "")][donor_key(row)].append(row)

    kept_rows: list[dict[str, str]] = []
    filtered_out_rows: list[dict[str, str]] = []
    selected_donors: list[dict[str, str]] = []
    no_survivor_accessions: list[str] = []
    total_donors = 0
    filtered_donors = 0

    for accession, donor_groups in sorted(donors_by_accession.items()):
        surviving: list[tuple[tuple[float, float, str], tuple[str, ...], list[dict[str, str]]]] = []
        for key, donor_rows in donor_groups.items():
            total_donors += 1
            identity = parse_float(donor_rows[0].get("identity"), -1.0)
            if identity > threshold:
                filtered_donors += 1
                filtered_out_rows.extend(donor_rows)
                continue
            ri = parse_float(donor_rows[0].get("reliability_index"), -1.0)
            distance = parse_float(donor_rows[0].get("distance"), float("inf"))
            protein_id = donor_rows[0].get("protein_id", "")
            surviving.append(((-ri, distance, protein_id), key, donor_rows))

        if not surviving:
            no_survivor_accessions.append(accession)
            continue

        _, _, best_rows = min(surviving)
        kept_rows.extend(best_rows)
        exemplar = best_rows[0]
        selected_donors.append(
            {
                "accession": accession,
                "protein_id": exemplar.get("protein_id", ""),
                "organism": exemplar.get("organism", ""),
                "gene_name": exemplar.get("gene_name", ""),
                "distance": exemplar.get("distance", ""),
                "reliability_index": exemplar.get("reliability_index", ""),
                "identity": exemplar.get("identity", ""),
                "query_len": exemplar.get("query_len", ""),
                "ref_len": exemplar.get("ref_len", ""),
                "source_file": exemplar.get("source_file", ""),
                "rows_kept_for_selected_donor": str(len(best_rows)),
            }
        )

    return {
        "kept_rows": kept_rows,
        "filtered_out_rows": filtered_out_rows,
        "selected_donors": selected_donors,
        "no_survivor_accessions": no_survivor_accessions,
        "summary": {
            "identity_threshold": threshold,
            "input_rows": len(rows),
            "kept_rows": len(kept_rows),
            "filtered_out_rows": len(filtered_out_rows),
            "selected_donors": len(selected_donors),
            "accessions_with_no_surviving_donor": len(no_survivor_accessions),
            "input_donors": total_donors,
            "filtered_out_donors": filtered_donors,
        },
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def threshold_label(threshold: float) -> str:
    return f"{threshold:.2f}".replace(".", "p")


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    rows, row_fieldnames = load_rows(input_dir)
    selected_fieldnames = [
        "accession",
        "protein_id",
        "organism",
        "gene_name",
        "distance",
        "reliability_index",
        "identity",
        "query_len",
        "ref_len",
        "source_file",
        "rows_kept_for_selected_donor",
    ]

    for threshold in sorted(set(args.thresholds)):
        results = select_best_donors(rows, threshold)
        label = threshold_label(threshold)
        target_dir = output_dir / f"global_identity_le_{label}"
        write_csv(target_dir / "selected_rows.csv", row_fieldnames, results["kept_rows"])
        write_csv(target_dir / "filtered_out_rows.csv", row_fieldnames, results["filtered_out_rows"])
        write_csv(target_dir / "selected_donors.csv", selected_fieldnames, results["selected_donors"])

        with (target_dir / "accessions_with_no_surviving_donor.txt").open(
            "w", encoding="utf-8"
        ) as handle:
            for accession in results["no_survivor_accessions"]:
                handle.write(f"{accession}\n")

        with (target_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(results["summary"], handle, indent=2, sort_keys=True)

        print(json.dumps(results["summary"], sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
