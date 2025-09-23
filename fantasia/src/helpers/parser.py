import argparse
from typing import Optional


# ---------------------------
# Helpers for CLI parsing
# ---------------------------
def _boolish(value: Optional[str]) -> Optional[bool]:
    """
    Parse common truthy/falsy CLI values into booleans.
    Accepted truthy: 1, true, t, yes, y, on
    Accepted falsy : 0, false, f, no, n, off
    """
    if value is None:
        return None
    v = str(value).strip().lower()
    if v in ("1", "true", "t", "yes", "y", "on"):
        return True
    if v in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean (true/false), got: {value!r}")


def build_parser():
    """
    Build the top-level argparse parser with two subcommands:

      • initialize  – prepare the system (download embeddings dump, set up dirs)
      • run         – execute the main pipeline (embedding + lookup, or lookup-only)

    Notes
    -----
    - Model selection and per-model parameters are configured **only** in YAML.
    - The CLI exposes:
        * embedding.device (via --device)
        * redundancy controls (identity/coverage/threads)
        * taxonomy filters
        * lookup limits and housekeeping flags
    """
    parser = argparse.ArgumentParser(
        prog="fantasia",
        description=(
            "FANTASIA: Functional Annotation and Similarity Analysis\n"
            "-------------------------------------------------------\n"
            "Command-line tool for protein functional annotation based on PLM embeddings."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    # ---------------------------------------------------------------------
    # Subcommand: initialize
    # ---------------------------------------------------------------------
    init = subparsers.add_parser(
        "initialize",
        help="Prepare the system: download embeddings dump and set up directories.",
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "Initialize FANTASIA environment:\n"
            " - Read YAML configuration.\n"
            " - Download the reference embeddings dump (if URL provided).\n"
            " - Ensure base directories exist.\n"
            "\n"
            "Use --embeddings_url to override the dataset URL from YAML."
        ),
        epilog=(
            "Examples:\n"
            "  Initialize with default from YAML (Final layer only)\n"
            "  fantasia initialize --config ./fantasia/config.yaml\n"
            "\n"
            "  Explicitly use Final layer only (smaller, faster)\n"
            "  fantasia initialize --config ./fantasia/config.yaml \\\n"
            "    --embeddings_url https://zenodo.org/records/17167843/files/"
            "FANTASIA_UniProt_Sep2025_Last_ExpOnly.dump?download=1\n"
            "\n"
            "  Explicitly use Final + intermediate layers (larger, more detailed)\n"
            "  fantasia initialize --config ./fantasia/config.yaml \\\n"
            "    --embeddings_url https://zenodo.org/records/17151847/files/"
            "FANTASIA_UniProt_Sep2025_Final+Interm_ExpOnly.dump?download=1\n"
        ),

    )

    init.add_argument(
        "--config",
        type=str,
        default="./fantasia/config.yaml",
        help="Path to the YAML configuration file. Default: './fantasia/config.yaml'.",
    )
    init.add_argument(
        "--embeddings_url",
        type=str,
        help="Override the embeddings dump URL (otherwise taken from YAML).",
    )
    init.add_argument(
        "--base_directory",
        type=str,
        help="Base directory used to place the embeddings/experiments folders.",
    )
    init.add_argument(
        "--log_path",
        type=str,
        help="Path or directory for logs. If a directory, a 'Logs_<timestamp>.log' file is created inside.",
    )

    # ---------------------------------------------------------------------
    # Subcommand: run
    # ---------------------------------------------------------------------
    run = subparsers.add_parser(
        "run",
        help="Execute the pipeline to process sequences, generate embeddings, and run lookups.",
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "Run FANTASIA pipeline:\n"
            " - Load configuration (YAML).\n"
            " - Process protein sequences from FASTA.\n"
            " - Generate embeddings using models defined in YAML.\n"
            " - Store embeddings (HDF5) and perform similarity-based lookups.\n"
            "\n"
            "NOTE: Model selection and per-model settings are configured ONLY in YAML."
        ),
        epilog=(
            "Example:\n"
            "  fantasia run \\\n"
            "    --config ./fantasia/config.yaml \\\n"
            "    --input ./data/proteins.fasta \\\n"
            "    --prefix demo_run \\\n"
            "    --base_directory ~/fantasia \\\n"
            "    --device cuda \\\n"
            "    --redundancy_identity 1.0 \\\n"
            "    --redundancy_coverage 0.7 \\\n"
            "    --threads 8 \\\n"
            "    --taxonomy_ids_to_exclude 559292,6239 \\\n"
            "    --get_descendants true \\\n"
            "    --log_path ~/fantasia/logs\n"
            "\n"
            "Lookup-only:\n"
            "  fantasia run \\\n"
            "    --config ./fantasia/config.yaml \\\n"
            "    --input /path/to/embeddings.h5 \\\n"
            "    --prefix lookup_only \\\n"
            "    --base_directory ~/fantasia \\\n"
            "    --only_lookup true \\\n"
            "    --device cpu \\\n"
            "    --redundancy_identity 1.0 \\\n"
            "    --redundancy_coverage 0.7"
        ),
    )

    # General I/O
    run.add_argument(
        "--config",
        type=str,
        default="./fantasia/config.yaml",
        help="Path to the YAML configuration file. Default: './fantasia/config.yaml'.",
    )
    run.add_argument(
        "--input",
        type=str,
        help="Path to the input FASTA (embedding stage) or an embeddings HDF5 when --only_lookup=true.",
    )
    run.add_argument("--prefix", type=str, help="Prefix used to name output artifacts.")
    run.add_argument(
        "--base_directory",
        type=str,
        help="Base directory where experiments and embeddings will be stored.",
    )
    run.add_argument(
        "--log_path",
        type=str,
        help="Path or directory for logs. If a directory, a 'Logs_<timestamp>.log' file is created inside.",
    )

    # Execution control
    run.add_argument("--limit_execution", type=int, help="Limit the number of sequences processed.")
    run.add_argument("--monitor_interval", type=int, help="Seconds between progress logs.")
    run.add_argument(
        "--only_lookup",
        type=_boolish,
        help="If true, skip the embedding stage and use --input as an embeddings HDF5.",
    )

    # Embedding / device (models are YAML-only)
    run.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        help="Primary device for PLMs (maps to embedding.device).",
    )
    run.add_argument(
        "--length_filter",
        type=int,
        help="Maximum accepted sequence length (maps to embedding.max_sequence_length).",
    )
    run.add_argument(
        "--sequence_queue_package",
        type=int,
        help="Queue batch size for sequence processing (maps to embedding.queue_batch_size).",
    )

    # Lookup / search
    run.add_argument(
        "--limit_per_entry",
        type=int,
        help="Maximum number of neighbors kept per query (maps to lookup.limit_per_entry).",
    )

    # Redundancy filter (enable + thresholds) with backward-compatible aliases
    # identity   -> dest='redundancy_filter'
    # coverage   -> dest='alignment_coverage'
    run.add_argument(
        "--redundancy_identity",
        "--redundancy_filter",
        dest="redundancy_filter",
        type=float,
        help="Sequence identity threshold in [0,1]. 0 disables; 1.0 = strict deduplication.",
    )
    run.add_argument(
        "--redundancy_coverage",
        "--alignment_coverage",
        dest="alignment_coverage",
        type=float,
        help="Alignment coverage threshold in (0,1]. Used for MMseqs2-based redundancy clustering.",
    )
    run.add_argument("--threads", type=int, help="Threads for redundancy clustering (MMseqs2).")

    # Taxonomy filters
    run.add_argument(
        "--taxonomy_ids_to_exclude",
        type=str,
        help="List of taxonomy IDs to exclude (comma/space separated).",
    )
    run.add_argument(
        "--taxonomy_ids_included_exclusively",
        type=str,
        help="List of taxonomy IDs to include exclusively (comma/space separated).",
    )
    run.add_argument(
        "--get_descendants",
        type=_boolish,
        help="If true, expand taxonomy filtering to include descendants.",
    )

    return parser
