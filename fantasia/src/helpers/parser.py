import argparse


def build_parser():
    parser = argparse.ArgumentParser(
        prog="fantasia",
        description=(
            """
FANTASIA: Functional Annotation and Similarity Analysis
-------------------------------------------------------
FANTASIA is a command-line tool for computing vector similarity and generating
functional annotations using pre-trained language models (PLMs). It supports:
  • ProtT5
  • ProstT5
  • ESM2

This system processes protein sequences by embedding them with these models,
storing the embeddings into an h5 Object, and performing efficient similarity searches over a vector database.

Pre-configured with UniProt 2024 data, FANTASIA integrates with an information system
for seamless data management. Protein data and Gene Ontology annotations (GOA) are
kept up to date, while proteins from the 2022 dataset remain for benchmarking (e.g., CAFA).

Reference Datasets:
--------------------
FANTASIA allows selecting between two datasets for reference annotation during the lookup stage:

1. **Default: UniProt 2024 + GOA full dump**
   - File URL: https://zenodo.org/records/14864851/files/uniprot_embeddings_goa_2024_with_2022_tags.tar?download=1
   - Zenodo page: https://zenodo.org/records/14864851

2. **Benchmark mode: GoPredSim 2022 filtered dataset**
   - File URL: https://zenodo.org/records/15095845/files/embeddings_fantasia_gopredsim_2022.dump?download=1
   - Zenodo page: https://zenodo.org/records/15095845

Use the `--embeddings_url` flag in the `initialize` command to override the default.

Requirements:
  • Relational Database with Vectors: PostgreSQL/PGVector (for storing annotations, embeddings and metadata)
  • Task Queue: RabbitMQ (for parallel task execution)
For setup instructions, refer to the documentation.
            """
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Subparser: initialize
    init_parser = subparsers.add_parser(
        "initialize",
        help="Set up the database and download the embeddings.",
        description=(
            """
FANTASIA: Functional Annotation and Similarity Analysis
-------------------------------------------------------
The 'initialize' command prepares the system for operation by:
  • Reading the configuration file.
  • Downloading the embeddings database (if specified).
  • Setting up the necessary directories.

By default, the configuration is loaded from './fantasia/config.yaml'.
You can select between UniProt 2024 full GOA dump or a filtered benchmark version.
            """
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )

    init_parser.add_argument(
        "--config", type=str, default="./fantasia/config.yaml",
        help="Path to the configuration file (YAML format). Default: './fantasia/config.yaml'."
    )

    init_parser.add_argument(
        "--embeddings_url", type=str,
        help="URL to download the embeddings database dump. If not provided, uses value from config."
    )

    init_parser.epilog = (
        "Examples:\n"
        "  # Default (UniProt 2024 + GOA full mirror)\n"
        "  python fantasia/main.py initialize --config my_config.yaml\n\n"
        "  # Benchmark mode (GoPredSim 2022 subset)\n"
        "  python fantasia/main.py initialize \\\n"
        "     --embeddings_url https://zenodo.org/records/15095845/files/embeddings_fantasia_gopredsim_2022.dump?download=1\n\n"
        "  # Explore Zenodo dataset documentation:\n"
        "  UniProt+GOA: https://zenodo.org/records/14864851\n"
        "  GoPredSim2022: https://zenodo.org/records/15095845"
    )

    # Subparser: run
    run_parser = subparsers.add_parser(
        "run",
        help="Execute the pipeline to process sequences, generate embeddings, and manage lookups.",
        description=(
            """
FANTASIA: Functional Annotation and Similarity Analysis
-------------------------------------------------------
The 'run' command executes the main pipeline, which includes:
  • Loading the configuration file.
  • Processing protein sequences from a FASTA file.
  • Generating sequence embeddings using selected models.
  • Storing embeddings in h5 file as input for similarity search through vectorial DB.
  • Running functional annotation lookups based on the embeddings.

By default, the configuration is loaded from './fantasia/config.yaml'.
Supported models include ProtT5, ProstT5, and ESM2.
            """
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )

    run_parser.add_argument("--config", type=str, default="./fantasia/config.yaml",
                            help="Path to the YAML configuration file. Default: './fantasia/config.yaml'.")
    run_parser.add_argument("--input", type=str, help="Path to the input FASTA file containing protein sequences.")
    run_parser.add_argument("--prefix", type=str, help="Prefix used to name the output files.")
    run_parser.add_argument("--base_directory", type=str,
                            help="Base directory where all results and embeddings will be stored.")
    run_parser.add_argument("--length_filter", type=int,
                            help="Filter sequences by length. Sequences longer than this will be ignored.")
    run_parser.add_argument("--redundancy_filter", type=float,
                            help="Apply redundancy filtering (e.g., 0.8 removes sequences >80%% identity).")
    run_parser.add_argument("--max_workers", type=int,
                            help="Number of parallel workers to process sequences. Affects only the database lookup stage.")
    run_parser.add_argument("--models", type=str,
                            help="Comma-separated list of embedding models to enable. Example: 'esm,prot'.")
    run_parser.add_argument("--distance_threshold", type=str,
                            help="Comma-separated model:threshold pairs. Example: 'esm:0.4,prot:0.6'.")
    run_parser.add_argument("--batch_size", type=str,
                            help="Comma-separated model:batch_size pairs. Example: 'esm:32,prot:64'.")
    run_parser.add_argument("--sequence_queue_package", type=int, help="Number of sequences to queue per batch.")
    run_parser.add_argument("--limit_per_entry", type=int,
                            help="Limit the number of retrieved reference entries per query.")
    run_parser.add_argument("--device", type=str, choices=["cuda", "cpu"], help="Device to use for embedding models.")
    run_parser.add_argument("--log_path", type=str, help="Path to the log file.")

    run_parser.epilog = (
        "Example usage:\n"
        "  python fantasia/main.py run \\\n"
        "     --config ./fantasia/config.yaml \\\n"
        "     --input ./data_sample/worm_test.fasta \\\n"
        "     --prefix test_run \\\n"
        "     --length_filter 300 \\\n"
        "     --redundancy_filter 0.8 \\\n"
        "     --max_workers 1 \\\n"
        "     --models esm,prot \\\n"
        "     --distance_threshold esm:0.4,prot:0.6 \\\n"
        "     --batch_size esm:32,prot:64 \\\n"
        "     --sequence_queue_package 100 \\\n"
        "     --limit_per_entry 5 \\\n"
        "     --device cuda \\\n"
        "     --log_path ~/fantasia/fantasia.log"
    )

    return parser
