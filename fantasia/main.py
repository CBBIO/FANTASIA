import os
import sys
from pprint import pprint
import yaml
import argparse
from datetime import datetime
from fantasia.src.helpers import download_embeddings, load_dump_to_db
from fantasia.src.embedder import SequenceEmbedder
from fantasia.src.lookup import EmbeddingLookUp
from protein_metamorphisms_is.helpers.config.yaml import read_yaml_config

import protein_metamorphisms_is.sql.model.model  # noqa: F401


def initialize(config_path, embeddings_url=None):
    """
    Initializes the system by downloading embeddings and loading the database dump.
    """
    with open(config_path, "r") as config_file:
        conf = yaml.safe_load(config_file)
    if embeddings_url:
        conf["embeddings_url"] = embeddings_url
    embeddings_dir = os.path.join(os.path.expanduser(conf["base_directory"]), "embeddings")
    os.makedirs(embeddings_dir, exist_ok=True)
    tar_path = os.path.join(embeddings_dir, "embeddings.tar")
    download_embeddings(conf["embeddings_url"], tar_path)
    print("Loading dump into the database...")
    load_dump_to_db(tar_path, conf)


def run_pipeline(conf):
    """
    Runs the main pipeline for sequence embedding and similarity lookup.
    """
    try:
        conf["embedding"]["types"] = [model for model, settings in conf["embedding"]["models"].items() if
                                      settings["enabled"]]
        current_date = datetime.now().strftime("%Y%m%d%H%M%S")
        conf = setup_experiment_directories(conf, current_date)
        print("Displaying configuration:")
        pprint(conf)

        embedder = SequenceEmbedder(conf, current_date)
        embedder.start()
        lookup = EmbeddingLookUp(conf, current_date)
        lookup.start()
    except Exception as ex:
        print(f"Unexpected Error: {ex}", file=sys.stderr)
        sys.exit(1)  # Detener el programa con código de error 1


def setup_experiment_directories(conf, timestamp):
    """
    Set up experiment directories using the format:
    base_dir/experiments/{prefix}_{timestamp}
    """
    base_directory = os.path.expanduser(conf.get("base_directory", "~/fantasia/"))
    experiments_dir = os.path.join(base_directory, "experiments")
    os.makedirs(experiments_dir, exist_ok=True)

    experiment_name = f"{conf.get('prefix', 'experiment')}_{timestamp}"
    experiment_path = os.path.join(experiments_dir, experiment_name)
    os.makedirs(experiment_path, exist_ok=True)

    conf['experiment_path'] = experiment_path

    # Guardar el YAML con los parámetros del experimento
    yaml_path = os.path.join(experiment_path, "experiment_config.yaml")
    with open(yaml_path, "w") as yaml_file:
        yaml.safe_dump(conf, yaml_file, default_flow_style=False)

    print(f"Configuración del experimento guardada en: {yaml_path}")

    return conf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "\nFANTASIA: Functional Annotation and Similarity Analysis\n"
            "-------------------------------------------------------\n"
            "FANTASIA is a command-line tool for computing vector similarity and generating\n"
            "functional annotations using pre-trained language models (PLMs). It supports:\n"
            "  • ProtT5\n"
            "  • ProstT5\n"
            "  • ESM2\n"
            "\nThis system processes protein sequences by embedding them with these models,\n"
            "storing the embeddings into an h5 Object, and performing efficient similarity searches over Vector Database.\n"
            "\nPre-configured with UniProt 2024 data, FANTASIA integrates with an information system\n"
            "for seamless data management. Protein data and Gene Ontology annotations (GOA) are\n"
            "kept up to date, while proteins from the 2022 dataset remain for benchmarking (e.g., CAFA).\n"
            "\nRequirements:\n"
            "  • Relational Database: PostgreSQL (for storing annotations and metadata)\n"
            "  • Vector Database: pgvector (for efficient similarity searches)\n"
            "  • Task Queue: RabbitMQ (for parallel task execution)\n"
            "\nFor setup instructions, refer to the documentation.\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    init_parser = subparsers.add_parser(
        "initialize",
        help="Set up the database and download the embeddings.",
        description=(
            "\nFANTASIA: Functional Annotation and Similarity Analysis\n"
            "-------------------------------------------------------\n"
            "The 'initialize' command prepares the system for operation by:\n"
            "  • Reading the configuration file.\n"
            "  • Downloading the embeddings database (if specified).\n"
            "  • Setting up the necessary directories.\n"
            "\n"
            "By default, the configuration is loaded from './fantasia/config.yaml'.\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )

    init_parser.add_argument(
        "--config", type=str, default="./fantasia/config.yaml",
        help=(
            "Path to the configuration file (YAML format).\n"
            "Default: './fantasia/config.yaml'."
        )
    )

    init_parser.add_argument(
        "--embeddings_url", type=str,
        help=(
            "URL to download the embeddings database dump.\n"
            "If not provided, the system will use the URL from the config file."
        )
    )

    init_parser.epilog = (
        "Examples:\n"
        "  python fantasia/main.py initialize --config my_config.yaml\n"
        "  python fantasia/main.py initialize --config my_config.yaml --embeddings_url https://example.com/embeddings.tar\n"
    )

    run_parser = subparsers.add_parser(
        "run",
        help="Execute the pipeline to process sequences, generate embeddings, and manage lookups.",
        description=(
            "\nFANTASIA: Functional Annotation and Similarity Analysis\n"
            "-------------------------------------------------------\n"
            "The 'run' command executes the main pipeline, which includes:\n"
            "  • Loading the configuration file.\n"
            "  • Processing protein sequences from a FASTA file.\n"
            "  • Generating sequence embeddings using selected models.\n"
            "  • Storing embeddings in h5 file as input for similarity search through vectorial DB.\n"
            "  • Running functional annotation lookups based on the embeddings.\n"
            "\n"
            "By default, the configuration is loaded from './fantasia/config.yaml'.\n"
            "Supported models include ProtT5, ProstT5, and ESM2.\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )

    run_parser.add_argument(
        "--config", type=str, default="./fantasia/config.yaml",
        help="Path to the YAML configuration file. Default: './fantasia/config.yaml'."
    )

    run_parser.add_argument(
        "--input", type=str,
        help="Path to the input FASTA file containing protein sequences."
    )

    run_parser.add_argument(
        "--prefix", type=str,
        help="Prefix used to name the output files."
    )

    run_parser.add_argument(
        "--base_directory", type=str,
        help="Base directory where all results, embeddings, and execution parameters will be stored."
    )

    run_parser.add_argument(
        "--length_filter", type=int,
        help="Filter sequences by length. Sequences longer than this value will be ignored."
    )

    run_parser.add_argument(
        "--redundancy_filter", type=float,
        help=(
            "Apply sequence redundancy filtering using clustering.\n"
            "Sequences that fall into the same cluster as reference sequences\n"
            "will be excluded from the lookup to prevent homolog contamination.\n"
            "Example: 0.8 filters sequences with >80 percent similarity."
        )
    )

    run_parser.add_argument(
        "--max_workers", type=int,
        help="Number of parallel workers to process sequences. Recommended value: 1 for default PostgreSQL settings. "
             "Increasing this requires configuring PostgreSQL to allocate more resources. This parameter does not "
             "affect embedding generation as it relies on GPU."
    )

    run_parser.add_argument(
        "--models", type=str,
        help="Comma-separated list of embedding models to enable. Example: 'esm,prot'."
    )

    run_parser.add_argument(
        "--distance_threshold", type=str,
        help="Comma-separated list of model:threshold pairs. Example: 'esm:0.4,prot:0.6'."
    )

    run_parser.add_argument(
        "--batch_size", type=str,
        help="Comma-separated list of model:size pairs defining batch sizes. Example: 'esm:32,prot:64'."
    )

    run_parser.add_argument(
        "--sequence_queue_package", type=int,
        help="Number of sequences to queue per processing batch."
    )

    run_parser.add_argument(
        "--limit_per_entry", type=int,
        help=(
            "Limit the number of reference proteins considered per query.\n"
            "The closest matches based on distance will be selected.\n"
            "Example: --limit_per_entry 5 ensures only the 5 closest references are used."
        )
    )

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
        "     --limit_per_entry 5\n"
    )

    args = parser.parse_args()

    if args.command == "initialize":
        print("Initializing embeddings and database...")
        initialize(args.config, args.embeddings_url)
    elif args.command == "run":
        print("Running the FANTASIA pipeline...")
        conf = read_yaml_config(args.config)
        for key, value in vars(args).items():
            if value is not None and key not in ["command", "config"]:
                conf[key] = value
        run_pipeline(conf)
    else:
        parser.print_help()
