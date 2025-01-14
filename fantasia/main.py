import os
import sys
import time
import yaml
import argparse
from datetime import datetime
import protein_metamorphisms_is.sql.model.model  # noqa: F401
from protein_metamorphisms_is.helpers.config.yaml import read_yaml_config
from fantasia.src.helpers import download_embeddings, load_dump_to_db
from fantasia.src.embedder import SequenceEmbedder
from fantasia.src.lookup import EmbeddingLookUp


def initialize(config_path):
    # Leer la configuración
    with open(config_path, "r") as config_file:
        conf = yaml.safe_load(config_file)

    embeddings_dir = os.path.expanduser(conf["embeddings_path"])
    os.makedirs(embeddings_dir, exist_ok=True)
    tar_path = os.path.join(embeddings_dir, "embeddings.tar")

    # Descargar embeddings
    print("Downloading embeddings...")
    download_embeddings(conf["embeddings_url"], tar_path)

    # Cargar el dump en la base de datos
    print("Loading dump into the database...")
    load_dump_to_db(tar_path, conf)


def run_pipeline(conf):
    # Validar configuraciones específicas de modelos
    embedding_types = conf["embedding"]["types"]
    distance_threshold = conf["embedding"].get("distance_threshold", {})
    batch_sizes = conf["embedding"].get("batch_size", {})

    for model_id in embedding_types:
        if model_id not in distance_threshold:
            raise ValueError(f"Distance threshold not defined for embedding type {model_id}")
        if model_id not in batch_sizes:
            raise ValueError(f"Batch size not defined for embedding type {model_id}")

    # Ejecutar el pipeline de fantasia
    current_date = datetime.now().strftime("%Y%m%d%H%M%S")
    embedder = SequenceEmbedder(conf, current_date)
    embedder.start()

    lookup = EmbeddingLookUp(conf, current_date)
    lookup.start()


def wait_forever():
    # Modo de espera
    print("Container is running and waiting for commands...")
    try:
        while True:
            time.sleep(3600)  # Espera indefinida
    except KeyboardInterrupt:
        print("Stopping container.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fantasia: Command Handler")
    parser.add_argument("command", type=str, nargs="?", default=None, help="Command to execute: initialize or run")
    parser.add_argument("--config", type=str, default="./fantasia/config.yaml", help="Path to the configuration YAML file.")
    parser.add_argument("--fasta", type=str, help="Path to the input FASTA file.")
    parser.add_argument("--prefix", type=str, help="Prefix for output files.")
    parser.add_argument("--length_filter", type=int, help="Length filter threshold for sequences.")
    parser.add_argument("--redundancy_filter", type=float, help="Redundancy filter threshold.")

    # CLI for embedding-specific parameters
    parser.add_argument("--esm", action="store_true", help="Use ESM model.")
    parser.add_argument("--prost", action="store_true", help="Use Prost model.")
    parser.add_argument("--prot", action="store_true", help="Use Prot model.")
    parser.add_argument("--distance_threshold", type=str, help="Comma-separated list of distance thresholds per model ID (e.g., 1:0.5,2:0.7,3:0.6).")
    parser.add_argument("--batch_size", type=str, help="Comma-separated list of batch sizes per model ID (e.g., 1:50,2:60,3:40).")
    parser.add_argument("--sequence_queue_package", type=int, help="Number of sequences to queue in each package.")

    args = parser.parse_args()

    if args.command == "initialize":
        print("Initializing embeddings and database...")
        initialize(args.config)
    elif args.command == "run":
        print("Running the fantasia pipeline...")

        # Leer la configuración una sola vez
        conf = read_yaml_config(args.config)

        # Sobrescribir parámetros con los valores del CLI
        if args.fasta:
            conf["fantasia_input_fasta"] = args.fasta
        if args.prefix:
            conf["fantasia_prefix"] = args.prefix
        if args.length_filter is not None:
            conf["length_filter"] = args.length_filter
        if args.redundancy_filter is not None:
            conf["redundancy_filter"] = args.redundancy_filter
        if args.sequence_queue_package is not None:
            conf["embedding_queue_size"] = args.sequence_queue_package

        # Filtrar los modelos según las opciones del CLI
        selected_models = []
        if args.esm:
            selected_models.append(1)  # ID para ESM
        if args.prost:
            selected_models.append(2)  # ID para Prost
        if args.prot:
            selected_models.append(3)  # ID para Prot

        if selected_models:
            conf["embedding"]["types"] = selected_models

        # Procesar distance_threshold desde CLI
        if args.distance_threshold:
            try:
                threshold_overrides = {
                    int(k): float(v) for k, v in (pair.split(":") for pair in args.distance_threshold.split(","))
                }
                conf["embedding"]["distance_threshold"].update(threshold_overrides)
            except ValueError as e:
                print(f"Error parsing distance_threshold: {e}")
                sys.exit(1)

        # Procesar batch_size desde CLI
        if args.batch_size:
            try:
                batch_size_overrides = {
                    int(k): int(v) for k, v in (pair.split(":") for pair in args.batch_size.split(","))
                }
                conf["embedding"]["batch_size"].update(batch_size_overrides)
            except ValueError as e:
                print(f"Error parsing batch_size: {e}")
                sys.exit(1)

        # Pasar la configuración modificada directamente
        print(conf)
        run_pipeline(conf)
    elif args.command is None:
        wait_forever()
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)
