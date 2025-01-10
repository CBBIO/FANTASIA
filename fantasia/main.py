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
    parser.add_argument("--config", type=str, default="./opt/venv/lib/python3.12/site-packages/fantasia/config.yaml", help="Path to the configuration YAML file.")
    parser.add_argument("--fasta", type=str, help="Path to the input FASTA file.")
    parser.add_argument("--prefix", type=str, help="Prefix for output files.")
    parser.add_argument("--max_distance", type=float, help="Maximum distance for similarity matching.")
    parser.add_argument("--length_filter", type=int, help="Length filter threshold for sequences.")
    parser.add_argument("--redundancy_filter", type=float, help="Redundancy filter threshold.")
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
        if args.max_distance is not None:
            conf["max_distance"] = args.max_distance
        if args.length_filter is not None:
            conf["length_filter"] = args.length_filter
        if args.redundancy_filter is not None:
            conf["redundancy_filter"] = args.redundancy_filter

        # Pasar la configuración modificada directamente
        run_pipeline(conf)
    elif args.command is None:
        wait_forever()
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)
