import os
import sys
import time
import yaml
import argparse
from datetime import datetime
import protein_metamorphisms_is.sql.model.model  # noqa: F401
from protein_metamorphisms_is.helpers.config.yaml import read_yaml_config
from FANTASIA.src.helpers import download_embeddings, load_dump_to_db
from FANTASIA.src.embedder import SequenceEmbedder
from FANTASIA.src.lookup import EmbeddingLookUp


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


def run_pipeline(config_path, fasta_path=None):
    # Leer la configuración
    conf = read_yaml_config(config_path)

    # Actualizar la ruta del FASTA si se proporciona
    if fasta_path:
        conf["fantasia_input_fasta"] = fasta_path

    # Ejecutar el pipeline de FANTASIA
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
    parser = argparse.ArgumentParser(description="FANTASIA: Command Handler")
    parser.add_argument("command", type=str, nargs="?", default=None, help="Command to execute: initialize or run")
    parser.add_argument("--config", type=str, default="./FANTASIA/config.yaml", help="Path to the configuration YAML file.")
    parser.add_argument("--fasta", type=str, help="Path to the input FASTA file.")
    args = parser.parse_args()

    if args.command == "initialize":
        print("Initializing embeddings and database...")
        initialize(args.config)
    elif args.command == "run":
        print("Running the FANTASIA pipeline...")
        run_pipeline(config_path=args.config, fasta_path=args.fasta)
    elif args.command is None:
        wait_forever()
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)
