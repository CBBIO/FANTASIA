import os
import sys
import urllib

import yaml
import logging
from datetime import datetime

from protein_metamorphisms_is.helpers.logger.logger import setup_logger

from fantasia.src.embedder import SequenceEmbedder
from fantasia.src.helpers.helpers import download_embeddings, load_dump_to_db, parse_unknown_args, check_services
from fantasia.src.lookup import EmbeddingLookUp
from protein_metamorphisms_is.helpers.config.yaml import read_yaml_config
import protein_metamorphisms_is.sql.model.model  # noqa: F401

from fantasia.src.helpers.parser import build_parser


def initialize(conf):
    logger = logging.getLogger("fantasia")
    embeddings_dir = os.path.join(os.path.expanduser(conf["experiment"]["base_directory"]), "embeddings")
    os.makedirs(embeddings_dir, exist_ok=True)

    # Nuevo: obtener nombre del archivo desde la URL
    filename = os.path.basename(urllib.parse.urlparse(conf["embeddings_url"]).path)
    tar_path = os.path.join(embeddings_dir, filename)

    logger.info(f"Downloading reference embeddings to {tar_path}...")
    download_embeddings(conf["embeddings_url"], tar_path)

    logger.info("Loading embeddings into the database...")
    load_dump_to_db(tar_path, conf)


def run_pipeline(conf):
    logger = logging.getLogger("fantasia")
    try:
        current_date = datetime.now().strftime("%Y%m%d%H%M%S")
        conf = setup_experiment_directories(conf, current_date)

        logger.info("Configuration loaded:")
        logger.debug(conf)

        # Ruta del archivo de embeddings generados por el embedder
        embeddings_path = os.path.join(conf["experiment_path"], "embeddings.h5")

        if conf.get("only_lookup", False) and os.path.exists(embeddings_path):
            conf["embeddings_path"] = embeddings_path
        else:
            # Ejecuta el módulo de embeddings y espera al archivo de salida
            embedder = SequenceEmbedder(conf, current_date)
            logger.info("Running embedding step to generate embeddings.h5...")
            embedder.start()

            if not os.path.exists(embeddings_path):
                raise FileNotFoundError(f"Expected embeddings file not found at {embeddings_path}")

            conf["embeddings_path"] = embeddings_path

        # Ejecuta la búsqueda de anotaciones funcionales basadas en embeddings
        lookup = EmbeddingLookUp(conf, current_date)
        lookup.start()

    except Exception:
        logger.error("Pipeline execution failed.", exc_info=True)
        sys.exit(1)



def setup_experiment_directories(conf, timestamp):
    logger = logging.getLogger("fantasia")

    base_dir = os.path.expanduser(conf["experiment"]["base_directory"])
    experiments_dir = base_dir
    os.makedirs(experiments_dir, exist_ok=True)

    prefix = conf["experiment"].get("prefix", "experiment")
    experiment_name = f"{prefix}_{timestamp}"
    experiment_path = os.path.join(experiments_dir, experiment_name)
    os.makedirs(experiment_path, exist_ok=True)

    conf["experiment_path"] = experiment_path
    conf["output_path"] = conf["experiment"].get("output_path", experiment_path)

    yaml_path = os.path.join(experiment_path, "experiment_config.yaml")
    with open(yaml_path, "w") as yaml_file:
        yaml.safe_dump(conf, yaml_file, default_flow_style=False)

    logger.info(f"Experiment configuration saved at: {yaml_path}")
    return conf


def load_and_merge_config(args, unknown_args):
    """
    Load the base configuration from a YAML file and apply overrides provided
    through explicit CLI arguments or additional unknown parameters.

    This method ensures proper integration between configuration layers, giving
    priority to CLI arguments while preserving default and runtime behaviors.

    Returns
    -------
    dict
        Merged configuration dictionary ready for pipeline execution.
    """
    conf = read_yaml_config(args.config)

    # 1. Apply known CLI arguments (e.g., --fasta_path, --device)
    for key, value in vars(args).items():
        if value is not None and key not in ["command", "config"]:
            conf[key] = value

    # 2. Apply unknown CLI overrides (e.g., --batch_size esm:16,prot_t5:8)
    unknown_args_dict = parse_unknown_args(unknown_args)
    conf.update(unknown_args_dict)

    # 3. Parse CLI-specified models and disable others explicitly
    if "models" in conf and isinstance(conf["models"], str):
        selected_models = [m.strip() for m in conf["models"].split(",")]
        conf.setdefault("embedding", {})
        conf["embedding"].setdefault("models", {})
        for model in conf["embedding"]["models"]:
            conf["embedding"]["models"][model]["enabled"] = model in selected_models

    # 4. Apply per-model batch size if provided (e.g., --batch_size esm:32,prot_t5:8)
    if "batch_size" in conf and isinstance(conf["batch_size"], str):
        for pair in conf["batch_size"].split(","):
            model, size = pair.split(":")
            conf["embedding"]["models"].setdefault(model, {})
            conf["embedding"]["models"][model]["batch_size"] = int(size)
        del conf["batch_size"]
    else:
        # Use runtime.batch_size as fallback for models without batch_size
        runtime_default = conf.get("runtime", {}).get("batch_size")
        if runtime_default is not None:
            for model in conf.get("embedding", {}).get("models", {}):
                conf["embedding"]["models"][model].setdefault("batch_size", runtime_default)

    # 5. Apply per-model distance thresholds (e.g., --distance_threshold esm:3.0)
    if "distance_threshold" in conf and isinstance(conf["distance_threshold"], str):
        for pair in conf["distance_threshold"].split(","):
            model, threshold = pair.split(":")
            conf["embedding"]["models"].setdefault(model, {})
            conf["embedding"]["models"][model]["distance_threshold"] = float(threshold)
        del conf["distance_threshold"]

    # 6. Set list of enabled model types for processing
    conf["embedding"]["types"] = [
        model for model, cfg in conf["embedding"]["models"].items()
        if cfg.get("enabled", False)
    ]

    # Map CLI-level `prefix` to nested `experiment.prefix` if needed
    if "prefix" in conf:
        conf.setdefault("experiment", {})
        conf["experiment"]["prefix"] = conf.pop("prefix")

    return conf


def main():
    parser = build_parser()
    args, unknown_args = parser.parse_known_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    conf = load_and_merge_config(args, unknown_args)

    current_date = datetime.now().strftime("%Y%m%d%H%M%S")
    logs_directory = os.path.expanduser(os.path.expanduser(conf.get("log_path", "~/fantasia/logs/")))
    log_name = f"Logs_{current_date}"
    conf['log_path'] = os.path.join(logs_directory, log_name)  # por ahora hace un archivo, no una carpeta
    logger = setup_logger("FANTASIA", conf.get("log_path", "fantasia.log"))

    check_services(conf, logger)

    if args.command == "initialize":
        logger.info("Starting initialization...")
        initialize(conf)

    elif args.command == "run":
        logger.info("Starting FANTASIA pipeline...")

        models_cfg = conf.get("embedding", {}).get("models", {})
        enabled_models = [name for name, model in models_cfg.items() if model.get("enabled")]

        if not enabled_models:
            raise ValueError(
                "At least one embedding model must be enabled in the configuration under 'embedding.models'.")

        if args.redundancy_filter is not None and not (0 < args.redundancy_filter <= 1):
            raise ValueError("redundancy_filter must be a decimal between 0 and 1 (e.g., 0.95 for 95%)")

        run_pipeline(conf)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
