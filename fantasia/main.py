
import os
import sys
import urllib

import yaml
import logging
from datetime import datetime

from protein_information_system.helpers.logger.logger import setup_logger

from fantasia.src.embedder import SequenceEmbedder
from fantasia.src.helpers.helpers import download_embeddings, load_dump_to_db, parse_unknown_args
from fantasia.src.lookup import EmbeddingLookUp
from protein_information_system.helpers.config.yaml import read_yaml_config
import protein_information_system.sql.model.model  # noqa: F401
from protein_information_system.helpers.services.services import check_services

from fantasia.src.helpers.parser import build_parser


def initialize(conf):
    logger = logging.getLogger("fantasia")
    embeddings_dir = os.path.join(os.path.expanduser(conf["base_directory"]), "embeddings")
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

        if conf["only_lookup"]:
            conf["embeddings_path"] = conf["input"]
        else:
            embedder = SequenceEmbedder(conf, current_date)
            logger.info("Running embedding step to generate embeddings.h5...")
            embedder.start()

            conf["embeddings_path"] = os.path.join(conf["experiment_path"], "embeddings.h5")

            if not os.path.exists(conf["embeddings_path"]):
                logger.error(
                    f"❌ The embedding file was not created: {conf['embeddings_path']}\n"
                    f"💡 Please ensure the embedding step ran correctly. "
                    f"You can try re-running with 'only_lookup: true' and 'input: <path_to_h5>'."
                )
                raise FileNotFoundError(
                    f"Missing HDF5 file after embedding step: {conf['embeddings_path']}"
                )

        lookup = EmbeddingLookUp(conf, current_date)
        lookup.start()
    except Exception:
        logger.error("Pipeline execution failed.", exc_info=True)
        sys.exit(1)


def setup_experiment_directories(conf, timestamp):
    logger = logging.getLogger("fantasia")
    base_directory = os.path.expanduser(conf.get("base_directory", "~/fantasia/"))
    experiments_dir = os.path.join(base_directory, "experiments")
    os.makedirs(experiments_dir, exist_ok=True)

    experiment_name = f"{conf.get('prefix', 'experiment')}_{timestamp}"
    experiment_path = os.path.join(experiments_dir, experiment_name)
    os.makedirs(experiment_path, exist_ok=True)

    conf['experiment_path'] = experiment_path

    yaml_path = os.path.join(experiment_path, "experiment_config.yaml")
    with open(yaml_path, "w") as yaml_file:
        yaml.safe_dump(conf, yaml_file, default_flow_style=False)

    logger.info(f"Experiment configuration saved at: {yaml_path}")
    return conf


def load_and_merge_config(args, unknown_args):
    """
    Loads the base configuration from YAML and applies any overrides provided via CLI arguments.

    This function ensures that any parameters passed through standard arguments or unknown
    arguments (parsed as key-value pairs) override those defined in the configuration file.

    Additionally, it restores legacy support for systems that rely on the presence of the
    `embedding.types` field, which lists all embedding model identifiers enabled for the run.

    Parameters
    ----------
    args : Namespace
        Parsed known arguments from argparse.
    unknown_args : list of str
        List of unknown CLI arguments in the format --key value.

    Returns
    -------
    dict
        A merged and final configuration dictionary.
    """
    conf = read_yaml_config(args.config)

    for key, value in vars(args).items():
        if value is not None and key not in ["command", "config"]:
            conf[key] = value

    unknown_args_dict = parse_unknown_args(unknown_args)
    for key, value in unknown_args_dict.items():
        if value is not None:
            conf[key] = value

    # ✅ Legacy compatibility: populate embedding.types with enabled model names
    # This list is used by some components (e.g., GPU task schedulers)
    conf.setdefault("embedding", {})
    conf["embedding"]["types"] = [
        model for model, settings in conf["embedding"].get("models", {}).items()
        if settings.get("enabled", False)
    ]

    import re

    def sanitize_taxonomy_lists(conf):
        """
        Ensures taxonomy ID fields are always lists of numeric strings (e.g., ["559292", "6239"]).
        Accepts input as list or single string, with space/comma/mixed separators.
        """
        for key in ["taxonomy_ids_to_exclude", "taxonomy_ids_included_exclusively"]:
            val = conf.get(key)

            if isinstance(val, list):
                # Asegura que todos los elementos sean strings de dígitos
                cleaned = []
                for item in val:
                    if isinstance(item, str):
                        tokens = re.split(r"[,\s]+", item.strip())
                        cleaned.extend(t for t in tokens if t.isdigit())
                    elif isinstance(item, int):
                        cleaned.append(str(item))
                conf[key] = cleaned

            elif isinstance(val, str):
                # Soporta separadores por espacio, coma o mezcla
                conf[key] = [t for t in re.split(r"[,\s]+", val.strip()) if t.isdigit()]

            elif val is None or val is False:
                conf[key] = []

            else:
                raise ValueError(f"Invalid format for {key}: expected list, string, or None.")

    sanitize_taxonomy_lists(conf)

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

        if args.redundancy_filter is not None and not (0 <= args.redundancy_filter <= 1):
            raise ValueError("redundancy_filter must be a decimal between 0 and 1 (e.g., 0.95 for 95%)")

        run_pipeline(conf)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
