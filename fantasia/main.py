"""
=================================
Main execution module
=================================

This module serves as the primary entry point for the **FANTASIA** system within
the *Protein Information System (PIS)*. It orchestrates the end-to-end workflow,
from initializing the reference embeddings database to running the functional
annotation pipeline.

Main Functions
--------------

- **initialize**: Downloads the reference embeddings and loads them into the database.
- **run_pipeline**: Executes the main FANTASIA pipeline, including sequence embedding
  and subsequent database lookup.
- **setup_experiment_directories**: Manages directory creation and experiment-specific
  configuration files.
- **load_and_merge_config**: Loads the base YAML configuration, applies CLI overrides,
  ensures backward compatibility, and performs early validation checks.
- **main**: CLI entry point that parses arguments, initializes logging, validates
  services, and dispatches subcommands.

Notes
-----
* Designed for CLI usage through the ``initialize`` and ``run`` subcommands.
* Configuration is driven by YAML files and command-line arguments.
"""

import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)  # noqa: E402
warnings.filterwarnings("ignore", category=UserWarning)  # noqa: E402

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
    """
    Initialize the FANTASIA environment by downloading and loading reference embeddings.

    This function:

      1. Creates the embeddings directory if it does not exist.
      2. Downloads the reference embeddings archive from the configured URL.
      3. Loads the extracted embeddings into the database for subsequent lookup.

    Parameters
    ----------
    conf : dict
        Configuration dictionary. Must contain:

        - ``base_directory`` (str): Base directory where embeddings and experiments are stored.
        - ``embeddings_url`` (str): URL of the reference embeddings archive to be downloaded.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If the embeddings archive cannot be found or accessed after download.
    RuntimeError
        If loading the embeddings into the database fails.
    """

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
    """
    Execute the main FANTASIA pipeline.

    This function coordinates the entire functional annotation workflow:

      1. Prepares experiment directories and saves the configuration.
      2. Runs the embedding step unless ``only_lookup`` is enabled.
      3. Validates that the embeddings file has been generated.
      4. Performs database lookup using the generated or provided embeddings.

    Parameters
    ----------
    conf : dict
        Configuration dictionary. Must include:

        - ``base_directory`` (str): Root path for storing experiments.
        - ``only_lookup`` (bool): If True, skip embedding and use provided input file.
        - ``input`` (str, optional): Path to an existing HDF5 embeddings file (required if
          ``only_lookup`` is True).
        - Other pipeline settings required by embedding and lookup components.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If the embeddings file is missing after the embedding step.
    SystemExit
        If a fatal error occurs during pipeline execution.
    Exception
        For any other unexpected runtime errors.
    """

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
            embedder.start()  # los workers del embedder hacen join dentro del start
            del embedder  # elimina referencias en el padre

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
    """
    Prepare and configure directories for a new experiment.

    This function:

      1. Expands the base directory and ensures an ``experiments`` folder exists.
      2. Creates a unique experiment directory using the provided timestamp.
      3. Stores the experiment configuration into ``experiment_config.yaml``.
      4. Updates the configuration dictionary with the generated experiment path.

    Parameters
    ----------
    conf : dict
        Configuration dictionary containing at least:

        - ``base_directory`` (str, optional): Root path for experiments.
          Defaults to ``~/fantasia/`` if not provided.
        - ``prefix`` (str, optional): Prefix for experiment naming. Defaults to ``experiment``.

    timestamp : str
        Unique identifier (usually ``YYYYMMDDHHMMSS``) appended to the experiment name.

    Returns
    -------
    dict
        Updated configuration dictionary including the key:
        - ``experiment_path`` (str): Path to the newly created experiment directory.

    Raises
    ------
    OSError
        If the experiment directory or configuration file cannot be created.
    """

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
    Load the base YAML configuration and apply CLI overrides.

    This function merges different sources of configuration into a normalized
    dictionary ready for pipeline execution. The process includes:

      1. Loading the YAML configuration file specified by ``--config``.
      2. Applying known CLI arguments as flat overrides.
      3. Parsing unknown CLI key-value pairs (``--key value``) into overrides.
      4. Mapping selected CLI flags into their canonical nested structure.
      5. Sanitizing taxonomy ID lists.
      6. Restoring legacy compatibility for ``embedding.types``.
      7. Validating redundancy thresholds and taxonomy lists.

    Parameters
    ----------
    args : argparse.Namespace
        Namespace of parsed known arguments from ``argparse``. Must include:

        - ``config`` (str): Path to the base YAML configuration.
        - Other optional CLI overrides such as ``device``, ``redundancy_filter``,
          ``alignment_coverage``, and ``threads``.

    unknown_args : list of str
        List of additional CLI arguments in the form
        ``["--key", "value", ...]``. These are parsed into dictionary entries.

    Returns
    -------
    dict
        A fully merged and validated configuration dictionary. Keys include:

        - ``embedding`` (dict): Embedding-related settings, including enabled models.
        - ``lookup`` (dict): Lookup and redundancy-related parameters.
        - ``taxonomy`` (dict): Taxonomy filtering options.
        - Other keys inherited from the YAML file and CLI overrides.

    Raises
    ------
    ValueError
        If redundancy thresholds are out of range, or taxonomy lists are provided
        in an invalid format.
    """

    # Load base YAML
    conf = read_yaml_config(args.config)

    # 1) Merge known CLI args as flat overrides (except control keys)
    for key, value in vars(args).items():
        if value is not None and key not in ("command", "config"):
            conf[key] = value

    # 2) Merge unknown --k v pairs as flat overrides
    unknown_args_dict = parse_unknown_args(unknown_args)
    for key, value in unknown_args_dict.items():
        if value is not None:
            conf[key] = value

    # 3) Canonical mappings (mirror CLI flags into nested structure expected downstream)
    # 3.1 Device → embedding.device (also keep flat 'device' for any legacy consumer)
    if conf.get("device") is not None:
        emb = conf.setdefault("embedding", {})
        emb["device"] = conf["device"]  # "cpu" | "cuda"

    # 3.2 Redundancy thresholds and threads → lookup.redundancy.*
    #     Keep flat duplicates for compatibility with components that read flat keys.
    ri = conf.get("redundancy_filter")  # identity in [0, 1]; 0 disables
    rc = conf.get("alignment_coverage")  # coverage in (0, 1]
    th = conf.get("threads")  # MMseqs2 threads

    if any(v is not None for v in (ri, rc, th)):
        lk = conf.setdefault("lookup", {})
        r = lk.setdefault("redundancy", {})
        if ri is not None:
            r["identity"] = float(ri)
        if rc is not None:
            r["coverage"] = float(rc)
        if th is not None:
            r["threads"] = int(th)

    # 3.3 Taxonomy filters → lookup.taxonomy.{exclude, include_only, get_descendants}
    tx_ex = conf.get("taxonomy_ids_to_exclude")
    tx_in = conf.get("taxonomy_ids_included_exclusively")
    tx_desc = conf.get("get_descendants")

    if any(v not in (None, [], "") for v in (tx_ex, tx_in, tx_desc)):
        lk = conf.setdefault("lookup", {})
        t = lk.setdefault("taxonomy", {})
        if tx_ex not in (None, []):
            t["exclude"] = tx_ex
        if tx_in not in (None, []):
            t["include_only"] = tx_in
        if tx_desc is not None:
            # Accept truthy/falsy shapes; coerce to bool
            t["get_descendants"] = bool(tx_desc)

    # 4) Sanitize taxonomy lists (always list[str] of digits like ["559292", "6239"])
    import re

    def _sanitize_taxonomy_lists(cfg: dict) -> None:
        keys = ("taxonomy_ids_to_exclude", "taxonomy_ids_included_exclusively")
        for k in keys:
            val = cfg.get(k)
            if isinstance(val, list):
                cleaned = []
                for item in val:
                    if isinstance(item, int):
                        cleaned.append(str(item))
                    elif isinstance(item, str):
                        tokens = re.split(r"[,\s]+", item.strip())
                        cleaned.extend(tok for tok in tokens if tok.isdigit())
                cfg[k] = cleaned
            elif isinstance(val, str):
                cfg[k] = [tok for tok in re.split(r"[,\s]+", val.strip()) if tok.isdigit()]
            elif val in (None, False):
                cfg[k] = []
            else:
                raise ValueError(f"Invalid format for {k}: expected list, string, or None; got {type(val).__name__}.")

    _sanitize_taxonomy_lists(conf)

    # 5) Legacy compatibility: populate embedding.types with names of enabled models (YAML-only)
    conf.setdefault("embedding", {})
    conf["embedding"]["types"] = [
        name for name, settings in conf["embedding"].get("models", {}).items()
        if isinstance(settings, dict) and settings.get("enabled", False)
    ]

    # 6) Early validations for redundancy thresholds (if supplied)
    if ri is not None:
        iri = float(ri)
        if not (0.0 <= iri <= 1.0):
            raise ValueError("redundancy_filter / redundancy_identity must be in [0, 1].")

    if rc is not None:
        irc = float(rc)
        if not (0.0 < irc <= 1.0):
            raise ValueError("alignment_coverage / redundancy_coverage must be in (0, 1].")

    return conf


def main():
    """
    Command-line interface (CLI) entry point for FANTASIA.

    This function:
      1. Builds the argument parser and reads CLI inputs.
      2. Loads and merges the configuration from YAML and CLI overrides.
      3. Sets up logging with timestamped log files.
      4. Verifies that required background services are available.
      5. Dispatches execution to the selected subcommand.

    Supported Subcommands
    ---------------------
    - ``initialize`` : Download and load reference embeddings into the database.
    - ``run``        : Execute the full FANTASIA pipeline (embedding + lookup).

    Behavior
    --------
    * If no command is provided, the function prints the help message and exits.
    * The ``run`` command requires at least one embedding model to be enabled
      in the configuration under ``embedding.models``.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If no embedding models are enabled, or if redundancy thresholds are invalid.
    SystemExit
        If the user requests help, or if a fatal error occurs during execution.
    """

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
