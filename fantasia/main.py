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
    Load the base configuration from YAML and apply CLI overrides, normalizing the
    structure expected by the pipeline.

    This function:
      1) Loads the YAML configuration specified by --config.
      2) Applies known CLI arguments and unknown key-value pairs (e.g., "--foo bar")
         as overrides on top of the YAML.
      3) Maps select CLI flags to their canonical nested locations in the configuration:
         - --device                      → embedding.device
         - --redundancy_identity|--redundancy_filter  → lookup.redundancy.identity (+ flat compatibility)
         - --redundancy_coverage|--alignment_coverage → lookup.redundancy.coverage (+ flat compatibility)
         - --threads                     → lookup.redundancy.threads (+ flat compatibility)
         - taxonomy filters              → lookup.taxonomy.{exclude,include_only,get_descendants}
      4) Sanitizes taxonomy ID lists so they are always lists of numeric strings.
      5) Restores legacy support for components that rely on `embedding.types`
         (the list of enabled model names in YAML).
      6) Performs early validations for redundancy thresholds.

    Notes
    -----
    * **Model selection and per-model settings are YAML-only.** This function does not
      enable/disable models from the CLI, nor does it accept per-model batch sizes,
      thresholds, or layer indices.
    * Redundancy settings are accepted from CLI and mirrored in both nested
      `lookup.redundancy.*` and flat keys for backward compatibility with consumers
      that read either form.

    Parameters
    ----------
    args : argparse.Namespace
        Known arguments parsed by argparse.
    unknown_args : list[str]
        Extra CLI arguments in the form ["--key", "value", ...] that are parsed into
        key-value pairs (using the project helper).

    Returns
    -------
    dict
        The merged, normalized configuration dictionary ready for the pipeline.

    Raises
    ------
    ValueError
        If redundancy thresholds are out of range or taxonomy lists have invalid formats.
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
    ri = conf.get("redundancy_filter")      # identity in [0, 1]; 0 disables
    rc = conf.get("alignment_coverage")     # coverage in (0, 1]
    th = conf.get("threads")                # MMseqs2 threads

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
