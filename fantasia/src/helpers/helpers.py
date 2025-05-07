"""
helpers.py

Utility functions for downloading files, restoring PostgreSQL dumps,
checking infrastructure services, running sequence alignments, and handling NCBI taxonomy.

All functions are documented in Sphinx-style and follow PEP8 for readability and maintainability.
"""

import os
import subprocess
from typing import List, Dict, Union

import pika
import requests
from ete3 import NCBITaxa
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from tqdm import tqdm
import parasail

from protein_metamorphisms_is.sql.base.database_manager import DatabaseManager


def download_embeddings(url: str, tar_path: str) -> None:
    """
    Download a TAR file containing embeddings from a given URL with a progress bar.

    Parameters
    ----------
    url : str
        The URL from which to download the embeddings.
    tar_path : str
        The local path where the TAR file will be saved.

    Raises
    ------
    Exception
        If the download fails due to a non-200 HTTP status code.
    """
    if os.path.exists(tar_path):
        print("Embeddings file already exists. Skipping download.")
        return

    print("Downloading embeddings...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))
        with open(tar_path, "wb") as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                progress_bar.update(len(chunk))
        print(f"Embeddings downloaded successfully to {tar_path}.")
    else:
        raise Exception(f"Failed to download embeddings. Status code: {response.status_code}")


def load_dump_to_db(dump_path: str, db_config: Dict[str, str]) -> None:
    """
    Load a PostgreSQL database dump (TAR format) into the database.

    Parameters
    ----------
    dump_path : str
        Path to the .tar database dump.
    db_config : dict
        Dictionary with keys: DB_HOST, DB_PORT, DB_USERNAME, DB_PASSWORD, DB_NAME.

    Raises
    ------
    Exception
        If subprocess execution fails.
    """
    print("Resetting and preparing the database...")

    url = (
        f"postgresql://{db_config['DB_USERNAME']}:{db_config['DB_PASSWORD']}"
        f"@{db_config['DB_HOST']}:{db_config['DB_PORT']}/{db_config['DB_NAME']}"
    )
    engine = create_engine(url)

    with engine.connect() as conn:
        conn.execution_options(isolation_level="AUTOCOMMIT")
        conn.execute(text("DROP SCHEMA public CASCADE;"))
        conn.execute(text("CREATE SCHEMA public;"))
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS VECTOR;"))
        print("✅ Schema reset and VECTOR extension created.")

    print("Loading dump into the database...")

    env = os.environ.copy()
    env["PGPASSWORD"] = db_config["DB_PASSWORD"]

    command = [
        "pg_restore",
        "--verbose",
        "-U", db_config["DB_USERNAME"],
        "-h", db_config["DB_HOST"],
        "-p", str(db_config["DB_PORT"]),
        "-d", db_config["DB_NAME"],
        dump_path
    ]

    print("Executing:", " ".join(command))
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env
        )
        stdout, stderr = process.communicate()

        if process.returncode == 0:
            print("✅ Database dump loaded successfully.")
        else:
            print(f"❌ Error while loading dump: {stderr}")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")


def parse_unknown_args(unknown_args: List[str]) -> Dict[str, Union[str, bool]]:
    """
    Parse unknown command-line arguments into a dictionary.

    Parameters
    ----------
    unknown_args : list of str
        List of arguments in the format ["--key", "value", "--flag"].

    Returns
    -------
    dict
        Parsed arguments where flags have boolean True and keys have corresponding values.
    """
    result = {}
    i = 0
    while i < len(unknown_args):
        arg = unknown_args[i]
        if arg.startswith("--"):
            key = arg[2:]
            if i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith("--"):
                result[key] = unknown_args[i + 1]
                i += 1
            else:
                result[key] = True
        i += 1
    return result


def check_services(conf: Dict[str, str], logger) -> None:
    """
    Check the connectivity of PostgreSQL and RabbitMQ services.

    Parameters
    ----------
    conf : dict
        Configuration with DB and RabbitMQ credentials.
    logger : logging.Logger
        Logger instance for logging status messages.

    Raises
    ------
    ConnectionError
        If connection to PostgreSQL or RabbitMQ fails.
    """
    # PostgreSQL check
    try:
        db_manager = DatabaseManager(conf)
        session = db_manager.get_session()
        session.execute(text("SELECT 1"))
        session.close()
        logger.info("PostgreSQL connection OK.")
    except OperationalError as e:
        raise ConnectionError(
            f"Could not connect to PostgreSQL at {conf['DB_HOST']}:{conf['DB_PORT']}. "
            f"Check your DB settings.\nDocs: https://fantasia.readthedocs.io"
        ) from e

    # RabbitMQ check
    try:
        connection_params = pika.ConnectionParameters(
            host=conf["rabbitmq_host"],
            port=conf.get("rabbitmq_port", 5672),
            credentials=pika.PlainCredentials(
                conf["rabbitmq_user"], conf["rabbitmq_password"]
            ),
            blocked_connection_timeout=3,
            heartbeat=600,
        )
        connection = pika.BlockingConnection(connection_params)
        connection.close()
        logger.info("RabbitMQ connection OK.")
    except Exception as e:
        raise ConnectionError(
            f"Could not connect to RabbitMQ at {conf['rabbitmq_host']}:{conf.get('rabbitmq_port', 5672)}. "
            f"Check your MQ settings.\nDocs: https://fantasia.readthedocs.io"
        ) from e


def run_needle_from_strings(seq1: str, seq2: str) -> Dict[str, float]:
    """
    Perform global alignment of two sequences using Parasail and return EMBOSS-style metrics.

    Parameters
    ----------
    seq1 : str
        First sequence (query).
    seq2 : str
        Second sequence (reference).

    Returns
    -------
    dict
        Dictionary with identity, similarity, gaps, and alignment score metrics.
    """
    result = parasail.nw_trace(seq1, seq2, 10, 1, parasail.blosum62)
    aligned_query = result.traceback.query
    aligned_ref = result.traceback.ref
    comp_line = result.traceback.comp

    alignment_length = len(aligned_query)
    matches = sum(a == b for a, b in zip(aligned_query, aligned_ref) if a != '-' and b != '-')
    similarity = sum(c in "|:" for c in comp_line)
    gaps = aligned_query.count('-') + aligned_ref.count('-')

    return {
        "identity_count": matches,
        "alignment_length": alignment_length,
        "identity_percentage": 100 * matches / alignment_length,
        "similarity_percentage": 100 * similarity / alignment_length,
        "gaps_percentage": 100 * gaps / alignment_length,
        "alignment_score": result.score,
    }


def get_descendant_ids(parent_ids: List[int]) -> List[str]:
    """
    Get all descendant taxon IDs (including intermediate nodes) from a list of parent taxon IDs.

    Parameters
    ----------
    parent_ids : list of int
        List of NCBI taxonomy IDs.

    Returns
    -------
    list of str
        List of descendant taxonomy IDs as strings (including the original parent IDs).
    """
    print(parent_ids)
    descendants_ids = []
    ncbi = NCBITaxa()
    ncbi.update_taxonomy_database()
    for taxon in parent_ids:
        descendants = ncbi.get_descendant_taxa(taxon, intermediate_nodes=True)
        descendants_ids.extend(str(tid) for tid in descendants)
    descendants_ids.extend(str(tid) for tid in parent_ids)
    return descendants_ids
