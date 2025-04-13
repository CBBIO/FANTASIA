import os

import pika
import requests
import subprocess
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from protein_metamorphisms_is.sql.base.database_manager import DatabaseManager
from tqdm import tqdm
from ete3 import NCBITaxa


def download_embeddings(url, tar_path):
    """
    Download the embeddings TAR file from the given URL with a progress bar.

    Parameters
    ----------
    url : str
        The URL to download the embeddings from.
    tar_path : str
        Path where the TAR file will be saved.
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


def load_dump_to_db(dump_path, db_config):
    """
    Load a database backup file (in TAR format) into the database.

    Parameters
    ----------
    dump_path : str
        Path to the database backup TAR file.
    db_config : dict
        Database configuration dictionary containing host, port, user, password, and db name.
    """
    print("Resetting and preparing the database...")

    from sqlalchemy import create_engine

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


def parse_unknown_args(unknown_args):
    """Convierte una lista de argumentos desconocidos en un diccionario."""
    result = {}
    i = 0
    while i < len(unknown_args):
        arg = unknown_args[i]
        if arg.startswith("--"):
            key = arg[2:]  # Elimina los dos guiones
            if i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith("--"):
                result[key] = unknown_args[i + 1]
                i += 1
            else:
                result[key] = True  # Si no tiene valor, se asume un flag booleano
        i += 1
    return result


def check_services(conf, logger):
    # Comprobación de PostgreSQL usando tu propio DatabaseManager
    try:
        db_manager = DatabaseManager(conf)
        session = db_manager.get_session()
        session.execute(text("SELECT 1"))  # ✅ Correcto
        session.close()
        logger.info("PostgreSQL connection OK.")
    except OperationalError as e:
        raise ConnectionError(
            f"Could not connect to PostgreSQL at {conf['DB_HOST']}:{conf['DB_PORT']}. "
            f"Verify your DB settings.\nDocs: https://fantasia.readthedocs.io"
        ) from e

    # Comprobación de RabbitMQ usando Pika (como haces tú mismo)
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
            f"Verify your MQ settings.\nDocs: https://fantasia.readthedocs.io"
        ) from e

import re
import tempfile
import subprocess
from pathlib import Path

def run_needle_from_strings(seq1, seq2):
    """
    Executes EMBOSS Needle from two protein sequences as strings.
    Returns official alignment metrics: identity, score, similarity, gaps, etc.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        fasta1 = tmpdir / "seq1.fasta"
        fasta2 = tmpdir / "seq2.fasta"
        outfile = tmpdir / "needle.txt"

        # Write sequences to temporary FASTA files
        fasta1.write_text(f">seq1\n{seq1}\n")
        fasta2.write_text(f">seq2\n{seq2}\n")

        # Run EMBOSS Needle
        result = subprocess.run([
            "needle",
            "-asequence", str(fasta1),
            "-bsequence", str(fasta2),
            "-gapopen", "10",
            "-gapextend", "0.5",
            "-outfile", str(outfile),
            "-auto"
        ], capture_output=True, text=True)

        # Check for errors in the execution
        if result.returncode != 0:
            raise RuntimeError(f"Needle failed:\n{result.stderr}")

        # Read the output
        content = outfile.read_text()
        metrics = {}

        try:
            # Parse relevant metrics from the output
            for line in content.splitlines():
                line = line.strip()

                # Extract Identity
                if line.startswith("# Identity:"):
                    match = re.search(r"# Identity:\s+(\d+)\s*/\s*(\d+)\s*\(\s*([\d.]+)%\)", line)
                    if match:
                        matches, total, percent = match.groups()
                        metrics["identity_count"] = int(matches)
                        metrics["alignment_length"] = int(total)
                        metrics["identity_percentage"] = float(percent)

                # Extract Similarity
                elif line.startswith("# Similarity:"):
                    match = re.search(r"# Similarity:\s+\d+/\d+\s*\(\s*([\d.]+)%\)", line)
                    if match:
                        metrics["similarity_percentage"] = float(match.group(1))

                # Extract Gaps
                elif line.startswith("# Gaps:"):
                    match = re.search(r"# Gaps:\s+\d+/\d+\s*\(\s*([\d.]+)%\)", line)
                    if match:
                        metrics["gaps_percentage"] = float(match.group(1))

                # Extract Score
                elif line.startswith("# Score:"):
                    match = re.search(r"# Score:\s*([\d.]+)", line)
                    if match:
                        metrics["alignment_score"] = float(match.group(1))

            # Ensure identity percentage is parsed
            if "identity_percentage" not in metrics:
                raise ValueError(f"Unable to parse Needle output. Unexpected format:\n{content}")

            # Default to None if gaps_percentage or other values are missing
            metrics["gaps_percentage"] = metrics.get("gaps_percentage", None)

            return metrics

        except Exception as e:
            raise ValueError(f"Error parsing Needle output:\n{content}") from e

def get_descendant_ids(parent_ids):
    descendants_ids = []
    ncbi = NCBITaxa()
    for taxon in parent_ids:
        descendants = ncbi.get_descendant_taxa(taxon, intermediate_nodes=True)
        descendants_ids.extend(str(tid) for tid in descendants)
    descendants_ids.extend(str(tid) for tid in parent_ids)
    return descendants_ids