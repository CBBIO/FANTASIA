import os

import requests
import subprocess

from ete3 import NCBITaxa
from sqlalchemy import text
from tqdm import tqdm

import parasail


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


def run_needle_from_strings(seq1, seq2):
    """
    Alinea dos secuencias con Parasail (global alignment) y extrae métricas estilo EMBOSS.
    """
    result = parasail.nw_trace_striped_32(seq1, seq2, 10, 1, parasail.blosum62)

    aligned_query = result.traceback.query
    aligned_ref = result.traceback.ref
    comp_line = result.traceback.comp

    alignment_length = len(aligned_query)
    matches = sum(a == b for a, b in zip(aligned_query, aligned_ref) if a != '-' and b != '-')
    similarity = sum(c in "|:" for c in comp_line)
    gaps = aligned_query.count('-') + aligned_ref.count('-')

    metrics = {
        "identity_count": matches,
        "alignment_length": alignment_length,
        "identity_percentage": 100 * matches / alignment_length,
        "similarity_percentage": 100 * similarity / alignment_length,
        "gaps_percentage": 100 * gaps / alignment_length,
        "alignment_score": result.score,
    }

    return metrics


def get_descendant_ids(parent_ids):
    descendants_ids = []
    ncbi = NCBITaxa()
    for taxon in parent_ids:
        descendants = ncbi.get_descendant_taxa(taxon, intermediate_nodes=True)
        descendants_ids.extend(descendants)
    descendants_ids.extend(parent_ids)
    return list(set(descendants_ids))  # Evita duplicados
