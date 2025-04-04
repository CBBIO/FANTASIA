import os

import pika
import requests
import subprocess
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from protein_metamorphisms_is.sql.base.database_manager import DatabaseManager
from tqdm import tqdm


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
