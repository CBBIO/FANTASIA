import os
import requests
import subprocess
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
    print("Loading dump into the database...")

    # Configura las variables de entorno necesarias para pg_restore
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
            env=env  # Usa las variables de entorno configuradas
        )
        stdout, stderr = process.communicate()

        if process.returncode == 0:
            print("Database dump loaded successfully.")
        else:
            print(f"Error while loading dump: {stderr}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
