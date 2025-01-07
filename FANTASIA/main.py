import os
import requests
import subprocess
from datetime import datetime
from tqdm import tqdm
from protein_metamorphisms_is.helpers.config.yaml import read_yaml_config
from FANTASIA.embedder import SequenceEmbedder
from FANTASIA.lookup import EmbeddingLookUp


def download_embeddings_once(url, tar_path):
    """
    Download the embeddings TAR file from the given URL if it doesn't already exist
    or if the file is incomplete, with a progress bar.

    Parameters
    ----------
    url : str
        The URL to download the embeddings from.
    tar_path : str
        Path where the TAR file will be saved.
    """
    # Check if the file already exists and is complete
    temp_file = tar_path + ".part"
    response = requests.head(url, allow_redirects=True)
    total_size = int(response.headers.get('content-length', 0))

    if os.path.exists(tar_path):
        if os.path.getsize(tar_path) == total_size:
            print("Embeddings file already exists and is complete. Skipping download.")
            return
        else:
            print("Embeddings file exists but is incomplete. Restarting download.")
            os.remove(tar_path)

    if os.path.exists(temp_file):
        downloaded_size = os.path.getsize(temp_file)
        print(f"Resuming download from {downloaded_size} bytes...")
    else:
        downloaded_size = 0

    print("Downloading embeddings...")
    headers = {"Range": f"bytes={downloaded_size}-"}
    response = requests.get(url, headers=headers, stream=True)

    mode = "ab" if downloaded_size > 0 else "wb"
    with open(temp_file, mode) as f, tqdm(
        desc="Downloading",
        total=total_size,
        initial=downloaded_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            progress_bar.update(len(chunk))

    # Rename the temp file to the final file
    os.rename(temp_file, tar_path)
    print(f"Embeddings downloaded successfully to {tar_path}.")


def load_dump_to_db_once(dump_path, db_config):
    """
    Load a SQL dump file into the database if it hasn't been loaded yet.

    Parameters
    ----------
    dump_path : str
        Path to the TAR file containing the SQL dump.
    db_config : dict
        Database configuration dictionary containing host, port, user, password, and db name.
    """
    flag_path = dump_path + ".loaded"

    if os.path.exists(flag_path):
        print("Database dump already loaded. Skipping this step.")
        return

    print("Loading dump into the database...")
    command = [
        "psql",
        f"-h{db_config['DB_HOST']}",
        f"-p{db_config['DB_PORT']}",
        f"-U{db_config['DB_USERNAME']}",
        "-d", db_config["DB_NAME"],
        "-f", dump_path
    ]
    env = os.environ.copy()
    env["PGPASSWORD"] = db_config["DB_PASSWORD"]

    subprocess.run(command, env=env, check=True)

    # Create a flag file to indicate that the dump has been loaded
    with open(flag_path, "w") as flag_file:
        flag_file.write(f"Dump loaded on {datetime.now()}\n")
    print("Database dump loaded successfully.")


def main(config_path, fasta_path=None):
    # Read configuration
    conf = read_yaml_config(config_path)

    # Ensure embeddings are downloaded and the database is loaded
    embeddings_dir = os.path.expanduser(conf["embeddings_path"])
    os.makedirs(embeddings_dir, exist_ok=True)  # Ensure directory exists

    tar_path = os.path.join(embeddings_dir, "embeddings.tar")  # Full path to the TAR file
    download_embeddings_once(conf["embeddings_url"], tar_path)
    load_dump_to_db_once(tar_path, conf)

    # Update the FASTA path if provided
    if fasta_path:
        conf["fantasia_input_fasta"] = fasta_path

    # FANTASIA pipeline
    current_date = datetime.now().strftime("%Y%m%d%H%M%S")
    embedder = SequenceEmbedder(conf, current_date)
    embedder.start()

    lookup = EmbeddingLookUp(conf, current_date)
    lookup.start()



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the pipeline with a specified configuration file.")
    parser.add_argument("--config", type=str, required=False, help="Path to the configuration YAML file.")
    parser.add_argument("--fasta", type=str, required=False, help="Path to the input FASTA file.")
    args = parser.parse_args()

    config_path = args.config if args.config else './config.yaml'
    fasta_path = args.fasta

    main(config_path=config_path, fasta_path=fasta_path)
