Single FANTASIA Job Execution on CESGA (GPU + Apptainer)
=========================================================

This section describes how to launch a **single, non-array job** of the FANTASIA pipeline on CESGA using SLURM with GPU acceleration and containerized execution via **Apptainer**.

This mode is suitable for individual experiments with one input file and fixed parameters.

SLURM Script Overview
---------------------

The job script performs the following actions:

- Launches **PostgreSQL with pgvector** via Apptainer
- Launches **RabbitMQ** via Apptainer
- Runs **FANTASIA** inside an Apptainer container with GPU support
- Mounts required volumes using **bind mounts**
- Uses **LUSTRE** for persistent storage of containers, models, and caches

SLURM Directives
----------------

.. code-block:: bash

   #SBATCH -p gpu
   #SBATCH --gres=gpu:a100:1
   #SBATCH -c 32
   #SBATCH --mem=64G
   #SBATCH -t 08:00:00
   #SBATCH --job-name=fantasia_job
   #SBATCH --output=fantasia_%j.out
   #SBATCH --error=fantasia_%j.err

Resources requested:

- **GPU**: 1 × A100
- **CPU**: 32 cores
- **Memory**: 64 GB
- **Walltime**: 8 hours

Input Parameters
----------------

For single-job execution, input parameters can be **hardcoded** within the script, or you can modify the configuration file.


Persistent Storage Paths (LUSTRE)
---------------------------------

The following environment variables are defined to cache models, containers, and intermediate files:

.. code-block:: bash

   export TRANSFORMERS_CACHE="/mnt/lustre/.../.transformers_cache"
   export HF_HOME="/mnt/lustre/.../.hf_cache"
   export UDOCKER_DIR="/mnt/lustre/.../.udocker_repo"
   export APPTAINER_CACHEDIR="/mnt/lustre/.../.singularity/cache"
   export APPTAINER_TMPDIR="/mnt/lustre/.../.singularity/tmp"
   export APPTAINER_LOCALCACHEDIR="/mnt/lustre/.../.singularity/local"
   export PIP_CACHE_DIR="/mnt/lustre/.../.pip_cache"

These should point to **your own directory** in LUSTRE.

Additional directories used during execution:

.. code-block:: bash

   PROJECT_DIR="$STORE/FANTASIA"
   EXECUTION_DIR="$STORE/fantasia"
   PGDATA_HOST="$STORE/pgdata_pgvector"
   RABBITMQ_DATA_DIR="$HOME/fantasia_rabbitmq"

Apptainer Container Setup
-------------------------

The following container images are used:

- ``fantasia_v4.1.2.sif`` — main pipeline (GPU-enabled)
- ``pgvector_pg18.sif`` — PostgreSQL 18 with the pgvector extension
- ``rabbitmq.sif`` — message broker

Images are built automatically if missing. The file names include the version,
so images cached by earlier releases (``fantasia.sif``, ``pgvector.sif``) are
not reused:

.. code-block:: bash

   apptainer build pgvector_pg18.sif docker://pgvector/pgvector:0.8.1-pg18
   apptainer build --disable-cache fantasia_v4.1.2.sif docker://cbbio/fantasia:v4.1.2

Reference dump
--------------

``fantasia initialize`` restores the reference into the job's database. The
packaged ``config.yaml`` leaves ``embeddings_url`` unset, so either set it in
``$CONFIG_FILE`` or skip ``initialize`` if the database is already loaded.
Download the dump once to shared storage and point to it, instead of letting
every job download several GB:

.. code-block:: yaml

   embeddings_url: "file:///path/on/lustre/BioData_Dec25_esm2_prott5_prostt5_ankh3_large_esm3c_Layer0.backup"

The dumps require PostgreSQL 18 client tools, which the FANTASIA 4.1.2 image
provides. If ``initialize`` fails, the script stops before ``fantasia run``.

Execution Phase
---------------

After services are launched in the background, the script waits until
PostgreSQL and RabbitMQ accept connections (up to 10 and 5 minutes), then
initializes and runs the pipeline:

.. code-block:: bash

   apptainer exec --nv --bind "$EXECUTION_DIR:/fantasia" "$FANTASIA_IMAGE" \
       fantasia initialize --config "$CONFIG_FILE" --base_directory /fantasia

   apptainer exec --nv --bind "$EXECUTION_DIR:/fantasia" "$FANTASIA_IMAGE" \
       fantasia run --config "$CONFIG_FILE" --base_directory /fantasia

- The ``--nv`` flag enables GPU passthrough.
- All outputs are written under ``$EXECUTION_DIR``.

Shutdown and Cleanup
--------------------

A ``cleanup`` function is registered with ``trap cleanup EXIT`` before the
services start, so it runs on any exit, including a failed ``initialize``. It
stops PostgreSQL and RabbitMQ by process ID (``SIGTERM``, then ``SIGKILL`` after
30 seconds) and does not affect other jobs on the node.

Launching the Job
-----------------

To launch the job:

.. code-block:: bash

   sbatch fantasia_single.sh

Where ``fantasia_single.sh`` is the name of your job script.

Log files are created as:

- ``fantasia_<jobid>.out`` — SLURM standard output
- ``fantasia_<jobid>.err`` — SLURM standard error
- ``postgres.log`` — PostgreSQL log
- ``rabbitmq.log`` — RabbitMQ log

Summary
-------

This job script enables fully reproducible, self-contained execution of the FANTASIA pipeline on CESGA's GPU nodes using Apptainer. No root access or external services are required. All data, services, and containers are managed within the node, ensuring high portability and reproducibility.
