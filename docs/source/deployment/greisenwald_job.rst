Single FANTASIA Job Execution on Greisenwald HPC (GPU + Singularity)
================================================================================================

This section describes how to launch a **single, non-array job** of the FANTASIA pipeline on the **Greisenwald HPC cluster**, using SLURM with GPU acceleration and containerized execution via **Singularity** (non-Apptainer).

This execution mode is suitable for standard, reproducible analyses with one input and full local service orchestration.

SLURM Script Overview
---------------------

This SLURM job script will:

- Load required modules (GCC, HDF5, Singularity)
- Clone the FANTASIA repository if missing
- Build the following Singularity images if not found locally:
  - `pgvector_pg18.sif`
  - `rabbitmq.sif`
  - `fantasia_v4.1.2.sif`
- Launch PostgreSQL with pgvector from container, it's also scaled vertically (Not needed).
- Launch RabbitMQ from container
- Initialize and run the FANTASIA pipeline
- Clean up temporary services and shared memory directories on exit

SLURM Directives
----------------

.. code-block:: bash

   #SBATCH --job-name=fantasia
   #SBATCH --output=fantasia_%j.log
   #SBATCH --error=fantasia_%j.err
   #SBATCH --partition=vision
   #SBATCH --gres=gpu:1
   #SBATCH --cpus-per-task=64
   #SBATCH --mem=128G
   #SBATCH --time=3-00:00:00
   #SBATCH --nodes=1
   #SBATCH --ntasks=1

Resources requested:

- **GPU**: 1 (any type)
- **CPU**: 64 cores
- **Memory**: 128 GB
- **Walltime**: 3 days

Persistent Paths and Variables
------------------------------

Environment variables and paths are set as follows:

.. code-block:: bash

   REPO_DIR="$HOME/FANTASIA"
   WORK_DIR="$HOME/FANTASIA"
   SHM_DIR="/tmp/fantasia_pgvector"
   DB_DIR="$SHM_DIR/data"
   DB_SOCKET="$SHM_DIR/socket"
   RABBIT_DIR="$HOME/fantasia_rabbitmq"
   FANTASIA_RUN_DIR="$HOME/fantasia"
   CONFIG_PATH="$REPO_DIR/fantasia/config.yaml"

Images are built to:

- ``$WORK_DIR/pgvector_pg18.sif``
- ``$WORK_DIR/rabbitmq.sif``
- ``$WORK_DIR/fantasia_v4.1.2.sif``

The file names include the version, so images cached by earlier releases are
not reused.

Container Build Commands
------------------------

If containers are missing, they are built with:

.. code-block:: bash

   singularity build pgvector_pg18.sif docker://pgvector/pgvector:0.8.1-pg18
   singularity build rabbitmq.sif docker://rabbitmq:management
   singularity build --disable-cache fantasia_v4.1.2.sif docker://cbbio/fantasia:v4.1.2

PostgreSQL Configuration
------------------------

A shared memory directory is prepared and PostgreSQL is started with:

.. code-block:: bash

   singularity exec "$PGVECTOR_SIF" initdb -D "$DB_DIR" --encoding=UTF8 --locale=C.UTF-8
   singularity exec "$PGVECTOR_SIF" postgres -D "$DB_DIR" ...

The script waits until PostgreSQL accepts connections on ``$PG_PORT`` before
creating the user and database with ``psql``, then applies the tuning settings
and restarts the server with ``pg_ctl restart``.

.. note::

   ``pg_ctl restart`` leaves PostgreSQL running outside the ``singularity exec``
   that started it. If the site mounts SIF images through FUSE (unprivileged
   Apptainer/Singularity), the image is unmounted when that command exits and
   PostgreSQL dies with ``Bus error``. Use a sandbox image instead:
   ``singularity build --sandbox pgvector_pg18 docker://pgvector/pgvector:0.8.1-pg18``
   and set ``PGVECTOR_SIF`` to that directory.

RabbitMQ Startup
----------------

RabbitMQ is started in background via:

.. code-block:: bash

   singularity exec --bind "$RABBIT_DIR:/var/lib/rabbitmq" "$RABBITMQ_SIF" rabbitmq-server &
   RABBITMQ_PID=$!

and the script waits until port 5672 accepts connections.

``fantasia initialize`` restores the reference only if ``embeddings_url`` is set
in ``$CONFIG_PATH`` (it is unset in the packaged configuration). Download the
dump once and use a ``file://`` URL, for example
``embeddings_url: "file:///home/user/BioData_Dec25_esm2_prott5_prostt5_ankh3_large_esm3c_Layer0.backup"``.
If ``initialize`` fails, the script stops before ``fantasia run``.

FANTASIA Pipeline Execution
---------------------------

The pipeline is executed with:

.. code-block:: bash

   singularity exec --nv --bind "$FANTASIA_RUN_DIR:/fantasia" "$FANTASIA_SIF" \
       fantasia initialize --config "$CONFIG_PATH" --base_directory /fantasia

   singularity exec --nv --bind "$FANTASIA_RUN_DIR:/fantasia" "$FANTASIA_SIF" \
       fantasia run --config "$CONFIG_PATH" --base_directory /fantasia

Cleanup
-------

``trap cleanup EXIT`` is registered before the services start, so the cleanup
runs on any exit, including a failed ``initialize``:

.. code-block:: bash

   singularity exec "$PGVECTOR_SIF" pg_ctl -D "$DB_DIR" stop -m fast -t 60
   stop_process RabbitMQ "$RABBITMQ_PID"   # SIGTERM, then SIGKILL after 30 s
   rm -rf "$SHM_DIR"

Only this job's services are stopped; other jobs on the node are not affected.

Logs
----

- SLURM output: ``fantasia_<jobid>.log``
- SLURM error: ``fantasia_<jobid>.err``
