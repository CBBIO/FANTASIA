Single FANTASIA Job Execution on Greisenwald HPC (GPU + Singularity)
=================================================================

This section describes how to launch a **single, non-array job** of the FANTASIA pipeline on the **Greisenwald HPC cluster**, using SLURM with GPU acceleration and containerized execution via **Singularity** (non-Apptainer).

This execution mode is suitable for standard, reproducible analyses with one input and full local service orchestration.

SLURM Script Overview
---------------------

This SLURM job script will:

- Load required modules (GCC, HDF5, Singularity)
- Clone the FANTASIA repository if missing
- Build the following Singularity images if not found locally:
  - `pgvector.sif`
  - `rabbitmq.sif`
  - `fantasia.sif`
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

- ``$WORK_DIR/pgvector.sif``
- ``$WORK_DIR/rabbitmq.sif``
- ``$WORK_DIR/fantasia.sif``

Container Build Commands
------------------------

If containers are missing, they are built with:

.. code-block:: bash

   singularity build pgvector.sif docker://pgvector/pgvector:pg16
   singularity build rabbitmq.sif docker://rabbitmq:management
   singularity build --disable-cache fantasia.sif docker://frapercan/fantasia:latest

PostgreSQL Configuration
------------------------

A shared memory directory is prepared and PostgreSQL is started with:

.. code-block:: bash

   singularity exec "$PGVECTOR_SIF" initdb -D "$DB_DIR"
   singularity exec "$PGVECTOR_SIF" postgres -D "$DB_DIR" ...

User and database are created manually via `psql` calls.

RabbitMQ Startup
----------------

RabbitMQ is started in background via:

.. code-block:: bash

   singularity exec --bind "$RABBIT_DIR:/var/lib/rabbitmq" "$RABBITMQ_SIF" rabbitmq-server &

FANTASIA Pipeline Execution
---------------------------

The pipeline is executed with:

.. code-block:: bash

   singularity exec --nv --bind "$FANTASIA_RUN_DIR:/fantasia" "$FANTASIA_SIF" \
       fantasia initialize

   singularity exec --nv --bind "$FANTASIA_RUN_DIR:/fantasia" "$FANTASIA_SIF" \
       fantasia run

Cleanup
-------

Upon exit, a `trap cleanup EXIT` command ensures graceful termination:

.. code-block:: bash

   pkill -f "$DB_DIR"
   pkill -f "rabbitmq-server"
   rm -rf "$SHM_DIR"

Logs
----

- SLURM output: ``fantasia_<jobid>.log``
- SLURM error: ``fantasia_<jobid>.err``


