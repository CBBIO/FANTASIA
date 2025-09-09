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
   SHARED_MEM_DIR="/tmp/fantasia"
   POSTGRESQL_DATA="$SHARED_MEM_DIR/data"
   POSTGRESQL_SOCKET="$SHARED_MEM_DIR/socket"
   RABBITMQ_DATA_DIR="$STORE/fantasia_rabbitmq"

Apptainer Container Setup
-------------------------

The following container images are used:

- ``fantasia.sif`` — main pipeline (GPU-enabled)
- ``pgvector.sif`` — PostgreSQL database with pgvector extension
- ``rabbitmq.sif`` — message broker

Images are built automatically if missing:

.. code-block:: bash

   apptainer build fantasia.sif docker://frapercan/fantasia:latest

Execution Phase
---------------

After services are launched in the background, the FANTASIA pipeline is initialized and executed:

.. code-block:: bash

   apptainer exec --nv --bind "$EXECUTION_DIR:/fantasia" "$FANTASIA_IMAGE" \
       fantasia initialize

   apptainer exec --nv --bind "$EXECUTION_DIR:/fantasia" "$FANTASIA_IMAGE" \
       fantasia run

- The ``--nv`` flag enables GPU passthrough.
- All outputs are written under ``$EXECUTION_DIR``.

Shutdown and Cleanup
--------------------

A cleanup routine is executed at the end to terminate services and remove temporary data:

.. code-block:: bash

   pkill -f "rabbitmq-server"
   pkill -f "$POSTGRESQL_DATA"
   rm -rf "$SHARED_MEM_DIR"

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

