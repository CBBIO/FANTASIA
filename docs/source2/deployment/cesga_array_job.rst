Running FANTASIA at CESGA (SLURM + Apptainer + GPU)
===================================================

This section describes how to execute the full **FANTASIA functional annotation pipeline** on the **CESGA supercomputer** using SLURM job arrays and GPU acceleration. All components (PostgreSQL, RabbitMQ, FANTASIA) are isolated via **Apptainer containers**, and run locally on each node without external dependencies.

Job script overview
-------------------

The recommended SLURM script (``cesga_array.sh``) is designed to:

- Read tab-separated parameters from a file (input FASTA, experiment name, and extra options)
- Set up all persistent paths on LUSTRE
- Launch services (PostgreSQL + RabbitMQ) inside containers
- Run ``fantasia initialize`` and ``fantasia run`` inside the GPU container
- Perform automated cleanup on exit

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

Requesting:

- **GPU**: A100
- **CPU**: 32 cores
- **Memory**: 64 GB
- **Time**: 8 hours

Parameter file format
---------------------

The script expects an input file (e.g., ``fantasia_input_list.txt``) with **tab-separated** fields:

.. code-block:: text

   <input_fasta>    <output_prefix>    <extra_arguments>

Example:

.. code-block:: text

   proteomes_uniprot/UP000000589.fasta    MOUSE_nr100_k1    --redundancy_filter 1.0 --taxonomy_ids_to_exclude 10090,9606

This file must be located at ``$HOME/fantasia_input_list.txt`` (can be customized).

Launching a job array
---------------------

Submit the job array with:

.. code-block:: bash

   sbatch --array=1-8%1 cesga_array.sh

Explanation:

- ``--array=1-8``: launches 8 jobs corresponding to lines 1–8 of the parameter file
- ``%1``: limits execution to 1 job at a time

Persistent Storage Paths (LUSTRE)
---------------------------------

These paths must be defined according to your CESGA environment:

.. code-block:: bash

   LUSTRE_BASE="/mnt/lustre/scratch/nlsas/home/csic/cbb/fpc"

   export TRANSFORMERS_CACHE="$LUSTRE_BASE/.transformers_cache"
   export HF_HOME="$LUSTRE_BASE/.hf_cache"
   export UDOCKER_DIR="$LUSTRE_BASE/.udocker_repo"
   export APPTAINER_CACHEDIR="$LUSTRE_BASE/.singularity/cache"
   export APPTAINER_TMPDIR="$LUSTRE_BASE/.singularity/tmp"
   export APPTAINER_LOCALCACHEDIR="$LUSTRE_BASE/.singularity/local"
   export PIP_CACHE_DIR="$LUSTRE_BASE/.pip_cache"

Execution paths and services
----------------------------

Containers used:

- ``fantasia.sif`` — main pipeline
- ``pgvector.sif`` — PostgreSQL with pgvector
- ``rabbitmq.sif`` — RabbitMQ broker

Services are launched in the background:

.. code-block:: bash

   apptainer run \
     --env POSTGRES_USER=usuario \
     --env POSTGRES_PASSWORD=clave \
     --env POSTGRES_DB=BioData \
     --bind $PGDATA_HOST:/var/lib/postgresql/data \
     --bind $PG_RUN:/var/run/postgresql \
     $PGVECTOR_IMAGE

   apptainer run \
     --bind $RABBITMQ_DATA_DIR:/var/lib/rabbitmq \
     $RABBITMQ_IMAGE

Pipeline execution
------------------

After a short delay, the pipeline is launched:

.. code-block:: bash

   apptainer exec --nv --bind "$EXECUTION_DIR:/fantasia" "$FANTASIA_IMAGE" \
       fantasia initialize

   apptainer exec --nv --bind "$EXECUTION_DIR:/fantasia" "$FANTASIA_IMAGE" \
       fantasia run --input "$INPUT" --prefix "$OUTPUT" $EXTRA

Cleanup
-------

Upon completion, background services are stopped and temporary files are removed:

.. code-block:: bash

   pkill -f "$POSTGRESQL_DATA"
   pkill -f "rabbitmq-server"
   rm -rf "$SHARED_MEM_DIR"

Logs
----

- SLURM stdout: ``fantasia_<jobid>.out``
- SLURM stderr: ``fantasia_<jobid>.err``
- PostgreSQL log: ``postgres.log``
- RabbitMQ log: ``rabbitmq.log``
