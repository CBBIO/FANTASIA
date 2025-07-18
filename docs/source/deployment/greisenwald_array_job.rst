FANTASIA Array Job Execution on Greisenwald HPC (GPU + Singularity)
====================================================================

This section explains how to execute **multiple FANTASIA jobs in parallel** on the **Greisenwald HPC system**, using **SLURM job arrays** and GPU acceleration. All services and containers (PostgreSQL, RabbitMQ, FANTASIA) are launched locally per job using **Singularity**, ensuring isolation and reproducibility.

This method is ideal for batch annotation of multiple input FASTA files, each with independent configuration.

Job Script Summary
------------------

The script performs the following for each job in the array:

- Reads parameters (FASTA file, output prefix, extra args) from a tab-separated input list
- Builds containers if missing
- Initializes PostgreSQL with `pgvector` extension
- Launches RabbitMQ
- Runs `fantasia initialize` and `fantasia run` with job-specific parameters
- Performs automatic cleanup

SLURM Directives
----------------

.. code-block:: bash

    #SBATCH --job-name=fantasia
    #SBATCH --output=fantasia_%A_%a.out
    #SBATCH --error=fantasia_%A_%a.err
    #SBATCH --partition=vision
    #SBATCH --gres=gpu:1
    #SBATCH --cpus-per-task=64
    #SBATCH --mem=128G
    #SBATCH --time=3-00:00:00
    #SBATCH --array=1-8%1

Explanation:

- `--array=1-8`: launch jobs for lines 1 to 8 in the input list
- `%1`: limit to **1 concurrent job** (adjust depending on GPU availability)

Parameter File Format
---------------------

The job array reads from a plain-text, tab-separated file located at:

.. code-block:: bash

    PARAM_FILE="$HOME/fantasia_input_list.txt"

Each line must contain:

.. code-block:: text

    <input_fasta>    <output_prefix>    <extra_arguments>

Example:

.. code-block:: text

    proteomes_uniprot/UP000000589.fasta    MOUSE_nr100_k1    --redundancy_filter 1.0 --taxonomy_ids_to_exclude 10090,9606

Containers and Paths
--------------------

Paths and image definitions:

.. code-block:: bash

    REPO_DIR="$HOME/FANTASIA"
    WORK_DIR="$HOME/FANTASIA"
    SHM_DIR="/tmp/fantasia_pgvector_\${SLURM_ARRAY_TASK_ID}"
    DB_DIR="$SHM_DIR/data"
    DB_SOCKET="$SHM_DIR/socket"
    RABBIT_DIR="$HOME/fantasia_rabbitmq_\${SLURM_ARRAY_TASK_ID}"
    FANTASIA_RUN_DIR="$HOME/fantasia_\${SLURM_ARRAY_TASK_ID}"

    PGVECTOR_SIF="$WORK_DIR/pgvector.sif"
    RABBITMQ_SIF="$WORK_DIR/rabbitmq.sif"
    FANTASIA_SIF="$WORK_DIR/fantasia.sif"

If missing, containers are built from Docker images.

Dynamic Parameter Parsing
-------------------------

Each job extracts its corresponding line from the parameter file using:

.. code-block:: bash

    LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$PARAM_FILE")
    INPUT=$(echo "$LINE" | cut -f1)
    OUTPUT=$(echo "$LINE" | cut -f2)
    EXTRA=$(echo "$LINE" | cut -f3-)

This allows job-specific execution.

Execution Phase
---------------

PostgreSQL and RabbitMQ are launched with dedicated folders per job. Then:

.. code-block:: bash

    singularity exec --nv --bind "$FANTASIA_RUN_DIR:/fantasia" "$FANTASIA_SIF" \
        fantasia initialize

    singularity exec --nv --bind "$FANTASIA_RUN_DIR:/fantasia" "$FANTASIA_SIF" \
        fantasia run --input "$INPUT" --prefix "$OUTPUT" $EXTRA

Cleanup
-------

The script defines a `cleanup()` function to:

- Kill PostgreSQL and RabbitMQ
- Remove temporary folders

Registered using `trap EXIT`.

Launching the Job
-----------------

Submit the array job with:

.. code-block:: bash

    sbatch greisenwald_array.sh

Logs for each task will be stored as:

- `fantasia_<arrayjobid>_<taskid>.out`
- `fantasia_<arrayjobid>_<taskid>.err`

