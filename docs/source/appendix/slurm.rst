HPC - SLURM Deployment
======================

FANTASIA provides several ready-to-use SLURM job scripts for deployment on high-performance computing (HPC) systems. These scripts are available in the `deployment/` folder of the repository and are tailored to different infrastructures and container runtimes.

To get started:

.. code-block:: bash

    git clone https://github.com/CBBIO/FANTASIA.git
    cd FANTASIA

Choose the appropriate script based on your computing environment (e.g., CESGA or Greisenwald) and the desired workload (single job or job array). Each script launches PostgreSQL, RabbitMQ, and the FANTASIA pipeline inside containers using either **Apptainer** or **Singularity**, depending on the cluster. You can use these scripts as a template to adapt the workflow to your own HPC environment.

.. note::

   Each script includes internal setup of containers, service initialization, and cleanup logic. You may customize them by editing execution parameters (e.g., input FASTA, number of neighbors, filtering options, output prefix, etc.).

Configuration File and Runtime Options
--------------------------------------

FANTASIA is primarily configured via a YAML file located at:

.. code-block:: bash

    config/config.yaml

This file allows you to specify key execution parameters, including:

- Input FASTA path
- Output prefix
- Number of nearest neighbors (`limit_per_entry`)
- Redundancy filters and taxonomy exclusions
- Resource usage parameters (batch size, workers, etc.)

.. note::

   Taxonomy filters are currently exact-ID based. If you want to exclude a full organism group
   for benchmarking, provide a manually curated list of the relevant taxonomy IDs instead of
   relying on descendant expansion.

.. note::

   The recent application-level updates do not change the overall Slurm or array-deployment
   strategy. The main changes are: explicit support for ``only_embedding: true``, clearer
   disabling of ``distance_threshold`` (for example ``false`` instead of legacy ``0``, while
   keeping backward compatibility), corrected and clarified taxonomy filtering behavior,
   recommendation to use decompressed FASTA files for full embedding and full-pipeline runs,
   and optional generation of ``query_index_mapping.csv`` for sequence-aware outputs.
   Existing deployment wrappers should therefore remain structurally valid, but may require
   small updates if they assume the previous threshold convention, gzipped FASTA inputs, or
   older output expectations.

.. note::

   This configuration file is automatically loaded during the execution of ``fantasia run``.

Alternatively, you may override specific parameters directly via the command-line interface (CLI) when invoking FANTASIA. Command-line arguments take precedence over values defined in the configuration file.

Example:

.. code-block:: bash

    fantasia run --input my_proteins.fasta --prefix TEST_RUN --redundancy_filter 1.0 --taxonomy_ids_to_exclude 9606,<additional_related_taxa> --k 5


HPC Deployment Examples
-----------------------

FANTASIA provides tailored SLURM job scripts for different HPC infrastructures. Each script adapts to the specific container engine (Apptainer or Singularity), available modules, storage layout, and GPU configuration.

CESGA (single job)
------------------

Runs **PostgreSQL**, **RabbitMQ**, and the **FANTASIA pipeline** within isolated **Apptainer** containers. All persistent data, cache directories, and outputs are mounted on the **LUSTRE** parallel file system.

📄 See: :doc:`/deployment/cesga_job`

CESGA (job array)
-----------------

Launches a SLURM array of jobs based on a tab-separated parameter file. Each task executes a separate FANTASIA run with GPU acceleration (A100) using **Apptainer**, with all caches and intermediate files stored under **LUSTRE**.

📄 See: :doc:`/deployment/cesga_array_job`

Greisenwald (single job)
------------------------

Designed for high-memory, long-running jobs. Uses **Singularity** as container runtime along with pre-loaded environment modules (e.g., ``gcc``, ``hdf5``, ``singularity``). All services (PostgreSQL, RabbitMQ) and containers are launched locally.

📄 See: :doc:`/deployment/greisenwald_job`

Greisenwald (job array)
-----------------------

Equivalent to the CESGA array version but adapted to **Greisenwald’s scheduler configuration** and Singularity runtime. It allows batch processing via SLURM arrays using HPC-appropriate paths and module setups.

📄 See: :doc:`/deployment/greisenwald_array_job`

All deployment scripts can also be found in the official repository:

🔗 https://github.com/CBBIO/FANTASIA/tree/main/deployment
