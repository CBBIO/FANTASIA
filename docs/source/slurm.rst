.. _fantasia_slurm_job:

Running FANTASIA via SLURM
===========================

This guide explains how to launch the FANTASIA tool through a SLURM job on an HPC environment. The process ensures proper directory configuration, leverages Singularity containers, and uses RAM-based PostgreSQL and in-memory loading for optimal performance.

See the documentation for Singularity [SingularityDocs]_ and SLURM [SLURMdocs]_, or consult the FANTASIA repository [FANTASIArepo]_.

Steps to Run FANTASIA on SLURM
------------------------------

1. Connect to the HPC Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Begin by connecting to the remote server or HPC environment using SSH:

.. code-block:: bash

    ssh user@server

2. Prepare the Job Script
~~~~~~~~~~~~~~~~~~~~~~~~~

There are **two ways** to prepare your working environment. Both will use the same `job.sh` script, but differ in how you retrieve it.

**Option A: Download and run `job.sh` from the main repository**

Use this if you haven’t cloned the repository yet:

.. code-block:: bash

    wget https://raw.githubusercontent.com/CBBIO/FANTASIA/main/job.sh
    bash job.sh

**Option B: Clone the repository manually and execute `job.sh` from there**

This method launches a default test job using a set of **four sequences**, which is ideal for debugging and validating the full process on your target machine:

.. code-block:: bash

    git clone https://github.com/CBBIO/FANTASIA.git
    cd FANTASIA
    bash job.sh

⚠️ **Important:** Make sure the environment variables `REPO_DIR` and `WORK_DIR` point to the **same directory** to avoid cloning the repository twice. The default values are:

.. code-block:: bash

    REPO_DIR="$HOME/FANTASIA"
    WORK_DIR="$HOME/FANTASIA"

3. What `job.sh` Actually Does
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `job.sh` script performs the following actions:

- Ensures the `FANTASIA` repository is present under ``$REPO_DIR``
- Builds the required Singularity (`.sif`) images from the Docker containers
- Loads the embeddings and metadata **directly into RAM**
- Mounts the PostgreSQL database in shared memory (`/dev/shm`)
- Submits the analysis job to SLURM using:

.. code-block:: bash

    sbatch job.sh

This means the actual analysis is performed in a SLURM job, **not during the execution of `job.sh` itself**.

4. Default Directory Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Code and containers: ``$HOME/FANTASIA``
- Output/results: ``$HOME/fantasia`` (note the lowercase)

These defaults can be modified inside `job.sh` to suit your own directory structure.

5. Database Configuration and Performance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FANTASIA mounts the internal PostgreSQL database under ``/dev/shm`` (shared memory), which provides faster I/O than disk-based storage.

However, **most of the heavy lifting is done in RAM**: embeddings and associated metadata are loaded entirely into memory at the beginning of the process, significantly improving performance during distance computations and annotation transfer.

The PostgreSQL configuration is also tuned for high-throughput performance, including:

- Increased buffer sizes
- Improved concurrency handling

6. Modifying Execution Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To customize how FANTASIA runs (number of workers, batch sizes, input sequences, etc.), you must edit the configuration file located in:

.. code-block:: bash

    config/config.yaml

Set the parameters according to your environment and job size.

**Default SLURM resources** (defined in `job.sh`) are as follows:

.. code-block:: bash

    #!/bin/bash
    #SBATCH --job-name=fantasia
    #SBATCH --output=fantasia_%j.log
    #SBATCH --error=fantasia_%j.err
    #SBATCH --partition=vision
    #SBATCH --gres=gpu:1
    #SBATCH --cpus-per-task=128
    #SBATCH --mem=128G
    #SBATCH --time=1-00:00:00
    #SBATCH --nodes=1
    #SBATCH --ntasks=1

Under this setup, the **recommended configuration** is:

- **80 workers**
- **Batch size of 200** for distance calculations
- **Batch size of 128** for embedding generation

Adjust these values in `config.yaml` depending on the available resources and the size of your dataset.


References
--------------------------------------------------

.. [FANTASIArepo] Ana Rojas Lab, *FANTASIA repository*, available at: https://github.com/CBBIO/FANTASIA.

.. [SingularityDocs] Sylabs, *Singularity Documentation*, available at: https://sylabs.io/guides/.

.. [SLURMdocs] SchedMD, *SLURM Workload Manager Documentation*, available at: https://slurm.schedmd.com/documentation.html.

