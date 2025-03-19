.. _fantasia_local_deployment:

=========================
FANTASIA Deployment Guide
=========================

This guide provides a step-by-step process for deploying **FANTASIA** locally.

Prerequisites
=============

Before proceeding, ensure you have the following dependencies installed:

System Requirements
-------------------
- **Operating System**: Linux (Ubuntu recommended)
- **Python**: Version 3.10 or higher
- **Docker**: Installed and running.
  If not installed, follow the `Docker installation guide <https://docs.docker.com/get-docker/>`_ and the `post-installation steps <https://docs.docker.com/engine/install/linux-postinstall/>`_ to run Docker without `sudo`.

Machine Learning Dependencies
-----------------------------
- **NVIDIA Driver**: Version 550.120 or newer (verify using ``nvidia-smi``).
- **CUDA**: Version 12.4 or newer (verify using ``nvcc --version``).

Database Dependencies
---------------------
- **PostgreSQL Client**: Version 16 or later, required to restore database backups without compatibility issues.

  .. warning:: 🚨 **Important for Ubuntu 22.04 and older** 🚨

     PostgreSQL 16 is **not available** in the default repositories for Ubuntu **22.04 and earlier**.
     If you try to restore a backup using `pg_restore`, you may encounter **incompatibility issues**.

Python Environment
------------------
- **Poetry**: Used for dependency management.

  .. code-block:: bash

     pip install poetry

------------------------------
Cloning the Repository
------------------------------

Clone the repository and navigate into the project directory:

.. code-block:: bash

   git clone https://github.com/CBBIO/FANTASIA.git
   cd FANTASIA

----------------------------------------------
Creating and Activating the Virtual Environment
----------------------------------------------

Use `poetry` to manage the virtual environment. Follow these steps:

1. **Ensure Poetry is installed and up to date:**

   .. code-block:: bash

      poetry self update

2. **If using Poetry 1.5 or later, install the required shell plugin:**

   .. code-block:: bash

      poetry self add poetry-plugin-shell

3. **Create and activate the virtual environment:**

   .. code-block:: bash

      poetry env use <python_version>  # Specify the desired Python version (e.g., 3.12)
      poetry install
      poetry env activate

.. note::

   If using Conda, avoid managing environments with both Poetry and Conda simultaneously to prevent dependency conflicts.

   We recommend using PyCharm for development due to its seamless integration with Poetry, making environment management and package handling more intuitive.


Starting Required Services
====================================

Ensure **PostgreSQL** and **RabbitMQ** services are running.

-----------------------------------
Starting PostgreSQL with pgvector
-----------------------------------

.. code-block:: bash

    docker run -d --name pgvectorsql \
        --shm-size=64g \
        -e POSTGRES_USER=usuario \
        -e POSTGRES_PASSWORD=clave \
        -e POSTGRES_DB=BioData \
        -p 5432:5432 \
        pgvector/pgvector:pg16 \
        -c shared_buffers=16GB \
        -c effective_cache_size=32GB \
        -c work_mem=64MB

PostgreSQL Configuration
------------------------

The configuration parameters provided above have been **optimized for a machine with 128GB of RAM and 32 CPU cores**, allowing **up to 20 concurrent workers**. These settings enhance PostgreSQL's performance when handling large datasets and computationally intensive queries.

- ``--shm-size=64g``:
  Allocates **64GB of shared memory** to the container, preventing PostgreSQL from running out of memory in high-performance environments.

- ``-c shared_buffers=16GB``:
  Allocates **16GB of RAM** for PostgreSQL's shared memory buffers. This should typically be **25-40% of total system memory**.

- ``-c effective_cache_size=32GB``:
  Sets PostgreSQL's **estimated available memory** for disk caching to **32GB**. This helps the query planner make better decisions.

- ``-c work_mem=64MB``:
  Defines **64MB of memory per worker** for operations like sorting and hashing. This is crucial when handling **parallel query execution**.

Adjusting the Configuration
---------------------------

These parameters should be adjusted based on **available hardware** and **workload requirements**:

- If running on a machine with **less RAM**, decrease ``shared_buffers`` and ``effective_cache_size`` proportionally.
- If running on a machine with **fewer CPU cores**, reduce the number of workers accordingly.
- For large parallel queries, increasing ``work_mem`` can improve performance, but setting it too high may exhaust memory.

For more details on PostgreSQL performance tuning, refer to the official guide:
`<https://www.postgresql.org/docs/current/runtime-config-resource.html>`_.

---------------------------------
Starting RabbitMQ
---------------------------------

.. code-block:: bash

   docker run -d --name rabbitmq \
       -p 15672:15672 \
       -p 5672:5672 \
       rabbitmq:management

You can access the RabbitMQ management interface at:
`http://localhost:15672 <http://localhost:15672>`_
(Default credentials: ``guest/guest``).


Configuration
==================================

Before proceeding, create the necessary directories with proper permissions:

.. code-block:: bash

   mkdir -p ~/fantasia/dumps ~/fantasia/embeddings ~/fantasia/results ~/fantasia/redundancy
   chmod -R 755 ~/fantasia

Ensure the following parameters are correctly set in `fantasia/config.yaml`:

.. code-block:: yaml

   DB_USERNAME: usuario
   DB_PASSWORD: clave
   DB_HOST: pgvectorsql
   DB_PORT: 5432
   DB_NAME: BioData

   rabbitmq_host: rabbitmq
   rabbitmq_user: guest
   rabbitmq_password: guest


Initialization
==================================

Download embeddings and initialize the database:

.. code-block:: bash

   python fantasia/main.py initialize --config ./fantasia/config.yaml

Verify that the embeddings are loaded into:

- The directory specified in `base_directory`.
- The configured PostgreSQL database.


Running the Pipeline
==================================

.. code-block:: bash

   python fantasia/main.py run


Arguments
---------

- ``--fasta``: Specifies the input FASTA file containing protein sequences to process. The path is relative to the mounted directory inside the container.
- ``--prefix``: Sets a prefix for output files. This helps organize results and logs for different runs.
- ``--length_filter``: Filters out sequences longer than the specified length (in this case, 50,000,000 base pairs). Sequences exceeding this length will be ignored.
- ``--redundancy_filter``: Specifies the redundancy threshold (0.0 in this case). Sequences with redundancy above this threshold will be excluded.
- ``--sequence_queue_package``: Determines the size of sequence batches (1000 sequences per package). This controls how many sequences are processed in each batch.
- ``--esm``, ``--prost``, ``--prot``: Enables different processing modes or models in the pipeline. These flags activate specific embedding models (ESM, ProstT5, and ProtT5, respectively).
- ``--limit_per_entry``: It is a number and sets the max number of proteins exhibiting the closest embedding distance. It is equivalent to the "k" parameter in the original GopredSim implementation for euclidean distance. Default should be "1". 
- ``--distance_threshold``: Sets thresholds for distances across different embedding types. IMPORTANT: this value cannot be "0" even if distance is not the selected criteria. The format is a comma-separated list of ``embedding_type:threshold`` pairs. For example, ``esm:1.2,prot:0.7,prost:0.7`` sets distance thresholds.
- ``--batch_size``: Specifies batch sizes for different embedding types. The format is a comma-separated list of ``embedding_type:size`` pairs. For example, ``esm:32,prot:32,prost:32`` sets batch sizes.
- ``--device``: Specifies the device to use for computation. Options are ``cuda`` (for GPU acceleration) or ``cpu`` (for CPU-only execution). Default is ``cuda`` if available.
- ``--base_directory``: Specifies the base directory where all experiments, results, and execution parameters will be stored. This is the root location for organizing output files and logs.



