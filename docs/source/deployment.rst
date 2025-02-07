=======================
Local Deployment Guide
=======================

This guide provides a step-by-step process for deploying **FANTASIA** locally.

Prerequisites
=============

Ensure you have the following dependencies installed:

- **Operating System**: Linux (Ubuntu recommended)
- **Python**: Version 3.10 or higher
- **Poetry**: Installed for dependency management:

  .. code-block:: bash

     pip install poetry

- **Docker**: Installed and running. If not installed, follow the `Docker installation guide <https://docs.docker.com/get-docker/>`_.
- **NVIDIA Driver**: Version 550.120 or newer (verify using ``nvidia-smi``).
- **CUDA**: Version 12.4 or newer (verify using ``nvcc --version``).

Cloning the Repository
======================

Clone the repository and navigate into the project directory:

.. code-block:: bash

   git clone https://github.com/CBBIO/FANTASIA.git
   cd FANTASIA

Creating and Activating the Virtual Environment
===============================================

Use `poetry` to manage the virtual environment:

.. code-block:: bash

   poetry install
   poetry shell

Starting Required Services
==========================

Ensure PostgreSQL and RabbitMQ services are running.

**Start PostgreSQL with pgvector:**

.. code-block:: bash

   docker run -d --name pgvectorsql \
       -e POSTGRES_USER=usuario \
       -e POSTGRES_PASSWORD=clave \
       -e POSTGRES_DB=BioData \
       -p 5432:5432 \
       pgvector/pgvector:pg16

**Start RabbitMQ:**

.. code-block:: bash

   docker run -d --name rabbitmq \
       -p 15672:15672 \
       -p 5672:5672 \
       rabbitmq:management

Access the RabbitMQ management interface at `http://localhost:15672 <http://localhost:15672>`_ (default credentials: ``guest/guest``).

Configuration
=============

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
==============

Download embeddings and initialize the database:

.. code-block:: bash

   python fantasia/main.py initialize --config ./fantasia/config.yaml

Verify that the embeddings are loaded into:

- The directory specified in `embeddings_path`.
- The configured PostgreSQL database.

Running the Pipeline
====================

Prepare an input FASTA file:

.. code-block:: bash

   mkdir -p ~/fantasia/input
   cp ./data_sample/worm_test.fasta ~/fantasia/input/worm_test.fasta

Run the pipeline:

.. code-block:: bash

   python fantasia/main.py run \
     --config ./fantasia/config.yaml \
     --fasta ./data_sample/worm_test.fasta \
     --prefix test_run \
     --length_filter 3000 \
     --redundancy_filter 0 \
     --max_workers 1 \
     --models esm,prot \
     --distance_threshold esm:1,prot:1 \
     --batch_size esm:32,prot:64 \
     --sequence_queue_package 100

Output Files
============

The pipeline outputs results as CSV files with the following naming format:

.. code-block:: bash

   <prefix>_<YYYYMMDD>.csv

- **prefix**: Set in ``config.yaml`` under ``fantasia_prefix``. If not provided, it defaults to ``"default"``. The ``--prefix`` argument overrides this setting.
- **YYYYMMDD**: The execution date.


Examples
--------

- If `fantasia_prefix` in `config.yaml` is:

  .. code-block:: yaml

     fantasia_prefix: worm_test_Prot_100_1.2

  The output file will be:

  .. code-block:: bash

     worm_test_Prot_100_1.2_20250206.csv  # (if executed on February 6, 2025)

- If `--prefix test_run` is passed in the command, the output will be:

  .. code-block:: bash

     test_run_20250206.csv

- If no prefix is set in `config.yaml` and `--prefix` is not provided, the default name is used:

  .. code-block:: bash

     default_20250206.csv

Storage Location
----------------

By default, CSV files are saved in:

.. code-block:: yaml

   directories:
     csv_outputs: results

This means the output files are stored in:

.. code-block:: bash

   ~/fantasia/results/

To change the output directory, modify `csv_outputs` in `config.yaml`.
