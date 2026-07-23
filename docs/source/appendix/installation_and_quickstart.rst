===========================
Installation and Quickstart
===========================

This page describes a complete local installation: Python environment,
database, message broker, reference restore, GPU/CPU selection, and the first
run. For the shortest runnable path, start with `Quickstart moved <../quickstart.rst>`_.

What you’ll set up
==================

- PostgreSQL with the ``pgvector`` extension (Docker)
- RabbitMQ message broker (Docker)
- External tool: **MMseqs2**
- Python environment managed with **Poetry**
- (Optional) GPU support: NVIDIA driver + CUDA Toolkit

Prerequisites
=============

System Requirements
-------------------
- **OS**: Linux (Ubuntu recommended)
- **Python**: 3.12 (project metadata constrains runtime support to ``>=3.12,<3.13``)
- **Docker**: installed and running (configured for non-root use)

External Tools
--------------

MMseqs2 (optional query-aware redundancy masking and clustering)::

   sudo apt-get update
   sudo apt-get install mmseqs2

.. note::
   ``parasail`` is used for alignment-based post-processing through its Python package.
   When FANTASIA is installed through its declared Python dependencies, no separate
   ``parasail`` command-line installation is required.

PostgreSQL client (host-side, v16)
----------------------------------
Needed to load dumps from the **host** into the containerized database.

Ubuntu/Debian::

   sudo apt-get update
   sudo apt-get install postgresql-client-16
   psql --version  # verify major version is 16

Poetry (host)
-------------
Official installer script::

   curl -sSL https://install.python-poetry.org | python3 -
   export PATH="$HOME/.local/bin:$PATH"   # add Poetry to PATH (Linux shells)
   poetry --version

GPU (optional)
--------------
- **NVIDIA Driver**: 550.120 or newer (check with ``nvidia-smi``)
- **CUDA Toolkit**: 12.4 or newer (check with ``nvcc --version``)

1) Clone the repository
=======================

.. code-block:: bash

   git clone https://github.com/CBBIO/FANTASIA.git
   cd FANTASIA

2) Install the environment (Poetry)
===================================

.. code-block:: bash

   poetry install

After installation, the ``fantasia`` CLI entrypoint is available within the Poetry
environment. You can open a Poetry shell (``poetry shell``) or prefix commands with
``poetry run``. Examples below assume the CLI is directly available.

.. important::

   The full FANTASIA pipeline accepts both plain and gzip-compressed FASTA
   inputs. Files ending in ``.gz`` or ``.gzip`` are decompressed on the fly
   during embedding and do not require manual preparation.

2b) Alternative: install as a package (``pip``)
===============================================

.. code-block:: bash

   pip3 install fantasia

Then provide your own configuration so that it **resolves a correct ``constants.yaml``**.
Use the repository as a reference for the expected configuration layout and defaults.

3) Start required services (Docker)
===================================

PostgreSQL with ``pgvector``::

   docker run -d --name pgvectorsql \
       -e POSTGRES_USER=usuario \
       -e POSTGRES_PASSWORD=clave \
       -e POSTGRES_DB=BioData \
       -p 5432:5432 \
       pgvector/pgvector:pg16

RabbitMQ (with management UI)::

   docker run -d --name rabbitmq \
       -p 15672:15672 \
       -p 5672:5672 \
       rabbitmq:management

RabbitMQ UI: ``http://localhost:15672`` (default credentials: ``guest/guest``).

4) Configure FANTASIA
=====================

Create the example input and working directories. They are not part of the Git
repository::

   mkdir -p data lookup/{logs,experiments,embeddings}

Minimal settings in ``fantasia/config.yaml``:

.. code-block:: yaml

   DB_USERNAME: usuario
   DB_PASSWORD: clave
   DB_HOST: localhost
   DB_PORT: 5432
   DB_NAME: BioData

   rabbitmq_host: localhost
   rabbitmq_user: guest
   rabbitmq_password: guest

   embedding:
     device: cuda

   lookup:
     use_gpu: true

For CPU-only deployments, set ``embedding.device: cpu`` and
``lookup.use_gpu: false`` before running the pipeline.

.. note::
   If running FANTASIA in a user-defined Docker network with the services,
   you may set hosts to the container names (e.g., ``pgvectorsql`` / ``rabbitmq``).

5) Initialize the database
==========================

.. code-block:: bash

   poetry run fantasia initialize \
     --config ./config/prott5_test.yaml \
     --base_directory ./lookup \
     --log_path ./lookup/logs \
     --embeddings_url 'https://zenodo.org/records/17795871/files/BioData_Dec25_esm2_prott5_prostt5_ankh3_large_esm3c_Layer0.backup?download=1'

During initialization, the reference dump is downloaded and restored into the
configured PostgreSQL database. This operation resets the database's ``public``
schema; use a dedicated FANTASIA database.

5.1) (Optional) Load dumps from the host
========================================

SQL dump (plain ``.sql``) with ``psql``::

   PGPASSWORD=clave psql \
     -h localhost -p 5432 -U usuario -d BioData \
     -f sample.sql

Custom-format dump (``pg_dump -Fc``) with ``pg_restore``::

   PGPASSWORD=clave pg_restore \
     -h localhost -p 5432 -U usuario -d BioData \
      sample.dump

6) Run the bundled example
==========================

.. code-block:: bash

   poetry run fantasia run \
     --config ./config/prott5_test.yaml \
     --input ./data_sample/sample.fasta \
     --prefix first_search \
     --base_directory ./lookup \
     --log_path ./lookup/logs \
     --device cuda \
     --limit_per_entry 1

The test config limits execution to 20 sequences. For a complete proteome, use
``config/prott5_full.yaml`` or set ``limit_execution: 0``.

7) CLI help
===========

.. code-block:: bash

   poetry run fantasia --help
   poetry run fantasia run --help

Notes
=====

- Docker should be usable without ``sudo`` (see Docker post-installation steps if needed).
- For GPU usage, check ``nvidia-smi`` and ``nvcc --version`` before running.
