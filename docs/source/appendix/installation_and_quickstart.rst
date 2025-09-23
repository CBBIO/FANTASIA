===========================
Installation and Quickstart
===========================

This quickstart brings up a **local development environment** for FANTASIA:
database, message broker, and core dependencies. For an end-user installation
(e.g., via ``pip``), refer to the production deployment section when available.

What you’ll set up
==================

- PostgreSQL with the ``pgvector`` extension (Docker)
- RabbitMQ message broker (Docker)
- External tools: **MMseqs2** and **Parasail**
- Python environment managed with **Poetry**
- (Optional) GPU support: NVIDIA driver + CUDA Toolkit

Prerequisites
=============

System Requirements
-------------------
- **OS**: Linux (Ubuntu recommended)
- **Python**: 3.10+
- **Docker**: installed and running (configured for non-root use)

External Tools
--------------

MMseqs2 (redundancy filtering and clustering)::

   sudo apt-get update
   sudo apt-get install mmseqs2

Parasail (SIMD-accelerated pairwise alignment)::

   sudo apt-get install parasail

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

Use the default workspace path and set permissions::

   mkdir -p ~/fantasia
   chmod -R 755 ~/fantasia

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

.. note::
   If running FANTASIA in a user-defined Docker network with the services,
   you may set hosts to the container names (e.g., ``pgvectorsql`` / ``rabbitmq``).

5) Initialize the database
==========================

.. code-block:: bash

   fantasia initialize

During initialization, required embeddings are downloaded and indexed.

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

6) Run the pipeline (development)
=================================

.. code-block:: bash

   fantasia run

7) CLI help
===========

.. code-block:: bash

   fantasia --help

Notes
=====

- Docker should be usable without ``sudo`` (see Docker post-installation steps if needed).
- For GPU usage, check ``nvidia-smi`` and ``nvcc --version`` before running.
