=======================================
Local Development Environment
=======================================

This guide describes how to deploy and configure a **local development environment** for FANTASIA.
If you're interested in installing FANTASIA as an end-user tool (e.g., via ``pip``), refer to the `production deployment section <#fantasia-production-deployment>`_.

Overview
========

The development environment includes the full backend required to run, test and extend the FANTASIA pipeline:

- PostgreSQL database with `pgvector` extension
- RabbitMQ message broker
- External services: MMseqs2, Parasail
- A Python environment managed by Poetry

We recommend using **PyCharm** for development, as it integrates seamlessly with Poetry and provides a rich debugging and navigation experience.


Prerequisites
=============

System Requirements
-------------------
- **Operating System**: Linux (Ubuntu recommended)
- **Python**: Version 3.10 or higher
- **Docker**: Installed and running.
  If not installed, follow the `Docker installation guide <https://docs.docker.com/get-docker/>`_
  and the `post-installation steps <https://docs.docker.com/engine/install/linux-postinstall/>`_ to run Docker without ``sudo``.

- **MMseqs2**: Required for redundancy filtering and clustering.

  Install with:

  .. code-block:: bash

     sudo apt-get update
     sudo apt-get install mmseqs2

- **Parasail**: SIMD-accelerated pairwise alignment library.

  Install with:

  .. code-block:: bash

     sudo apt-get install parasail


Machine Learning Dependencies
-----------------------------
- **NVIDIA Driver**: Version 550.120 or newer (check with ``nvidia-smi``).
- **CUDA Toolkit**: Version 12.4 or newer (check with ``nvcc --version``).

Python Environment
------------------
- **Poetry**: Python dependency and virtual environment manager.

  .. code-block:: bash

     curl -sSL https://install.python-poetry.org | python3 -
     export PATH="$HOME/.local/bin:$PATH"
     source ~/.bashrc  # or ~/.zshrc

.. note::

   PyCharm users can directly import the `pyproject.toml` file and enable Poetry support from the IDE.


Repository Setup
================

Clone the repository and enter the project directory:

.. code-block:: bash

   git clone https://github.com/CBBIO/FANTASIA.git
   cd FANTASIA


Creating the Virtual Environment
================================

.. code-block:: bash

   poetry self update
   poetry self add poetry-plugin-shell  # required for Poetry ≥1.5
   poetry env use 3.12  # or your Python version
   poetry install
   poetry env activate


Starting Required Services
==========================

Start PostgreSQL with pgvector support:

.. code-block:: bash

   docker run -d --name pgvectorsql \
       -e POSTGRES_USER=usuario \
       -e POSTGRES_PASSWORD=clave \
       -e POSTGRES_DB=BioData \
       -p 5432:5432 \
       pgvector/pgvector:pg16

Start RabbitMQ (with UI):

.. code-block:: bash

   docker run -d --name rabbitmq \
       -p 15672:15672 \
       -p 5672:5672 \
       rabbitmq:management

You can access the interface at http://localhost:15672 (default login: ``guest/guest``).


Configuration
=============

Make sure the following directories exist:

.. code-block:: bash

   mkdir -p ~/fantasia/dumps ~/fantasia/embeddings ~/fantasia/results ~/fantasia/redundancy
   chmod -R 755 ~/fantasia

Update `fantasia/config.yaml`:

.. code-block:: yaml

   DB_USERNAME: usuario
   DB_PASSWORD: clave
   DB_HOST: pgvectorsql
   DB_PORT: 5432
   DB_NAME: BioData

   rabbitmq_host: rabbitmq
   rabbitmq_user: guest
   rabbitmq_password: guest


Database Initialization
=======================

.. code-block:: bash

   python3 fantasia/main.py initialize

Embeddings will be downloaded automatically and indexed in the database.


Running the Pipeline (Development Mode)
=======================================

Use the CLI:

.. code-block:: bash

   python3 fantasia/main.py run

Or check all available commands:

.. code-block:: bash

   python3 fantasia/main.py --help

