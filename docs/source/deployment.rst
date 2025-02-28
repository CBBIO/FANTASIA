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

Use `poetry` to manage the virtual environment. Follow these steps:

1. **Ensure Poetry is installed and up to date:**

   .. code-block:: bash

      poetry self update

2. **If using Poetry 1.5 or later, install the required shell plugin:**

   .. code-block:: bash

      poetry self add poetry-plugin-shell

3. **Create and activate the virtual environment:**

   .. code-block:: bash

      poetry env use <python_version>  # Specify the desired Python version (3.12)
      poetry install
      poetry env activate


.. note::

   If using Conda, avoid managing environments with both Poetry and Conda simultaneously to prevent dependency conflicts.


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

.. code-block:: bash

   python fantasia/main.py run

