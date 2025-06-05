FANTASIA: Installation from PyPI
=================================================

This section explains how to install and run FANTASIA directly from **PyPI**, as an end-user tool.

It is the recommended option for **functional annotation workflows**, avoiding the need for local code changes or development environments.

Installation
------------

FANTASIA requires Python **3.10 or newer**.

We strongly recommend installing FANTASIA inside a virtual environment to avoid conflicts with system-wide packages.

To install the latest release from PyPI:

.. code-block:: bash

   pip install fantasia

After installation, you can verify the CLI is available by running:

.. code-block:: bash

   fantasia --help


Service Requirements
--------------------

FANTASIA depends on two external services:

- **PostgreSQL with pgvector**: used for storing and querying precomputed embeddings.
- **MMseqs2**: used for redundancy filtering during benchmarking (optional).
- **Parasail**: used for sequence alignment post-processing (optional but recommended).

Start the required services using Docker:

.. code-block:: bash

   docker run -d --name pgvectorsql \
       -e POSTGRES_USER=usuario \
       -e POSTGRES_PASSWORD=clave \
       -e POSTGRES_DB=BioData \
       -p 5432:5432 \
       pgvector/pgvector:pg16

You may also install MMseqs2 and Parasail via APT:

.. code-block:: bash

   sudo apt-get update
   sudo apt-get install mmseqs2 parasail


Directory Setup


FANTASIA expects several working directories to exist. You can create them with:

.. code-block:: bash

   mkdir -p ~/fantasia
   chmod -R 755 ~/fantasia


Configuration
-------------

By default, FANTASIA loads its configuration from ``./fantasia/config.yaml``.

This default path is hardcoded in both ``initialize`` and ``run`` commands (via the ``--config`` argument). To use a custom path, override it explicitly:

.. code-block:: bash

   fantasia run --config /path/to/your/config.yaml

The configuration file must include a valid path to the ``constants.yaml`` file, typically specified via the ``constants:`` field.

You have two options to provide these files:

* **Option 1: Clone the repository**

  .. code-block:: bash

     git clone https://github.com/CBBIO/FANTASIA.git
     cd FANTASIA
     fantasia run --config fantasia/config.yaml

* **Option 2: Download the files manually**

  .. code-block:: bash

     wget https://github.com/CBBIO/FANTASIA/raw/main/fantasia/config.yaml
     wget https://github.com/CBBIO/FANTASIA/raw/main/fantasia/constants.yaml

  Then update the ``constants:`` field in ``config.yaml`` if needed:

  .. code-block:: yaml

     constants: "./constants.yaml"


Running the Pipeline
------------------------------------------------
Once installed and configured, you can run the annotation pipeline with:

.. code-block:: bash

   fantasia run --input my_sequences.fasta

You can also initialize the database (for embedding indexing):

.. code-block:: bash

   fantasia initialize

For more options:

.. code-block:: bash

   fantasia --help
