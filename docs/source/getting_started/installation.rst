Installation
============

Repository installation
-----------------------

.. code-block:: bash

   git clone https://github.com/CBBIO/FANTASIA.git
   cd FANTASIA
   poetry install
   poetry run fantasia --help

Create the untracked working folders used by the documentation:

.. code-block:: bash

   mkdir -p data lookup/{logs,experiments,embeddings}

Start local services:

.. code-block:: bash

   docker compose up -d
   docker compose ps

The Compose credentials are development defaults. Change them for shared or
production deployments and update the YAML configuration accordingly.

Package installation
--------------------

``pip install fantasia`` installs the Python package, but does not remove the
need for PostgreSQL, pgvector, RabbitMQ, a compatible reference database, and a
configuration resolving ``constants.yaml``. New users should use the repository
workflow above.

Developer installation
----------------------

Use ``poetry install`` and see `Testing and contributions <../development/testing.rst>`_. This release does
not include a ``poetry.lock`` file, so record the resolved environment for
reproducible development and publication runs.
