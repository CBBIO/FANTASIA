==========================
Run your first GO search
==========================

This page is the shortest supported path from a fresh clone to a FANTASIA
annotation result. It runs the complete pipeline on the bundled sample:

``protein FASTA → ProtT5 embeddings → cosine lookup (k=1) → GO tables``

Prerequisites
=============

- Linux with Python 3.12
- Poetry
- Docker with the Compose plugin
- An NVIDIA GPU for the commands as written
- Enough disk space for the selected reference database and model cache

For CPU execution or manual service setup, see
:doc:`../appendix/installation_and_quickstart`.

1. Install and start services
=============================

.. code-block:: bash

   git clone https://github.com/CBBIO/FANTASIA.git
   cd FANTASIA
   poetry install
   docker compose up -d
   docker compose ps

Wait until both ``fantasia-postgres`` and ``fantasia-rabbitmq`` are healthy.

2. Create local working folders
================================

``data/`` and ``lookup/`` are deliberately excluded from the repository.
Create them before running the examples:

.. code-block:: bash

   mkdir -p data lookup/{logs,experiments,embeddings}

Use ``data/`` for your FASTA files. This guide uses ``lookup/`` as the base
directory for the reference download, logs, query embeddings, and results.

3. Load the reference database
==============================

Initialization is required once for a new PostgreSQL database. The following
command downloads the approximately 3.1 GB final-layer reference dataset and
restores it into PostgreSQL:

.. code-block:: bash

   poetry run fantasia initialize \
     --config ./config/prott5_test.yaml \
     --base_directory ./lookup \
     --log_path ./lookup/logs \
     --embeddings_url 'https://zenodo.org/records/17795871/files/BioData_Dec25_esm2_prott5_prostt5_ankh3_large_esm3c_Layer0.backup?download=1'

.. warning::

   Initialization resets the ``public`` schema of the configured database
   before restoring the reference dump. Use a dedicated FANTASIA database.

The bundled Docker service uses these values:

.. list-table::
   :header-rows: 1

   * - Setting
     - Value
   * - Host / port
     - ``localhost:5432``
   * - Database
     - ``BioData``
   * - User / password
     - ``usuario`` / ``clave``

4. Run the bundled annotation example
======================================

.. code-block:: bash

   poetry run fantasia run \
     --config ./config/prott5_test.yaml \
     --input ./data_sample/sample.fasta \
     --prefix first_search \
     --base_directory ./lookup \
     --log_path ./lookup/logs \
     --device cuda \
     --limit_per_entry 1

``--limit_per_entry 1`` is the command-line form of ``k=1``. Model selection
and ``lookup.use_gpu`` are YAML settings; the command uses ProtT5 and GPU lookup
as defined in ``config/prott5_test.yaml``.

5. Find and interpret the result
================================

Each invocation creates a timestamped directory:

.. code-block:: text

   lookup/experiments/first_search_<YYYYMMDDHHMMSS>/
   ├── embeddings.h5
   ├── experiment_config.yaml
   ├── model_provenance.yaml
   ├── raw_results/prot-t5/layer_0/*.csv
   ├── summary.csv
   └── topgo/  (only when lookup.topgo is true)

``summary.csv``
   Consolidated accession-by-GO output after post-processing.

``raw_results/<model>/layer_0/*.csv``
   Detailed per-query assignments. Each row links a query, donor, and GO term
   and includes embedding distance and reliability information.

``topgo/``
   Optional TopGO-compatible exports, enabled by ``lookup.topgo: true``.

``experiment_config.yaml``
   The resolved configuration saved with the run. Retain it for
   reproducibility.

Run your own FASTA
==================

Copy a plain or gzip-compressed protein FASTA into ``data/`` and replace the
input path:

.. code-block:: bash

   poetry run fantasia run \
     --config ./config/prott5_full.yaml \
     --input ./data/my_proteome.faa.gz \
     --prefix my_proteome \
     --base_directory ./lookup \
     --log_path ./lookup/logs \
     --device cuda \
     --limit_per_entry 1

The test config processes only 20 sequences (``limit_execution: 20``). Use
``config/prott5_full.yaml`` or set ``limit_execution: 0`` for a complete
proteome.

Reuse query embeddings
======================

Embedding only
--------------

.. code-block:: bash

   poetry run fantasia run \
     --config ./config/prott5_full.yaml \
     --input ./data/my_proteome.faa.gz \
     --prefix embed_only \
     --base_directory ./lookup \
     --log_path ./lookup/logs \
     --only_embedding true

Lookup only
-----------

.. code-block:: bash

   poetry run fantasia run \
     --config ./config/prott5_full.yaml \
     --input ./lookup/experiments/embed_only_<timestamp>/embeddings.h5 \
     --prefix lookup_only \
     --base_directory ./lookup \
     --log_path ./lookup/logs \
     --only_lookup true \
     --limit_per_entry 1

The enabled model and layer in the lookup config must match those stored in
the HDF5 file and available in the reference database.

Benchmark / leakage-control example
===================================

Annotation mode normally uses ``k=1`` without self-exclusion. For a benchmark,
retrieve several candidates and explicitly exclude query taxa:

.. code-block:: bash

   poetry run fantasia run \
     --config ./config/prott5_full.yaml \
     --input ./data/mouse_proteome.fasta \
     --prefix mouse_benchmark_k5 \
     --base_directory ./lookup \
     --log_path ./lookup/logs \
     --limit_per_entry 5 \
     --taxonomy_ids_to_exclude 10090,10091,57486

Taxonomy matching uses the exact IDs supplied. Descendant expansion is
currently disabled. For stricter leakage control, inspect or filter retrieved
donors using sequence identity after lookup.

Next steps
==========

- :doc:`installation` — package installation
- :doc:`reference_data` — reference database setup
- :doc:`/user_guide/configuration` — models, layers, filters, and run modes
- :doc:`first_results` — output tables and validation
- :doc:`/deployment/hpc_slurm` — HPC and Slurm examples
