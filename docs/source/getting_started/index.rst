Getting started
===============

FANTASIA Full annotates protein FASTA files by embedding each query, finding
nearby experimentally annotated reference proteins, and transferring their GO
terms. A first search therefore needs the Python package, PostgreSQL and
RabbitMQ services, a downloaded reference database, and a compatible protein
language model.

Start here
----------

For the shortest complete test, follow :doc:`quickstart`. It runs 20 bundled
sequences with ProtT5, cosine distance, and ``k=1`` and shows how to locate the
result. Before using your own proteome, read :doc:`requirements` and
:doc:`reference_data` because model downloads and the reference database require
substantial disk, RAM, and GPU memory.

The minimal workflow is:

.. code-block:: text

   install → start PostgreSQL/RabbitMQ → initialize reference data
           → run FASTA → inspect summary.csv and raw_results/

A successful run creates a timestamped experiment directory containing:

- ``summary.csv``: consolidated accession–GO assignments;
- ``raw_results/<model>/<layer>/<protein>.csv``: one detailed file per query
  protein, mergeable with the supplied merge tool;
- ``embeddings.h5``: reusable query embeddings;
- ``experiment_config.yaml``: effective run settings;
- ``model_provenance.yaml``: model repositories, revisions, layers, and software
  versions recorded automatically;
- optional ``topgo/`` exports only when ``lookup.topgo: true``.

Choose the relevant path
------------------------

- **New installation:** :doc:`requirements` → :doc:`installation` →
  :doc:`reference_data` → :doc:`quickstart`.
- **First output interpretation:** :doc:`first_results`.
- **Reuse existing embeddings:** use the embedding-only and lookup-only examples
  in :doc:`quickstart`.
- **No PostgreSQL/RabbitMQ:** consider FANTASIA-Lite after reading
  :doc:`choosing_version`.
- **All configuration defaults:** :doc:`../reference/configuration_reference`.
- **Supported model identifiers and revisions:**
  :doc:`../reference/supported_models`.

.. note::

   ``data/`` and ``lookup/`` are not tracked by Git. Create them from the
   repository root before the examples with
   ``mkdir -p data lookup/{logs,experiments,embeddings}``.

Detailed sequence
-----------------

.. toctree::
   :maxdepth: 1

   choosing_version
   requirements
   installation
   reference_data
   quickstart
   first_results
