Getting started
===============

FANTASIA Full annotates protein FASTA files by embedding each query, finding
nearby experimentally annotated reference proteins, and transferring their GO
terms. A first search therefore needs the Python package, PostgreSQL and
RabbitMQ services, a downloaded reference database, and a compatible protein
language model.

Start here
----------

For the shortest complete test, follow the `quick-start guide <quickstart.rst>`_.
It runs 20 bundled
sequences with ProtT5, cosine distance, and ``k=1`` and shows how to locate the
result. Before using your own proteome, read `requirements <requirements.rst>`_
and the `reference data guide <reference_data.rst>`_ because model downloads and
the reference database require
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

- **New installation:** `requirements <requirements.rst>`_ →
  `installation guide <installation.rst>`_ →
  `reference data guide <reference_data.rst>`_ → `quick-start guide <quickstart.rst>`_.
- **First output interpretation:** `first-results guide <first_results.rst>`_.
- **Reuse existing embeddings:** use the embedding-only and lookup-only examples
  in `quick-start guide <quickstart.rst>`_.
- **No PostgreSQL/RabbitMQ:** consider FANTASIA-Lite after reading
  `version guide <choosing_version.rst>`_.
- **All configuration defaults:**
  `configuration defaults <../reference/configuration_reference.rst>`_.
- **Supported model identifiers and revisions:**
  `supported-model reference <../reference/supported_models.rst>`_.

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
