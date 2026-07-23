FANTASIA 4.1.1
==============

**Functional annotation through protein-language-model embedding similarity.**

FANTASIA embeds protein sequences, searches an experimentally annotated
reference database, and transfers Gene Ontology terms from nearby reference
proteins. Full FANTASIA supports five models and detailed database-backed
workflows; `FANTASIA-Lite <https://github.com/CBBIO/FANTASIA-Lite>`_ is the
standalone alternative without PostgreSQL or RabbitMQ.

New user?
---------

1. `Choose Full or Lite <getting_started/choosing_version.rst>`_.
2. `Confirm requirements <getting_started/requirements.rst>`_.
3. `Install FANTASIA <getting_started/installation.rst>`_.
4. `Set up reference data <getting_started/reference_data.rst>`_.
5. `Run the included example <getting_started/quickstart.rst>`_.
6. `Inspect the first results <getting_started/first_results.rst>`_.

.. note::

   Full FANTASIA requires service deployment and an approximately 3.1 GB
   reference download before the first annotation. It is not a five-minute
   standalone demonstration.

Common tasks
------------

- `Annotate a proteome <user_guide/annotation_mode.rst>`_
- `Select models and settings <user_guide/configuration.rst>`_
- `Reuse embeddings <user_guide/lookup_only.rst>`_
- `Control benchmark leakage <user_guide/benchmark_mode.rst>`_
- `Run on GPU <deployment/gpu.rst>`_ or `CPU <deployment/cpu.rst>`_
- `Deploy with Docker <deployment/docker.rst>`_ or `Slurm <deployment/hpc_slurm.rst>`_
- `Understand output files <reference/output_files.rst>`_
- `Diagnose failures <troubleshooting/index.rst>`_

.. toctree::
   :caption: Getting started
   :maxdepth: 2

   getting_started/index

.. toctree::
   :caption: User guide
   :maxdepth: 2

   user_guide/index

.. toctree::
   :caption: Deployment
   :maxdepth: 2

   deployment/index

.. toctree::
   :caption: Reference
   :maxdepth: 2

   reference/index

.. toctree::
   :caption: How FANTASIA works
   :maxdepth: 2

   explanation/index

.. toctree::
   :caption: Benchmarks and validation
   :maxdepth: 2

   benchmarks/index

.. toctree::
   :caption: Performance
   :maxdepth: 2

   performance/index

.. toctree::
   :caption: Troubleshooting
   :maxdepth: 2

   troubleshooting/index

.. toctree::
   :caption: Development
   :maxdepth: 2

   development/index

.. toctree::
   :caption: Citation and licence
   :maxdepth: 1

   citation
   changelog

.. toctree::
   :caption: Legacy pages
   :hidden:

   quickstart
   appendix/index
   method/index
   abstract
   introduction
   discussion
   conclusions
   references
   acknowledgments
   contact
