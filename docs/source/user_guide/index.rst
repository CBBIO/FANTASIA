User guide
==========

This section covers routine use after FANTASIA and its reference database are
installed. Begin with `running FANTASIA <running_fantasia.rst>`_ for a
complete command, then use
the task-specific pages below.

Choose a workflow
-----------------

- **Annotate one proteome:** `input preparation <preparing_input.rst>`_ →
  `configuration <configuration.rst>`_ →
  `annotation mode <annotation_mode.rst>`_.
- **Compare models or control reference leakage:**
  `benchmark mode <benchmark_mode.rst>`_.
- **Generate embeddings without annotation:** `embedding-only mode <embedding_only.rst>`_.
- **Reuse an existing embeddings file:** `lookup-only mode <lookup_only.rst>`_.
- **Process several FASTA files:**
  `multiple-proteome guide <multiple_proteomes.rst>`_.

The usual full run is:

.. code-block:: bash

   poetry run fantasia run \
     --config ./config/prott5_full.yaml \
     --input ./data/my_proteome.faa.gz \
     --prefix my_proteome \
     --base_directory ./lookup \
     --log_path ./lookup/logs

Configuration controls the enabled model, layer, truncation, distance metric,
number of neighbours, taxonomy filters, and optional post-processing. Every run
records the effective settings in ``experiment_config.yaml`` and model/software
metadata in ``model_provenance.yaml``.

Guide contents
--------------

.. toctree::
   :maxdepth: 1

   running_fantasia
   preparing_input
   configuration
   annotation_mode
   benchmark_mode
   embedding_only
   lookup_only
   multiple_proteomes
