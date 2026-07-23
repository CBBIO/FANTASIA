User guide
==========

This section covers routine use after FANTASIA and its reference database are
installed. Begin with :doc:`running_fantasia` for a complete command, then use
the task-specific pages below.

Choose a workflow
-----------------

- **Annotate one proteome:** :doc:`preparing_input` → :doc:`configuration` →
  :doc:`annotation_mode`.
- **Compare models or control reference leakage:** :doc:`benchmark_mode`.
- **Generate embeddings without annotation:** :doc:`embedding_only`.
- **Reuse an existing embeddings file:** :doc:`lookup_only`.
- **Process several FASTA files:** :doc:`multiple_proteomes`.

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
