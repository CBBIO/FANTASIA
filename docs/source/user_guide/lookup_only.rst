Lookup-only mode
================

.. code-block:: bash

   poetry run fantasia run --config ./config/prott5_full.yaml \
     --input ./lookup/experiments/embed_only_<timestamp>/embeddings.h5 \
     --prefix lookup_only --base_directory ./lookup \
     --log_path ./lookup/logs --only_lookup true --limit_per_entry 1

The enabled model and layers must match the HDF5 and the restored reference
database. Changing ``k``, taxonomy filters, or lookup device does not require
regenerating compatible query embeddings. ``only_lookup`` and
``only_embedding`` cannot both be true.
