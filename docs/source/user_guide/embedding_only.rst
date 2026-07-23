Embedding-only mode
===================

.. code-block:: bash

   poetry run fantasia run --config ./config/prott5_full.yaml \
     --input ./data/my_proteome.faa.gz --prefix embed_only \
     --base_directory ./lookup --log_path ./lookup/logs \
     --only_embedding true

The experiment contains ``embeddings.h5`` and the resolved configuration but
no lookup results. Keep the HDF5 to reuse the expensive embedding stage.
