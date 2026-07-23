Reference data setup
====================

Current releases
----------------

.. list-table::
   :header-rows: 1

   * - Reference
     - Zenodo record
     - Download
     - Use
   * - Final layer
     - `17795871 <https://zenodo.org/records/17795871>`_
     - approximately 3.1 GB
     - Recommended for routine annotation
   * - Multiple layers
     - `17793273 <https://zenodo.org/records/17793273>`_
     - approximately 17.1 GB
     - Layer-comparison workflows

The reference database contains experimentally annotated proteins and their
model/layer embeddings. It is distinct from manuscript benchmark-output
archives.

Restore the final-layer reference
---------------------------------

Start PostgreSQL and RabbitMQ first, then run:

.. code-block:: bash

   poetry run fantasia initialize \
     --config ./config/prott5_test.yaml \
     --base_directory ./lookup \
     --log_path ./lookup/logs \
     --embeddings_url 'https://zenodo.org/records/17795871/files/BioData_Dec25_esm2_prott5_prostt5_ankh3_large_esm3c_Layer0.backup?download=1'

.. warning::

   The initializer resets the configured database's ``public`` schema before
   restoring the dump. Use a dedicated FANTASIA database.

The downloaded dump is retained below ``lookup/embeddings/``. Software model
and layer selections must exist in the restored reference. The repository does
not currently implement automatic checksum verification; retain Zenodo
metadata and calculate a local SHA-256 checksum when provenance is required.
