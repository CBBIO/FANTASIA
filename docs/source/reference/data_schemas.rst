Data schemas
============

Query HDF5
----------

New files use
``/accession_<id>/type_<embedding_type_id>/layer_<n>/embedding`` with sequence
data at accession level when retained. Lookup also recognizes a legacy layout
without explicit layers.

TopGO
-----

TopGO files contain accession, GO term, and reliability index in tab-separated
form.

CSV
---

Raw CSV schemas are stable around query–donor–GO rows; sequence and alignment
fields depend on configuration. Summary CSV columns depend on configured
models, layers, metrics, aliases, and weights. See `Output files <output_files.rst>`_ for the
canonical description. The old `Schemas <../appendix/schemas.rst>`_ page is retained only
for URL compatibility.
