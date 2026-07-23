Similarity search and GO transfer
=================================

For every compatible model/layer, lookup compares query vectors with eligible
reference vectors and keeps up to ``limit_per_entry`` nearest sequences. Each
reference sequence can map to one or more protein records, and every
experimental GO annotation attached to those records is expanded into a raw
row. Consequently, ``k`` counts reference sequence neighbours, not GO terms or
necessarily unique protein accessions.

Cosine and Euclidean distance are supported. Lower distance is better.
``reliability_index`` reverses the direction for convenient ranking but is not
a calibrated confidence or correctness probability.
