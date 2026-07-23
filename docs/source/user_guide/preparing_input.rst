Prepare input FASTA
===================

Full and embedding-only runs accept plain or gzip-compressed protein FASTA.
Supported compressed examples include ``.fa.gz``, ``.faa.gz``, and
``.fasta.gz``. Lookup-only mode instead expects FANTASIA's ``embeddings.h5``.

Use unique, non-empty accession headers. The parser normalizes headers for file
names and output accessions; preserve the original input and inspect
``query_index_mapping.csv`` when internal ``Q`` identifiers appear. Empty or
invalid files cannot produce embeddings. With
``embedding.max_sequence_length: 0``, FANTASIA does not truncate queries, but a
model may still reject sequences beyond its own supported length.

Quick checks:

.. code-block:: bash

   test -s data/my_proteome.faa.gz
   gzip -cd data/my_proteome.faa.gz | awk '/^>/{n++} END{print n}'
