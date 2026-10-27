Input-file failures
===================

Empty or unreadable FASTA
   Use ``test -s FILE`` and inspect the first header. Confirm read permissions.

Compressed input fails
   Ensure the file is valid gzip (``gzip -t FILE``), not merely renamed with a
   ``.gz`` suffix.

Empty sequences
   A FASTA record without residues is still embedded and annotated, but its
   alignment metrics (``identity``, ``similarity`` and their ``_sw`` variants)
   are left empty and the log reports how many pairs were skipped. Remove empty
   records from the input when they are not intended.

Duplicate or unsafe identifiers
   Use unique accessions. Inspect ``query_index_mapping.csv`` and sanitized raw
   filenames when matching outputs back to inputs.

Long sequences missing from HDF5
   ``max_sequence_length: 0`` disables FANTASIA truncation but does not remove
   upstream model limits. Inspect embedding logs and compare FASTA accessions
   with ``tests/benchmark/report_missing_embeddings.py``.
