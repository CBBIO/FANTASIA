Reference databases
===================

Current final-layer and multilayer releases are listed in
:doc:`/getting_started/reference_data`. They contain PostgreSQL/PIS records for
reference sequences, experimentally supported GO annotations, model metadata,
and vector embeddings. During lookup, the selected model/layer vectors are
loaded into memory; pgvector provides storage while the application performs
vectorized CPU/GPU distance computation.

Compatibility requires matching model identifiers, dimensions, and layers.
Initialization restores a complete PostgreSQL backup and resets ``public``. It
does not consume manuscript benchmark outputs. Historical releases should be
cited by version-specific DOI; use the current record pages for new setup.
