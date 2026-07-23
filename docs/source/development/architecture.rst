Developer architecture
======================

``fantasia.main`` parses/normalizes config, validates services, creates a
timestamped experiment, and orchestrates embedding then lookup.
``fantasia.src.embedder`` queues PIS model tasks and writes HDF5.
``fantasia.src.lookup`` loads reference metadata/vectors, computes neighbours,
expands GO rows, aligns sequences, and exports results. ``helpers`` contains
download, restore, parsing, taxonomy, and Parasail routines.

PostgreSQL/PIS owns persistent biological/reference data; RabbitMQ coordinates
embedding workers. Generated query/results data remain experiment-local.
