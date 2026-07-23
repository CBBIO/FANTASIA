Configure a run
===============

Start from ``config/prott5_test.yaml`` for a 20-sequence test or
``config/prott5_full.yaml`` for a complete ProtT5 run. Copy the file before
editing a study-specific configuration.

The main decisions are:

- enable one or more entries under ``embedding.models``;
- choose model layers (``0`` is the final/output layer);
- choose ``embedding.device`` and ``lookup.use_gpu`` independently;
- set ``embedding.max_sequence_length`` (``0`` means no FANTASIA truncation);
- set ``lookup.distance_metric`` and ``lookup.limit_per_entry``;
- add exact taxonomy IDs where exclusions are required;
- enable MMseqs2 redundancy masking only when intended;
- select post-processing metrics and weights.

Use one model per launch for predictable resource use. CLI model selection is
not supported; model enablement is YAML-only. See
`Configuration defaults <../reference/configuration_reference.rst>`_ for paths, types, defaults, and
interactions.
