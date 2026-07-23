Configuration defaults
======================

What “default” means
--------------------

FANTASIA reads a YAML file first and then applies explicitly supplied CLI
overrides. Apart from ``--config``, CLI options default to *unset* and therefore
inherit the YAML value; they do not replace it with a second hidden CLI
default. The values below are those shipped in ``config/prott5_full.yaml``.
``config/prott5_test.yaml`` differs principally in ``prefix: prott5_test`` and
``limit_execution: 20``.

Global, service and execution parameters
----------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 31 17 19 33

   * - Parameter
     - Type
     - Packaged default
     - Meaning
   * - ``log_path``
     - path
     - ``~/FANTASIA/logs/``
     - Log directory
   * - ``constants``
     - path
     - ``./fantasia/constants.yaml``
     - Model and embedding-type registry
   * - ``base_directory``
     - path
     - ``~/fantasia_runs/``
     - Root for experiments and embeddings
   * - ``prefix``
     - string
     - ``prott5_full``
     - Timestamped experiment-name prefix
   * - ``limit_execution``
     - integer
     - ``0``
     - Number of queries; 0 processes all
   * - ``monitor_interval``
     - integer
     - ``10``
     - Progress-log interval in seconds
   * - ``input``
     - path
     - ``data_sample/sample.fasta``
     - FASTA, or HDF5 in lookup-only mode
   * - ``only_lookup`` / ``only_embedding``
     - boolean
     - ``false`` / ``false``
     - Mutually exclusive partial modes
   * - ``DB_USERNAME`` / ``DB_PASSWORD``
     - string
     - ``usuario`` / ``clave``
     - Local Compose credentials; replace outside local use
   * - ``DB_HOST`` / ``DB_PORT`` / ``DB_NAME``
     - scalar
     - ``localhost`` / ``5432`` / ``BioData``
     - PostgreSQL connection
   * - ``rabbitmq_host`` / ``rabbitmq_port``
     - scalar
     - ``localhost`` / ``5672``
     - RabbitMQ endpoint
   * - ``rabbitmq_user`` / ``rabbitmq_password``
     - string
     - ``guest`` / ``guest``
     - Local RabbitMQ credentials
   * - ``delete_queues``
     - boolean
     - ``true``
     - Remove stale queues between runs
   * - ``embeddings_url``
     - URL/unset
     - unset
     - Reference dump used by ``initialize``

Embedding parameters
--------------------

.. list-table::
   :header-rows: 1
   :widths: 35 15 18 32

   * - Parameter
     - Type
     - Packaged default
     - Meaning
   * - ``embedding.device``
     - ``cuda``/``cpu``
     - ``cuda``
     - PLM execution device
   * - ``embedding.queue_batch_size``
     - integer
     - ``100``
     - Sequences per RabbitMQ package
   * - ``embedding.max_sequence_length``
     - integer
     - ``0``
     - 0 means uncapped; positive values truncate
   * - ``embedding.models.ESM.enabled``
     - boolean
     - ``false``
     - Enable ESM-2
   * - ``embedding.models.ESM3c.enabled``
     - boolean
     - ``false``
     - Enable ESM Cambrian 600M
   * - ``embedding.models.Ankh3-Large.enabled``
     - boolean
     - ``false``
     - Enable Ankh3-Large
   * - ``embedding.models.Prot-T5.enabled``
     - boolean
     - ``true``
     - Enable ProtT5
   * - ``embedding.models.Prost-T5.enabled``
     - boolean
     - ``false``
     - Enable ProstT5
   * - ``embedding.models.<name>.batch_size``
     - integer
     - ``1`` for every model
     - PLM forward-pass batch size
   * - ``embedding.models.<name>.layer_index``
     - list[integer]
     - ``[0]`` for every model
     - 0 is final, 1 penultimate, and so on
   * - ``embedding.models.<name>.distance_threshold``
     - false/positive float
     - ``false`` for every model
     - Optional maximum distance before top-k; false disables

At least one model must be enabled. The enabled model/layer must exist in the
restored reference database.

Lookup, filtering and post-processing parameters
------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 38 15 20 27

   * - Parameter
     - Type
     - Packaged default
     - Meaning
   * - ``lookup.use_gpu``
     - boolean
     - ``true``
     - GPU distance computation
   * - ``lookup.batch_size``
     - integer
     - ``516``
     - Query distance batch size
   * - ``lookup.distance_metric``
     - enum
     - ``cosine``
     - ``cosine`` or ``euclidean``
   * - ``lookup.limit_per_entry``
     - integer
     - ``1``
     - Reference embedding neighbours per query (k)
   * - ``lookup.lookup_cache_max``
     - integer
     - ``1``
     - Cached model/layer reference tables
   * - ``lookup.topgo``
     - boolean
     - ``true``
     - Write TopGO-compatible exports
   * - ``lookup.precision``
     - integer
     - ``4``
     - Decimal places in numeric outputs
   * - ``redundancy.identity``
     - float [0,1]
     - ``0``
     - MMseqs2 identity; 0 disables redundancy masking
   * - ``redundancy.coverage``
     - float (0,1]
     - ``0.7``
     - MMseqs2 alignment coverage
   * - ``redundancy.threads``
     - integer
     - ``10``
     - MMseqs2 CPU threads
   * - ``taxonomy.exclude``
     - list[taxonomy ID]
     - ``[]``
     - Exact reference taxonomy IDs to exclude
   * - ``taxonomy.include_only``
     - list[taxonomy ID]
     - ``[]``
     - Exact reference allow-list; takes precedence
   * - ``taxonomy.get_descendants``
     - boolean
     - ``false``
     - **Deprecated and disabled.** True is rejected
   * - ``postprocess.keep_sequences``
     - boolean
     - ``true``
     - Retain sequences for alignment-aware output
   * - ``postprocess.summary.include_counts``
     - boolean
     - ``true``
     - Add neighbour support count
   * - ``postprocess.summary.metrics``
     - mapping
     - RI max; global/local identity min/max/mean
     - Summary aggregations
   * - ``postprocess.summary.aliases``
     - mapping
     - ``ri``, ``id_g``, ``id_l``
     - Output-column abbreviations
   * - ``postprocess.summary.weights``
     - mapping
     - RI 0.4; global ID 0.2; local ID 0.2; count 0.2
     - Components normalized into ``final_score``
   * - ``postprocess.summary.weighted_prefix``
     - string
     - ``w_``
     - Prefix for weighted columns
   * - ``postprocess.summary.normalize_count_by_limit_per_entry``
     - boolean
     - ``true``
     - Compatibility key; current summarizer always normalizes by k
   * - ``postprocess.summary.export_raw_count``
     - boolean
     - ``true``
     - Compatibility key; current summarizer does not emit a separate raw-count column

Canonical and legacy paths
--------------------------

``fantasia/config.yaml`` nests redundancy and taxonomy under ``lookup``.
The named configs retain top-level ``redundancy`` and ``taxonomy`` blocks for
compatibility; the loader maps them to ``lookup.redundancy`` and
``lookup.taxonomy``. Prefer nested paths in new configurations.

``get_descendants`` is retained only so old configurations fail clearly rather
than silently changing meaning. Descendant expansion is no longer executed.
Use explicit IDs in ``exclude`` or ``include_only``. Any true value supplied via
CLI, top-level legacy YAML, or ``lookup.taxonomy.get_descendants`` raises an
error; the resolved ``experiment_config.yaml`` always records false.

No general resume or overwrite parameter exists. Every invocation creates a
timestamped experiment directory. Retain its ``experiment_config.yaml`` as the
record of effective values after CLI overrides.
