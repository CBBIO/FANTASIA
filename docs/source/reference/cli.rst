Command-line interface
======================

FANTASIA reads YAML first. Except for ``--config``, omitted CLI options are
unset and inherit YAML; the table therefore lists ``YAML`` as their CLI
default. Model enablement and per-model settings are YAML-only.

``initialize`` options
----------------------

.. list-table::
   :header-rows: 1

   * - Option
     - CLI default
     - Meaning
   * - ``--config PATH``
     - ``./fantasia/config.yaml``
     - Base YAML configuration
   * - ``--embeddings_url URL``
     - YAML/unset
     - Override reference-dump URL
   * - ``--base_directory PATH``
     - YAML
     - Embeddings and experiment root
   * - ``--log_path PATH``
     - YAML
     - Log file or directory

``run`` options
---------------

.. list-table::
   :header-rows: 1

   * - Option
     - CLI default
     - Meaning
   * - ``--config PATH``
     - ``./fantasia/config.yaml``
     - Base YAML configuration
   * - ``--input PATH`` / ``--prefix TEXT``
     - YAML
     - Input and experiment prefix
   * - ``--base_directory PATH`` / ``--log_path PATH``
     - YAML
     - Output and log locations
   * - ``--limit_execution INT`` / ``--monitor_interval INT``
     - YAML
     - Query limit and progress interval
   * - ``--only_embedding BOOL`` / ``--only_lookup BOOL``
     - YAML
     - Mutually exclusive partial modes
   * - ``--device {cpu,cuda}``
     - YAML
     - Override ``embedding.device``
   * - ``--length_filter INT``
     - YAML
     - Override sequence cap; 0 is uncapped
   * - ``--sequence_queue_package INT``
     - YAML
     - Override embedding queue-package size
   * - ``--limit_per_entry INT``
     - YAML
     - Override neighbours per query (k)
   * - ``--redundancy_identity FLOAT``
     - YAML
     - MMseqs2 identity [0,1]; 0 disables
   * - ``--redundancy_coverage FLOAT`` / ``--threads INT``
     - YAML
     - MMseqs2 coverage and threads
   * - ``--taxonomy_ids_to_exclude IDS``
     - YAML/empty
     - Exact comma/space-separated exclusion IDs
   * - ``--taxonomy_ids_included_exclusively IDS``
     - YAML/empty
     - Exact allow-list, taking precedence
   * - ``--get_descendants BOOL``
     - YAML/false
     - **Deprecated and disabled**; true raises an error

The historical aliases ``--redundancy_filter`` and ``--alignment_coverage``
remain accepted. Consult live help for the installed version:

.. code-block:: bash

   poetry run fantasia --help
   poetry run fantasia initialize --help
   poetry run fantasia run --help
