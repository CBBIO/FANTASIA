Output files
============

.. list-table::
   :header-rows: 1
   :widths: 26 14 20 40

   * - Path
     - Status
     - One item/row per
     - Purpose
   * - ``experiment_config.yaml``
     - final metadata
     - run
     - Resolved configuration; retain
   * - ``embeddings.h5``
     - reusable intermediate
     - query/model/layer dataset
     - Required for lookup-only reuse
   * - ``raw_results/<model>/layer_<n>/<accession>.csv``
     - detailed result
     - query–donor–GO association
     - Donor, GO, distance, evidence, and optional alignment data
   * - ``summary.csv``
     - main result
     - accession–GO pair
     - Configured wide metrics and donor list
   * - ``topgo/<model>/layer_<n>/<category>.topgo``
     - downstream export
     - accession–GO pair
     - TopGO input
   * - ``topgo/ensemble/<category>.topgo``
     - downstream export
     - best accession–GO pair
     - Best reliability across models/layers
   * - ``sequences.fasta`` / ``query_index_mapping.csv``
     - auxiliary
     - sequence / query index
     - Alignment and identifier provenance
   * - timestamped log
     - final metadata
     - run
     - Progress, failures, timing

Raw-result columns
------------------

Core columns include ``accession``, ``model_name``, ``embedding_type_id``,
``layer_index``, ``distance``, ``reliability_index``, ``go_id``, ``category``,
``evidence_code``, ``go_description``, ``protein_id``, ``organism``,
``gene_name``, query/reference indices, and lengths. When sequences are
available, Parasail adds global ``identity``, ``similarity``,
``alignment_score``, ``gaps_percentage``, ``alignment_length`` and local
counterparts suffixed ``_sw``.

For cosine, ``reliability_index = 1 - distance``. For Euclidean,
``reliability_index = 0.5 / (0.5 + distance)``. It is not a probability.

Summary columns
---------------

Rows begin with ``accession``, ``go_id``, ``term_count``, and pipe-separated
``proteins`` donor IDs. Configured metrics are pivoted into columns such as
``max_ri_ProtT5_L0``. Weighted columns and ``final_score`` depend on the saved
configuration, so the summary schema is intentionally dynamic.

Raw CSV files are written per query protein. A file can contain multiple rows
because a query can have several retrieved donors and transferred GO terms.
They must therefore be merged when a single proteome-level raw table is needed.
Merge one model/layer directory at a time:

.. code-block:: bash

   python scripts/merge_raw_results.py \
     EXPERIMENT/raw_results/prot-t5/layer_0 \
     --output EXPERIMENT/raw_results/prot-t5/layer_0_merged.csv \
     --add-source-file

The utility concatenates all rows and checks that their headers agree. It does
not compute minima, maxima, means, or other aggregates. By default, its output
is ``raw_results/<model>/layer_<n>_merged.csv``; ``--add-source-file`` adds the
original per-protein filename for provenance. Repeat this operation for every
model/layer that should have its own consolidated table.
