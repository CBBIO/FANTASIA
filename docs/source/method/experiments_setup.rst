Experiments Setup
=================

Scope
-------
This page explains how to plan and configure a **FANTASIA v4** experiment. It focuses on
execution modes, similarity-search controls, taxonomy filtering, and embedding settings,
with practical guidance for batching and model/layer selection.

Recommended defaults
--------------------
.. note::

   The **default configuration shipped in the repository** is the reference for all runs.
   It is the setup we are most comfortable with and the one that has delivered the
   **best overall performance** (accuracy vs. runtime vs. memory) in practice.
   Unless you have a specific reason to deviate, **use the defaults as-is**.

Quick sanity check
------------------
Run the bundled sample **from the project root**, so the default relative paths
(e.g., ``./config/experiment.yaml`` and ``./fantasia/constants.yaml``) are found:

.. code-block:: bash

   cd /path/to/FANTASIA
   fantasia initialize
   fantasia run

.. important::
   If you execute the CLI **outside** the project root, you must provide explicit
   paths (at least to the experiment config), for example:

   .. code-block:: bash

      fantasia initialize --config /abs/path/to/experiment.yaml
      fantasia run --config /abs/path/to/experiment.yaml


Plan your experiment
--------------------
- Choose the **execution mode**.
- Select **models** and **hidden layers**.
- Size **lookup batches** according to available RAM/VRAM.
- Decide on **neighbor budget** and **filters** (redundancy, taxonomy).
- Set a meaningful **prefix** to preserve traceability.

Inputs & execution modes
------------------------
input / only_lookup
  - ``only_lookup: false`` → ``input`` is **FASTA**; the pipeline computes embeddings (HDF5) and then runs lookup.
  - ``only_lookup: true`` → ``input`` is **HDF5** containing per-layer embeddings; only lookup runs.

prefix
  - Name used for the experiment folder and outputs.
  - Keep it unique and descriptive to maintain a clean trace of results.

Similarity-search settings
--------------------------
batch_size
  - Lookup processing batch size for distance computation (RAM/VRAM bound).
  - Larger batches increase throughput but may exhaust memory.

limit_per_entry
  - Number of nearest neighbors retained per query (per model/layer).
  - Increasing this typically **raises recall**; it may **lower precision** and increase runtime.

embedding.distance_metric
  - Distance metric used by the **lookup** stage (configured under ``embedding``).
  - Recommended: **cosine** (scale-invariant, better comparability across models/layers).
  - Note: Euclidean distance depends on vector magnitudes; unless vectors are normalized,
    cross-model/layer comparisons can be distorted.

redundancy_filter / alignment_coverage
  - Optional MMseqs2-based redundancy control on the reference side.
  - ``redundancy_filter``: identity threshold (e.g., ``0.95`` = 95%).
  - ``alignment_coverage``: minimum fractional coverage to accept an alignment.
  - Set both to 0 if redundancy control is not required.

Taxonomy filtering
------------------
taxonomy_ids_to_exclude
  - List of taxonomy IDs to **exclude** from the reference set.

taxonomy_ids_included_exclusively
  - If non-empty, only these IDs are **included** (overrides the exclude list).

get_descendants
  - If ``true``, the filter applies to **all descendants** of the listed IDs
    (NCBI Taxonomy expansion).

Examples
  - ``559292`` — *Saccharomyces cerevisiae*
  - ``6239`` — *Caenorhabditis elegans*

Embedding settings (query side)
-------------------------------
embedding.device
  - ``cuda`` or ``cpu``; GPU is recommended when available.

embedding.queue_batch_size
  - **Enqueuing** granularity for RabbitMQ/internal queues.
  - This is **not** the per-model forward-pass batch size.
  - Per-model batches (see below) should be **strictly smaller** than this value; if not,
    the effective batch will be bounded by the queue packet size.

embedding.max_sequence_length
  - Truncation length (tokens/residues) applied **before** embedding.
  - ``0`` = **no truncation**; sequences exceeding the model’s internal limit will **error out**
    and no embedding will be written.
  - Positive values (e.g., 512, 1024, …) depend on VRAM and your use case.

embedding.models.<ModelKey>.batch_size
  - Per-model embedding batch size (forward pass).
  - Must be **strictly less than** ``embedding.queue_batch_size``.

embedding.models.<ModelKey>.layer_index
  - Hidden layers to extract for that model.
  - Prefer **late layers** for function-oriented signals; optionally add 1–2 **intermediate** layers
    for robustness.
  - Ensure the indices exist in the selected reference (lookup) table; non-existing layers will not
    yield neighbors.

embedding.models.<ModelKey>.distance_threshold
  - Optional per-model cutoff applied **before** capping to ``limit_per_entry``.
  - ``0`` → pure top-k selection; ``> 0`` → filter by threshold first, then cap by top-k.

Model & layer selection
-----------------------
FANTASIA processes queries **per model and per layer**. At any time it holds in memory
only the **current (model, layer)**: the query embeddings for that layer plus the
in-memory reference table for that model.

Memory model (what actually sits in RAM/VRAM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Peak memory is driven by the **selected model** (and current layer), not by how many
  models/layers are configured overall.
- More models/layers mainly increase **wall-clock time**, not peak memory—provided each
  (model, layer) is processed **individually**.
- Each (model, layer) must fit simultaneously with: the model’s **reference table**,
  the **lookup batch** being compared, and temporary **distance buffers**.

Selection guidelines
^^^^^^^^^^^^^^^^^^^^
- Choose models for their **predictive value**; plan around the **largest** model’s
  reference footprint you enable.
- Prefer **late layers** first (often stronger functional signal); add 1–2 **intermediate**
  layers if you need extra stability across datasets.
- Keep ``limit_per_entry`` reasonable; raising it increases recall but may reduce precision.

Batching & limits
^^^^^^^^^^^^^^^^^
- **Lookup OOM** → lower global ``batch_size`` (and, if needed, reduce ``limit_per_entry``).
- **Embedding OOM** → lower per-model ``embedding.models.<ModelKey>.batch_size`` or select fewer layers.
- Very long sequences cost more memory/time; cap via ``embedding.max_sequence_length`` if required.
- ``embedding.queue_batch_size`` controls **enqueue granularity** (not the forward batch). It should be
  **greater than** every per-model ``batch_size`` to avoid being bottlenecked by the queue packet size.

Verification
^^^^^^^^^^^^
- Ensure requested ``layer_index`` values exist in the chosen lookup table; non-existent layers yield
  no neighbors. To verify, **query PIS** (SQL/ORM) for available ``(model, layer)`` pairs or consult the
  loaded reference metadata.
