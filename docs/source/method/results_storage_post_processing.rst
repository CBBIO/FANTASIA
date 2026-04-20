Result Storage and Post-processing
==================================

Scope
-----
This section covers the two final stages of FANTASIA:

1. **Result storage** —  neighbor hits are expanded and written to disk.
2. **Post-processing** — stored results are aggregated, scored, and collapsed into final outputs.

Store Entry
-----------
The ``store_entry`` method expands lookup hits into full annotation rows
and persists them to disk for each query accession per model & layer. It bridges the **lookup phase**
and the **post-processing phase** by generating reproducible raw result files that
can be reused without recomputing lookup.

**Workflow**

1. **Input**

   - Compact hits from lookup: ``(accession, ref_sequence_id, distance, model, layer)``.

2. **Expansion**

   - For each hit, fetch the associated GO annotations (and sequences if enabled).
   - Attach metadata such as ``model_name``, ``layer_index``, and a
     ``reliability_index`` derived from the distance.
   - Expand into full annotation rows ready for storage.

3. **Persistence**

   - Write results into per-accession CSVs under ``raw_results/{model}/layer_{k}/``.
   - Each row represents one transferred GO annotation associated with one retrieved reference hit
     for one query protein.
   - A combined ``sequences.fasta`` is also written if sequences are kept.
   - If redundancy masking (e.g., MMseqs2) is enabled, cluster definitions
     are generated.

**Outputs**

- Per-accession raw files under ``raw_results/``.
- ``sequences.fasta`` with all query and reference sequences (optional).
- Cluster structures, only if redundancy masking is active.


Post-processing
---------------
The ``post_processing`` method aggregates the raw per-accession CSVs, computes
weighted scores, and produces a global summary along with enrichment-ready
exports. This is the final consolidation step of the pipeline.

**Workflow**

1. **Locate inputs**

   - Collect all CSV shards under ``raw_results/**``.
   - Group them by accession.

2. **Load configuration**

   - Parameters are taken from ``conf['postprocess']['summary']``.
   - Defines metrics, aliases, inclusion of counts, weights, and weighted prefix.

3. **Aggregation per accession**

   - Concatenate all shards for an accession.
   - Group rows by ``(accession, go_id, model_name, layer_index)``.
   - Compute aggregation metrics (``mean``, ``max``, ``min``) for each configured column.
   - Add normalized support counts (neighbors / k) if enabled.

4. **Weighting and scoring**

   - Resolve weights from configuration, normalize them, and apply them to aggregated metrics.
   - Produce weighted columns (``w_<metric>``) and a composite ``final_score``.
   - ``final_score`` is a configuration-driven heuristic ranking score, not a calibrated
     probability.
   - Preserve both global (``final_score``) and per-model/layer scores.

5. **Summarization**

   - Join aggregated values with counts and protein lists per GO term.
   - Pivot metrics into wide format to produce a concise accession–GO table.
   - Write incrementally to ``summary.csv``.

6. **Exports**
   - Write a global ``summary.csv``.
   - Generate TopGO-compatible files for per-model/layer and ensemble configurations (``topgo/...``).

Output interpretation notes
---------------------------

- ``distance`` is the embedding-space nearest-neighbor distance, so lower values are better.
- ``reliability_index`` is derived from ``distance`` and is the easiest raw column to rank by.
- ``summary.csv`` aggregates rows by ``(accession, go_id, model_name, layer_index)`` and reports
  configured summary metrics.
- By default, support ``count`` is normalized by ``limit_per_entry`` and should be interpreted as a
  support-strength signal, not as a probability.
- TopGO exports are written as three-column tab-separated files:
  accession, GO term, and reliability index.
