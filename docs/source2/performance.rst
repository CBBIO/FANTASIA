Execution Benchmarks
=====================

The following table summarizes execution times for FANTASIA functional annotation experiments. Each experiment corresponds to the complete proteome of a model organism. All were executed on a single NVIDIA A100 GPU with identical pipeline configuration. The main difference between runs is the number of nearest neighbors (`k`) used in the lookup phase.

+------------------------+-------+------------+---+---------+-------------+------------+------------+--------------+-----------+
| Log file               | Org.  | #Sequences | k | ESM (s) | ProSTT5 (s) | ProtT5 (s) | Lookup (s) | Postproc (s) | Total (s) |
+========================+=======+============+===+=========+=============+============+============+==============+===========+
| fantasia_5634510_1.err | MOUSE | 54,727     | 1 | 926.9   | 943.1       | 937.6      | 2929.2     | 954.1        | 6691.0    |
+------------------------+-------+------------+---+---------+-------------+------------+------------+--------------+-----------+
| fantasia_5634510_2.err | YEAST | 6,066      | 1 | 127.4   | 129.3       | 128.5      | 485.3      | 84.3         | 954.8     |
+------------------------+-------+------------+---+---------+-------------+------------+------------+--------------+-----------+
| fantasia_5634510_3.err | DROME | 22,010     | 1 | 466.4   | 473.5       | 469.4      | 1518.3     | 452.1        | 3379.6    |
+------------------------+-------+------------+---+---------+-------------+------------+------------+--------------+-----------+
| fantasia_5634510_4.err | ARATH | 39,275     | 1 | 833.5   | 846.2       | 844.6      | 2648.9     | 594.0        | 5767.2    |
+------------------------+-------+------------+---+---------+-------------+------------+------------+--------------+-----------+
| fantasia_5634510_5.err | MOUSE | 54,727     | 5 | 963.7   | 978.7       | 970.4      | 3036.3     | 3079.9       | 9028.9    |
+------------------------+-------+------------+---+---------+-------------+------------+------------+--------------+-----------+
| fantasia_5634510_6.err | YEAST | 6,066      | 5 | 131.2   | 132.9       | 131.3      | 493.7      | 340.9        | 1230.0    |
+------------------------+-------+------------+---+---------+-------------+------------+------------+--------------+-----------+
| fantasia_5634510_7.err | DROME | 22,010     | 5 | 517.4   | 522.3       | 511.9      | 1663.0     | 1654.7       | 4869.2    |
+------------------------+-------+------------+---+---------+-------------+------------+------------+--------------+-----------+
| fantasia_5634510_8.err | ARATH | 39,275     | 5 | 872.0   | 885.8       | 870.9      | 2755.6     | 2210.0       | 7594.3    |
+------------------------+-------+------------+---+---------+-------------+------------+------------+--------------+-----------+

.. note::

   - Embeddings were generated sequentially for each model (ESM, ProSTT5, ProtT5).
   - Lookup and post-processing duration increases substantially with higher `k` values.
   - Post-processing includes hit collapsing, reliability scoring and pairwise alignments.

.. warning::

   The **lookup phase** operates over the combined predictions of the three embedding models (ESM, ProSTT5, ProtT5).
   Therefore, overall execution time—particularly in post-processing—will vary if models are added or removed from the pipeline.


Performance Analysis
--------------------

All experiments queried a shared reference table containing **126,582 sequence embeddings** and **623,134 GO term annotations**, corresponding exclusively to proteins with **experimental evidence** (i.e., excluding annotations inferred electronically).

The number of neighbors (`k`) used during the lookup step has a direct impact on execution time, especially in the **post-processing phase**. Increasing `k` results in:

- More hits to evaluate per query.

- Greater redundancy in annotations.

- A larger number of pairwise alignments.

A clear linear trend is observed in the **lookup** and **post-processing** durations as `k` increases from 1 to 5, while the **embedding generation time** remains unaffected.

**Key Observations:**

- For large proteomes like *Mus musculus* (~55k sequences), increasing `k` from 1 to 5 adds over **2000 seconds** to the post-processing step alone.
- For small proteomes like *S. cerevisiae* (~6k sequences), this overhead is proportionally smaller but still significant (~+250s in post-processing).

This suggests a trade-off between **annotation depth** (more neighbors = more GO candidates) and **execution efficiency**, depending on the target proteome and available computational resources.


General Statistics
------------------

The following metrics summarize the computational scope and cost of the full set of benchmarking experiments:

- **Total number of sequences processed:** 244,156
- **Total runtime across all experiments:** 39,515 seconds (~11 h)
- **Average runtime per sequence:** 0.16 seconds

**Average time per sequence (ms):**

- **ESM embedding:** 19.8 ms
- **ProSTT5 embedding:** 20.1 ms
- **ProtT5 embedding:** 19.9 ms
- **Lookup phase:** 63.6 ms
- **Post-processing:** 38.4 ms

These values reflect the average per-sequence cost of each model and pipeline stage when annotating full proteomes from different model organisms under controlled hardware conditions.
