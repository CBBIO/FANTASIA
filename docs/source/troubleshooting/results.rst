Missing or empty results
========================

``embeddings.h5`` missing
   Embedding did not finish. The pipeline logs ``The embedding file was not
   created``. Inspect model/download/worker errors.

No raw results
   Confirm HDF5 model/layers match the reference and enabled config. Check
   taxonomy filters and distance thresholds for an empty eligible set.

No ``summary.csv``
   Post-processing writes no summary when no raw CSVs exist. Inspect
   ``raw_results/`` first.

Unexpectedly few donors after filtering
   Taxonomy and identity filters can exhaust a small k-neighbourhood. Inspect
   filter reports and consider a larger ``limit_per_entry``.

Output permission error
   Verify write access to ``base_directory`` and ``log_path`` before rerunning.
