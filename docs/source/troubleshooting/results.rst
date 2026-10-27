Missing or empty results
========================

``embeddings.h5`` missing
   Embedding did not finish. The pipeline logs ``The embedding file was not
   created``. Inspect model/download/worker errors.

No raw results
   Confirm HDF5 model/layers match the reference and enabled config. Check
   taxonomy filters and distance thresholds for an empty eligible set.

``The reference database contains no embeddings``
   ``fantasia run`` stops before embedding because the database is empty or
   was never initialized. Run ``fantasia initialize`` and check its log for
   ``pg_restore`` errors (see `PostgreSQL troubleshooting <postgresql.rst>`_).

Run exits with ``TaskFailuresError``
   Some tasks failed and were discarded, or a worker process crashed (for
   example out of memory). The errors are logged above the final message.
   Results for the other proteins are still written, so the outputs are
   incomplete. Fix the cause and rerun; set ``fail_on_task_errors: false`` to
   accept incomplete results with an error in the log instead.

No ``summary.csv``
   Post-processing writes no summary when no raw CSVs exist. Inspect
   ``raw_results/`` first.

Unexpectedly few donors after filtering
   Taxonomy and identity filters can exhaust a small k-neighbourhood. Inspect
   filter reports and consider a larger ``limit_per_entry``.

Output permission error
   Verify write access to ``base_directory`` and ``log_path`` before rerunning.
