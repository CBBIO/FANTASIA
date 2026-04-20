# Benchmark Example

This directory is intended for compact, benchmarking-oriented examples.

Suggested contents:

- an input FASTA used for benchmark or leakage-control mode
- the corresponding `embeddings.h5` files when helpful
- representative raw result files
- identity-filtered outputs
- a short note describing the model, `k`, and filtering rule used

Recommended mode:

- `k > 1`
- explicit sequence-identity exclusion against the query
- keep the best remaining donor by highest `reliability_index`

Useful helper scripts:

- [`../../scripts/merge_raw_results.py`](../../scripts/merge_raw_results.py)
- [`../../scripts/filter_raw_results_by_identity.py`](../../scripts/filter_raw_results_by_identity.py)
- [`../../scripts/run_sequential_proteomes.sh`](../../scripts/run_sequential_proteomes.sh)

When GPU competition is a concern, sequential launches are recommended so
embedding and lookup jobs do not contend with one another across concurrent runs.
