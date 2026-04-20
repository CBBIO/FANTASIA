# Test Layout

This directory groups small, reproducible examples that illustrate the two main
intended FANTASIA workflows:

- `annotation/`: lightweight annotation-oriented examples
- `benchmark/`: leakage-control and benchmarking-oriented examples

The helper utilities used by these examples remain in the canonical
[`scripts/`](../scripts) directory so they can be maintained in one place and
reused across workflows.

Recommended helper scripts:

- [`scripts/merge_raw_results.py`](../scripts/merge_raw_results.py)
  Merge many per-accession raw CSV files into a single table for downstream analysis.

- [`scripts/run_sequential_proteomes.sh`](../scripts/run_sequential_proteomes.sh)
  Run proteomes sequentially. This is recommended when GPU contention may become
  a concern.

- [`scripts/filter_raw_results_by_identity.py`](../scripts/filter_raw_results_by_identity.py)
  Apply explicit sequence-identity filtering for benchmark / leakage-control workflows.

Each subdirectory should include:

- the input FASTA or a small representative input
- the configuration used
- expected key outputs or example outputs
- a short note describing the intended workflow
