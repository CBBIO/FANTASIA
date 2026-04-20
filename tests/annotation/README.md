# Annotation Example

This directory is intended for compact, annotation-oriented examples.

Suggested contents:

- an input FASTA used for annotation mode
- the corresponding `embeddings.h5` when helpful
- representative output files such as `summary.csv` or selected raw results
- a short note describing the model and configuration used

Recommended mode:

- `k = 1`
- no explicit self-exclusion identity filtering
- maximize coverage for unknown genomes or proteomes not present in the reference set

Useful helper scripts:

- [`../../scripts/merge_raw_results.py`](../../scripts/merge_raw_results.py)
- [`../../scripts/run_sequential_proteomes.sh`](../../scripts/run_sequential_proteomes.sh)

If GPU competition is a concern, sequential launches are recommended.
