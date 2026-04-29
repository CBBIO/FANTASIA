# Test Layout

This directory groups small, reproducible examples that illustrate the two main
intended FANTASIA workflows:

- `annotation/`: lightweight annotation-oriented examples
- `benchmark/`: leakage-control and benchmarking-oriented examples

The helper utilities used by these examples remain in the canonical
[`scripts/`](../scripts) directory so they can be maintained in one place and
reused across workflows.

Runnable examples:

```bash
# Unit/regression tests
python -m pytest tests

# Small annotation example: embedding + lookup on data_sample/PRUB1_longiso_10random.pep
./tests/annotation/run_annotation_example.sh

# PRUB annotation: k=1, cosine, all models
./tests/annotation/run_prub_all_models_annotation.sh

# Small benchmark example: Prot-T5, cosine, k=5, exclude taxonomy 10090
./tests/benchmark/run_benchmark_example.sh

# Full no-truncation benchmark rerun for mouse and worm
./tests/benchmark/run_no_truncation_mouse_worm_benchmark.sh
```

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

## Recorded Full-Run Results

The release validation runs were written under ignored local output directories
so the repository can document the results without committing generated CSV,
HDF5, FASTA, log, or topGO artifacts.

- `benchmark/`: full mouse and worm no-truncation benchmark results are
  summarized in [`benchmark/README.md`](benchmark/README.md).
- `annotation/`: full PRUB all-model annotation results are summarized in
  [`annotation/README.md`](annotation/README.md).
