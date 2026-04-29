# Benchmark Example

This directory contains benchmark/leakage-control workflows for the packaged
mouse and worm proteomes.

Run it from the repository root:

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate fantasia-py312
./tests/benchmark/run_benchmark_example.sh
```

Equivalent explicit command:

```bash
python -m fantasia.main run \
  --config ./config/musm_benchmark_prott5_k5.yaml \
  --input ./data_sample/MUSM_10090.fasta \
  --prefix musm_benchmark_prott5_k5 \
  --base_directory ./fantasia_benchmark_runs \
  --log_path ./fantasia_benchmark_runs/logs \
  --device cuda \
  --DB_HOST localhost \
  --DB_PORT 5432 \
  --limit_per_entry 5 \
  --taxonomy_ids_to_exclude 10090 \
  --only_lookup false \
  --only_embedding false
```

This is a full mouse Prot-T5 benchmark run: Stage A generates embeddings, then
Stage B performs cosine lookup with `k = 5` while excluding mouse taxonomy ID
`10090`.

Start the services first with `docker compose up -d`, and initialize or restore
the reference database before running the example.

## Full Benchmark

To rerun the full mouse and worm benchmark matrix with truncation disabled and
record proteins that are not embedded, use:

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate fantasia-py312
./tests/benchmark/run_no_truncation_mouse_worm_benchmark.sh
```

Inputs:

- Mouse: `data_sample/MUSM_10090.fasta` (`taxid = 10090`)
- Worm: `data_sample/UP000001940_6239.fasta` (`taxid = 6239`)

Outputs are written under `fantasia_no_truncation_benchmark_runs/`.
Per-run missing-embedding reports are written under:

```text
fantasia_no_truncation_benchmark_runs/missing_embeddings/
```

The runner covers the previous matrix: baseline, self-exclude, and
self-exclude-redundancy modes for all five models at `k = 1`, plus the Prot-T5
`k = 3` supplemental runs. To run only one slice:

```bash
./tests/benchmark/run_no_truncation_mouse_worm_benchmark.sh self-exclude prot-t5
```

Equivalent explicit lookup-only command for the Prot-T5 mouse
self-exclude+redundancy table:

```bash
python -m fantasia.main run \
  --config /tmp/lookup_full_mouse_self-exclude-redundancy_prot-t5_k5.yaml \
  --input /path/to/embed_full_mouse_prot-t5/embeddings.h5 \
  --prefix lookup_full_mouse_self-exclude-redundancy_prot-t5_k5 \
  --base_directory ./fantasia_full_proteome_benchmark_runs \
  --log_path ./fantasia_full_proteome_benchmark_runs/logs \
  --device cuda \
  --DB_HOST localhost \
  --DB_PORT 5432 \
  --only_lookup true \
  --limit_per_entry 5 \
  --taxonomy_ids_to_exclude 10090 \
  --redundancy_identity 0.95 \
  --redundancy_coverage 0.7 \
  --threads 10
```

The important leakage-control switches are `--taxonomy_ids_to_exclude 10090`
for species self-exclusion and `--redundancy_identity 0.95
--redundancy_coverage 0.7` for MMseqs2-based redundancy masking. For worm,
use taxonomy ID `6239` and the corresponding worm Prot-T5 `embeddings.h5`.

Useful environment overrides for the script:

- `FANTASIA_DEVICE=cpu` to force CPU embedding when CUDA is unavailable
- `FANTASIA_DB_PORT=5433`
- `FANTASIA_K=10`
- `FANTASIA_TAXONOMY_EXCLUDE=10090`
- `FANTASIA_BASE_DIR=/path/to/fantasia_benchmark_runs`

Recommended mode:

- `k > 1`
- explicit sequence-identity exclusion against the query
- keep the best remaining donor by highest `reliability_index`

Useful helper scripts:

- [`run_benchmark_example.sh`](run_benchmark_example.sh)
- [`../../scripts/merge_raw_results.py`](../../scripts/merge_raw_results.py)
- [`../../scripts/filter_raw_results_by_identity.py`](../../scripts/filter_raw_results_by_identity.py)
- [`../../scripts/run_sequential_proteomes.sh`](../../scripts/run_sequential_proteomes.sh)

When GPU competition is a concern, sequential launches are recommended so
embedding and lookup jobs do not contend with one another across concurrent runs.

## Recorded Full-Proteome Results

These release-validation benchmark outputs were generated with sequence
truncation disabled. The generated run directories and CSV outputs are ignored
by git.

Baseline per-model timing summary:

| Organism | Model | Input proteins | Embedded proteins | Failed sequences | Embedding time | k=1 lookup | k=5 lookup | Total with k=1 | Total with k=5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mouse | Prot-T5 | 21,852 | 21,830 | 22 | 26m 45s | 6m 45s | 9m 57s | 33m 29s | 36m 41s |
| Mouse | ESM | 21,852 | 21,851 | 1 | 24m 14s | 7m 14s | 10m 09s | 31m 28s | 34m 23s |
| Mouse | ESM3c | 21,852 | 21,852 | 0 | 15m 44s | 7m 08s | 10m 20s | 22m 53s | 26m 04s |
| Mouse | Ankh3-Large | 21,852 | 21,851 | 1 | 29m 11s | 7m 05s | 10m 08s | 36m 17s | 39m 19s |
| Mouse | Prost-T5 | 21,852 | 21,830 | 22 | 27m 35s | 6m 40s | 9m 43s | 34m 15s | 37m 18s |
| Worm | Prot-T5 | 19,831 | 19,822 | 9 | 17m 39s | 5m 50s | 7m 38s | 23m 29s | 25m 17s |
| Worm | ESM | 19,831 | 19,829 | 2 | 17m 04s | 6m 01s | 7m 52s | 23m 06s | 24m 56s |
| Worm | ESM3c | 19,831 | 19,831 | 0 | 14m 14s | 5m 57s | 7m 57s | 20m 11s | 22m 11s |
| Worm | Ankh3-Large | 19,831 | 19,828 | 3 | 19m 54s | 6m 08s | 8m 13s | 26m 03s | 28m 07s |
| Worm | Prost-T5 | 19,831 | 19,822 | 9 | 17m 45s | 5m 46s | 7m 32s | 23m 31s | 25m 17s |

Aggregate all-model baseline timings were `2h 38m 22s` for mouse k=1,
`2h 53m 46s` for mouse k=5, `1h 56m 20s` for worm k=1, and `2h 05m 48s`
for worm k=5.

Prot-T5 self-exclusion timing summary:

| Organism | Mode | Embedded proteins | k=1 lookup | k=5 lookup |
| --- | --- | ---: | ---: | ---: |
| Mouse | self taxonomy excluded | 21,830 | 7m 27s | 9m 19s |
| Mouse | self taxonomy excluded + redundancy | 21,830 | 16m 44s | 18m 14s |
| Worm | self taxonomy excluded | 19,822 | 7m 11s | 7m 51s |
| Worm | self taxonomy excluded + redundancy | 19,822 | 16m 45s | 18m 22s |

Summary row counts for the Prot-T5 leakage-control runs:

| Organism | Mode | k | Summary rows | Accessions | GO terms |
| --- | --- | ---: | ---: | ---: | ---: |
| Mouse | self-exclude | 1 | 130,561 | 21,830 | 13,574 |
| Mouse | self-exclude | 5 | 448,754 | 21,830 | 18,624 |
| Mouse | self-exclude + redundancy | 1 | 122,438 | 21,830 | 13,673 |
| Mouse | self-exclude + redundancy | 5 | 437,409 | 21,830 | 18,459 |
| Worm | self-exclude | 1 | 80,723 | 19,822 | 9,042 |
| Worm | self-exclude | 5 | 324,006 | 19,822 | 15,353 |
| Worm | self-exclude + redundancy | 1 | 80,675 | 19,822 | 9,038 |
| Worm | self-exclude + redundancy | 5 | 323,721 | 19,822 | 15,331 |

Column-aware checks of the raw `protein_id` field confirmed that same-organism
hits were absent from the self-exclude runs: no `_MOUSE` hits in mouse
self-exclude outputs and no `_CAEEL` hits in worm self-exclude outputs.
