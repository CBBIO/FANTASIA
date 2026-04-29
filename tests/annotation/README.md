# Annotation Example

This directory contains an annotation-oriented example using the full
`PRUB1_longiso.pep` sample proteome and the Prot-T5 test config.

Run it from the repository root:

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate fantasia-py312
./tests/annotation/run_annotation_example.sh
```

Equivalent explicit command:

```bash
python -m fantasia.main run \
  --config ./config/prott5_test.yaml \
  --input ./data_sample/PRUB1_longiso.pep \
  --prefix annotation_example \
  --base_directory ./fantasia_test_runs \
  --log_path ./fantasia_test_runs/logs \
  --device cuda \
  --DB_HOST localhost \
  --DB_PORT 5432 \
  --only_lookup false \
  --only_embedding false
```

This is a full annotation run: Stage A generates embeddings and Stage B performs
lookup. Start the services first with `docker compose up -d`, and initialize or
restore the reference database before running the example.

Useful environment overrides for the script:

- `FANTASIA_DEVICE=cpu` to force CPU embedding when CUDA is unavailable
- `FANTASIA_DB_PORT=5433`
- `FANTASIA_BASE_DIR=/path/to/fantasia_test_runs`

Recommended mode:

- `k = 1`
- no explicit self-exclusion identity filtering
- maximize coverage for unknown genomes or proteomes not present in the reference set

Useful helper scripts:

- [`run_annotation_example.sh`](run_annotation_example.sh)
- [`../../scripts/merge_raw_results.py`](../../scripts/merge_raw_results.py)
- [`../../scripts/run_sequential_proteomes.sh`](../../scripts/run_sequential_proteomes.sh)

If GPU competition is a concern, sequential launches are recommended.

## PRUB All-Model Annotation

For the minimal PRUB annotation run (`k = 1`, cosine, all models), use:

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate fantasia-py312
./tests/annotation/run_prub_all_models_annotation.sh
```

This uses the full `data_sample/PRUB1_longiso.pep` proteome, enables all five
models, keeps truncation disabled, and performs full embedding + lookup.

For full proteomes on a GPU machine, running those models sequentially is
recommended over loading several large models in the same launch.

## Recorded PRUB All-Model Results

The full PRUB annotation validation run used `k = 1`, cosine distance, all five
models, and sequence truncation disabled. Generated outputs are kept under an
ignored local run directory.

Final embedding counts:

| Model type | Embedded accessions |
| --- | ---: |
| ESM (`type_1`) | 20,223 |
| Prost-T5 (`type_2`) | 20,218 |
| Prot-T5 (`type_3`) | 20,221 |
| Ankh3-Large (`type_4`) | 20,223 |
| ESM3c (`type_5`) | 20,223 |

Raw lookup outputs:

| Model | Raw CSV files | Raw rows |
| --- | ---: | ---: |
| Ankh3-Large | 20,223 | 84,062 |
| ESM | 20,223 | 83,040 |
| ESM3c | 20,223 | 82,903 |
| Prost-T5 | 20,218 | 84,823 |
| Prot-T5 | 20,221 | 79,338 |

Final summarized annotation output:

| Output | Count |
| --- | ---: |
| Accessions in `summary.csv` | 20,223 |
| Summary rows | 298,225 |
| Unique GO terms | 15,429 |
| Ensemble topGO cellular component rows | 91,009 |
| Ensemble topGO molecular function rows | 48,357 |
| Ensemble topGO biological process rows | 158,859 |

Recorded timing for the same run:

| Phase | Start | End | Wall time |
| --- | --- | --- | ---: |
| Full pipeline | 2026-04-28 14:41:30 | 2026-04-28 16:37:16 | 1h 55m 46s |
| Embedding stage, all five models | 2026-04-28 14:41:30 | 2026-04-28 16:09:27 | 1h 27m 57s |
| Lookup, summary, and TopGO export | 2026-04-28 16:09:28 | 2026-04-28 16:37:16 | 27m 48s |
| Lookup search and raw CSV writing | 2026-04-28 16:09:28 | 2026-04-28 16:19:36 | 10m 08s |
| Summary aggregation | 2026-04-28 16:19:48 | 2026-04-28 16:27:06 | 7m 18s |
| TopGO exports | 2026-04-28 16:27:06 | 2026-04-28 16:37:16 | 10m 10s |

This validation used GPU execution (`embedding.device: cuda`,
`lookup.use_gpu: true`) with `lookup.batch_size: 516`.

Per-model embedding timing, measured from the first to the last stored embedding
for each model type in the same log:

| Model | Model type | Embedded accessions | Start | End | Wall time |
| --- | --- | ---: | --- | --- | ---: |
| ESM | `type_1` | 20,223 | 2026-04-28 14:41:57 | 2026-04-28 14:58:50 | 16m 53s |
| ESM3c | `type_5` | 20,223 | 2026-04-28 14:59:08 | 2026-04-28 15:12:30 | 13m 22s |
| Ankh3-Large | `type_4` | 20,223 | 2026-04-28 15:13:26 | 2026-04-28 15:33:07 | 19m 41s |
| Prot-T5 | `type_3` | 20,221 | 2026-04-28 15:33:50 | 2026-04-28 15:51:14 | 17m 24s |
| Prost-T5 | `type_2` | 20,218 | 2026-04-28 15:51:54 | 2026-04-28 16:09:14 | 17m 20s |

Per-model lookup timing, measured from lookup matrix load to the final stored
raw-result acknowledgement for that model:

| Model | Model ID | Reference rows | Dimension | Raw CSV files | Raw rows | Lookup wall time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ESM | 1 | 124,362 | 1,280 | 20,223 | 83,040 | 1m 12s |
| ESM3c | 5 | 124,397 | 1,152 | 20,223 | 82,903 | 1m 05s |
| Ankh3-Large | 4 | 124,336 | 1,536 | 20,223 | 84,062 | 1m 10s |
| Prot-T5 | 3 | 123,977 | 1,024 | 20,221 | 79,338 | 1m 11s |
| Prost-T5 | 2 | 123,375 | 1,024 | 20,218 | 84,823 | 1m 07s |
