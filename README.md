[![PyPI version](https://img.shields.io/pypi/v/fantasia)](https://pypi.org/project/fantasia/)
[![Documentation](https://readthedocs.org/projects/fantasia/badge/?version=latest)](https://fantasia.readthedocs.io/en/latest/)
[![Linting](https://github.com/CBBIO/FANTASIA/actions/workflows/test-lint.yml/badge.svg?branch=main)](https://github.com/CBBIO/FANTASIA/actions/workflows/test-lint.yml)

# FANTASIA

Current release: **4.1.1**

**Functional ANnoTAtion based on embedding space SImilArity**

FANTASIA annotates protein FASTA files by embedding query sequences with a
protein language model, comparing them with experimentally annotated reference
proteins, and transferring Gene Ontology (GO) terms from nearby references. The
Full workflow uses PostgreSQL/pgvector and RabbitMQ and supports ProtT5,
ProstT5, ESM-2, ESM3c, and Ankh3-Large.

## Choose Full or Lite

| Use | Choose |
|---|---|
| Proteome-scale runs, five embedding models, database-backed references, taxonomy filters, detailed donor and alignment outputs | **Full FANTASIA** (this repository) |
| A standalone local workflow without PostgreSQL or RabbitMQ | [**FANTASIA-Lite**](https://github.com/CBBIO/FANTASIA-Lite) |

Full FANTASIA is not a lightweight demonstration: the recommended reference
dump is approximately 3.1 GB before database restoration, and the first model
run downloads model weights. Use Lite for classroom, laptop, or simple local
use when the Full infrastructure is unnecessary.

## Quick start

The commands below run a complete 20-sequence ProtT5 annotation example with
cosine distance and `k=1`. Run them from the repository root.

### Requirements

- Linux
- Python `>=3.12,<3.13`
- [Poetry](https://python-poetry.org/)
- Docker with the Compose plugin
- NVIDIA GPU for the commands as written; see the [CPU guide](docs/source/deployment/cpu.rst) for CPU configuration
- Disk space for the reference database and model cache

Recommended starting resources for **one model at a time** are:

| Workload | Free disk | System RAM | GPU VRAM |
|---|---:|---:|---:|
| 20-protein test | 30 GB | 16 GB | 12 GB |
| Full proteome, final-layer reference | 100 GB | 32 GB | 16 GB minimum; 24 GB recommended |
| Full proteome, multilayer/all-model work | 200 GB | 64 GB | 24 GB recommended |

These are operational starting points, not hard guarantees. Peak memory depends
strongly on the selected model, sequence length, model batch size and lookup
batch size. Uncapped long proteins may require more VRAM. CPU execution needs
no VRAM but remains RAM- and disk-bound and is substantially slower. Lower the
model `batch_size` after an embedding out-of-memory error and
`lookup.batch_size` after a lookup out-of-memory error.

The 100/200 GB disk recommendations include headroom beyond the compressed
3.1/17.1 GB downloads for PostgreSQL restoration, model caches, embeddings,
raw per-protein CSVs and consolidated results. Do not size storage from the
compressed archive alone.

MMseqs2 is required only when redundancy masking is enabled. Parasail is
installed as a Python dependency.

### 1. Clone, install, and start services

```bash
git clone https://github.com/CBBIO/FANTASIA.git
cd FANTASIA
poetry install
docker compose up -d
docker compose ps
```

Wait until `fantasia-postgres` and `fantasia-rabbitmq` report a healthy status.
The bundled credentials are for local development only; use managed secrets
and non-default credentials for shared or production deployments.

### 2. Create local working folders

`data/` and `lookup/` are intentionally absent from Git:

```bash
mkdir -p data lookup/{logs,experiments,embeddings}
```

### 3. Load the reference database once

```bash
poetry run fantasia initialize \
  --config ./config/prott5_test.yaml \
  --base_directory ./lookup \
  --log_path ./lookup/logs \
  --embeddings_url 'https://zenodo.org/records/17795871/files/BioData_Dec25_esm2_prott5_prostt5_ankh3_large_esm3c_Layer0.backup?download=1'
```

> **Warning:** initialization resets the `public` schema in the configured
> database. Use the dedicated FANTASIA database created by
> `docker-compose.yml`, never a database containing unrelated data.

### 4. Run the included example

```bash
poetry run fantasia run \
  --config ./config/prott5_test.yaml \
  --input ./data_sample/sample.fasta \
  --prefix first_search \
  --base_directory ./lookup \
  --log_path ./lookup/logs \
  --device cuda \
  --limit_per_entry 1
```

### 5. Verify and inspect the result

```bash
experiment=$(find ./lookup/experiments -maxdepth 1 -type d \
  -name 'first_search_*' | sort | tail -n 1)
test -s "$experiment/summary.csv"
head -n 5 "$experiment/summary.csv"
```

A successful run produces:

```text
lookup/experiments/first_search_<timestamp>/
├── embeddings.h5
├── experiment_config.yaml
├── model_provenance.yaml
├── raw_results/prot-t5/layer_0/*.csv
├── summary.csv
├── sequences.fasta
├── query_index_mapping.csv
└── topgo/  (only when `lookup.topgo: true`)
```

`summary.csv` is the main consolidated accession-by-GO result. It is not one
row per protein: a protein can have many GO rows. The CSV files in each
`raw_results/<model>/layer_<n>/` directory are **per query protein**, not one
proteome-level table. Each file can contain multiple donor–GO rows. Merge the
raw files for one model and layer into a single proteome table with:

```bash
python scripts/merge_raw_results.py \
  "$experiment/raw_results/prot-t5/layer_0" \
  --output "$experiment/raw_results/prot-t5/layer_0_merged.csv" \
  --add-source-file
```

The merge concatenates the original rows without calculating minima, maxima,
means, or other aggregates. `final_score`, when present in `summary.csv`, is a
configuration-dependent ranking score, not a probability.

## Basic usage

For a complete proteome, use the full config (`limit_execution: 0`) and replace
the FASTA path. Plain and gzip-compressed protein FASTA files are accepted.

```bash
poetry run fantasia run \
  --config ./config/prott5_full.yaml \
  --input ./data/my_proteome.faa.gz \
  --prefix my_proteome \
  --base_directory ./lookup \
  --log_path ./lookup/logs \
  --device cuda \
  --limit_per_entry 1
```

## Understand the configuration

The YAML file is part of the scientific definition of a run; do not treat it
only as a list of paths. Start from [`config/prott5_test.yaml`](config/prott5_test.yaml)
for the 20-protein check or [`config/prott5_full.yaml`](config/prott5_full.yaml)
for a complete proteome, copy it, and record your changes.

The main sections are:

| Section/key | Controls |
|---|---|
| Top-level paths and services | Input/output locations and PostgreSQL/RabbitMQ connections |
| `limit_execution` | Number of input sequences processed; `0` means all sequences |
| `embedding` | Device, queue size, sequence-length cap, enabled models, model batch sizes, and layers |
| `lookup` | CPU/GPU lookup, distance metric, lookup batch size, and `limit_per_entry` (`k`) |
| `redundancy` | Optional MMseqs2 identity/coverage filtering; `identity: 0` disables it |
| `taxonomy` | Exact reference taxonomy IDs; `get_descendants` is deprecated/disabled and true is rejected |
| `postprocess` | Sequence retention, summary metrics, aliases, counts, and heuristic score weights |

Critical points:

- At least one entry under `embedding.models` must have `enabled: true`.
- `layer_index: [0]` means the final model layer. The enabled model and layer
  must exist in the restored reference database; query and reference
  embeddings are not interchangeable across models or layers.
- `embedding.max_sequence_length: 0` applies no FANTASIA length cap. A positive
  value truncates sequences before embedding.
- `lookup.distance_metric` selects `cosine` or `euclidean` distance.
- `lookup.limit_per_entry` is the number of nearest reference embeddings
  retained per query. It does not guarantee that many unique donor accessions
  or GO-bearing donors after expansion and filtering.
- `postprocess.summary.metrics` controls the min/max/mean columns in
  `summary.csv`; these aggregations are separate from raw-file merging.
- `taxonomy.get_descendants` is retained only for compatibility. It is
  deprecated and disabled; any true value in CLI, legacy YAML, or nested YAML
  raises an error. List every taxonomy ID explicitly.
- Database passwords in the example files are local Compose defaults. Replace
  them for shared or production deployments and never commit secrets.

### Packaged full-run defaults

`config/prott5_full.yaml` processes all inputs (`limit_execution: 0`) using
uncapped ProtT5 final-layer embeddings (`max_sequence_length: 0`, batch size 1)
on CUDA. Lookup uses GPU cosine distance, batch size 516, `k=1`, one cached
model/layer table, four-decimal output, and TopGO disabled by default.
Redundancy masking is
disabled (`identity: 0`; coverage 0.7 and 10 threads apply only when enabled),
and both exact taxonomy lists are empty. Sequence-aware post-processing is
enabled; it summarizes reliability by maximum and global/local identities by
minimum, maximum and mean, with configured weights 0.4/0.2/0.2/0.2.

All other models are disabled by default; each retains batch size 1, final
layer `[0]`, and no distance threshold. The test config changes the prefix and
sets `limit_execution: 20`. The complete parameter-by-parameter table—including
paths, local service credentials, initialization URL behavior, types, defaults,
and compatibility keys—is the
[configuration defaults reference](docs/source/reference/configuration_reference.rst).

CLI options such as `--input`, `--device`, and `--limit_per_entry` override the
corresponding supported YAML values. See all overrides with:

```bash
poetry run fantasia run --help
```

Every experiment saves the resolved `experiment_config.yaml` and an automatic
`model_provenance.yaml`. The latter records model repositories and immutable
revision identifiers, requested layers, and relevant package versions. Keep
both with the results. The revision is an audit record; current upstream loaders
may not enforce it in every code path. See the complete
[configuration reference](docs/source/reference/configuration_reference.rst).

## Execution modes

| Mode | Setting | Intended use |
|---|---|---|
| Annotation | `limit_per_entry: 1`; no self-exclusion | Annotate unknown proteomes and maximize coverage |
| Benchmark/leakage control | `limit_per_entry > 1`; explicit taxonomy and post-hoc identity filtering | Retain alternative donors after exclusion |
| Embedding only | `--only_embedding true` | Produce `embeddings.h5` without lookup |
| Lookup only | `--only_lookup true --input <embeddings.h5>` | Reuse compatible query embeddings |

Taxonomy filters match exact IDs. `get_descendants` is deprecated and disabled; any true value is rejected.
For benchmark filtering, retrieve several neighbours and use
[`scripts/filter_raw_results_by_identity.py`](scripts/filter_raw_results_by_identity.py).
To combine the per-query raw CSVs into one proteome-level file for each
model/layer, use
[`scripts/merge_raw_results.py`](scripts/merge_raw_results.py), as shown above.

## Reference data

- [Final-layer reference, recommended](https://zenodo.org/records/17795871): approximately 3.1 GB download.
- [Multilayer reference](https://zenodo.org/records/17793273): approximately 17.1 GB download.
- [Versioned benchmark companion dataset](https://doi.org/10.5281/zenodo.20305840).

The reference database and benchmark outputs are different resources. See the
[reference-data guide](docs/source/getting_started/reference_data.rst)
before initialization.

## Documentation

- [Getting started](docs/source/getting_started/index.rst)
- [User guide](docs/source/user_guide/index.rst)
- [Configuration reference](docs/source/reference/configuration_reference.rst)
- [Output files](docs/source/reference/output_files.rst)
- [Deployment](docs/source/deployment/index.rst)
- [Troubleshooting](docs/source/troubleshooting/index.rst)

## Citation

If you use FANTASIA, cite:

1. Martínez-Redondo GI, Barrios I, Vázquez-Valls M, Rojas AM, Fernández R.
   *Illuminating the functional landscape of the dark proteome across the
   Animal Tree of Life.* [doi:10.1101/2024.02.28.582465](https://doi.org/10.1101/2024.02.28.582465)
2. Barrios-Núñez I, Martínez-Redondo GI, Medina-Burgos P, Cases I, Fernández R,
   Rojas AM. *Decoding proteome functional information in model organisms
   using protein language models.*
   [doi:10.1101/2024.02.14.580341](https://doi.org/10.1101/2024.02.14.580341)

## Licence

FANTASIA is distributed under the
[GNU Affero General Public License v3.0](LICENSE).

## Project team and acknowledgements

FANTASIA is developed by Ana Rojas' Lab (CBBIO, CABD-CSIC) and Rosa
Fernández's Lab (Metazoa Phylogenomics Lab, IBE-CSIC-UPF). Project team: Ana M.
Rojas, Rosa Fernández, Belén Carbonetto, and Àlex Domínguez Rodríguez. Past
contributors include Gemma I. Martínez-Redondo, Francisco Miguel Pérez
Canales, and Francisco J. Ruiz Mota.

The project builds on the original
[Metazoa Phylogenomics Lab FANTASIA](https://github.com/MetazoaPhylogenomicsLab/FANTASIA),
[`bio_embeddings`](https://github.com/sacdallago/bio_embeddings),
[`GoPredSim`](https://github.com/Rostlab/goPredSim),
[`MMseqs2`](https://github.com/soedinglab/MMseqs2),
[`Parasail`](https://github.com/jeffdaily/parasail), and the
[`protein-information-system`](https://github.com/CBBIO/protein-information-system).
