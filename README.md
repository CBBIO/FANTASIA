![FANTASIA Logo](docs/source/_static/FANTASIA.png)

[![PyPI - Version](https://img.shields.io/pypi/v/fantasia)](https://pypi.org/project/fantasia/)
[![Documentation Status](https://readthedocs.org/projects/fantasia/badge/?version=latest)](https://fantasia.readthedocs.io/en/latest/?badge=latest)
![Linting Status](https://github.com/CBBIO/fantasia/actions/workflows/test-lint.yml/badge.svg?branch=main)



# FANTASIA v4.1

**Functional ANnoTAtion based on embedding space SImilArity**

FANTASIA is an advanced pipeline for the automatic functional annotation of protein sequences using state-of-the-art protein language models. It integrates deep learning embeddings and in-memory similarity searches, retrieving reference vectors from a PostgreSQL database with pgvector-backed storage, to associate Gene Ontology (GO) terms with proteins.

> [!NOTE]
> For the full FANTASIA workflow, input proteomes should be provided as **decompressed FASTA files**.
> If your source file is gzip-compressed, such as `.fa.gz` or `.fasta.gz`, decompress it before
> running embedding or full-pipeline jobs.

> [!IMPORTANT]
> **Two intended usage modes**
>
> **1. Annotation mode**
> - Use `k = 1`
> - Do **not** apply self-exclusion-style identity filtering
> - Goal: maximize annotation coverage for unknown genomes or proteomes not present in the reference set
>
> **2. Benchmark / leakage-control mode**
> - Use `k > 1`
> - Apply explicit sequence-identity exclusion against the query
> - Keep the best remaining donor by highest `reliability_index` after filtering
> - Goal: reduce near-self or near-orthologue leakage during benchmarking

For full documentation, visit [FANTASIA Documentation](https://fantasia.readthedocs.io/en/latest/).

For users who need a lightweight, standalone alternative, FANTASIA-Lite provides fast Gene Ontology annotation directly from local FASTA files, without requiring a database server or the full FANTASIA infrastructure. It leverages protein language model embeddings and nearest-neighbor similarity in embedding space to deliver high-quality functional annotations with minimal setup.

For FANTASIA-Lite, visit https://github.com/CBBIO/FANTASIA-Lite

## Reference Datasets
Two packaged reference datasets are available; select one depending on your analysis needs:

- **Main Reference (last layer, default)**  
  Embeddings extracted only from the **final hidden layer** of each PLM.  
  Recommended for most annotation tasks (smaller, faster to load).  
  *Record*: https://zenodo.org/records/17795871

- **Multilayer Reference (early layers + final layers)**  
  Embeddings extracted from **multiple hidden layers** (including intermediate and final).  
  Suitable for comparative and exploratory analyses requiring layer-wise representations.  
  *Record*: https://zenodo.org/records/17793273


## Key Features

**Available Embedding Models**  
Supports protein language models: **ESM-2**, **ProtT5**, **ProstT5**, **Ankh3-Large**, and **ESM3c** for sequence representation.

- **Redundancy Filtering**  
  Provides optional **MMseqs2-based query-aware redundancy masking** during lookup. This masks donor
  sequences that MMseqs2 assigns to the same cluster as the query, but it does **not** guarantee
  removal of all clearly similar or high-identity donors. For benchmark or leakage-control workflows,
  retrieving more neighbors and applying explicit sequence-identity filtering is the more reliable approach.

- **Optimized Data Storage**  
  Embeddings are stored in **HDF5 format** for input sequences. The reference table, however, is hosted in a **public
  relational PostgreSQL database** using **pgvector**.

- **Efficient Similarity Lookup**  
  High-throughput similarity search with a **hybrid approach**: reference embeddings are stored in a **PostgreSQL + pgvector** database, then loaded **per model/layer into memory** so similarities can be computed efficiently in the application with vectorized CPU or GPU operations. In the repository default configuration, lookup runs on **CPU** unless `lookup.use_gpu: true` is enabled.

- **Sequential Embedding + Lookup**  
  FANTASIA first computes query embeddings and stores them in `embeddings.h5`, then runs the lookup stage. These stages execute sequentially within a run, so embedding and lookup do not compete for GPU resources unless multiple FANTASIA jobs are launched at the same time.

- **Global & Local Alignment of Hits**  
  Candidate hits from the reference table are **aligned both globally and locally** against the input protein for validation and scoring.

- **Multi-layer Embedding Support**  
  Optional support for **intermediate + final layers** to enable layer-wise analyses and improved exploration.
  Layer indices are specified per model in the YAML config under `embedding.models.<Model>.layer_index`.
  Indexing is relative to the output end of the network: `0 = final/output layer`, `1 = penultimate layer`, `2 = second-to-last`, and so on.

- **Raw Outputs & Flexible Post-processing**  
  Exposes **raw result tables** for custom analyses and includes a **flexible post-processing & scoring system** that produces **TopGO-ready** files.  
  Performs high-speed searches using **in-memory computations**. Reference vectors are retrieved from a PostgreSQL database with pgvector-backed storage for comparison.

- **Functional Annotation by Similarity**  
  Assigns Gene Ontology (GO) terms to proteins based on **embedding space similarity**, using pre-trained embeddings from all supported models.

## Pipeline Overview (Simplified)

1. **Embedding Generation**  
   Computes protein embeddings using deep learning models (**ProtT5**, **ProstT5**, **ESM-2**, **Ankh3-Large**, and **ESM3c**).

2. **GO Term Lookup**  
   Performs vector similarity searches using **in-memory computations** to assign Gene Ontology terms. Reference
   embeddings are retrieved from a **PostgreSQL database with pgvector-backed storage** and loaded per model/layer into memory. In the default configuration, this stage runs on **CPU** (`lookup.use_gpu: false`). Only experimental evidence codes are used for transfer.

## GPU Recommendation

The repository default is **CPU lookup** (`lookup.use_gpu: false`). For single-run workflows on CUDA-capable systems, enabling GPU lookup with `lookup.use_gpu: true` is recommended. In the current pipeline, embeddings are generated first and lookup runs afterward, so Stage A and Stage B do not overlap within the same run.

When processing multiple proteomes on a single GPU-equipped machine, a [sequential launcher script](scripts/run_sequential_proteomes.sh) is recommended. Running one proteome at a time preserves the same non-overlapping execution model used within a single FANTASIA run and avoids GPU contention between concurrent jobs. This is often the simplest and most reliable strategy for small-to-medium batches of proteomes.

Example:

```bash
./scripts/run_sequential_proteomes.sh ../config/prott5_full.yaml /path/to/proteomes /path/to/experiments prott5
```

The GPU memory required by the lookup stage depends mainly on:

- the size of the reference embedding matrix
- the lookup query batch size
- the embedding dimensionality
- temporary tensors created during cosine or euclidean distance computation

Because FANTASIA runs embeddings first and lookup afterward, GPU lookup memory requirements do **not** depend on the embedding step being active within the same run.

For a typical single-model **Prot-T5 layer-0** lookup on a proteome, the reference matrix may be on the order of `123,977 x 1024`, with lookup batches such as `516 x 1024` using `float32` tensors. In practice, this fits comfortably on a `24 GB` GPU and is generally expected to fit on a `16 GB` GPU as well. Actual memory requirements still depend on the selected reference dataset, enabled layers/models, and lookup batch size.

### Example Benchmark: CPU vs GPU Lookup

The table below summarizes a lookup-only benchmark on a single proteome using the same precomputed Prot-T5 embeddings and the same reference table. Only the lookup execution device was changed.

Benchmark hardware for the GPU run:

- GPU: `NVIDIA GeForce RTX 3090 Ti`
- VRAM: `24 GB`
- CUDA available in the runtime environment: `True`
- PyTorch build used for the benchmark: `2.11.0+cu130`

| Proteome | Input proteins | Mean protein length (aa) | Max protein length (aa) | Embedded proteins | Lookup tasks | Lookup device | Distance time (total) | Distance time / batch | Lookup wall time |
|----------|----------------|--------------------------|--------------------------|-------------------|--------------|---------------|------------------------|-----------------------|------------------|
| A proteome (Prot-T5, layer 0) | 20,223 | 392.25 | 8,215 | 20,223 | 20,223 | CPU | 1,835.89 s | 45.90 s | 1,933.08 s |
| A proteome (Prot-T5, layer 0) | 20,223 | 392.25 | 8,215 | 20,223 | 20,223 | GPU | 17.05 s | 0.43 s | 126.95 s |

Observed speedup in this benchmark:

- Distance kernel: about `108x` faster on GPU (`1835.89 s` → `17.05 s`)
- Lookup wall time: about `15x` faster on GPU (`1933.08 s` → `126.95 s`)

In this benchmark, no proteins were discarded before embedding: the input FASTA contained `20,223` proteins and the generated `embeddings.h5` also contained `20,223` embedded accessions.

Long proteins were not removed either. Instead, when `embedding.max_sequence_length` is set, FANTASIA truncates sequences longer than that limit before embedding. This means lookup can still cover the full proteome while controlling the memory cost of the embedding stage.

## Interpreting Outputs

FANTASIA writes lookup results in three main forms:

- Per-accession raw CSV files under `raw_results/{model}/layer_{k}/`
- A global `summary.csv` produced during post-processing
- TopGO-ready files under `topgo/`

If you need to consolidate many per-accession raw CSV files into a single table for downstream analysis, use the [merge utility](scripts/merge_raw_results.py).

Example:

```bash
python scripts/merge_raw_results.py \
  /path/to/experiment/raw_results/prot-t5/layer_0 \
  -o /path/to/experiment/raw_results/prot-t5/layer_0_merged.csv \
  --add-source-file
```

### Raw per-accession CSV files

The raw CSVs are the most detailed output. Each row represents one transferred GO annotation associated with one retrieved reference hit for one query protein.

Typical columns include:

- `accession`: query protein accession
- `go_id`: transferred GO term
- `go_description`: GO term name
- `category`: GO namespace, typically `BP`, `MF`, or `CC`
- `distance`: embedding-space distance between the query and the selected reference hit
- `reliability_index`: similarity-derived score computed from `distance`
- `model_name`: embedding model used for the lookup
- `layer_index`: model layer used for the lookup
- `protein_id`, `organism`, `gene_name`: metadata from the matched reference protein
- `evidence_code`: evidence code associated with the transferred annotation
- `query_len`, `ref_len`: query and reference sequence lengths

If sequence-aware storage is enabled, the raw CSVs can also include alignment-derived metrics:

- `identity`, `similarity`, `alignment_score`, `gaps_percentage`: global alignment metrics
- `identity_sw`, `similarity_sw`, `alignment_score_sw`, `gaps_percentage_sw`: local Smith-Waterman-style alignment metrics
- `alignment_length`, `alignment_length_sw`: aligned lengths for the global and local alignments

### Distance and reliability_index

`distance` is the nearest-neighbor distance in embedding space, so lower values indicate a closer reference match.

`reliability_index` is derived from `distance` so that higher values indicate stronger support:

- cosine lookup: `reliability_index = 1 - distance`
- euclidean lookup: `reliability_index = 0.5 / (0.5 + distance)`
- other metrics: `reliability_index = 1 / (1 + distance)`

In practice:

- lower `distance` is better
- higher `reliability_index` is better
- `reliability_index` is the easiest column to rank by in the raw files

### Global versus local alignment metrics

When alignment metrics are present:

- `identity` and related columns summarize the global end-to-end alignment
- `identity_sw` and related columns summarize the best local alignment segment

This is useful because some hits may share only a conserved local region. A protein can therefore have:

- moderate global identity but high local identity
- strong embedding similarity together with weak sequence alignment, or the reverse

These fields are best interpreted as complementary evidence rather than strict pass/fail filters.

### summary.csv

`summary.csv` is the post-processed accession-by-GO summary table. It should be interpreted as the output of a heuristic ranking procedure, not as a table of probabilities. In particular, `final_score` is not a probability score and should not be read as a calibrated confidence value. The table aggregates all raw rows belonging to the same `(accession, go_id, model_name, layer_index)` combination and computes configured statistics such as `min`, `max`, and `mean`.

When sequence-aware exports are enabled, FANTASIA can also write auxiliary files such as `sequences.fasta` and `query_index_mapping.csv` to help relate internal `Q*` identifiers back to parsed query accessions. These mapping aids are provided as optional conveniences for downstream inspection. Their use is left to the user, since FANTASIA's primary goal is to provide a solid and flexible annotation framework rather than to impose a single interpretation or accuracy-estimation workflow.

By default, the repository configuration summarizes:

- `reliability_index`
- `identity`
- `identity_sw`
- support count normalized by `limit_per_entry`

The default aliases are:

- `ri` for `reliability_index`
- `id_g` for global identity
- `id_l` for local identity

In the current code, the support `count` metric is derived from the number of raw rows supporting the same `(accession, go_id, model_name, layer_index)` group, normalized by `limit_per_entry`. This means `count` acts as a support-strength signal rather than a probability: GO terms supported repeatedly across raw hits receive a larger value.

So columns such as `max_ri_ProtT5_L0`, `mean_id_g_ProtT5_L0`, or `max_id_l_ProtT5_L0` in `summary.csv` represent aggregated per-model, per-layer evidence for the same accession and GO term.

If weights are configured, FANTASIA also writes:

- weighted columns prefixed by `w_`
- a composite `final_score`

`final_score` is a configuration-driven heuristic ranking score, not a universal probability or calibrated confidence value. Its objective is to combine several evidence signals into one sortable value so candidate GO terms can be prioritized within the same run and configuration.

In the repository default configuration, `final_score` is built from a weighted combination of:

- the best embedding-derived support (`max_ri`)
- the best global alignment identity (`max_id_g`)
- the best local alignment identity (`max_id_l`)
- the support `count`

This makes `final_score` useful for ranking candidate GO terms, filtering outputs, and downstream prioritization, but its numerical value should not be interpreted as a probability of correctness. Changing the configured metrics or weights changes the meaning of the score.

### TopGO exports

If `lookup.topgo: true`, FANTASIA also exports TopGO-compatible files under `topgo/`.

- Per-model/layer exports keep rows separated by model, layer, and GO category
- Ensemble exports keep the best `reliability_index` per `(accession, go_id, category)` across all models and layers

These files contain three columns in tab-separated form:

- accession
- GO term
- reliability index

## Setting Up Required Services with Docker Compose

FANTASIA requires two key services:
- **PostgreSQL 16 with pgvector**: Stores reference protein embeddings used by the lookup stage
- **RabbitMQ**: Message broker for distributed embedding task processing

### Prerequisites
- **Python 3.12** (the project metadata specifies `>=3.12,<4.0`)
  A Conda environment based on Python 3.12 is a suitable local setup option.
- Docker and Docker Compose installed

Additional dependency notes:

- **MMseqs2** is required if you enable redundancy filtering during lookup. FANTASIA invokes the external `mmseqs` executable, so it must be installed separately and available in your `PATH`.
  In the current workflow, this feature should be interpreted as query-aware redundancy masking rather
  than a guaranteed exclusion of all clearly similar donors.
- **Parasail** is used for alignment-based post-processing through its Python package. When FANTASIA is installed through its declared Python dependencies, `parasail` is provided by the runtime environment and does not need to be invoked as a separate command-line tool.
- **Taxonomy descendant expansion** (`taxonomy.get_descendants: true`) is currently disabled. The original implementation relied on `ete3.NCBITaxa` and its local NCBI taxonomy database rather than on the FANTASIA PostgreSQL reference database, which introduced environment-dependent behavior. The current taxonomy filter therefore works on the exact taxonomy IDs you provide. For benchmark-style exclusions, manually list the relevant species, subspecies, or related taxa in `taxonomy_ids_to_exclude` and keep `get_descendants: false`.

Execution modes:

- Default run: embedding + lookup
- `only_lookup: true`: skip embedding and use an existing `embeddings.h5`
- `only_embedding: true`: stop after generating `embeddings.h5`
- `only_lookup: true` and `only_embedding: true` cannot be used together

> **Deployment note**
> These updates do not change the overall deployment strategy for Docker, Slurm, or array-based cluster execution. The main changes are at the application level:
> - explicit support for `only_embedding: true`
> - clearer disabling of `distance_threshold` (for example `false` instead of legacy `0`, while keeping backward compatibility)
> - corrected and clarified taxonomy filtering behavior
> - recommendation to use decompressed FASTA files for full embedding and full-pipeline runs
> - optional generation of `query_index_mapping.csv` for sequence-aware outputs
>
> Existing deployment wrappers should therefore remain structurally valid, but may require small updates if they assume the previous threshold convention, gzipped FASTA inputs, or older output expectations.

### Quick Start

1. **Start services** (from the FANTASIA directory):
   ```bash
   docker-compose up -d
   ```

2. **Verify services are running**:
   ```bash
   docker-compose ps
   ```

   Expected output:
   ```
   CONTAINER ID   IMAGE                           STATUS
   xxx            pgvector/pgvector:0.7.0-pg16   Up (healthy)
   xxx            rabbitmq:3.13-management       Up (healthy)
   ```

3. **Test database connection**:
   ```bash
   PGPASSWORD=clave psql -h localhost -U usuario -d BioData -c "SELECT 1"
   ```

### Service Credentials

The `docker-compose.yml` is configured with the following default credentials (matching `config.yaml`):

| Service    | Host       | Port  | User     | Password | Database |
|------------|-----------|-------|----------|----------|----------|
| PostgreSQL | localhost | 5432  | usuario  | clave    | BioData  |
| RabbitMQ   | localhost | 5672  | guest    | guest    | -        |

`BioData` is the default local PostgreSQL database name used for the restored reference lookup table downloaded from Zenodo. It is a configurable database name, not a separate repository requirement.

RabbitMQ Management UI is available at: http://localhost:15672 (user: guest, password: guest)

### Troubleshooting

**Connection refused error**:
```bash
# Check if containers are running
docker-compose ps

# If stopped, restart them
docker-compose restart

# View logs
docker-compose logs postgres
docker-compose logs rabbitmq
```

**Password authentication failed**:
Ensure the credentials in `docker-compose.yml` match those in `config.yaml`:
```bash
# Current values in docker-compose.yml
POSTGRES_USER: usuario
POSTGRES_PASSWORD: clave
POSTGRES_DB: BioData
```

**Cleaning up**: To remove containers and volumes:
```bash
docker-compose down -v
```

## Supported Embedding Models

| Name         | Model ID                                 | Params | Architecture      | Description                                                                 |
|--------------|-------------------------------------------|--------|-------------------|-----------------------------------------------------------------------------|
| **ESM-2**     | `facebook/esm2_t33_650M_UR50D`            | 650M   | Encoder (33L)     | Learns structure/function from UniRef50. No MSAs. Optimized for accuracy.  |
| **ProtT5**    | `Rostlab/prot_t5_xl_uniref50`             | 1.2B   | Encoder-Decoder   | Trained on UniRef50. Strong transfer for structure/function tasks.         |
| **ProstT5**   | `Rostlab/ProstT5`                         | 1.2B   | Multi-modal T5     | Learns 3Di structural states + function. Enhances contact/function tasks.  |
| **Ankh3-Large** | `ElnaggarLab/ankh3-large`              | 620M   | Encoder (T5-style)| Fast inference. Good semantic/structural representation.                   |
| **ESM3c**     | `esmc_600m`                               | 600M   | Encoder (36L)     | New gen. model trained on UniRef + MGnify + JGI. High precision & speed.   |


## Acknowledgments

FANTASIA is the result of a collaborative effort between **Ana Rojas’ Lab (CBBIO)** (Andalusian Center for Developmental
Biology, CSIC) and **Rosa Fernández’s Lab** (Metazoa Phylogenomics Lab, Institute of Evolutionary Biology, CSIC-UPF).
This project demonstrates the synergy between research teams with diverse expertise.

This version of FANTASIA builds upon previous work from:

- [`Metazoa Phylogenomics Lab's FANTASIA`](https://github.com/MetazoaPhylogenomicsLab/FANTASIA)  
  The original implementation of FANTASIA for functional annotation.

- [`bio_embeddings`](https://github.com/sacdallago/bio_embeddings)  
  A state-of-the-art framework for generating protein sequence embeddings.

- [`GoPredSim`](https://github.com/Rostlab/goPredSim)  
  A similarity-based approach for Gene Ontology annotation.

- [`MMseqs2`](https://github.com/soedinglab/MMseqs2)  
  Used for optional query-aware redundancy masking during lookup workflows.

- [`Parasail`](https://github.com/jeffdaily/parasail)  
  Provides high-performance pairwise sequence alignment routines used in hit validation and post-processing.

- [`protein-information-system`](https://github.com/CBBIO/protein-information-system)  
  Serves as the **reference biological information system**, providing a robust data model and curated datasets for
  protein structural and functional analysis.

We also extend our gratitude to **LifeHUB-CSIC** for inspiring this initiative and fostering innovation in computational
biology.

## Citing FANTASIA

If you use **FANTASIA** in your research, please cite the following publications:

1. Martínez-Redondo, G. I., Barrios, I., Vázquez-Valls, M., Rojas, A. M., & Fernández, R. (2024).  
   *Illuminating the functional landscape of the dark proteome across the Animal Tree of Life.*  
   [DOI: 10.1101/2024.02.28.582465](https://doi.org/10.1101/2024.02.28.582465)

2. Barrios-Núñez, I., Martínez-Redondo, G. I., Medina-Burgos, P., Cases, I., Fernández, R., & Rojas, A. M. (2024).  
   *Decoding proteome functional information in model organisms using protein language models.*  
   [DOI: 10.1101/2024.02.14.580341](https://doi.org/10.1101/2024.02.14.580341)


## License

FANTASIA is distributed under the terms of the [GNU Affero General Public License v3.0](LICENSE).


---

### Project Team

- **Ana M. Rojas**: [a.rojas.m@csic.es](mailto:a.rojas.m@csic.es)
- **Rosa Fernández**: [rosa.fernandez@ibe.upf-csic.es](mailto:rosa.fernandez@ibe.upf-csic.es)
- **Belén Carbonetto**: [belen.carbonetto.metazomics@gmail.com](mailto:belen.carbonetto.metazomics@gmail.com)
- **Àlex Domínguez Rodríguez**: [adomrod4@upo.es](mailto:adomrod4@upo.es)

### Past Contributors

- **Gemma I. Martínez-Redondo**: [gemma.martinez@ibe.upf-csic.es](mailto:gemma.martinez@ibe.upf-csic.es)
- **Francisco Miguel Pérez Canales**: [fmpercan@upo.es](mailto:fmpercan@upo.es)
- **Francisco J. Ruiz Mota**: [fraruimot@alum.us.es](mailto:fraruimot@alum.us.es)

---
