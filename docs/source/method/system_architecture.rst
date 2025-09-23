System Architecture
===================

Scope
-----
This page describes the runtime architecture of **FANTASIA v2**: components, services, deployment targets, and data stores.
It intentionally avoids method details (e.g., scoring, ontology logic) and focuses on how the system is structured and executed.

Runtime Stack (at a glance)
---------------------------
- **CLI / Runner**: loads YAML config, prepares an experiment directory, and orchestrates stages.
- **ORM layer (PostgreSQL + SQLAlchemy)**: domain entities and reference data served by the pre-existing **Protein Information System (PIS)**; reference tables are **loaded into memory** for throughput.
- **Message broker (RabbitMQ)**: queues embedding workloads and decouples producers/consumers.
- **Embedding back-ends**: model/tokenizer resolution from a registry; **hidden-layer selection** persisted per layer into HDF5.
- **Lookup & Post-processing**: in-memory lookup tables (per model) + GPU/CPU distance kernels; optional taxonomy filters and redundancy control.
- **Data stores**: PostgreSQL (reference) + experiment-local **HDF5** (query embeddings).
- **Model sources**: integrates with **Hugging Face** to load models/tokenizers.

Services & Data Stores
----------------------
**PostgreSQL (SQLAlchemy ORM)**

- The pipeline imports ORM models from PIS and executes queries to obtain sequences, proteins, GO annotations, and embedding types. Data needed for lookup is **materialized in memory** per enabled model to minimize DB round-trips.
- The reference system is vector-aware (e.g., **pgvector**) for curated storage and indexing of embedding vectors.

**RabbitMQ**

- Embedding tasks are **enqueued** and consumed by workers, enabling back-pressure control and multi-model batching.

**Experiment-local HDF5**

* The embedding stage persists **per-layer** embeddings in an HDF5 layout
  `/accession_<ID>/type_<embedding_type_id>/layer_<k>/embedding`.
  Each layer group includes a `shape` attribute, and the accession group may contain a single `sequence` dataset—both available to downstream stages.

Core Components
---------------
**CLI / Runner**

- Commands: ``initialize`` (fetch and load reference embeddings), ``run`` (full pipeline or lookup-only).
- Creates an experiment directory, persists the effective YAML (``experiment_config.yaml``), and wires stage execution.

**Embedding Service (Sequence Embedder)**

- Loads model back-ends via a registry, batches input sequences, supports **hidden-layer selection**, and writes one dataset per layer to HDF5.
- Handles truncation, per-model batch sizes, queue batch size, and device selection.
- Back-ends leverage **Hugging Face** ecosystems (model/tokenizer load).

**Lookup, Store & Post-processing (Embedding LookUp)**

- Builds **in-memory lookup tables** per enabled model (IDs + dense arrays) from the ORM layer; supports **cosine** or **euclidean** distance on GPU (PyTorch) or CPU (SciPy).
- Optional **taxonomy include/exclude** filters (with descendants) and **redundancy filtering** via MMseqs2.
- Produces experiment-scoped CSV/TSV artifacts (details outside this page).

Packaging & Deployment Targets
------------------------------
**pip / PyPI**

- Installable with ``pip`` (from **PyPI** or source). CLI entry points drive the pipeline end-to-end.

**Containers (Docker / Apptainer)**

- Docker images encapsulate runtime dependencies (Python, PyTorch, system libs).
- **Apptainer/Singularity** is used in HPC clusters to run the same container images with GPU passthrough where available.

**HPC batch jobs (SLURM)**

- SLURM job templates (GPU partitions) prepare the environment, bind persistent volumes, and launch the pipeline inside Apptainer.
- Auxiliary containers for **PostgreSQL/pgvector** and **RabbitMQ** can be started within the job prologue and torn down on exit.

Execution Modes
---------------
- **Full pipeline**: FASTA → per-layer HDF5 → lookup → post-processing (default).
- **Lookup-only**: skip embedding and point ``embeddings_path`` to an existing HDF5 file.

Configuration Surfaces (non-exhaustive)
---------------------------------------
- **Embedding**: enabled models, per-model batch size, list of **hidden layers**, device, queue batch size, max sequence length.
- **Lookup**: distance metric (cosine/euclidean), per-model thresholds, taxonomy filters, redundancy filter (MMseqs2), workers.
- **IO & reproducibility**: experiment directory layout and persisted YAML.
- **Services**: DB connection, message broker endpoints, and optional containerized service startup.


System Workflow Overview
------------------------------------------

.. code-block:: text

    ─────────────────────────── STAGE A — EMBEDDING ───────────────────────────
    FASTA ──▶ SequenceEmbedder ──▶ Embedding Workers (PLMs) ──▶ embeddings.h5


    ─────────────────────────── STAGE B — LOOKUP ──────────────────────────────
    embeddings.h5 + PIS (PostgreSQL) ──▶ Lookup Workers ──▶ raw results


    ───────────────────────── POST-PROCESSING ────────────────────────────────
    raw results ──▶ collapse, reliability index, scoring ──▶ final results
