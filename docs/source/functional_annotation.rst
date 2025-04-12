.. _functional_annotation:

==========================================
Functional Annotation
==========================================

Objective
---------
This use case describes the **functional annotation process** in **FANTASIA**.
The goal is to predict **functional annotations for unknown sequences**, enabling their classification based on similarity to known protein functions.

FANTASIA leverages **embedding-based approaches** to transfer functional information from well-characterized proteins to unannotated sequences.
This method provides a reliable annotation strategy, especially for proteins with no clear homologs.

The annotation is performed using the three **Gene Ontology (GO)** domains:

- **F**: Molecular Function
- **B**: Biological Process
- **C**: Cellular Component

Annotations are assigned based on similarity to reference datasets following **CAFA** standards (https://geneontology.org/docs/guide-go-evidence-codes/):

- **EXP, IDA, IPI, IMP, IGI, IEP, TAS, IC**

Functional Annotation Procedure
--------------------------------

1. **Input a set of unknown protein sequences**.
2. **Generate embeddings** for each sequence using **ProtT5**, **ProstT5**, or **ESM2**.
3. **Retrieve reference embeddings** from a PostgreSQL + pgvector database.
4. **Compute distances in-memory** to identify most similar annotated proteins.
5. **Transfer GO terms** using model-specific thresholds and redundancy filtering (optional).
6. **Export results** in standard CSV and TopGO-compatible TSV formats.

Input Data
----------

The input must be a single FASTA file containing **protein sequences**.

Example of **FILENAME_query.fasta**:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

   >query1 Unknown protein sequence
   MVKFTASDLKQGERTSLP...
   >query2 Hypothetical protein
   MLFTGASDVKNQTWPAL...

**Note:** Input must consist of **amino acid sequences**, not DNA.

Functional Annotation Configuration
-----------------------------------

Pipeline Configuration
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   input: data_sample/FILENAME_query.fasta
   only_lookup: false
   limit_per_entry: 5
   batch_size: 1
   sequence_queue_package: 64
   length_filter: 5000000
   redundancy_filter: 0
   fantasia_prefix: FILENAME_query_annotated
   delete_queues: true

Embedding Configuration
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   embedding:
     device: cuda  # Options: "cpu", "cuda", or "cuda:0", etc.
     distance_metric: euclidean  # Options: "euclidean", "cosine"
     models:
       esm:
         enabled: false
         distance_threshold: 3
         batch_size: 32
       prost_t5:
         enabled: false
         distance_threshold: 3
         batch_size: 32
       prot_t5:
         enabled: true
         distance_threshold: 3
         batch_size: 32

Functional Analysis
^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   topgo: true

Directory Configuration
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   base_directory: ~/fantasia/
   log_path: ~/fantasia/logs/

Execution Modes
---------------

FANTASIA operates in two main phases, controlled via command-line arguments:

1. **System Initialization** *(optional)*

   Downloads the reference embeddings archive from Zenodo and loads it into a PostgreSQL + pgvector database.

   .. code-block:: console

      fantasia initialize --config config.yaml

   To override the default reference source:

   .. code-block:: yaml

      embeddings_url: <ZENODO_URL>

2. **Pipeline Execution**

   Runs the embedding and GO term annotation steps. Behavior depends on the `only_lookup` setting:

   - `only_lookup: false` → expects input in **FASTA format** and computes new embeddings.
   - `only_lookup: true`  → expects input in **HDF5 format** with precomputed embeddings.

   Run with:

   .. code-block:: console

      fantasia run --config config.yaml

Redundancy Filtering (CD-HIT)
-----------------------------

To avoid assigning GO terms from highly similar proteins in the LOOKUP table, FANTASIA supports optional **redundancy filtering** via **CD-HIT**.

This step is activated by setting an identity threshold:

.. code-block:: yaml

   redundancy_filter: 0.95  # Only keep annotations below 95% sequence identity

CD-HIT will:

- Combine reference sequences and query sequences
- Cluster them based on identity and coverage
- Exclude annotations coming from sequences in the same cluster as the query

This ensures more robust and non-redundant functional transfers.

Lookup-Only Mode (`only_lookup`)
--------------------------------

FANTASIA can skip the embedding calculation step and directly use precomputed embeddings stored in **HDF5 format**.

.. code-block:: yaml

   only_lookup: true
   input: path/to/precomputed_embeddings.h5

This is useful when:

- Embeddings were computed in a previous run
- You want to re-run the annotation with different parameters
- You only want to test the lookup performance

In contrast:

.. code-block:: yaml

   only_lookup: false
   input: path/to/sequences.fasta

In this case, the pipeline will generate embeddings from the input FASTA file.


Results
-------

FANTASIA produces experiment-specific output files stored in a timestamped directory under `~/fantasia/experiments/`.

Main output files:

1. **results.csv**
   Predicted GO annotations for each query sequence:

   - `accession`, `sequence_query`, `sequence_reference`, `go_id`, `category`, `distance`, `reliability_index`, `model_name`
   - Additional info: `evidence_code`, `organism`, `go_description`, etc.

2. **results_topgo.tsv** *(optional)*
   One row per query with comma-separated GO terms to produce **TopGO** input ready-to-use files.

3. **experiment_config.yaml**
   Snapshot of the full configuration used in the run.

4. **embeddings.h5**
   HDF5 file with embeddings and sequences. Required if `only_lookup: true`.

5. **redundancy.fasta**, **filtered.fasta.clstr** *(optional)*
   Intermediate files for CD-HIT clustering (if redundancy filtering is enabled).

Logging
-------

All logs are saved in:

.. code-block:: text

   ~/fantasia/logs/Logs_<timestamp>.log

They include:

- Experiment configuration and parameters
- Pipeline status and batch processing
- Warnings (e.g., missing sequences, threshold filters)
- Embedding memory usage and lookup summaries
- CD-HIT execution info
- Error tracebacks

Advanced Configuration
----------------------

.. code-block:: yaml

   # Worker threads
   max_workers: 1

   # Internal polling interval (in seconds)
   monitor_interval: 10

   # Path to constants file
   constants: ./fantasia/constants.yaml

   # PostgreSQL credentials
   DB_USERNAME: usuario
   DB_PASSWORD: clave
   DB_HOST: localhost
   DB_PORT: 5432
   DB_NAME: BioData

   # RabbitMQ setup
   rabbitmq_host: localhost
   rabbitmq_port: 5672
   rabbitmq_user: guest
   rabbitmq_password: guest
