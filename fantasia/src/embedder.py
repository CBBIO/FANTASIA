"""
Sequence Embedding Module
=========================

Overview
--------
The **Sequence Embedding Module** provides a high-throughput workflow for computing and
persisting protein sequence embeddings from FASTA input files. It serves as an orchestration
layer around model loading, batching, task publication, and structured storage in HDF5.

Responsibilities
----------------
This module defines :class:`SequenceEmbedder`, a concrete implementation built on top of
:class:`protein_information_system.operation.embedding.sequence_embedding.SequenceEmbeddingManager`.
Its main responsibilities are:

- Parsing input sequences from FASTA files, with optional truncation by length.
- Enqueuing embedding tasks for **all configured hidden-layer indices** of each model.
- Executing model-specific embedding routines with dynamic model loading.
- Writing embeddings to HDF5 with a stable hierarchy and minimal metadata.

Processing Pipeline
-------------------
1. **Ingest**: Parse sequences from the configured FASTA file using Biopython.
2. **Batch**: Partition sequences into queue batches (``queue_batch_size``) to control message size.
3. **Dispatch**: For each enabled model, publish a task containing all batch sequences and
   the full list of requested layer indices.
4. **Embed**: Load the appropriate model, tokenizer, and module, then compute embeddings.
5. **Persist**: Store results in ``embeddings.h5`` under per-accession, per-model, per-layer groups.

Input / Output
--------------
**Input**: A single- or multi-sequence FASTA file.
**Output**: An HDF5 file named ``embeddings.h5`` with the structure:

.. code-block::

   /accession_<ID>/
       /type_<embedding_type_id>/
           /layer_<k>/
               embedding   (dataset)
               shape       (attribute)
       sequence            (dataset, optional; stored once per accession)

Configuration
-------------
The module expects a dictionary ``conf`` with at least the following keys:

- ``input`` : Path to the input FASTA file.
- ``experiment_path`` : Output directory where ``embeddings.h5`` will be written.
- ``embedding.models`` : Model-level configuration:
  - ``enabled`` (bool) : Whether this model should be enqueued.
  - ``layer_index`` (list[int]) : Hidden-layer indices to extract.
- ``embedding.batch_size`` (dict[str,int]) : Per-model batch sizes at embedding time.
- ``embedding.queue_batch_size`` (int) : Number of sequences per published message.
- ``embedding.max_sequence_length`` (int | None) : Optional truncation length.

Operational Notes
-----------------
- **No DB dependency**: Enqueueing does not query a database or require sequence IDs in advance.
- **All layers extracted**: For each model, all configured layers are included (no aggregation).
- **Device selection**: Defaults to ``"cuda"`` unless overridden by ``conf["embedding"]["device"]``.
- **Idempotency**: Existing per-layer datasets are skipped rather than overwritten.

Error Handling & Logging
------------------------
- Missing FASTA or I/O errors are raised and logged.
- Inconsistent batches (multiple ``embedding_type_id`` values) trigger a ``ValueError``.
- Each storage operation logs whether a dataset was created or skipped.

Dependencies
------------
- `Biopython <https://biopython.org/>`_ (FASTA parsing via ``Bio.SeqIO``).
- `h5py <https://www.h5py.org/>`_ (structured storage).
- Model registry and dynamic loading provided by
  :class:`protein_information_system.operation.embedding.sequence_embedding.SequenceEmbeddingManager`.

Public API
----------
- :meth:`SequenceEmbedder.enqueue`
    Read FASTA, batch sequences, and enqueue per-model tasks with all configured layers.
- :meth:`SequenceEmbedder.process`
    Load the appropriate model/tokenizer/module, embed a batch, and return records.
- :meth:`SequenceEmbedder.store_entry`
    Persist per-layer embeddings and metadata into ``embeddings.h5``.

Intended Use
------------
This module is the **first stage** in an embedding-driven functional annotation pipeline.
Downstream consumers typically perform similarity search, annotation transfer, or clustering
using the stored embeddings.

"""

# --- Standard library ---
import os
import traceback

# --- Third-party libraries ---
import h5py
from Bio import SeqIO

# --- Project-specific imports ---
from protein_information_system.operation.embedding.sequence_embedding import SequenceEmbeddingManager


class SequenceEmbedder(SequenceEmbeddingManager):
    """
    High-throughput computation of protein sequence embeddings from FASTA input.

    The :class:`SequenceEmbedder` orchestrates model loading, batching, optional
    sequence truncation, and storage of per-layer embeddings into HDF5. It supports
    multiple embedding models in parallel and produces structured outputs suitable
    for downstream similarity search, annotation transfer, or clustering.

    Parameters
    ----------
    conf : dict
        Configuration dictionary with input paths, model definitions, batch sizes,
        and optional filters.
    current_date : str
        Timestamp string for naming outputs and logs.

    Attributes
    ----------
    fasta_path : str
        Path to the input FASTA file with sequences to embed.
    experiment_path : str
        Directory where ``embeddings.h5`` and logs are written.
    queue_batch_size : int
        Number of sequences per published task message.
    max_sequence_length : int
        Optional truncation length (0 disables truncation).
    batch_sizes : dict
        Per-model embedding batch sizes.
    model_instances : dict
        Dynamically loaded model objects, keyed by ``embedding_type_id``.
    tokenizer_instances : dict
        Tokenizer objects, keyed by ``embedding_type_id``.
    types : dict
        Metadata for each enabled model (e.g. thresholds, batch size, module).
    results : list
        In-memory embedding results (used for aggregation/debugging).
    """

    def __init__(self, conf, current_date):
        """
        Initialize the SequenceEmbedder with configuration settings and paths.

        Loads the selected embedding models, sets file paths and filters, and prepares
        internal structures for managing embeddings and batching.

        Parameters
        ----------
        conf : dict
            Configuration dictionary containing input paths, model settings, and batch parameters.
        current_date : str
            Timestamp used for generating unique output names and logging.
        """
        super().__init__(conf)
        self.current_date = current_date
        self.reference_attribute = "sequence_embedder_from_fasta"

        # Debug / test mode
        self.limit_execution = conf.get("limit_execution")

        # Input and output paths
        self.fasta_path = conf.get("input")
        self.experiment_path = conf.get("experiment_path")

        # Optional batch and filtering settings
        self.queue_batch_size = conf.get("embedding", {}).get("queue_batch_size", 1)
        self.max_sequence_length = conf.get("embedding", {}).get("max_sequence_length", 0)

        # --- Logging of configuration ---
        if not self.fasta_path:
            self.logger.warning("No FASTA input path provided in conf['input']")
        else:
            self.logger.info("FASTA input path: %s", self.fasta_path)

        if not self.experiment_path:
            self.logger.warning("No experiment_path set → outputs may fail")
        else:
            self.logger.info("Experiment outputs will be written under: %s", self.experiment_path)

        self.logger.info(
            "Embedding configuration: queue_batch_size=%d | max_sequence_length=%s | limit_execution=%s",
            self.queue_batch_size,
            str(self.max_sequence_length) if self.max_sequence_length else "disabled",
            str(self.limit_execution) if self.limit_execution else "unlimited"
        )

        # Optional: log enabled models summary
        enabled_models = [k for k, v in conf.get("embedding", {}).get("models", {}).items() if v.get("enabled")]
        if enabled_models:
            self.logger.info("Enabled models: %s", ", ".join(enabled_models))
        else:
            self.logger.warning("No embedding models enabled in configuration")

    def enqueue(self) -> None:
        """
        Read the input FASTA and enqueue *all* sequences for *all* enabled models,
        emitting *all* configured layers for each model in a single message per model.
        """
        try:
            self.logger.info("enqueue: starting embedding enqueue process (all models, all layers).")

            # --- 0) Truncation limit ---
            max_len = getattr(self, "max_sequence_length", None)
            if max_len is not None and (not isinstance(max_len, int) or max_len <= 0):
                self.logger.warning("enqueue: invalid 'max_sequence_length'=%r → ignoring truncation.", max_len)
                max_len = None

            # --- 1) Load FASTA ---
            input_fasta = os.path.expanduser(self.fasta_path)
            if not os.path.exists(input_fasta):
                raise FileNotFoundError(f"FASTA file not found at: {input_fasta}")

            sequences = list(SeqIO.parse(input_fasta, "fasta"))

            # Optional cap
            limit_exec = getattr(self, "limit_execution", None)
            if isinstance(limit_exec, int) and limit_exec > 0:
                sequences = sequences[:limit_exec]
                self.logger.info("enqueue: limit_execution=%d → truncated to %d sequences.", limit_exec, len(sequences))

            if not sequences:
                self.logger.warning("enqueue: no sequences found in FASTA → nothing to enqueue.")
                return

            self.logger.info("enqueue: loaded %d sequences from %s", len(sequences), input_fasta)

            # --- 2) Partition ---
            queue_batch_size = int(getattr(self, "queue_batch_size", 1)) or 1
            sequence_batches = [
                sequences[i: i + queue_batch_size]
                for i in range(0, len(sequences), queue_batch_size)
            ]
            self.logger.info("enqueue: partitioned into %d batches (size=%d).",
                             len(sequence_batches), queue_batch_size)

            # --- 3) Iterate batches ---
            total_published = 0
            for batch_idx, batch in enumerate(sequence_batches, start=1):
                model_batches: dict[str, list[dict]] = {}

                for seq_record in batch:
                    accession = seq_record.id
                    seq_str = str(seq_record.seq)

                    if max_len and len(seq_str) > max_len:
                        self.logger.debug("enqueue: truncating sequence %s from %d → %d residues.",
                                          accession, len(seq_str), max_len)
                        seq_str = seq_str[:max_len]

                    models_cfg = self.conf.get("embedding", {}).get("models", {}) or {}
                    if not models_cfg:
                        self.logger.error("enqueue: config missing 'embedding.models'. Aborting.")
                        return

                    for model_name, model_cfg in models_cfg.items():
                        if not model_cfg.get("enabled", False):
                            continue

                        type_info = self.types.get(model_name)
                        if not type_info:
                            self.logger.warning("enqueue: model '%s' missing in types registry. Skipping.", model_name)
                            continue

                        embedding_type_id = type_info.get("id")
                        backend_model_name = type_info.get("model_name")
                        if embedding_type_id is None or not backend_model_name:
                            self.logger.warning(
                                "enqueue: incomplete type info for model '%s' (id=%r, model_name=%r). Skipping.",
                                model_name, embedding_type_id, backend_model_name
                            )
                            continue

                        desired_layers = (
                                model_cfg.get("layer_index") or
                                type_info.get("layer_index") or []
                        )

                        if not isinstance(desired_layers, (list, tuple)) or not desired_layers:
                            self.logger.warning("enqueue: model '%s' has no 'layer_index' configured. Skipping.",
                                                model_name)
                            continue

                        try:
                            normalized_layers = sorted({int(x) for x in desired_layers})
                        except Exception:
                            self.logger.warning("enqueue: invalid 'layer_index' values for model '%s': %r. Skipping.",
                                                model_name, desired_layers)
                            continue

                        task_data = {
                            "sequence": seq_str,
                            "accession": accession,
                            "model_name": backend_model_name,
                            "embedding_type_id": embedding_type_id,
                            "layer_index": normalized_layers,
                        }
                        model_batches.setdefault(model_name, []).append(task_data)

                # --- 4) Publish ---
                for model_name, batch_data in model_batches.items():
                    if not batch_data:
                        continue
                    try:
                        self.publish_task(batch_data, model_name)
                        total_published += len(batch_data)
                        self.logger.info(
                            "enqueue: batch %d/%d → published %d seqs to model '%s' (id=%s, layers=%d).",
                            batch_idx, len(sequence_batches),
                            len(batch_data), model_name, self.types[model_name]["id"],
                            len(batch_data[0]["layer_index"])
                        )
                        self.logger.debug("enqueue: batch %d/%d, model '%s', accessions=%s",
                                          batch_idx, len(sequence_batches),
                                          model_name, [d['accession'] for d in batch_data])
                    except Exception as pub_err:
                        self.logger.error("enqueue: failed to publish batch for model '%s': %s", model_name, pub_err)
                        raise

            self.logger.info("enqueue: finished. Published %d sequences across %d batches.",
                             total_published, len(sequence_batches))

        except FileNotFoundError:
            self.logger.exception("enqueue: FASTA file not found.")
            raise
        except Exception as e:
            self.logger.error("enqueue: unexpected error → %s", e)
            self.logger.debug("enqueue: traceback:\n%s", traceback.format_exc())
            raise

    def process(self, task_data):
        """
        Computes embeddings for a batch of protein sequences using a specific model.

        Each task in the batch must reference the same `embedding_type_id`, which is used
        to retrieve the appropriate model, tokenizer, and embedding module. The method
        delegates the actual embedding logic to the dynamically loaded module.

        Parameters
        ----------
        task_data : list of dict
            A batch of embedding tasks. Each task should include:
            - 'sequence': str, amino acid sequence.
            - 'accession': str, identifier of the sequence.
            - 'embedding_type_id': str, key for the embedding model.

        Returns
        -------
        list of dict
            A list of embedding records. Each record includes the embedding vector, shape,
            accession, and embedding_type_id.

        Raises
        ------
        ValueError
            If the batch includes multiple embedding types.
        Exception
            For any other error during embedding generation.
        """
        try:

            if not task_data:
                self.logger.warning("No task data provided for embedding. Skipping batch.")
                return []

            # Ensure all tasks belong to the same model
            embedding_type_id = task_data[0]["embedding_type_id"]
            if not all(task["embedding_type_id"] == embedding_type_id for task in task_data):
                raise ValueError("All tasks in the batch must have the same embedding_type_id.")

            # Load model, tokenizer and embedding logic

            model_type = self.types_by_id[embedding_type_id]['name']
            model = self.model_instances[model_type]
            tokenizer = self.tokenizer_instances[model_type]
            module = self.types[model_type]['module']

            device = self.conf["embedding"].get("device", "cuda")

            batch_size = self.types[model_type]["batch_size"]

            layer_index_list = self.types[model_type].get('layer_index', [0])

            # Prepare input: list of {'sequence', 'sequence_id'}
            sequence_batch = [
                {"sequence": task["sequence"], "sequence_id": task["accession"]}
                for task in task_data
            ]

            # Run embedding task
            embeddings = module.embedding_task(
                sequence_batch,
                model=model,
                tokenizer=tokenizer,
                batch_size=batch_size,
                embedding_type_id=embedding_type_id,
                device=device,
                layer_index_list=layer_index_list
            )

            # Enrich results with task metadata
            for record, task in zip(embeddings, task_data):
                record["accession"] = task["accession"]
                record["embedding_type_id"] = embedding_type_id
            return embeddings

        except Exception as e:
            self.logger.error(f"Error during embedding computation: {e}\n{traceback.format_exc()}")
            raise

    def store_entry(self, results):
        """
        Persist per-layer embeddings into an HDF5 file using a stable, idempotent group hierarchy.
        """
        try:
            # --- Normalize input ----------------------------------------------------
            if isinstance(results, dict):
                results = [results]
            elif not isinstance(results, (list, tuple)):
                raise TypeError(f"store_entry: expected dict or list[dict], got {type(results)}")

            output_h5 = os.path.join(self.experiment_path, "embeddings.h5")

            self.logger.info("store_entry: writing %d record(s) → %s", len(results), output_h5)

            # --- Open HDF5 in append mode -------------------------------------------
            with h5py.File(output_h5, "a") as h5file:
                for record in results:
                    # --- Minimal validations ----------------------------------------
                    for key in ("sequence_id", "embedding_type_id", "layer_index", "embedding"):
                        if key not in record:
                            raise KeyError(f"store_entry: missing required key '{key}' in record")

                    accession = record["sequence_id"].replace("|", "_")
                    embedding_type_id = record["embedding_type_id"]
                    layer_index = int(record["layer_index"])

                    accession_group = h5file.require_group(f"accession_{accession}")
                    type_group = accession_group.require_group(f"type_{embedding_type_id}")
                    layer_group = type_group.require_group(f"layer_{layer_index}")

                    # --- Write embedding dataset (idempotent) -----------------------
                    if "embedding" not in layer_group:
                        layer_group.create_dataset("embedding", data=record["embedding"])
                        layer_group.attrs["shape"] = tuple(
                            record.get("shape", getattr(record.get("embedding"), "shape", ()))
                        )
                        self.logger.info(
                            "store_entry: stored embedding → accession=%s | type=%s | layer=%d | shape=%s",
                            accession, embedding_type_id, layer_index, layer_group.attrs["shape"]
                        )
                    else:
                        self.logger.debug(
                            "store_entry: embedding already exists → accession=%s | type=%s | layer=%d (skipped)",
                            accession, embedding_type_id, layer_index
                        )

                    # --- Store raw sequence once per accession ----------------------
                    if "sequence" in record and "sequence" not in accession_group:
                        accession_group.create_dataset("sequence", data=record["sequence"].encode("utf-8"))
                        self.logger.info("store_entry: stored sequence → accession=%s", accession)

            self.logger.info("store_entry: successfully committed %d record(s) to HDF5", len(results))

        except Exception as e:
            self.logger.error("store_entry: error while writing to HDF5 → %s", e)
            self.logger.debug("store_entry: traceback:\n%s", traceback.format_exc())
            raise
