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
import hashlib

# --- Third-party libraries ---
import h5py

# --- Project-specific imports ---
from protein_information_system.operation.embedding.sequence_embedding import SequenceEmbeddingManager
from protein_information_system.sql.model.entities.sequence.sequence import Sequence
from protein_information_system.sql.model.entities.embedding.sequence_embedding import SequenceEmbedding


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

    def _parse_fasta_robust(self, fasta_path: str) -> list:
        """
        Robust FASTA parser that handles header continuations properly.

        This parser can handle malformed FASTA files where headers are split across
        multiple lines, which causes issues with BioPython's SeqIO.parse().

        Parameters
        ----------
        fasta_path : str
            Path to the FASTA file to parse

        Returns
        -------
        list
            List of Bio.SeqRecord-like objects with .id and .seq attributes
        """
        import re
        from collections import namedtuple

        # Define amino acid pattern
        AA_RE = re.compile(r'^[ACDEFGHIKLMNPQRSTVWYBXZJOUacdefghiklmnpqrstvwybxzjou]+$')

        # Create a simple SeqRecord-like object
        SeqRecord = namedtuple('SeqRecord', ['id', 'seq'])

        records = []
        cur_header = None
        cur_seq_parts = []
        seq_started = False

        with open(fasta_path, 'r', encoding='utf-8') as handle:
            for raw in handle:
                line = raw.rstrip('\n')
                if not line:
                    continue

                if line.startswith('>'):
                    # Flush previous record
                    if cur_header is not None:
                        seq_str = ''.join(cur_seq_parts)
                        # Extract ID from header (first word)
                        record_id = cur_header.split()[0] if cur_header else "unknown"
                        records.append(SeqRecord(id=record_id, seq=seq_str))

                    cur_header = line[1:].strip()
                    cur_seq_parts = []
                    seq_started = False
                    continue

                s = line.strip()
                if not seq_started:
                    # Check if line looks like a sequence (only amino acid letters)
                    if AA_RE.match(s):
                        seq_started = True
                        cur_seq_parts.append(s.upper())
                    else:
                        # Header continuation (contains spaces, '=', digits, parentheses, etc.)
                        if cur_header is None:
                            cur_header = ''
                        cur_header += ' ' + s
                else:
                    # Sequence continuation lines
                    cur_seq_parts.append(s.replace(' ', '').upper())

        # Flush last record
        if cur_header is not None:
            seq_str = ''.join(cur_seq_parts)
            record_id = cur_header.split()[0] if cur_header else "unknown"
            records.append(SeqRecord(id=record_id, seq=seq_str))

        return records

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

            sequences = self._parse_fasta_robust(input_fasta)

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
            sequence_id (FASTA accession), and embedding_type_id. When possible, embeddings
            are retrieved from the BioData database; otherwise they are computed on-the-fly.

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
            # Configured layers for this model (e.g. [0, 1, 2])
            layer_index_list = list(self.types[model_type].get('layer_index', [0]))

            db_records = []
            sequences_for_embedding_by_layers = {}

            # --- 1) Try to retrieve embeddings from BioData (local DB) ---------
            try:
                self.session_init()
                session = self.session

                # Map each FASTA task to a stable sequence hash (MD5) and accession
                hash_to_tasks = {}
                for task in task_data:
                    seq_str = task["sequence"]
                    accession = task["accession"]
                    seq_hash = hashlib.md5(seq_str.encode("utf-8")).hexdigest()
                    hash_to_tasks.setdefault(seq_hash, []).append({
                        "accession": accession,
                        "sequence": seq_str,
                    })

                # Resolve hashes to Sequence IDs in a single query
                seq_rows = (
                    session.query(Sequence.id, Sequence.sequence_hash)
                    .filter(Sequence.sequence_hash.in_(list(hash_to_tasks.keys())))
                    .all()
                )
                hash_to_seq_id = {row.sequence_hash: row.id for row in seq_rows}

                # Collect all sequence_ids that are present in BioData
                seq_ids_present = {sid for sid in hash_to_seq_id.values() if sid is not None}

                # Fetch existing embeddings for all (sequence_id, embedding_type_id, layer_index)
                emb_by_seq_id = {}
                if seq_ids_present:
                    emb_rows = (
                        session.query(
                            SequenceEmbedding.sequence_id,
                            SequenceEmbedding.layer_index,
                            SequenceEmbedding.embedding,
                        )
                        .filter(
                            SequenceEmbedding.sequence_id.in_(seq_ids_present),
                            SequenceEmbedding.embedding_type_id == embedding_type_id,
                            SequenceEmbedding.layer_index.in_(layer_index_list),
                        )
                        .all()
                    )

                    for seq_id, layer_idx, emb in emb_rows:
                        emb_by_seq_id.setdefault(seq_id, {})[int(layer_idx)] = emb

                # For each task, decide which layers come from DB vs need computation
                for task in task_data:
                    accession = task["accession"]
                    seq_str = task["sequence"]
                    seq_hash = hashlib.md5(seq_str.encode("utf-8")).hexdigest()
                    seq_id = hash_to_seq_id.get(seq_hash)

                    cached_by_layer = emb_by_seq_id.get(seq_id, {}) if seq_id is not None else {}
                    missing_layers = [li for li in layer_index_list if li not in cached_by_layer]

                    # Build records for layers already stored in DB
                    for li, emb in cached_by_layer.items():
                        try:
                            vec = emb.to_numpy()
                        except Exception:
                            vec = emb

                        shape = getattr(vec, "shape", None)
                        if shape is None and hasattr(vec, "__len__"):
                            shape = (len(vec),)

                        db_records.append({
                            "sequence_id": accession,  # keep FASTA accession for HDF5 hierarchy
                            "embedding_type_id": embedding_type_id,
                            "layer_index": int(li),
                            "sequence": seq_str,
                            "embedding": vec,
                            "shape": tuple(shape) if shape is not None else (),
                        })

                    # Log per-accession DB vs compute decision for debugging
                    self.logger.info(
                        "process: accession=%s | model_type=%s | layers_from_db=%s | layers_to_compute=%s",
                        accession,
                        model_type,
                        sorted(cached_by_layer.keys()) if cached_by_layer else [],
                        missing_layers,
                    )

                    # Any missing layers must be computed on-the-fly
                    if missing_layers:
                        key = tuple(sorted(int(li) for li in missing_layers))
                        sequences_for_embedding_by_layers.setdefault(key, []).append({
                            "sequence": seq_str,
                            "sequence_id": accession,
                        })

                session.close()

                self.logger.info(
                    "process: DB lookup completed → %d record(s) served from cache, %d distinct layer-sets to compute.",
                    len(db_records),
                    len(sequences_for_embedding_by_layers),
                )

            except Exception as db_err:
                # If DB is unavailable or query fails, fall back to on-the-fly embeddings
                self.logger.warning(
                    "process: DB lookup failed or unavailable, falling back to on-the-fly embeddings. Error: %s",
                    db_err,
                )
                db_records = []
                sequences_for_embedding_by_layers = {
                    tuple(layer_index_list): [
                        {"sequence": task["sequence"], "sequence_id": task["accession"]}
                        for task in task_data
                    ]
                }

            # --- 2) Compute embeddings for any remaining (sequence, layer) pairs ---
            computed_records = []

            for layers_key, seq_batch in sequences_for_embedding_by_layers.items():
                if not seq_batch:
                    continue

                current_layers = list(layers_key)

                embeddings = module.embedding_task(
                    sequences=seq_batch,
                    model=model,
                    tokenizer=tokenizer,
                    batch_size=batch_size,
                    embedding_type_id=embedding_type_id,
                    device=device,
                    layer_index_list=current_layers,
                )

                # Ensure embedding_type_id is always present
                for record in embeddings:
                    record.setdefault("embedding_type_id", embedding_type_id)

                self.logger.info(
                    "process: computed %d embedding record(s) on-the-fly for model_type=%s, layers=%s",
                    len(embeddings),
                    model_type,
                    current_layers,
                )

                computed_records.extend(embeddings)

            all_records = db_records + computed_records

            self.logger.info(
                "process: batch summary for model_type=%s → %d record(s) from DB, %d record(s) computed, %d total.",
                model_type,
                len(db_records),
                len(computed_records),
                len(all_records),
            )

            if not all_records:
                self.logger.warning("process: no embeddings produced for batch (model_type=%s)", model_type)

            return all_records

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
