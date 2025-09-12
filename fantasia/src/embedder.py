"""
Sequence Embedding Module
=========================

Summary
-------
The **Sequence Embedding Module** exposes a high-throughput workflow to compute and
persist protein sequence embeddings from FASTA input files. It provides a thin orchestration
layer around model loading, batching, task publication, and structured storage in HDF5.

Scope
-----
This module defines :class:`SequenceEmbedder`, a concrete implementation built on top of
:class:`protein_information_system.operation.embedding.sequence_embedding.SequenceEmbeddingManager`.
It is responsible for:

- Reading FASTA inputs and optional truncation by length.
- Enqueuing per-model tasks, including **all** configured hidden-layer indices.
- Executing model-specific embedding routines.
- Writing per-accession, per-type, per-layer embeddings to an HDF5 hierarchy.

Processing Pipeline
-------------------
1. **Ingest**: Parse sequences from the configured FASTA file using Biopython.
2. **Batch**: Partition sequences into queue batches (``queue_batch_size``) to control message size.
3. **Dispatch**: For each enabled model, publish a single message containing all tasks in the batch,
   with the full list of requested layer indices (multi-layer extraction).
4. **Embed**: During processing, load the appropriate model/tokenizer/module and run the embedding task.
5. **Persist**: Store results in ``embeddings.h5`` using a stable hierarchy and minimal metadata.

Input / Output
--------------
**Input**: A FASTA file (single or multi-sequence).

**Output**: An HDF5 file named ``embeddings.h5`` with the structure:

.. code-block::

   /accession_<ID>/
       /type_<embedding_type_id>/
           /layer_<k>/
               embedding  (dataset)
               shape      (attribute)
       sequence           (dataset, optional; stored once per accession)

Configuration (Required Keys)
-----------------------------
The module reads its configuration from ``conf`` (``dict``), expecting at least:

- ``input``: Path to the input FASTA file.
- ``experiment_path``: Directory where ``embeddings.h5`` will be written.
- ``embedding.models``: Per-model configuration with:
  - ``enabled`` (``bool``): Whether to enqueue this model.
  - ``layer_index`` (``list[int]``): Hidden-layer indices to extract for this model.
- ``embedding.batch_size`` (``dict[str,int]``): Model-level batch sizes used at embedding time.
- ``embedding.queue_batch_size`` (``int``): Number of sequences grouped per published message.
- ``embedding.max_sequence_length`` (``int`` or ``None``): Optional truncation length.

Operational Notes
-----------------
- **No DB lookup**: ``enqueue`` does not query a database nor require pre-existing sequence IDs.
- **All layers**: For each model, *all* configured layers are sent (no aggregation at enqueue time).
- **Device selection**: The processing step defaults to ``"cuda"`` unless overridden in ``conf["embedding"]["device"]``.
- **Idempotency**: Existing per-layer datasets are not overwritten; attempts are logged and skipped.

Error Handling & Logging
------------------------
- File system and parsing errors (e.g., missing FASTA) are surfaced and logged.
- Inconsistent batches (multiple ``embedding_type_id`` values) raise a ``ValueError``.
- Each storage operation logs whether a dataset was created or skipped.

Dependencies
------------
- `Biopython <https://biopython.org/>`_ for FASTA parsing (``Bio.SeqIO``).
- `h5py <https://www.h5py.org/>`_ for structured storage.
- A model registry and dynamic loading provided by
  :class:`protein_information_system.operation.embedding.sequence_embedding.SequenceEmbeddingManager`.

Public API (Overview)
---------------------
- :meth:`SequenceEmbedder.enqueue`
    Read FASTA, form queue-sized batches, and publish per-model tasks with all configured layers.
- :meth:`SequenceEmbedder.process`
    Resolve model/tokenizer/module, run the embedding task over a batch, and return records.
- :meth:`SequenceEmbedder.store_entry`
    Write per-layer embeddings and metadata into ``embeddings.h5`` under a stable hierarchy.

Intended Use
------------
This component is the **first stage** in an embedding-driven functional annotation pipeline.
Downstream consumers typically perform similarity search, annotation transfer, or clustering
using the stored representations.

See Also
--------
- `BioEmbeddings Project Documentation <https://docs.bioembeddings.com>`_
"""

import os
import traceback

from Bio import SeqIO

import h5py

from protein_information_system.operation.embedding.sequence_embedding import SequenceEmbeddingManager


class SequenceEmbedder(SequenceEmbeddingManager):
    """
    SequenceEmbedder computes protein embeddings from FASTA-formatted sequences and stores them in HDF5 format.

    This class supports dynamic model loading, batch-based processing, optional sequence filtering,
    and structured output suitable for downstream similarity-based annotation. It is designed to integrate
    seamlessly with a database of embedding model definitions and can enqueue embedding tasks across multiple models.

    Parameters
    ----------
    conf : dict
        Configuration dictionary specifying input paths, enabled models, batch sizes, and filters.
    current_date : str
        Timestamp used for naming outputs and logging purposes.

    Attributes
    ----------
    fasta_path : str
        Path to the input FASTA file containing sequences to embed.
    experiment_path : str
        Directory for writing output files (e.g., embeddings.h5).
    batch_sizes : dict
        Dictionary of batch sizes per model, controlling how sequences are grouped during embedding.
    model_instances : dict
        Loaded model objects, keyed by embedding_type_id.
    tokenizer_instances : dict
        Loaded tokenizer objects, keyed by embedding_type_id.
    types : dict
        Metadata for each enabled model, including threshold, batch size, and loaded module.
    results : list
        List of processed embedding results (used optionally during aggregation or debugging).
    """

    def __init__(self, conf, current_date):
        """
        Initializes the SequenceEmbedder with configuration settings and paths.

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

        # Debug mode
        self.limit_execution = conf.get("limit_execution")

        # Input and output paths
        self.fasta_path = conf.get("input")  # Actual input FASTA
        self.experiment_path = conf.get("experiment_path")

        # Optional batch and filtering settings
        self.queue_batch_size = conf.get('embedding', {}).get("queue_batch_size", 1)
        self.max_sequence_length = conf.get("embedding", {}).get("max_sequence_length", 0)

    def enqueue(self) -> None:
        """
        Read the input FASTA and enqueue *all* sequences for *all* enabled models,
        emitting *all* configured layers for each model in a single message per model.

        Design & constraints
        --------------------
        • No database lookups. No use of sequence_id.
        • Requires configuration keys:
            self.conf["embedding"]["models"][<model_name>]["enabled"] -> bool
            self.conf["embedding"]["models"][<model_name>]["layer_index"] -> list[int]
          and a type registry entry:
            self.types[<model_name>] -> {"id": <embedding_type_id>, "model_name": <backend_model_name>, ...}

        Behavior
        --------
        1) Load the FASTA file (optionally truncate sequences to `self.max_sequence_length`).
        2) Partition the input into batches of size `self.queue_batch_size`.
        3) For each batch, group tasks by model:
           - Skip models that are disabled or missing type info.
           - Validate that `layer_index` is a non-empty list of integers.
           - Enqueue one compact message per model with all sequences in the batch and the full list of layers.
        4) Log progress and corner cases; raise on hard failures (e.g., missing FASTA).

        Side effects
        ------------
        • Calls `self.publish_task(batch_data, model_name)` once per model for each batch.

        Raises
        ------
        FileNotFoundError
            If the FASTA file does not exist.
        Exception
            Propagates unexpected errors after logging.

        Notes
        -----
        • This method *requires* layered embeddings downstream (no legacy paths).
        • Keep messages compact: sequences + metadata + list of `layer_index` integers (no embeddings here).
        """

        try:
            self.logger.info("Starting embedding enqueue process (FASTA, all layers, no DB).")

            # --- 0) Truncation limit from config (may be None) ---
            max_len = getattr(self, "max_sequence_length", None)
            if max_len is not None and (not isinstance(max_len, int) or max_len <= 0):
                self.logger.warning("'max_sequence_length' (%r). Ignoring truncation (Up to VRAM avaliable).", max_len)
                max_len = None

            # --- 1) Load FASTA (no pre-filter by length) ---
            input_fasta = os.path.expanduser(self.fasta_path)
            if not os.path.exists(input_fasta):
                raise FileNotFoundError(f"FASTA file not found at: {input_fasta}")

            sequences = list(SeqIO.parse(input_fasta, "fasta"))

            # Optional execution cap for dry-runs or smoke tests
            limit_exec = getattr(self, "limit_execution", None)
            if isinstance(limit_exec, int) and limit_exec > 0:
                sequences = sequences[:limit_exec]

            if not sequences:
                self.logger.warning("No sequences found. Finishing embedding enqueue process.")
                return

            # --- 2) Partition into batches to bound message size ---
            queue_batch_size = int(getattr(self, "queue_batch_size", 1)) or 1
            sequence_batches = [
                sequences[i: i + queue_batch_size]
                for i in range(0, len(sequences), queue_batch_size)
            ]

            # --- 3) Iterate batches and build per-model payloads ---
            for batch_idx, batch in enumerate(sequence_batches, start=1):
                model_batches: dict[str, list[dict]] = {}

                for seq_record in batch:
                    accession = seq_record.id
                    seq_str = str(seq_record.seq)

                    # Truncate if exceeding max_len (when defined)
                    if max_len and len(seq_str) > max_len:
                        seq_str = seq_str[:max_len]

                    # Iterate configured models
                    models_cfg = self.conf.get("embedding", {}).get("models", {}) or {}
                    if not models_cfg:
                        self.logger.error("Config missing 'embedding.models'. Aborting enqueue.")
                        return

                    for model_name, model_cfg in models_cfg.items():
                        # Skip disabled models
                        if not model_cfg.get("enabled", False):
                            continue

                        type_info = self.types.get(model_name)
                        if not type_info:
                            self.logger.warning(
                                "Model '%s' not present in 'types' registry. Skipping.", model_name
                            )
                            continue

                        embedding_type_id = type_info.get("id")
                        backend_model_name = type_info.get("model_name")
                        if embedding_type_id is None or not backend_model_name:
                            self.logger.warning(
                                "Type info incomplete for model '%s' (id=%r, model_name=%r). Skipping.",
                                model_name, embedding_type_id, backend_model_name
                            )
                            continue

                        # Resolve desired layers: prefer config; fallback to type registry
                        desired_layers = (
                            model_cfg.get("layer_index") or
                            type_info.get("layer_index") or
                            []
                        )

                        # Validate layers: must be a non-empty list of integers
                        if not isinstance(desired_layers, (list, tuple)) or not desired_layers:
                            self.logger.warning(
                                "No 'layer_index' configured for model '%s'. Skipping.", model_name
                            )
                            continue

                        # Normalize, deduplicate, sort and validate integers
                        try:
                            normalized_layers = sorted({int(x) for x in desired_layers})
                        except Exception:
                            self.logger.warning(
                                "Invalid 'layer_index' values for model '%s': %r. Skipping.",
                                model_name, desired_layers
                            )
                            continue

                        task_data = {
                            "sequence": seq_str,
                            "accession": accession,
                            "model_name": backend_model_name,
                            "embedding_type_id": embedding_type_id,
                            "layer_index": normalized_layers,  # full layer list for downstream expansion
                        }
                        model_batches.setdefault(model_name, []).append(task_data)

                # --- 4) Publish one message per model (more efficient than one per sequence) ---
                for model_name, batch_data in model_batches.items():
                    if not batch_data:
                        continue
                    try:
                        self.publish_task(batch_data, model_name)
                        self.logger.info(
                            "Batch %d/%d · Published %d sequences to model '%s' "
                            "(type_id=%s) with %d configured layers.",
                            batch_idx, len(sequence_batches),
                            len(batch_data), model_name, self.types[model_name]["id"],
                            len(batch_data[0]["layer_index"])
                        )
                    except Exception as pub_err:
                        self.logger.error(
                            "Failed to publish batch for model '%s': %s", model_name, pub_err
                        )
                        raise

        except FileNotFoundError:
            # Re-raise after logging for upstream handling
            self.logger.exception("FASTA file not found.")
            raise
        except Exception as e:
            self.logger.error("Unexpected error during enqueue: %s", e)
            self.logger.debug("Traceback:\n%s", traceback.format_exc())
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
        Persist per-layer embeddings into an HDF5 file using a stable, idempotent
        group hierarchy. The storage layout is:

            /accession_<ACCESSION>/type_<EMBEDDING_TYPE_ID>/layer_<LAYER_INDEX>/embedding
            └─ each `layer_*` group includes an attribute: attrs["shape"] = (<rows>, <cols>)
            └─ once per accession (optional): /accession_.../sequence  (UTF-8 bytes)

        Behavior
        --------
        - Accepts a single record (dict) or a sequence of records (list/tuple of dicts).
        - Creates the file `<experiment_path>/embeddings.h5` if it does not exist.
        - Creates missing HDF5 groups on demand (idempotent).
        - Skips writing the dataset if an `embedding` dataset already exists for a given
          (accession, embedding_type_id, layer_index) triple.
        - Writes the `sequence` dataset once per accession (if provided and not already present).
        - Persists the embedding shape as `layer_group.attrs["shape"]`.

        Required Record Keys
        --------------------
        Each record (dict) MUST contain:
          - sequence_id (str): Identifier for the sequence; `|` characters will be replaced with `_`.
          - embedding_type_id (int | str): Model/type identifier for the embedding.
          - layer_index (int): Zero-based layer index associated with the embedding.
          - embedding (np.ndarray): 1D or 2D array. If 1D, treated as a single vector;
            if 2D, treated as per-residue / stacked vectors.

        Optional Record Keys
        --------------------
          - shape (tuple): Explicit shape to store; if absent, inferred from `embedding.shape`.
          - sequence (str): Raw amino-acid sequence; stored once per accession as bytes (UTF-8).

        Idempotency & Concurrency
        -------------------------
        - Idempotent at the (accession, embedding_type_id, layer_index) level:
          existing `embedding` datasets are not overwritten.
        - This method is not inherently thread-safe. If multiple processes/threads can
          write to the same HDF5 file, coordinate external locks to avoid I/O races.

        Parameters
        ----------
        results : dict | list[dict]
            A single embedding record or a collection of embedding records.

        Raises
        ------
        TypeError
            If `results` is neither a dict nor a list/tuple of dicts.
        KeyError
            If a required key is missing from a record (e.g., `layer_index`, `sequence_id`,
            `embedding_type_id`, or `embedding`).
        Exception
            Propagates any I/O or HDF5 errors encountered while writing.

        Returns
        -------
        None

        Examples
        --------
        Single record:
            store_entry({
                "sequence_id": "sp|Q9Y2X3|FOO_HUMAN",
                "embedding_type_id": "ProtT5",
                "layer_index": 18,
                "embedding": np.random.rand(1024),   # 1D vector
                "shape": (1024,),                    # optional; inferred if omitted
                "sequence": "MSEQNNTEMTFQIQRIYTKDIS..."
            })

        Batch of records:
            store_entry([
                {"sequence_id": "A0A0123456", "embedding_type_id": 1, "layer_index": 0, "embedding": E0},
                {"sequence_id": "A0A0123456", "embedding_type_id": 1, "layer_index": 1, "embedding": E1},
            ])
        """
        try:
            # Normalize to a list for uniform iteration.
            if isinstance(results, dict):
                results = [results]
            elif not isinstance(results, (list, tuple)):
                raise TypeError(f"store_entry expects dict or list[dict], got: {type(results)}")

            output_h5 = os.path.join(self.experiment_path, "embeddings.h5")

            # Open (or create) the HDF5 file in append mode. This is safe for idempotent writes,
            # but not inherently safe for concurrent writers—ensure external locking if needed.
            with h5py.File(output_h5, "a") as h5file:
                for record in results:
                    # ---- Minimal validations (fail fast on malformed input) --------------------
                    if "layer_index" not in record:
                        raise KeyError(
                            "Missing 'layer_index' in embedding record. "
                            "Ensure your upstream 'process' includes it per record."
                        )
                    if "sequence_id" not in record:
                        raise KeyError("Missing 'sequence_id' in embedding record.")
                    if "embedding_type_id" not in record:
                        raise KeyError("Missing 'embedding_type_id' in embedding record.")
                    if "embedding" not in record:
                        raise KeyError("Missing 'embedding' in embedding record.")

                    # Sanitize accession: map pipes to underscores to keep HDF5 path safe.
                    accession = record["sequence_id"].replace("|", "_")
                    embedding_type_id = record["embedding_type_id"]
                    layer_index = int(record["layer_index"])

                    # ---- Create or fetch the group hierarchy -----------------------------------
                    accession_group = h5file.require_group(f"accession_{accession}")
                    type_group = accession_group.require_group(f"type_{embedding_type_id}")
                    layer_group = type_group.require_group(f"layer_{layer_index}")

                    # ---- Write the embedding dataset (idempotent) ------------------------------
                    if "embedding" not in layer_group:
                        layer_group.create_dataset("embedding", data=record["embedding"])

                        # Set shape metadata: prefer explicit `shape`, otherwise infer from array.
                        layer_group.attrs["shape"] = tuple(
                            record.get("shape", getattr(record.get("embedding"), "shape", ()))
                        )

                        self.logger.info(
                            f"Stored embedding for accession {accession}, "
                            f"type {embedding_type_id}, layer {layer_index}."
                        )
                    else:
                        self.logger.warning(
                            f"Embedding already exists for accession {accession}, "
                            f"type {embedding_type_id}, layer {layer_index}. Skipping."
                        )

                    # ---- Persist the raw sequence exactly once per accession (if provided) -----
                    if "sequence" in record and "sequence" not in accession_group:
                        accession_group.create_dataset("sequence", data=record["sequence"].encode("utf-8"))
                        self.logger.info(f"Stored sequence for accession {accession}.")

        except Exception as e:
            # Log full traceback for diagnostics and re-raise to preserve the call site's control flow.
            self.logger.error(f"Error while storing embeddings to HDF5: {e}\n{traceback.format_exc()}")
            raise
