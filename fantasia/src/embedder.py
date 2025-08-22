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
    length_filter : int or None
        Optional maximum sequence length. Sequences longer than this are excluded.
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
        self.batch_sizes = conf.get("embedding", {}).get("batch_size", {})
        self.queue_batch_size = conf.get('embedding', {}).get("queue_batch_size", 1)
        self.length_filter = conf.get("embedding", {}).get("max_sequence_length", 0)

    def enqueue(self):
        """
        Lee el FASTA de entrada y encola *todas* las secuencias para *todos* los modelos habilitados,
        enviando *todas* las capas definidas en la configuración para cada modelo.

        No consulta BD. No usa sequence_id.
        Requiere que en la configuración exista:
            self.conf["embedding"]["models"][<model_name>]["enabled"] -> bool
            self.conf["embedding"]["models"][<model_name>]["layer_index"] -> list[int]
        y que self.types[<model_name>] contenga:
            {"id": <embedding_type_id>, "model_name": <backend_model_name>, ...}
        """
        try:
            self.logger.info("Starting embedding enqueue process (FASTA, all layers, no DB).")

            # --- 0) Límite de truncado desde la config ---
            max_len = self.conf.get("embedding", {}).get("max_sequence_length")  # puede ser None

            # --- 1) Cargar FASTA (SIN filtrar por longitud) ---
            input_fasta = os.path.expanduser(self.fasta_path)
            if not os.path.exists(input_fasta):
                raise FileNotFoundError(f"FASTA file not found at: {input_fasta}")

            sequences = list(SeqIO.parse(input_fasta, "fasta"))

            # Límite opcional de ejecución
            if getattr(self, "limit_execution", None):
                sequences = sequences[:self.limit_execution]

            if not sequences:
                self.logger.warning("No sequences found. Finishing embedding enqueue process.")
                return

            # --- 2) Particionar en lotes para controlar tamaño de mensajes ---
            queue_batch_size = self.queue_batch_size
            sequence_batches = [
                sequences[i:i + queue_batch_size]
                for i in range(0, len(sequences), queue_batch_size)
            ]

            # --- 3) Recorrer lotes y agrupar mensajes por modelo ---
            for batch in sequence_batches:
                model_batches = {}

                for seq_record in batch:
                    accession = seq_record.id
                    seq_str = str(seq_record.seq)

                    # 🔪 Truncado si excede max_len (si está definido)
                    if max_len and len(seq_str) > max_len:
                        seq_str = seq_str[:max_len]

                    for model_name, model_cfg in self.conf["embedding"]["models"].items():
                        # Saltar modelos deshabilitados
                        if not model_cfg.get("enabled", False):
                            continue

                        type_info = self.types.get(model_name)
                        if not type_info:
                            self.logger.warning(f"Model '{model_name}' not found in loaded types. Skipping.")
                            continue

                        embedding_type_id = type_info["id"]
                        backend_model_name = type_info["model_name"]

                        # Tomar TODAS las capas desde la config para este modelo
                        desired_layers = model_cfg.get("layer_index") or type_info.get("layer_index") or []
                        if not desired_layers:
                            self.logger.warning(
                                f"No 'layer_index' configured for model '{model_name}'. Skipping."
                            )
                            continue

                        task_data = {
                            "sequence": seq_str,  # ← ya truncada si excede max_len
                            "accession": accession,
                            "model_name": backend_model_name,
                            "embedding_type_id": embedding_type_id,
                            "layer_index": list(desired_layers),  # lista completa de capas
                        }
                        model_batches.setdefault(model_name, []).append(task_data)

                # --- 4) Publicar un mensaje por modelo (más eficiente) ---
                for model_name, batch_data in model_batches.items():
                    if batch_data:
                        self.publish_task(batch_data, model_name)
                        self.logger.info(
                            "Published batch with %d sequences to model '%s' (type ID %s) with ALL configured layers.",
                            len(batch_data), model_name, self.types[model_name]['id']
                        )

        except FileNotFoundError as e:
            self.logger.error(f"File not found: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error during enqueue: {e}\n{traceback.format_exc()}")
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
        Guarda embeddings por capa en HDF5 con la jerarquía:
          /accession_<ID>/type_<type_id>/layer_<k>/embedding
        y un atributo 'shape' en cada grupo de capa.

        results: dict o list[dict] con claves mínimas:
          - accession: str
          - embedding_type_id: int/str
          - embedding: np.ndarray (1D o 2D si ya agregaste algo)
          - shape: tuple
          - sequence: str (opcional, se guarda una vez por accession)
          - layer_index: int  <-- REQUERIDO para guardar por capa
        """
        try:
            # Normaliza a lista
            if isinstance(results, dict):
                results = [results]
            elif not isinstance(results, (list, tuple)):
                raise TypeError(f"store_entry expects dict or list[dict], got: {type(results)}")

            output_h5 = os.path.join(self.experiment_path, "embeddings.h5")

            with h5py.File(output_h5, "a") as h5file:
                for record in results:
                    # Validaciones mínimas
                    if "layer_index" not in record:
                        raise KeyError(
                            "Missing 'layer_index' in embedding record. "
                            "Asegúrate de que 'process' lo incluya por registro."
                        )
                    accession = record["sequence_id"].replace("|", "_")
                    embedding_type_id = record["embedding_type_id"]
                    layer_index = int(record["layer_index"])

                    accession_group = h5file.require_group(f"accession_{accession}")
                    type_group = accession_group.require_group(f"type_{embedding_type_id}")
                    layer_group = type_group.require_group(f"layer_{layer_index}")

                    # Dataset del embedding por capa (idempotente)
                    if "embedding" not in layer_group:
                        layer_group.create_dataset("embedding", data=record["embedding"])
                        # Metadatos útiles
                        layer_group.attrs["shape"] = tuple(
                            record.get("shape", getattr(record.get("embedding"), "shape", ())))
                        # opcional: versión/modelo/etc.
                        # layer_group.attrs["backend_model_name"] = record.get("model_name","")
                        self.logger.info(
                            f"Stored embedding for accession {accession}, "
                            f"type {embedding_type_id}, layer {layer_index}."
                        )
                    else:
                        self.logger.warning(
                            f"Embedding already exists for accession {accession}, "
                            f"type {embedding_type_id}, layer {layer_index}. Skipping."
                        )

                    # Guardar la secuencia una sola vez por accession (si viene)
                    if "sequence" in record and "sequence" not in accession_group:
                        accession_group.create_dataset("sequence", data=record["sequence"].encode("utf-8"))
                        self.logger.info(f"Stored sequence for accession {accession}.")

        except Exception as e:
            self.logger.error(f"Error while storing embeddings to HDF5: {e}\n{traceback.format_exc()}")
            raise

