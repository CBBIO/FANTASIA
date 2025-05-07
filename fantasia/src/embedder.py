"""
Sequence Embedding Module
==========================

This module defines the `SequenceEmbedder` class, which computes protein embeddings
from input FASTA files using preconfigured language models.

Given a FASTA file, the system filters and batches sequences, applies the selected
embedding models, and stores the resulting representations and metadata in HDF5 format.
It supports optional sequence length filtering and is designed for high-throughput pipelines.

Background
----------

The implementation draws inspiration from the BioEmbeddings project:
- https://docs.bioembeddings.com

Enhancements include:
- Efficient batch-level task handling and queuing.
- Dynamic model loading via modular architecture.
- Integration with a SQL-based model registry (SequenceEmbeddingType).
- Optional redundancy filtering support (via CD-HIT, managed externally).

This component is intended to serve as the first stage of a larger embedding-based
functional annotation pipeline.
"""

import importlib
import os
import traceback

from Bio import SeqIO

import h5py

from protein_metamorphisms_is.operation.embedding.sequence_embedding import SequenceEmbeddingManager
from protein_metamorphisms_is.sql.model.entities.embedding.sequence_embedding import SequenceEmbeddingType


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

        self.limit_execution = conf.get("limit_execution")

        self.model_instances = {}
        self.tokenizer_instances = {}
        self.types = {}
        self.results = []

        self.base_module_path = "protein_metamorphisms_is.operation.embedding.proccess.sequence"
        self.fetch_models_info()

        # Config paths
        self.fasta_path = conf["input"]["fasta_path"]
        self.experiment_path = conf["experiment_path"]

        # Optional config
        self.batch_sizes = {
            k: v.get("batch_size") for k, v in conf["embedding"]["models"].items()
        }
        self.sequence_queue_package = conf["runtime"].get("sequence_queue_package", 512)
        self.length_filter = conf["input"].get("length_filter")

    def fetch_models_info(self):
        """
        Loads metadata and modules for all enabled embedding models.

        This method queries the database for all available embedding types and checks which
        ones are enabled in the configuration. For each enabled model, it dynamically imports
        the corresponding Python module and stores relevant metadata in `self.types`.

        Raises
        ------
        Exception
            If an error occurs while querying the database or importing a model module.

        Notes
        -----
        - `self.types` maps each task_name to its metadata and module.
        - ⚠ TODO: Move this method to a shared base class to avoid duplication
          across embedding-related components (e.g. SequenceEmbedder, EmbeddingLookUp).
        """
        self.types = {}
        enabled_models = self.conf.get("embedding", {}).get("models", {})

        try:
            self.session_init()
            embedding_types = self.session.query(SequenceEmbeddingType).all()
        except Exception as e:
            self.logger.error(f"Error querying SequenceEmbeddingType table: {e}")
            raise
        finally:
            self.session.close()
            del self.engine

        for embedding_type in embedding_types:
            task_name = embedding_type.task_name

            model_config = enabled_models.get(task_name)
            if not model_config or not model_config.get("enabled", False):
                continue

            try:
                module_name = f"{self.base_module_path}.{task_name}"
                module = importlib.import_module(module_name)

                self.types[task_name] = {
                    "module": module,
                    "model_name": embedding_type.model_name,
                    "id": embedding_type.id,
                    "task_name": task_name,
                    "distance_threshold": model_config.get("distance_threshold"),
                    "batch_size": model_config.get("batch_size"),
                }

                self.logger.info(f"Loaded model: {task_name} ({embedding_type.model_name})")

            except ImportError as e:
                self.logger.error(f"Failed to import module '{module_name}': {e}")
                raise

    def enqueue(self):
        """
        Reads the input FASTA file, filters and batches the sequences, and enqueues embedding tasks.

        This method performs the following steps:
        1. Parses the input FASTA file using BioPython.
        2. Optionally filters sequences by length if a `length_filter` is defined.
        3. For each enabled model, splits the full sequence list into batches of configurable size.
        4. Enqueues each batch for embedding computation using `publish_task`.

        Raises
        ------
        FileNotFoundError
            If the input FASTA file does not exist.
        Exception
            For any unexpected errors during file parsing or batching.

        Notes
        -----
        - The batch size for all models is currently defined by the global `sequence_queue_package` value,
          which may override model-specific `batch_size` settings.
        - Each task entry includes: `sequence`, `accession`, `model_name`, and `embedding_type_id`.

        Example
        -------
        >>> embedder = SequenceEmbedder(conf, current_date)
        >>> embedder.enqueue()
        INFO: Starting embedding enqueue process.
        INFO: Published batch with 32 sequences to model type esm.
        INFO: Published batch with 32 sequences to model type prot_t5.
        """
        try:
            self.logger.info("Starting embedding enqueue process.")

            input_fasta = os.path.expanduser(self.fasta_path)
            if not os.path.exists(input_fasta):
                raise FileNotFoundError(f"FASTA file not found at: {input_fasta}")

            sequences = [
                record for record in SeqIO.parse(input_fasta, "fasta")
                if not self.length_filter or len(record.seq) <= self.length_filter
            ]

            if self.limit_execution:
                sequences = sequences[:self.limit_execution]

            if not sequences:
                self.logger.warning("No sequences found. Finishing embedding enqueue process.")
                return

            for model_id in self.types.keys():
                model_info = self.types.get(model_id)
                if model_info is None:
                    self.logger.warning(f"Model '{model_id}' not found in loaded types. Skipping.")
                    continue

                batch_size = self.sequence_queue_package
                sequence_batches = [
                    sequences[i:i + batch_size]
                    for i in range(0, len(sequences), batch_size)
                ]

                for batch in sequence_batches:
                    task_batch = [
                        {
                            "sequence": str(seq_record.seq),
                            "accession": seq_record.id,
                            "model_name": model_info["model_name"],
                            "embedding_type_id": model_id
                        }
                        for seq_record in batch
                    ]

                    self.publish_task(task_batch, model_id)
                    self.logger.info(
                        f"Published batch with {len(task_batch)} sequences to model '{model_id}'."
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
            model = self.model_instances[embedding_type_id]
            tokenizer = self.tokenizer_instances[embedding_type_id]
            module = self.types[embedding_type_id]["module"]

            device = self.conf["embedding"].get("device", "cuda")
            batch_size = self.types[embedding_type_id]["batch_size"]

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
                device=device
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
        Stores computed embeddings and associated sequences into an HDF5 file.

        For each embedding result, this method creates or updates the HDF5 structure
        following a hierarchical organization:
        - /accession_{id}/type_{embedding_type_id}/embedding : stores the embedding vector.
        - /accession_{id}/type_{embedding_type_id}/attrs     : stores metadata like shape.
        - /accession_{id}/sequence                            : stores the original sequence.

        If a dataset already exists, the method skips overwriting it.

        Parameters
        ----------
        results : list of dict
            A list of embedding records. Each record must include:
            - 'accession' (str): sequence identifier.
            - 'embedding_type_id' (str): model identifier.
            - 'embedding' (np.ndarray): embedding vector.
            - 'shape' (tuple): shape of the embedding.
            - 'sequence' (str): original amino acid sequence.

        Raises
        ------
        Exception
            If any error occurs while writing to the HDF5 file.
        """
        try:
            output_h5 = os.path.join(self.experiment_path, "embeddings.h5")

            with h5py.File(output_h5, "a") as h5file:
                for record in results:
                    accession = record["accession"]
                    embedding_type_id = record["embedding_type_id"]

                    accession_group = h5file.require_group(f"accession_{accession}")
                    type_group = accession_group.require_group(f"type_{embedding_type_id}")

                    # Store embedding
                    if "embedding" not in type_group:
                        type_group.create_dataset("embedding", data=record["embedding"])
                        type_group.attrs["shape"] = record["shape"]
                        self.logger.info(
                            f"Stored embedding for accession {accession}, type {embedding_type_id}."
                        )
                    else:
                        self.logger.warning(
                            f"Embedding for accession {accession}, type {embedding_type_id} already exists. Skipping."
                        )

                    # Store sequence
                    if "sequence" not in accession_group:
                        accession_group.create_dataset("sequence", data=record["sequence"].encode("utf-8"))
                        self.logger.info(f"Stored sequence for accession {accession}.")

        except Exception as e:
            self.logger.error(f"Error while storing embeddings to HDF5: {e}\n{traceback.format_exc()}")
            raise
