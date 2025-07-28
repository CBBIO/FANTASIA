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
- Optional redundancy filtering support via MMSeqs2.

This component is intended to serve as the first stage of a larger embedding-based
functional annotation pipeline.
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

            for model_name in self.conf["embedding"]["models"]:
                model_info = self.types.get(model_name)

                if model_info is None:
                    self.logger.warning(f"Model '{model_name}' not found in loaded types. Skipping.")
                    continue

                if not self.conf["embedding"]["models"][model_name]["enabled"]:
                    continue

                queue_batch_size = self.queue_batch_size
                sequence_batches = [
                    sequences[i:i + queue_batch_size]
                    for i in range(0, len(sequences), queue_batch_size)
                ]

                for batch in sequence_batches:
                    task_batch = [
                        {
                            "sequence": str(seq_record.seq),
                            "accession": seq_record.id,
                            "model_name": model_info["model_name"],
                            "embedding_type_id": model_info["id"]
                        }
                        for seq_record in batch
                    ]

                    self.publish_task(task_batch, model_info["name"])
                    self.logger.info(
                        f"Published batch with {len(task_batch)} sequences to model '{model_info["id"]}'."
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
                    accession = record["accession"].replace("|", "_")
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
