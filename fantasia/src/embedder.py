"""
Sequence Embedding Module
=========================

This module contains the `SequenceEmbedder` class, which processes protein sequences to generate embeddings
using protein language models, applies optional filters (like length and redundancy), and stores the embeddings
in HDF5 format.

Background
----------

This module includes functionalities inspired by:

- **BioEmbeddings**: Techniques for embedding generation and model handling are adapted from the
  BioEmbeddings framework. For more details, visit https://docs.bioembeddings.com.

Custom enhancements allow for efficient batch processing and integration with CD-HIT for redundancy filtering.

"""

import os
import traceback
import shutil

from Bio import SeqIO

import h5py

from protein_metamorphisms_is.operation.embedding.sequence_embedding import SequenceEmbeddingManager


class SequenceEmbedder(SequenceEmbeddingManager):
    """
    Processes protein sequences to compute embeddings and store them in HDF5 format.

    Parameters
    ----------
    conf : dict
        Configuration dictionary containing parameters for embedding generation.
    current_date : str
        A timestamp used to generate unique output file names.

    Attributes
    ----------
    fasta_path : str
        Path to the input FASTA file containing protein sequences.
    output_csv : str
        Path to store the embedding results in CSV format.
    output_h5 : str
        Path to store the embeddings in HDF5 format.
    batch_sizes : dict
        Number of sequences to process per batch, specific to each model.
    length_filter : int or None
        Optional filter to exclude sequences longer than the specified length.
    """

    def __init__(self, conf, current_date):
        """
        Initializes the SequenceEmbedder with configuration and output paths.

        Parameters
        ----------
        conf : dict
            The configuration dictionary containing paths and parameters.
        current_date : str
            The timestamp to uniquely identify output files.
        """
        super().__init__(conf)
        self.current_date = current_date
        self.reference_attribute = 'sequence_embedder_from_fasta'
        self.model_instances = {}
        self.tokenizer_instances = {}
        self.base_module_path = 'protein_metamorphisms_is.operation.embedding.proccess.sequence'
        self.fetch_models_info()
        self.sequence_queue_package = conf.get('sequence_queue_package')
        self.batch_sizes = conf['embedding'].get('batch_size', {})  # Store batch sizes as a dict
        self.fasta_path = conf.get('fantasia_input_fasta')
        self.output_csv = conf.get("fantasia_output_csv")
        self.length_filter = conf.get('length_filter', None)
        self.output_h5 = os.path.join(
            conf.get("fantasia_output_h5"),
            f"{conf.get('fantasia_prefix', 'default')}_embeddings_{self.current_date}.h5"
        )

        self.results = []

    def enqueue(self):
        """
        Reads the input FASTA file, applies optional redundancy and length filters,
        and prepares batches of sequences for embedding generation.

        Steps:

        - Runs CD-HIT for redundancy filtering if configured.
        - Splits sequences into batches based on batch size for each model.
        - Publishes tasks for embedding computation.

        Raises
        ------
        Exception
            If there is an error during the enqueue process.
        """
        try:
            self.logger.info("Starting embedding enqueue process.")
            sequences = []

            # Determinar el archivo FASTA de entrada dependiendo del filtro de redundancia
            input_fasta = self.fasta_path
            if self.conf.get('redundancy_filter'):
                self.logger.info("Running CD-HIT for redundancy filtering.")

                if shutil.which("cd-hit") is None:
                    raise EnvironmentError("CD-HIT is not installed or not found in the PATH.")

                os.system(
                    f"cd-hit -i {self.fasta_path} -o {self.conf.get('redundancy_file')} -c {self.conf.get('redundancy_filter')}"
                )
                input_fasta = self.conf.get('redundancy_file')  # Usar el archivo generado por CD-HIT

            # Leer las secuencias del archivo FASTA (filtradas o no)
            for record in SeqIO.parse(os.path.expanduser(input_fasta), "fasta"):
                if self.length_filter and len(record.seq) > self.length_filter:
                    continue
                sequences.append(record)

            # Dividir en lotes específicos por modelo
            for model_id in self.conf['embedding']['types']:
                sequence_queue_package = self.sequence_queue_package  # Default batch size if not specified
                sequence_batches = [sequences[i:i + sequence_queue_package] for i in
                                    range(0, len(sequences), sequence_queue_package)]

                for batch in sequence_batches:
                    model_batches = []
                    for sequence in batch:
                        task_data = {
                            'sequence': str(sequence.seq),
                            'accession': sequence.id,
                            'model_name': self.types[model_id]['model_name'],
                            'embedding_type_id': model_id
                        }
                        model_batches.append(task_data)

                    self.publish_task(model_batches, model_id)
                    self.logger.info(
                        f"Published batch with {len(model_batches)} sequences to model type {model_id}.")

        except Exception as e:
            self.logger.error(f"Error during enqueue process: {e}")
            raise

    def process(self, task_data):
        """
        Processes a batch of sequences to compute embeddings using the specified model and tokenizer.

        Parameters
        ----------
        task_data : list of dict
            A list of dictionaries containing sequence information and embedding parameters.

        Returns
        -------
        list of dict
            A list of embedding records with metadata, including accession ID and embedding type.

        Raises
        ------
        Exception
            If an error occurs during the embedding process.
        """
        try:
            results = []
            for data in task_data:
                embedding_type_id = data['embedding_type_id']
                model = self.model_instances[embedding_type_id]
                tokenizer = self.tokenizer_instances[embedding_type_id]
                module = self.types[embedding_type_id]['module']

                # Prepare input for embedding_task
                sequence_info = [{
                    'sequence': data['sequence'],
                    'sequence_id': data['accession']  # Propagating accession as sequence_id
                }]

                embedding_records = module.embedding_task(sequence_info, model, tokenizer,
                                                          batch_size=self.batch_sizes[embedding_type_id],
                                                          embedding_type_id=embedding_type_id)

                for record in embedding_records:
                    record['embedding_type_id'] = embedding_type_id
                    record['accession'] = data['accession']  # Propagar el accession
                    results.append(record)
            return results
        except Exception as e:
            self.logger.error(f"Error during embedding process: {e}\n{traceback.format_exc()}")
            raise

    def store_entry(self, results):
        """
        Stores the computed embeddings in an HDF5 file.

        Parameters
        ----------
        results : list of dict
            A list of embedding records, each containing metadata and embedding data.

        Raises
        ------
        Exception
            If an error occurs during file storage.
        """
        try:
            output_dir = os.path.dirname(self.output_h5)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                self.logger.info(f"Created directory: {output_dir}")

            with h5py.File(os.path.expanduser(self.output_h5), "a") as h5file:
                for record in results:
                    accession = record['accession']
                    embedding_type_id = record['embedding_type_id']

                    accession_group = h5file.require_group(f"accession_{accession}")

                    type_group = accession_group.require_group(f"type_{embedding_type_id}")

                    if "embedding" in type_group:
                        self.logger.warning(
                            f"Embedding for type {embedding_type_id} already exists in accession {accession}. Skipping.")
                        continue

                    type_group.create_dataset("embedding", data=record['embedding'])
                    type_group.attrs['shape'] = record['shape']
                    self.logger.info(f"Stored embedding for accession {accession}, type {embedding_type_id}.")
        except Exception as e:
            self.logger.error(f"Error storing results in HDF5: {e}")
            raise
