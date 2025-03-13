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
import importlib
import os
import traceback

from Bio import SeqIO

import h5py

from protein_metamorphisms_is.operation.embedding.sequence_embedding import SequenceEmbeddingManager
from protein_metamorphisms_is.sql.model.entities.embedding.sequence_embedding import SequenceEmbeddingType
from protein_metamorphisms_is.sql.model.entities.protein.accesion import Accession


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
        self.batch_sizes = conf['embedding'].get('batch_size',
                                                 {})  # Store batch sizes as a dict        self.fasta_path = conf.get('fantasia_input_fasta')
        self.length_filter = conf.get('length_filter', None)

        self.fasta_path = conf.get('input')
        self.experiment_path = conf.get('experiment_path')

        self.results = []

    def fetch_models_info(self):
        """
        Retrieves and initializes embedding models based on configuration.

        Queries the `SequenceEmbeddingType` table to fetch available embedding models.
        Modules are dynamically imported and stored in the `types` attribute.
        """
        self.session_init()
        try:
            embedding_types = self.session.query(SequenceEmbeddingType).all()
        except Exception as e:
            self.logger.error(f"Error querying SequenceEmbeddingType table: {e}")
            raise
        finally:
            self.session.close()
            del self.engine

        self.types = {}

        enabled_models = self.conf.get('embedding', {}).get('models', {})

        for type_obj in embedding_types:
            if type_obj.task_name in enabled_models:
                model_config = enabled_models[type_obj.task_name]
                if model_config.get('enabled', False):
                    try:
                        module_name = f"{self.base_module_path}.{type_obj.task_name}"
                        module = importlib.import_module(module_name)
                        self.types[type_obj.task_name] = {
                            'module': module,
                            'model_name': type_obj.model_name,
                            'id': type_obj.id,
                            'task_name': type_obj.task_name,
                            'distance_threshold': model_config.get('distance_threshold'),
                            'batch_size': model_config.get('batch_size'),
                        }
                        self.logger.info(f"Loaded model: {type_obj.task_name} ({type_obj.model_name})")
                    except ImportError as e:
                        self.logger.error(f"Failed to import module {module_name}: {e}")
                        raise

        if not self.types:
            self.logger.warning("No matching models found between the database and the configuration.")
        else:
            self.logger.info("Loaded model types:", self.types)


    def enqueue(self):
        """
        Reads the input FASTA file, applies optional redundancy and length filters,
        and prepares batches of sequences for embedding generation.

        The method performs the following steps:

        1. **File existence check**: Ensures the input FASTA file exists before proceeding.
        2. **Sequence filtering**: Reads the FASTA file and applies an optional length filter.
        3. **Batch preparation**: Splits sequences into batches based on the configured batch size.
        4. **Task publishing**: Sends batches to the embedding pipeline.

        If the input file is missing or any critical error occurs, the program will log the error and terminate.

        Raises
        ------
        SystemExit
            If the input FASTA file does not exist or an unexpected error occurs during the process.

        Examples
        --------
        >>> embedder = SequenceEmbedder(conf, current_date)
        >>> embedder.enqueue()
        Starting embedding enqueue process.
        Published batch with 32 sequences to model type esm.
        Published batch with 32 sequences to model type prot_t5.

        Notes
        -----
        - The batch size for each model is determined by the `sequence_queue_package` parameter.
        - The method stops execution (`sys.exit(1)`) if the input FASTA file is missing.
        """

        try:
            self.logger.info("Starting embedding enqueue process.")
            sequences = []

            input_fasta = self.fasta_path


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

        except FileNotFoundError as e:
            self.logger.error(f"File not found: {e}")
            raise e
        except Exception as e:
            self.logger.error(f"Error during enqueue process: {e}\n{traceback.format_exc()}")
            raise e

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
            if not task_data:
                self.logger.warning("No task data provided for processing.")
                return []

            # Extract embedding type and verify uniformity
            embedding_type_id = task_data[0]['embedding_type_id']
            if not all(data['embedding_type_id'] == embedding_type_id for data in task_data):
                raise ValueError("All sequences in the batch must have the same embedding_type_id.")

            # Retrieve model, tokenizer, and module
            model = self.model_instances[embedding_type_id]
            tokenizer = self.tokenizer_instances[embedding_type_id]
            module = self.types[embedding_type_id]['module']

            # Prepare batch input
            sequence_info = [
                {'sequence': data['sequence'], 'sequence_id': data['accession']}
                for data in task_data
            ]
            # Call embedding_task for the entire batch
            device = self.conf['embedding'].get('device', "cuda")
            embedding_records = module.embedding_task(
                sequence_info,
                model,
                tokenizer,
                batch_size=self.types[embedding_type_id]['batch_size'],  # Use batch size as the number of sequences
                embedding_type_id=embedding_type_id,
                device=device
            )

            # Add additional metadata to each record
            for record, data in zip(embedding_records, task_data):
                record['embedding_type_id'] = embedding_type_id
                record['accession'] = data['accession']  # Propagate accession

            return embedding_records

        except Exception as e:
            self.logger.error(f"Error during embedding process: {e}\n{traceback.format_exc()}")
            raise

    def store_entry(self, results):
        """
        Stores the computed embeddings and sequences in an HDF5 file.

        Parameters
        ----------
        results : list of dict
            A list of embedding records, each containing metadata, embedding data, and the sequence.

        Raises
        ------
        Exception
            If an error occurs during file storage.
        """
        try:
            output_h5 = os.path.join(self.conf['experiment_path'], "embeddings.h5")

            with h5py.File(output_h5, "a") as h5file:
                for record in results:
                    accession = record['accession']
                    embedding_type_id = record['embedding_type_id']

                    # Crear grupo para el accession
                    accession_group = h5file.require_group(f"accession_{accession}")

                    # Crear grupo para el tipo de embedding
                    type_group = accession_group.require_group(f"type_{embedding_type_id}")

                    # Almacenar el embedding
                    if "embedding" not in type_group:
                        type_group.create_dataset("embedding", data=record['embedding'])
                        type_group.attrs['shape'] = record['shape']
                        self.logger.info(f"Stored embedding for accession {accession}, type {embedding_type_id}.")
                    else:
                        self.logger.warning(
                            f"Embedding for type {embedding_type_id} already exists in accession {accession}. Skipping embedding storage."
                        )

                    # Almacenar la secuencia
                    if "sequence" not in accession_group:
                        accession_group.create_dataset("sequence", data=record['sequence'].encode('utf-8'))
                        self.logger.info(f"Stored sequence for accession {accession}.")

        except Exception as e:
            self.logger.error(f"Error storing results in HDF5: {e}")
            raise

    def tag_goa(self):
        accessions = set()

        # Leer el archivo FASTA y extraer los IDs de los accessions
        for record in SeqIO.parse('/home/bioxaxi/PycharmProjects/FANTASIA/data_sample/goa_2022.fasta', "fasta"):
            accessions.add(record.id)

        # Consultar en la base de datos los accessions que existen
        self.logger.info(f"Se encontraron {len(accessions)} accessions en el archivo FASTA.")

        # Consultar en la base de datos los accessions que existen
        found_accessions = self.session.query(Accession).filter(Accession.code.in_(accessions)).all()

        if not found_accessions:
            self.logger.info("No se encontraron accessions en la base de datos.")
            return

        # Actualizar las etiquetas (tag) de los accessions encontrados
        for acc in found_accessions:
            acc.tag = "GOA2022"  # Modificación del tag

        # Confirmar los cambios en la base de datos
        self.session.commit()

        self.logger.info(f"Se actualizaron {len(found_accessions)} accessions en la base de datos.")
