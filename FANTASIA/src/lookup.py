"""
Embedding LookUp Module
=======================

This module contains the `EmbeddingLookUp` class, which handles querying embeddings stored in HDF5 format,
calculating distances to identify similar proteins, and storing the resulting GO terms in CSV format.

Background
----------

This module integrates functionalities inspired by:

- **GoPredSim**: The GO term similarity and distance-based lookup functionalities are adapted from GoPredSim
  (https://github.com/Rostlab/goPredSim).

Additionally, customizations have been made to ensure seamless integration with
the vectorial database and HDF5-based embedding storage used in this pipeline.

"""

import os

import pandas as pd
from sqlalchemy import text
import h5py

from protein_metamorphisms_is.sql.model.entities.embedding.sequence_embedding import SequenceEmbeddingType
from protein_metamorphisms_is.tasks.queue import QueueTaskInitializer


class EmbeddingLookUp(QueueTaskInitializer):
    """
    A class to process embeddings from an HDF5 file, query GO terms based on similarity,
    and store results in a CSV file.

    Parameters
    ----------
    conf : dict
        Configuration dictionary containing paths and thresholds for processing.
    current_date : str
        A timestamp used for generating unique output file names.

    Attributes
    ----------
    h5_path : str
        Path to the input HDF5 file containing embeddings.
    output_csv : str
        Path to store the resulting GO terms in CSV format.
    max_distance : float
        Maximum allowed distance for similarity-based GO term retrieval.
    """

    def __init__(self, conf, current_date):
        """
        Initializes the EmbeddingLookUp class with configuration settings and output paths.

        Parameters
        ----------
        conf : dict
            The configuration dictionary containing paths and parameters.
        current_date : str
            The timestamp used to uniquely identify output files.
        """
        super().__init__(conf)
        self.current_date = current_date
        self.logger.info("EmbeddingLookUp initialized")
        self.h5_path = os.path.join(
            conf.get("fantasia_output_h5"),
            f"{conf.get('fantasia_prefix', 'default')}_embeddings_{self.current_date}.h5"
        )
        self.output_csv = os.path.join(
            conf.get("fantasia_output_csv"),
            f"{conf.get('fantasia_prefix', 'default')}_results_{self.current_date}.csv"
        )
        self.max_distance = conf.get('max_distance', 3)  # Distancia máxima permitida para el cálculo
        self.fetch_models_info()

    def fetch_models_info(self):
        """
        Retrieves and initializes embedding models based on configuration.

        Queries the `SequenceEmbeddingType` table to fetch available embedding models.
        Modules are dynamically imported and stored in the `types` attribute.
        """
        self.session_init()
        embedding_types = self.session.query(SequenceEmbeddingType).all()
        self.session.close()
        self.types = {}

        for type_obj in embedding_types:
            if type_obj.id in self.conf['embedding']['types']:
                self.types[type_obj.id] = {
                    'model_name': type_obj.model_name,
                    'id': type_obj.id,
                    'task_name': type_obj.task_name,
                }

    def enqueue(self):
        """
        Reads embeddings from an HDF5 file and enqueues tasks for processing.

        Raises
        ------
        Exception
            If any error occurs while reading the HDF5 file or publishing tasks.
        """
        try:
            self.logger.info(f"Reading embeddings from HDF5: {self.h5_path}")

            tasks = []
            with h5py.File(os.path.expanduser(self.h5_path), "r") as h5file:
                for accession, accession_group in h5file.items():
                    for embedding_type, type_group in accession_group.items():
                        # Verificar si el dataset 'embedding' existe
                        if 'embedding' in type_group:
                            embedding = type_group['embedding'][:]  # Leer el dataset como array
                            embedding_type_id = int(embedding_type.split('_')[1])  # Extraer el ID del tipo

                            task_data = {
                                'accession': accession,
                                'embedding': embedding,
                                'embedding_type_id': embedding_type_id
                            }
                            tasks.append(task_data)
                            self.logger.info(
                                f"Enqueued task for accession {accession} and embedding type {embedding_type_id}")
                        else:
                            self.logger.warning(f"Dataset 'embedding' not found in {embedding_type}")

            for task in tasks:
                self.publish_task(task)

            self.logger.info(f"Enqueued {len(tasks)} tasks based on HDF5 embeddings.")

        except Exception as e:
            self.logger.error(f"Error enqueuing tasks from HDF5: {e}")
            raise

    def process(self, task_data):
        """
        Processes a single task by querying GO terms based on embedding similarity.

        Parameters
        ----------
        task_data : dict
            Dictionary containing the embedding, accession ID, and embedding type ID.

        Returns
        -------
        list of dict
            A list of dictionaries containing GO term results and metadata.

        Raises
        ------
        Exception
            If any error occurs during the query execution.
        """
        try:
            accession = task_data['accession'].removeprefix('accession_')
            embedding_type_id = int(task_data['embedding_type_id'])  # Convertir a tipo nativo de Python
            embedding = task_data['embedding']

            # Convertir el vector NumPy a una lista de Python y formatear como cadena para pgvector
            vector_string = "[" + ",".join(f"{float(v):.8f}" for v in embedding) + "]"

            # Construir la consulta SQL con el vector embebido
            query = text("""
                WITH target_embedding AS (
                    SELECT :vector_string ::vector AS embedding
                )
                SELECT
                    s.sequence,
                    (se.embedding <-> te.embedding) AS distance,
                    p.id AS protein_id,
                    p.gene_name AS gene_name,
                    p.organism AS organism,
                    pgo.go_id AS go_term_id,
                    gt.category as category,
                    gt.description AS go_term_description,
                    pgo.evidence_code
                FROM
                    sequence_embeddings se
                    JOIN target_embedding te ON TRUE
                    JOIN sequence s ON se.sequence_id = s.id
                    JOIN protein p ON s.id = p.sequence_id
                    LEFT JOIN protein_go_term_annotation pgo ON p.id = pgo.protein_id
                    LEFT JOIN go_terms gt ON pgo.go_id = gt.go_id
                WHERE
                    se.embedding_type_id = :embedding_type_id
                    AND (se.embedding <-> te.embedding) < :max_distance
                ORDER BY
                    distance ASC;
            """)

            self.logger.info(f"Executing query for accession {accession} and embedding type {embedding_type_id}.")
            # Ejecutar la consulta
            with self.engine.connect() as connection:
                results = connection.execute(query, {
                    'vector_string': vector_string,
                    'embedding_type_id': embedding_type_id,
                    'max_distance': float(self.max_distance)
                }).fetchall()

            if not results:
                self.logger.info(f"No results found for accession {accession}.")
                return []

            go_terms = []
            for row in results:
                go_terms.append({
                    'accession': accession,
                    'go_id': row.go_term_id,
                    'category': row.category,
                    'evidence_code': row.evidence_code,
                    'go_description': row.go_term_description,
                    'distance': row.distance,
                    'model_name': self.types[embedding_type_id].get('model_name'),
                    'protein_id': row.protein_id,
                    'organism': row.organism,
                    'sequence': row.sequence
                })

            self.logger.info(
                f"Found {len(go_terms)} GO terms for accession {accession} and embedding type {embedding_type_id}.")
            return go_terms

        except Exception as e:
            self.logger.error(
                f"Error processing task for accession {accession} and embedding type {embedding_type_id}: {e}")
            raise

    def store_entry(self, go_terms):
        """
        Stores the retrieved GO terms in a CSV file.

        Parameters
        ----------
        go_terms : list of dict
            List of dictionaries containing GO term results.

        Raises
        ------
        Exception
            If an error occurs while writing to the CSV file.
        """
        if not go_terms:
            self.logger.info("No valid GO terms to store.")
            return

        try:
            output_csv_path = os.path.expanduser(self.output_csv)  # Asegúrate de usar la ruta expandida

            # Verificar y crear el directorio "results" si no existe
            output_dir = os.path.dirname(output_csv_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                self.logger.info(f"Created directory: {output_dir}")

            # Convertir go_terms a un DataFrame
            df = pd.DataFrame(go_terms)

            # Escribir al archivo
            if os.path.exists(output_csv_path) and os.path.getsize(output_csv_path) > 0:
                # Archivo existe y tiene contenido, añade sin encabezado
                df.to_csv(output_csv_path, mode='a', index=False, header=False)
                self.logger.info(f"Appended {len(go_terms)} entries to {output_csv_path}.")
            else:
                # Archivo no existe o está vacío, escribe con encabezado
                df.to_csv(output_csv_path, mode='w', index=False, header=True)
                self.logger.info(f"Created new file and stored {len(go_terms)} entries in {output_csv_path}.")

        except Exception as e:
            self.logger.error(f"Error storing results in CSV: {e}")
            raise
