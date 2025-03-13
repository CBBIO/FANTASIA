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
import importlib
import os

import pandas as pd
from goatools.base import get_godag
from pycdhit import cd_hit, read_clstr
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

        # Usar rutas desde conf
        self.experiment_path = self.conf.get("experiment_path")

        self.limit_per_entry = self.conf.get("limit_per_entry", 200)

        self.fetch_models_info()

        self.embeddings_path = os.path.join(self.conf.get("experiment_path"), "embeddings.h5")
        self.results_path = os.path.join(self.conf.get("experiment_path"), "results.csv")

        self.topgo_path = os.path.join(self.conf.get("experiment_path"), "results_topgo.tsv")
        self.topgo_enabled = self.conf.get("topgo", False)  # Comprobar si TopGO está activado

        # Check if redundancy filter is active
        redundancy_filter = self.conf.get("redundancy_filter", 0)
        if redundancy_filter > 0:
            self.generate_clusters()

        self.go = get_godag('go-basic.obo', optional_attrs='relationship')

        self.distance_metric = self.conf.get("embedding", {}).get("distance_metric", "<->")

        valid_metrics = ["<->", "<=>"]
        if self.distance_metric not in valid_metrics:
            self.logger.warning(f"Invalid distance metric '{self.distance_metric}', defaulting to '<->' (Euclidean).")
            self.distance_metric = "<->"

    def fetch_models_info(self):
        """
        Retrieves and initializes embedding models based on configuration.

        Queries the `SequenceEmbeddingType` table to fetch available embedding models.
        Modules are dynamically imported and stored in the `types` attribute.
        """
        try:
            embedding_types = self.session.query(SequenceEmbeddingType).all()
        except Exception as e:
            self.logger.error(f"Error querying SequenceEmbeddingType table: {e}")
            raise

        self.types = {}

        enabled_models = self.conf.get('embedding', {}).get('models', {})

        for type_obj in embedding_types:
            if type_obj.task_name in enabled_models:
                model_config = enabled_models[type_obj.task_name]
                if model_config.get('enabled', False):
                    try:
                        self.base_module_path = 'protein_metamorphisms_is.operation.embedding.proccess.sequence'
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

    def generate_clusters(self):
        """
        Genera un archivo FASTA de referencia, ejecuta CD-HIT para agrupar secuencias,
        y almacena los resultados de los clústeres en memoria.

        Raises
        ------
        Exception
            Si ocurre un error durante el proceso.
        """
        input_h5 = os.path.join(self.conf['experiment_path'], "embeddings.h5")
        try:
            self.reference_fasta = os.path.join(self.experiment_path, "redundancy.fasta")
            filtered_fasta = os.path.join(self.experiment_path, "filtered.fasta")
            # Crear el archivo FASTA combinando secuencias de la base de datos y HDF5
            with open(self.reference_fasta, "w") as ref_file:
                self.logger.info("Adding sequences from the database and HDF5 to the reference FASTA...")
                with self.engine.connect() as connection:
                    query = text("SELECT id, sequence FROM sequence")
                    result = connection.execute(query)
                    for row in result:
                        ref_file.write(f">{row.id}\n{row.sequence}\n")

                with h5py.File(input_h5, "r") as h5file:
                    for accession, accession_group in h5file.items():
                        if "sequence" in accession_group:
                            sequence = accession_group["sequence"][()].decode("utf-8")
                            clean_accession = accession.removeprefix("accession_")
                            ref_file.write(f">{clean_accession}\n{sequence}\n")

            self.logger.info("Running CD-HIT using py-cdhit...")

            cdhit_params = {
                "input_file": self.reference_fasta,
                "output_file": filtered_fasta,
                "sequence_identity_threshold": self.conf.get('redundancy_filter', 0.95),
                "alignment_coverage": self.conf.get('alignment_coverage', 0.95),
                "memory_usage": self.conf.get('memory_usage', 32000),
                "threads": self.conf.get('threads', 0),
                "most_representative_search": self.conf.get('most_representative_search', 1)
            }
            self.logger.info(f"CD-HIT parameters:\n"
                             f"  Input File: {cdhit_params['input_file']}\n"
                             f"  Output File: {cdhit_params['output_file']}\n"
                             f"  Sequence Identity Threshold: {cdhit_params['sequence_identity_threshold']}\n"
                             f"  Alignment Coverage: {cdhit_params['alignment_coverage']}\n"
                             f"  Memory Usage: {cdhit_params['memory_usage']} MB\n"
                             f"  Threads: {cdhit_params['threads']}\n"
                             f"  Most Representative Search: {cdhit_params['most_representative_search']}")

            # Ejecución de CD-HIT
            cd_hit(
                i=cdhit_params["input_file"],
                o=cdhit_params["output_file"],
                c=cdhit_params["sequence_identity_threshold"],
                d=0,
                aL=cdhit_params["alignment_coverage"],
                M=cdhit_params["memory_usage"],
                T=cdhit_params["threads"],
                g=cdhit_params["most_representative_search"]
            )

            # Finalizar y cargar los clústeres
            self.logger.info(f"CD-HIT completed. Clusters saved at: {filtered_fasta}.clstr")
            self.clusters = read_clstr(f"{filtered_fasta}.clstr")
            self.logger.info(f"Loaded {len(self.clusters)} clusters into memory.")

        except Exception as e:
            self.logger.error(f"Error generating clusters: {e}")
            raise

    def enqueue(self):
        """
        Reads embeddings and sequences from HDF5 file and enqueues tasks for processing.

        Raises
        ------
        Exception
            If any error occurs while reading the HDF5 file or publishing tasks.
        """
        try:
            self.logger.info(f"Reading embeddings from HDF5: {self.embeddings_path}")

            tasks = []
            with h5py.File(self.embeddings_path, "r") as h5file:
                for accession, accession_group in h5file.items():
                    if "sequence" not in accession_group:
                        self.logger.warning(f"No sequence found for accession {accession}. Skipping.")
                        continue

                    sequence = accession_group["sequence"][()].decode("utf-8")  # Decodificar la secuencia
                    for embedding_type, type_group in accession_group.items():
                        if not embedding_type.startswith("type_") or "embedding" not in type_group:
                            continue

                        embedding = type_group["embedding"][:]
                        embedding_type_id = embedding_type.replace("type_", "")
                        task_data = {
                            'accession': accession,
                            'sequence': sequence,
                            'embedding': embedding,
                            'embedding_type_id': self.types[embedding_type_id].get('id'),
                            'model_name': embedding_type_id,
                            'distance_threshold': self.types[embedding_type_id].get('distance_threshold')
                        }
                        tasks.append(task_data)
                        self.logger.info(
                            f"Enqueued task for accession {accession} and embedding type {embedding_type_id}.")

            for task in tasks:
                self.publish_task(task)

            self.logger.info(f"Enqueued {len(tasks)} tasks based on HDF5 embeddings.")

        except Exception as e:
            self.logger.error(f"Error enqueuing tasks from HDF5: {e}")
            raise

    def process(self, task_data):
        """
        Procesa una tarea, calcula distancias y filtra por redundancia utilizando CD-HIT.

        Parameters
        ----------
        task_data : dict
            Diccionario que contiene `accession`, `embedding` y `embedding_type_id`.

        Returns
        -------
        list of dict
            Lista de diccionarios con términos GO y metadatos asociados, sin redundancias.

        Raises
        ------
        Exception
            Si ocurre un error durante el procesamiento.
        """
        try:
            accession = task_data['accession'].removeprefix('accession_')
            embedding_type_id = task_data['embedding_type_id']
            embedding = task_data['embedding']

            # Generar vector de la secuencia
            vector_string = "[" + ",".join(f"{float(v):.8f}" for v in embedding) + "]"

            # Añadir la secuencia de consulta al archivo FASTA y aplicar CD-HIT
            if self.conf.get("redundancy_filter", 0) > 0:
                self.logger.info(f"Applying redundancy filter for accession {accession}.")
                redundant_ids = self.retrieve_cluster_members(accession)
                self.logger.info(f"Filtered {len(redundant_ids)} redundant sequences for accession {accession}.")
            else:
                redundant_ids = set()

            # Construir la cláusula opcional para redundant_ids
            not_in_clause = "AND s.id NOT IN :redundant_ids" if redundant_ids else ""

            # Construir la cláusula opcional para lookup_reference_tag
            tag_filter = "AND ac.tag = :lookup_reference_tag" if self.conf.get("lookup_reference_tag", False) else ""

            # Construir la consulta SQL
            query = text(f"""
                WITH target_embedding AS (
                    SELECT :vector_string ::vector AS embedding
                ),
                annotated_results AS (
                    SELECT
                        s.sequence,
                        (se.embedding {self.distance_metric} te.embedding) AS distance,
                        p.id AS protein_id,
                        p.gene_name AS gene_name,
                        p.organism AS organism,
                        pgo.go_id AS go_term_id,
                        gt.category AS category,
                        gt.description AS go_term_description,
                        pgo.evidence_code
                    FROM
                        sequence_embeddings se
                        JOIN target_embedding te ON TRUE
                        JOIN sequence s ON se.sequence_id = s.id
                        JOIN protein p ON s.id = p.sequence_id
                        JOIN protein_go_term_annotation pgo ON p.id = pgo.protein_id
                        JOIN go_terms gt ON pgo.go_id = gt.go_id
                        JOIN accession ac ON p.id = ac.protein_id
                    WHERE
                        se.embedding_type_id = :embedding_type_id
                        {not_in_clause}
                        {tag_filter}
                ),
                limited_proteins AS (
                    SELECT
                        protein_id,
                        MIN(distance) AS min_distance
                    FROM
                        annotated_results
                    where distance <= :max_distance
                    GROUP BY
                        protein_id
                    ORDER BY
                        min_distance ASC
                    LIMIT :limit_per_entry
                )
                SELECT
                    ar.sequence,
                    ar.distance,
                    ar.protein_id,
                    ar.gene_name,
                    ar.organism,
                    ar.go_term_id,
                    ar.category,
                    ar.go_term_description,
                    ar.evidence_code
                FROM
                    annotated_results ar
                    JOIN limited_proteins lp ON ar.protein_id = lp.protein_id
                ORDER BY
                    ar.distance ASC;
            """)
            max_distance = task_data['distance_threshold']
            limit_per_entry = self.conf.get("limit_per_entry", 1000)

            self.logger.info(f"Executing query for accession {accession} and embedding type {embedding_type_id}.")
            with self.engine.connect() as connection:
                results = connection.execute(query, {
                    'vector_string': vector_string,
                    'embedding_type_id': embedding_type_id,
                    'max_distance': float(max_distance),
                    'limit_per_entry': limit_per_entry,
                    'redundant_ids': tuple(redundant_ids),
                    'lookup_reference_tag': self.conf.get("lookup_reference_tag")
                }).fetchall()

            if not results:
                self.logger.info(f"No results found for accession {accession}.")
                return []

            # Procesar resultados
            go_terms = []
            for row in results:
                go_terms.append({
                    'accession': accession,
                    'go_id': row.go_term_id,
                    'category': row.category,
                    'evidence_code': row.evidence_code,
                    'go_description': row.go_term_description,
                    'distance': row.distance,
                    'model_name': task_data['model_name'],
                    'protein_id': row.protein_id,
                    'organism': row.organism,
                })

            self.logger.info(
                f"Found {len(go_terms)} GO terms for accession {accession} and embedding type {embedding_type_id}.")
            return go_terms

        except Exception as e:
            self.logger.error(
                f"Error processing task for accession {accession} and embedding type {embedding_type_id}: {e}")
            raise

    def store_entry(self, annotations):
        if not annotations:
            self.logger.info("No valid GO terms to store.")
            return

        try:
            df = pd.DataFrame(annotations)

            # Compute reliability index based on the selected distance metric
            if self.distance_metric == '<=>':
                df["reliability_index"] = 1 - df["distance"]
            if self.distance_metric == '<->':
                df["reliability_index"] = 0.5 / (0.5 + df["distance"])

            # Keep only the GO term with the highest reliability for each (accession, go_id)
            df = df.loc[df.groupby(["accession", "go_id"])["reliability_index"].idxmax()]

            # Identify parent GO terms using goatools
            parent_terms = set()
            for go_id in df["go_id"].unique():
                if go_id in self.go:  # Ensure the GO term exists in the ontology
                    parent_terms.update(p.id for p in self.go[go_id].parents)

            # Remove parent GO terms if a more specific child term is already present
            df = df[~df["go_id"].isin(parent_terms)]

            # Sort results by reliability index (higher values first)
            df = df.sort_values(by="reliability_index", ascending=False)

            # Save results to CSV
            results_path = self.results_path
            if os.path.exists(results_path) and os.path.getsize(results_path) > 0:
                df.to_csv(results_path, mode='a', index=False, header=False)  # Append to existing file
            else:
                df.to_csv(results_path, mode='w', index=False, header=True)  # Create new file with header

            self.logger.info(f"Stored {len(df)} collapsed entries in CSV.")

            # ✅ Generate TopGO file if enabled (exact format match)
            if self.topgo_enabled:
                # Format the data correctly for TopGO (GENE_ID<TAB>GO:XXXXXXX, GO:XXXXXXX, ...)
                df_topgo = df.groupby("accession")["go_id"].apply(lambda x: ", ".join(x)).reset_index()

                # Append to the TopGO file without header
                with open(self.topgo_path, "a") as f:
                    df_topgo.to_csv(f, sep="\t", index=False, header=False)

        except Exception as e:
            self.logger.error(f"Error storing results: {e}")
            raise

    def retrieve_cluster_members(self, accession):
        """
        Recupera todos los miembros del grupo/clúster asociado al accession para excluirlos de la consulta.

        Parameters
        ----------
        accession : str
            Identificador de la secuencia de consulta.

        Returns
        -------
        set
            Conjunto de IDs de secuencias que pertenecen al mismo grupo del accession dado.

        Raises
        ------
        Exception
            Si ocurre un error durante la consulta de los clústeres.
        """
        try:
            if not hasattr(self, "clusters"):
                raise ValueError("Clusters are not loaded into memory. Please ensure 'generate_clusters' was called.")

            self.logger.info(f"Retrieving cluster members for accession {accession}...")

            # Filtrar el clúster al que pertenece el accession
            cluster_row = self.clusters[self.clusters['identifier'] == accession]
            if cluster_row.empty:
                self.logger.warning(f"Accession {accession} not found in any cluster.")
                return set()

            # Recuperar todos los miembros del mismo clúster
            cluster_id = cluster_row.iloc[0]['cluster']
            cluster_members = set(self.clusters[self.clusters['cluster'] == cluster_id]['identifier'])

            # Filtrar IDs no numéricos
            filtered_members = {member for member in cluster_members if member.isdigit()}

            self.logger.info(
                f"Found {len(filtered_members)} members in the cluster for accession {accession} after filtering non-numeric IDs.")
            return filtered_members

        except Exception as e:
            self.logger.error(f"Error retrieving cluster members for accession {accession}: {e}")
            raise
