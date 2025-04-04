"""
EmbeddingLookUp Module
=======================

This module defines the `EmbeddingLookUp` class, which enables functional annotation of proteins
based on embedding similarity.

Given a set of query embeddings stored in HDF5 format, the class computes distances to reference
embeddings stored in a database, retrieves associated GO term annotations, and stores the results
in standard formats (CSV and optionally TopGO-compatible TSV). It also supports redundancy filtering
via CD-HIT and flexible integration with custom embedding models.

Background
----------

The design and logic are inspired by the GoPredSim tool:
- GoPredSim: https://github.com/Rostlab/goPredSim

Enhancements have been made to integrate the lookup process with:
- a vector-aware relational database,
- embedding models dynamically loaded from modular pipelines,
- and GO ontology support via the goatools package.

The system is designed for scalability, interpretability, and compatibility
with downstream enrichment analysis tools.
"""

import importlib
import os
import numpy as np
import pandas as pd
from goatools.base import get_godag
from protein_metamorphisms_is.sql.model.entities.sequence.sequence import Sequence
from pycdhit import cd_hit, read_clstr
from sklearn.metrics import euclidean_distances
from sklearn.metrics.pairwise import cosine_distances

from sqlalchemy import text
import h5py

from protein_metamorphisms_is.sql.model.entities.embedding.sequence_embedding import SequenceEmbeddingType, \
    SequenceEmbedding
from protein_metamorphisms_is.tasks.queue import QueueTaskInitializer
from protein_metamorphisms_is.helpers.clustering.cdhit import calculate_cdhit_word_length


class EmbeddingLookUp(QueueTaskInitializer):
    """
    EmbeddingLookUp handles the similarity-based annotation of proteins using precomputed embeddings.

    This class reads sequence embeddings from an HDF5 file, computes similarity to known embeddings
    stored in a database, retrieves GO term annotations from similar sequences, and writes
    the predicted annotations to a CSV file. It also supports optional redundancy filtering
    via CD-HIT and generation of a TopGO-compatible TSV file.

    Parameters
    ----------
    conf : dict
        Configuration dictionary with paths, thresholds, model definitions, and flags.
    current_date : str
        Timestamp used to generate unique file names for outputs.

    Attributes
    ----------
    experiment_path : str
        Base path for output files and temporary data.
    embeddings_path : str
        Path to the input HDF5 file containing embeddings and sequences.
    results_path : str
        Path to write the final CSV file containing GO term predictions.
    topgo_path : str
        Path to write the optional TopGO-compatible TSV file.
    topgo_enabled : bool
        Flag indicating whether TopGO output should be generated.
    limit_per_entry : int
        Maximum number of neighbors considered per query during lookup.
    distance_metric : str
        Metric used to compute similarity between embeddings ("<->" or "<=>").
    types : dict
        Metadata and modules for each enabled embedding model.
    lookup_tables : dict
        Preloaded embeddings used for distance computations, organized by model.
    go : GODag
        Gene Ontology DAG loaded via goatools.
    clusters : pandas.DataFrame, optional
        Cluster assignments used for redundancy filtering (if enabled).
    """

    def __init__(self, conf, current_date):
        """
        Initializes the EmbeddingLookUp class with configuration, paths, model metadata,
        and preloaded resources required for embedding-based GO annotation transfer.

        Parameters
        ----------
        conf : dict
            Configuration dictionary with paths, thresholds, and embedding model settings.
        current_date : str
            Timestamp used for uniquely identifying output files.
        """
        super().__init__(conf)

        self.current_date = current_date
        self.logger.info("Initializing EmbeddingLookUp...")

        # Paths
        self.experiment_path = self.conf.get("experiment_path")
        self.embeddings_path = self.conf.get("embeddings_path") or os.path.join(self.experiment_path, "embeddings.h5")
        self.results_path = os.path.join(self.experiment_path, "results.csv")
        self.topgo_path = os.path.join(self.experiment_path, "results_topgo.tsv")

        # Limits and optional features
        self.limit_per_entry = self.conf.get("limit_per_entry", 200)
        self.topgo_enabled = self.conf.get("topgo", False)

        # Initialize embedding models
        self.fetch_models_info()

        # Redundancy filtering setup
        redundancy_filter_threshold = self.conf.get("redundancy_filter", 0)
        if redundancy_filter_threshold > 0:
            self.generate_clusters()

        # Load GO ontology
        self.go = get_godag("go-basic.obo", optional_attrs="relationship")

        # Select distance metric
        self.distance_metric = self.conf.get("embedding", {}).get("distance_metric", "euclidean")
        if self.distance_metric not in ("euclidean", "cosine"):
            self.logger.warning(
                f"Invalid distance metric '{self.distance_metric}', defaulting to 'euclidean'."
            )
            self.distance_metric = "euclidean"

        # Load embedding lookup tables into memory
        self.lookup_table_into_memory()

        self.logger.info("EmbeddingLookUp initialization complete.")

    def fetch_models_info(self):
        """
        Loads embedding model definitions from the database and dynamically imports associated modules.

        This method retrieves all embedding types stored in the `SequenceEmbeddingType` table and checks
        which ones are enabled in the configuration. For each enabled model, it dynamically imports the
        embedding module and stores the metadata in the `self.types` dictionary.

        Raises
        ------
        Exception
            If the database query fails or a model module cannot be imported.

        Notes
        -----
        - `self.types` stores metadata per model task_name, including embedding type ID, module reference,
          and thresholds.
        - ⚠ TODO: This method should be factorized into parent class to avoid duplications.

        """

        try:
            embedding_types = self.session.query(SequenceEmbeddingType).all()
        except Exception as e:
            self.logger.error(f"Error querying SequenceEmbeddingType table: {e}")
            raise

        self.types = {}
        enabled_models = self.conf.get("embedding", {}).get("models", {})

        for embedding_type in embedding_types:
            task_name = embedding_type.task_name
            if task_name not in enabled_models:
                continue

            model_config = enabled_models[task_name]
            if not model_config.get("enabled", False):
                continue

            try:
                base_module_path = "protein_metamorphisms_is.operation.embedding.proccess.sequence"
                module_name = f"{base_module_path}.{task_name}"
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
        Reads embeddings and sequences from an HDF5 file and enqueues tasks in batches.

        Each task includes a protein accession, its amino acid sequence, and a set of embeddings
        generated by one or more models. Embeddings are grouped by model type and published in
        configurable batches for downstream processing.

        Raises
        ------
        Exception
            If any error occurs while reading the HDF5 file or publishing tasks.
        """
        try:
            self.logger.info(f"Reading embeddings from HDF5: {self.embeddings_path}")

            if not os.path.exists(self.embeddings_path):
                raise FileNotFoundError(
                    f"❌ The HDF5 file '{self.embeddings_path}' does not exist.\n"
                    f"💡 Make sure the embedding step has been completed, or that the path is correct "
                    f"(e.g., use 'only_lookup: true' with a valid 'input' path in the config)."
                )

            batch_size = self.conf.get("batch_size", 4)
            batch = []
            total_batches = 0

            with h5py.File(self.embeddings_path, "r") as h5file:
                for accession, group in h5file.items():
                    # Ensure sequence is available
                    if "sequence" not in group:
                        self.logger.warning(f"Missing sequence for accession '{accession}'. Skipping.")
                        continue

                    sequence = group["sequence"][()].decode("utf-8")

                    # Iterate through available embeddings
                    for item_name, item_group in group.items():
                        if not item_name.startswith("type_") or "embedding" not in item_group:
                            continue

                        model_key = item_name.replace("type_", "")
                        if model_key not in self.types:
                            self.logger.warning(
                                f"Unrecognized model '{model_key}' for accession '{accession}'. Skipping.")
                            continue

                        embedding = item_group["embedding"][:]
                        model_info = self.types[model_key]

                        task_data = {
                            "accession": accession,
                            "sequence": sequence,
                            "embedding": embedding,
                            "embedding_type_id": model_info["id"],
                            "model_name": model_key,
                            "distance_threshold": model_info["distance_threshold"]
                        }
                        batch.append(task_data)

                        # Publish batch if size is reached
                        if len(batch) == batch_size:
                            self.publish_task(batch)
                            total_batches += 1
                            self.logger.info(f"Published batch {total_batches} with {batch_size} tasks.")
                            batch = []

            # Publish any remaining entries
            if batch:
                self.publish_task(batch)
                total_batches += 1
                self.logger.info(f"Published final batch {total_batches} with {len(batch)} tasks.")

            self.logger.info(f"Enqueued a total of {total_batches} batches for processing.")
        except OSError:
            self.logger.error(f"Failed to read HDF5 file: '{self.embeddings_path}'. "
                              f"Make sure that to perform the only lookup, an embedding file in H5 format is required as input.")
            raise
        except Exception as e:
            import traceback
            self.logger.error(f"Error enqueuing tasks from HDF5: {e}\n{traceback.format_exc()}")
            raise

    def process(self, task_data):
        """
        Processes a batch of embedding-based lookup tasks and retrieves associated GO term annotations.

        For each input embedding, the method:
        - Computes distances against preloaded embeddings.
        - Selects similar sequences under a configured threshold.
        - Fetches GO annotations from the database.
        - Optionally filters out redundant annotations via clustering.
        - Returns GO terms with distance metadata for each matched protein.

        Parameters
        ----------
        task_data : list of dict
            A list of input entries, each with:
            - accession : str
            - embedding : np.ndarray
            - embedding_type_id : int
            - model_name : str
            - distance_threshold : float

        Returns
        -------
        list of dict
            A list of GO annotations with metadata and distance information.
        """
        try:
            limit_per_entry = self.conf.get("limit_per_entry", 1000)

            accession_list = []
            embeddings = []
            thresholds = []
            model_names = {}
            embedding_type_ids = []

            for task in task_data:
                accession = task["accession"].removeprefix("accession_")
                accession_list.append(accession)
                embeddings.append(np.array(task["embedding"]))
                thresholds.append(task["distance_threshold"])
                model_names[accession] = task["model_name"]
                embedding_type_ids.append(task["embedding_type_id"])

            selected_sequence_ids = set()
            distance_map = {}

            # Distance computation and selection
            for idx, accession in enumerate(accession_list):
                embedding_vector = embeddings[idx]
                threshold = thresholds[idx]
                type_id = embedding_type_ids[idx]

                lookup = self.lookup_tables.get(type_id)
                if lookup is None:
                    self.logger.warning(f"No lookup table for embedding_type_id {type_id}. Skipping {accession}.")
                    continue

                if self.distance_metric == "euclidean":
                    distances = euclidean_distances([embedding_vector], lookup["embeddings"])[0]
                elif self.distance_metric == "cosine":
                    distances = cosine_distances([embedding_vector], lookup["embeddings"])[0]
                else:
                    raise ValueError(f"Unsupported distance metric: {self.distance_metric}")

                sorted_indices = np.argsort(distances)
                selected = sorted_indices[distances[sorted_indices] <= threshold][:limit_per_entry]

                for i in selected:
                    seq_id = lookup["ids"][i]
                    dist = distances[i]
                    selected_sequence_ids.add(seq_id)
                    distance_map[(accession, seq_id)] = dist

            if not selected_sequence_ids:
                self.logger.info("No sequence IDs passed distance threshold filtering.")
                return []

            # Fetch GO annotations for selected sequence IDs
            sql = text("""
                SELECT
                    s.id AS sequence_id,
                    p.id AS protein_id,
                    p.gene_name,
                    p.organism,
                    pgo.go_id,
                    gt.category,
                    gt.description AS go_term_description,
                    pgo.evidence_code
                FROM
                    sequence s
                    JOIN protein p ON s.id = p.sequence_id
                    JOIN protein_go_term_annotation pgo ON p.id = pgo.protein_id
                    JOIN go_terms gt ON pgo.go_id = gt.go_id
                WHERE
                    s.id IN :sequence_ids
            """)

            with self.engine.connect() as connection:
                rows = connection.execute(sql, {
                    "sequence_ids": tuple(int(sid) for sid in selected_sequence_ids)
                }).fetchall()

            # Filter redundancy (optional)
            redundant_ids_by_accession = {}
            if self.conf.get("redundancy_filter", 0) > 0:
                for accession in accession_list:
                    redundant_ids_by_accession[accession] = self.retrieve_cluster_members(accession)

            # Construct output list
            go_terms = []
            for accession in accession_list:
                for row in rows:
                    seq_id = row.sequence_id

                    if accession in redundant_ids_by_accession:
                        if str(seq_id) in redundant_ids_by_accession[accession]:
                            continue

                    if (accession, seq_id) not in distance_map:
                        continue

                    go_terms.append({
                        "accession": accession,
                        "go_id": row.go_id,
                        "category": row.category,
                        "evidence_code": row.evidence_code,
                        "go_description": row.go_term_description,
                        "distance": distance_map[(accession, seq_id)],
                        "model_name": model_names[accession],
                        "protein_id": row.protein_id,
                        "organism": row.organism,
                    })

            self.logger.info(f"Processed {len(go_terms)} GO terms for batch of {len(task_data)} entries.")
            return go_terms

        except Exception as e:
            self.logger.error(f"Error during GO annotation processing: {e}")
            raise

    def store_entry(self, annotations):
        """
        Processes and stores GO term annotations for a given set of accessions.

        This method computes a reliability score for each annotation, filters out
        redundant or less reliable entries, and saves the final results to a CSV file.
        If enabled, it also generates a TSV file compatible with TopGO.

        Parameters
        ----------
        annotations : list of dict
            A list of GO annotations, each containing metadata such as accession, GO ID,
            model name, distance, and evidence code.

        Raises
        ------
        Exception
            If any error occurs during processing or file writing.
        """
        if not annotations:
            self.logger.info("No valid GO terms to store.")
            return

        try:
            df = pd.DataFrame(annotations)

            # Compute reliability index based on the selected distance metric
            if self.distance_metric == "cosine":
                df["reliability_index"] = 1 - df["distance"]
            elif self.distance_metric == "euclidean":
                df["reliability_index"] = 0.5 / (0.5 + df["distance"])

            # Retain only the most reliable annotation per (accession, GO term)
            df = df.loc[df.groupby(["accession", "go_id"])["reliability_index"].idxmax()]

            # Identify all parent GO terms using the GO DAG
            parent_go_terms = set()
            for go_id in df["go_id"].unique():
                if go_id in self.go:
                    parent_go_terms.update(p.id for p in self.go[go_id].parents)

            # Filter out parent terms if their children are already present
            df = df[~df["go_id"].isin(parent_go_terms)]

            # Sort by reliability (descending)
            df = df.sort_values(by="reliability_index", ascending=False)

            # Save final results to CSV
            write_mode = "a" if os.path.exists(self.results_path) and os.path.getsize(self.results_path) > 0 else "w"
            include_header = write_mode == "w"
            df.to_csv(self.results_path, mode=write_mode, index=False, header=include_header)

            self.logger.info(f"Stored {len(df)} GO annotations to CSV.")

            # If enabled, generate TopGO-compatible TSV file
            if self.topgo_enabled:
                df_topgo = (
                    df.groupby("accession")["go_id"]
                    .apply(lambda x: ", ".join(x))
                    .reset_index()
                )

                with open(self.topgo_path, "a") as f:
                    df_topgo.to_csv(f, sep="\t", index=False, header=False)

        except Exception as e:
            self.logger.error(f"Error storing results: {e}")
            raise

    def generate_clusters(self):
        """
        Generates non-redundant sequence clusters using CD-HIT.

        This method builds a FASTA reference file by combining sequences from the database
        and the HDF5 embedding file. It then runs CD-HIT to cluster sequences based on identity
        and coverage thresholds. The resulting clusters are loaded into memory for redundancy filtering.

        Raises
        ------
        Exception
            If any error occurs during FASTA creation, CD-HIT execution, or cluster parsing.
        """
        try:
            input_h5_path = os.path.join(self.conf["experiment_path"], "embeddings.h5")
            self.reference_fasta = os.path.join(self.experiment_path, "redundancy.fasta")
            filtered_fasta = os.path.join(self.experiment_path, "filtered.fasta")

            # Step 1: Build combined reference FASTA file
            self.logger.info("Generating reference FASTA file from DB and HDF5...")
            with open(self.reference_fasta, "w") as ref_file:
                # Add sequences from the SQL database
                with self.engine.connect() as connection:
                    query = text("SELECT id, sequence FROM sequence")
                    for row in connection.execute(query):
                        ref_file.write(f">{row.id}\n{row.sequence}\n")

                print(ref_file)
                # Add sequences from the HDF5 file
                with h5py.File(input_h5_path, "r") as h5file:
                    for accession, group in h5file.items():
                        if "sequence" in group:
                            sequence = group["sequence"][()].decode("utf-8")
                            clean_id = accession.removeprefix("accession_")
                            ref_file.write(f">{clean_id}\n{sequence}\n")

            # Step 2: Prepare CD-HIT parameters
            identity = self.conf.get("redundancy_filter", 0.95)
            coverage = self.conf.get("alignment_coverage", 0.95)
            memory = self.conf.get("memory_usage", 32000)
            threads = self.conf.get("threads", 0)
            search_mode = self.conf.get("most_representative_search", 1)
            word_length = calculate_cdhit_word_length(identity, self.logger)

            self.logger.info("Running CD-HIT with parameters:")
            self.logger.info(f"  Identity threshold: {identity}")
            self.logger.info(f"  Coverage: {coverage}")
            self.logger.info(f"  Memory: {memory} MB")
            self.logger.info(f"  Threads: {threads}")
            self.logger.info(f"  Word length: {word_length}")

            # Step 3: Execute CD-HIT
            cd_hit(
                i=self.reference_fasta,
                o=filtered_fasta,
                c=identity,
                d=0,
                aL=coverage,
                M=memory,
                T=threads,
                g=search_mode,
                n=word_length
            )

            # Step 4: Load resulting clusters
            clstr_path = f"{filtered_fasta}.clstr"
            if not os.path.exists(clstr_path) or os.path.getsize(clstr_path) == 0:
                raise ValueError(f"CD-HIT .clstr file missing or empty: {clstr_path}")

            self.logger.info(f"CD-HIT completed. Loading clusters from: {clstr_path}")
            self.clusters = read_clstr(clstr_path)
            self.logger.info(f"{len(self.clusters)} clusters loaded into memory.")

        except Exception as e:
            self.logger.error(f"Error while generating CD-HIT clusters: {e}")
            raise

    def retrieve_cluster_members(self, accession):
        """
        Retrieves all members from the cluster associated with the given accession.

        This method is used to identify potentially redundant sequences belonging to
        the same cluster as the query accession. These members can then be excluded
        from downstream annotation transfer.

        Parameters
        ----------
        accession : str
            The query accession for which cluster members will be retrieved.

        Returns
        -------
        set of str
            A set of sequence identifiers (as strings) that belong to the same cluster
            as the given accession. Only numeric identifiers are returned.

        Raises
        ------
        Exception
            If cluster information is missing or an error occurs during filtering.
        """
        try:
            if not hasattr(self, "clusters"):
                raise ValueError(
                    "Cluster data not loaded. Make sure 'generate_clusters()' has been called before using this method."
                )

            self.logger.info(f"Retrieving cluster members for accession '{accession}'...")

            # Find the cluster to which this accession belongs
            cluster_row = self.clusters[self.clusters["identifier"] == accession]
            if cluster_row.empty:
                self.logger.warning(f"Accession '{accession}' not found in any cluster.")
                return set()

            cluster_id = cluster_row.iloc[0]["cluster"]

            # Extract all identifiers in the same cluster
            cluster_members = self.clusters[self.clusters["cluster"] == cluster_id]["identifier"]

            # Keep only numeric IDs (likely to be database sequence IDs)
            numeric_members = {member for member in cluster_members if member.isdigit()}

            self.logger.info(
                f"Cluster {cluster_id}: found {len(numeric_members)} numeric members for accession '{accession}'."
            )

            return numeric_members

        except Exception as e:
            self.logger.error(f"Failed to retrieve cluster members for accession '{accession}': {e}")
            raise

    def lookup_table_into_memory(self):
        """
        Loads sequence embeddings from the database into memory for each enabled model (embedding_type_id).

        Embeddings are fetched from the `SequenceEmbedding` table and stored in memory as NumPy arrays,
        allowing fast similarity searches during processing. Optionally, a limit can be applied per model
        to reduce memory usage or for testing purposes.

        Raises
        ------
        Exception
            If any error occurs during database query or memory allocation.
        """
        try:
            self.logger.info("🔄 Starting lookup table construction: loading embeddings into memory per model...")

            self.lookup_tables = {}
            limit_execution = self.conf.get("limit_execution")

            for task_name, model_info in self.types.items():
                embedding_type_id = model_info["id"]
                self.logger.info(f"📥 Model '{task_name}' (ID: {embedding_type_id}): retrieving embeddings...")

                # Build the query to retrieve sequence ID and its embedding vector
                query = (
                    self.session
                    .query(Sequence.id, SequenceEmbedding.embedding)
                    .join(Sequence, Sequence.id == SequenceEmbedding.sequence_id)
                    .filter(SequenceEmbedding.embedding_type_id == embedding_type_id)
                )

                if isinstance(limit_execution, int) and limit_execution > 0:
                    self.logger.info(f"⛔ SQL limit applied: {limit_execution} entries for model '{task_name}'")
                    query = query.limit(limit_execution)

                results = query.all()
                if not results:
                    self.logger.warning(f"⚠️ No embeddings found for model '{task_name}' (ID: {embedding_type_id})")
                    continue

                sequence_ids = np.array([row[0] for row in results])
                embeddings = np.vstack([row[1].to_numpy() for row in results])
                mem_mb = embeddings.nbytes / (1024 ** 2)

                self.lookup_tables[embedding_type_id] = {
                    "ids": sequence_ids,
                    "embeddings": embeddings
                }

                self.logger.info(
                    f"✅ Model '{task_name}': loaded {len(sequence_ids)} embeddings "
                    f"with shape {embeddings.shape} (~{mem_mb:.2f} MB in memory)."
                )

            self.logger.info(f"🏁 Lookup table construction completed for {len(self.lookup_tables)} model(s).")

        except Exception:
            import traceback
            self.logger.error("❌ Failed to load lookup tables:\n" + traceback.format_exc())
            raise
