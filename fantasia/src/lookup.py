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
import time
from concurrent.futures import ProcessPoolExecutor

from protein_metamorphisms_is.tasks.base import BaseTaskInitializer

import numpy as np
import pandas as pd
from goatools.base import get_godag
from protein_metamorphisms_is.sql.model.entities.sequence.sequence import Sequence
from pycdhit import cd_hit, read_clstr

from sqlalchemy import text
import h5py
from protein_metamorphisms_is.sql.model.entities.embedding.sequence_embedding import SequenceEmbeddingType, \
    SequenceEmbedding
from protein_metamorphisms_is.sql.model.entities.protein.protein import Protein
from protein_metamorphisms_is.helpers.clustering.cdhit import calculate_cdhit_word_length

from fantasia.src.helpers.helpers import run_needle_from_strings, get_descendant_ids


def compute_metrics(row):
    seq1 = row["sequence_query"]
    seq2 = row["sequence_reference"]
    metrics = run_needle_from_strings(seq1, seq2)
    return {
        "sequence_query": seq1,
        "sequence_reference": seq2,
        "identity": metrics["identity_percentage"],
        "similarity": metrics.get("similarity_percentage"),
        "alignment_score": metrics["alignment_score"],
        "gaps_percentage": metrics.get("gaps_percentage"),
        "alignment_length": metrics["alignment_length"],
        "length_query": len(seq1),
        "length_reference": len(seq2),
    }


class EmbeddingLookUp(BaseTaskInitializer):
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

        self.raw_results_path = os.path.join(self.experiment_path, "raw_results.csv")
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

        self.logger.info("EmbeddingLookUp initialization complete.")

    def start(self):
        self.logger.info("Starting embedding-based GO annotation process.")

        self.logger.info("Preloading GO annotations from the database.")
        self.preload_annotations()

        self.logger.info("Loading reference embeddings into memory.")
        self.lookup_table_into_memory()

        self.logger.info(f"Processing query embeddings from HDF5: {self.embeddings_path}")
        try:
            batch_size = self.conf.get("batch_size", 4)
            batches_by_model = {}
            total_batches = 0

            if not os.path.exists(self.embeddings_path):
                raise FileNotFoundError(
                    f"HDF5 file not found: {self.embeddings_path}. "
                    f"Ensure embeddings have been generated prior to annotation."
                )

            with h5py.File(self.embeddings_path, "r") as h5file:
                for accession, group in h5file.items():
                    if "sequence" not in group:
                        self.logger.warning(f"Sequence missing for accession '{accession}'. Skipping.")
                        continue

                    sequence = group["sequence"][()].decode("utf-8")

                    for item_name, item_group in group.items():
                        if not item_name.startswith("type_") or "embedding" not in item_group:
                            continue

                        model_key = item_name.replace("type_", "")
                        if model_key not in self.types:
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

                        batches_by_model.setdefault(model_key, []).append(task_data)

            for model_key, tasks in batches_by_model.items():
                for i in range(0, len(tasks), batch_size):
                    batch = tasks[i:i + batch_size]
                    annotations = self.process(batch)
                    self.store_entry(annotations)
                    total_batches += 1
                    self.logger.info(
                        f"Processed batch {total_batches} for model '{model_key}' with {len(batch)} entries.")

            self.logger.info(f"All batches completed successfully. Total batches: {total_batches}.")

        except Exception as e:
            self.logger.error(f"Unexpected error during batch processing: {e}", exc_info=True)
            raise

        self.logger.info("Starting post-processing of annotation results.")
        self.post_process_results()
        self.logger.info("Embedding lookup pipeline completed.")

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
        import torch
        import numpy as np
        from scipy.spatial.distance import cdist

        task = task_data[0]
        model_id = task["embedding_type_id"]
        model_name = task["model_name"]
        threshold = task["distance_threshold"]
        use_gpu = self.conf.get("use_gpu", True)
        limit = self.conf.get("limit_per_entry", 1000)

        lookup = self.lookup_tables.get(model_id)
        if lookup is None:
            self.logger.warning(f"No lookup table for embedding_type_id {model_id}. Skipping batch.")
            return []

        embeddings = np.stack([np.array(t["embedding"]) for t in task_data])
        accessions = [t["accession"].removeprefix("accession_") for t in task_data]
        sequences = {t["accession"].removeprefix("accession_"): t["sequence"] for t in task_data}

        if use_gpu:
            queries = torch.tensor(embeddings, dtype=torch.float16).cuda()
            targets = torch.tensor(lookup["embeddings"], dtype=torch.float16).cuda()

            if self.distance_metric == "euclidean":
                q2 = (queries ** 2).sum(dim=1).unsqueeze(1)
                t2 = (targets ** 2).sum(dim=1).unsqueeze(0)
                d2 = q2 + t2 - 2 * torch.matmul(queries, targets.T)
                dist_matrix = torch.sqrt(torch.clamp(d2, min=0.0)).cpu().numpy()
            elif self.distance_metric == "cosine":
                qn = torch.nn.functional.normalize(queries, p=2, dim=1)
                tn = torch.nn.functional.normalize(targets, p=2, dim=1)
                dist_matrix = (1 - torch.matmul(qn, tn.T)).cpu().numpy()
            else:
                raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
        else:
            dist_matrix = cdist(embeddings, lookup["embeddings"], metric=self.distance_metric)

        redundancy = self.conf.get("redundancy_filter", 0)
        redundant_ids = {}
        if redundancy > 0:
            for acc in accessions:
                redundant_ids[acc] = self.retrieve_cluster_members(acc)

        go_terms = []
        total_transfers = 0
        total_neighbors = 0

        for i, accession in enumerate(accessions):
            all_distances = dist_matrix[i]
            all_seq_ids = lookup["ids"]

            if redundancy > 0 and accession in redundant_ids:
                mask = ~np.isin(all_seq_ids.astype(str), list(redundant_ids[accession]))
                distances = all_distances[mask]
                seq_ids = all_seq_ids[mask]
            else:
                distances = all_distances
                seq_ids = all_seq_ids

            if len(distances) == 0:
                continue

            sorted_idx = np.argsort(distances)
            selected_idx = sorted_idx[distances[sorted_idx] <= threshold][:limit]
            total_neighbors += len(selected_idx)

            for idx in selected_idx:
                seq_id = seq_ids[idx]
                if seq_id not in self.go_annotations:
                    continue

                annotations = self.go_annotations[seq_id]
                total_transfers += len(annotations)

                for ann in annotations:
                    go_terms.append({
                        "accession": accession,
                        "sequence_query": sequences[accession],
                        "sequence_reference": ann["sequence"],
                        "go_id": ann["go_id"],
                        "category": ann["category"],
                        "evidence_code": ann["evidence_code"],
                        "go_description": ann["go_description"],
                        "distance": distances[idx],
                        "model_name": model_name,
                        "protein_id": ann["protein_id"],
                        "organism": ann["organism"],
                        "gene_name": ann["gene_name"],
                    })

        self.logger.info(
            f"✅ Batch processed ({len(accessions)} entries): {total_neighbors} neighbors found, "
            f"{total_transfers} GO annotations transferred."
        )
        return go_terms

    def store_entry(self, annotations):
        if not annotations:
            self.logger.info("No valid GO terms to store.")
            return

        try:
            df = pd.DataFrame(annotations)
            write_mode = "a" if os.path.exists(self.raw_results_path) else "w"
            include_header = write_mode == "w"
            df.to_csv(self.raw_results_path, mode=write_mode, index=False, header=include_header)
            self.logger.info(f"Stored {len(df)} raw GO annotations.")
        except Exception as e:
            self.logger.error(f"Error writing raw results: {e}")
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
                l=4,
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
            self.clusters_by_id = self.clusters.set_index("identifier")
            self.clusters_by_cluster = self.clusters.groupby("cluster")["identifier"].apply(set).to_dict()

            self.logger.info(f"{len(self.clusters)} clusters loaded into memory.")

        except Exception as e:
            self.logger.error(f"Error while generating CD-HIT clusters: {e}")
            raise

    def retrieve_cluster_members(self, accession):
        try:
            cluster_id = self.clusters_by_id.loc[accession, "cluster"]
            members = self.clusters_by_cluster.get(cluster_id, set())
            return {m for m in members if m.isdigit()}
        except KeyError:
            self.logger.warning(f"Accession '{accession}' not found in clusters.")
            return set()

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
            get_descendants = self.conf.get("get_descendants", False)

            # Procesar filtros de taxonomía
            def expand_tax_ids(key):
                ids = self.conf.get(key, [])
                if get_descendants and ids:
                    return get_descendant_ids([int(tid) for tid in ids])
                return [str(tid) for tid in ids]

            exclude_taxon_ids = expand_tax_ids("taxonomy_ids_to_exclude")
            include_taxon_ids = expand_tax_ids("taxonomy_ids_included_exclusively")

            if exclude_taxon_ids and include_taxon_ids:
                self.logger.warning(
                    "⚠️ Both 'taxonomy_ids_to_exclude' and 'taxonomy_ids_included_exclusively' are set. This may lead to conflicting filters.")

            self.logger.info(
                f"🧬 Taxonomy filters — Exclude: {exclude_taxon_ids}, Include: {include_taxon_ids}, Descendants: {get_descendants}")

            for task_name, model_info in self.types.items():
                embedding_type_id = model_info["id"]
                self.logger.info(f"📥 Model '{task_name}' (ID: {embedding_type_id}): retrieving embeddings...")

                # Build the query to retrieve sequence ID and its embedding vector
                query = (
                    self.session
                    .query(Sequence.id, SequenceEmbedding.embedding)
                    .join(Sequence, Sequence.id == SequenceEmbedding.sequence_id)
                    .join(Protein, Sequence.id == Protein.sequence_id)
                    .filter(SequenceEmbedding.embedding_type_id == embedding_type_id)
                )

                if exclude_taxon_ids:
                    query = query.filter(~Protein.taxonomy_id.in_(exclude_taxon_ids))
                if include_taxon_ids:
                    query = query.filter(Protein.taxonomy_id.in_(include_taxon_ids))
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

    def preload_annotations(self):
        sql = text("""
                   SELECT s.id           AS sequence_id,
                          s.sequence,
                          pgo.go_id,
                          gt.category,
                          gt.description AS go_term_description,
                          pgo.evidence_code,
                          p.id           AS protein_id,
                          p.organism,
                          p.gene_name
                   FROM sequence s
                            JOIN protein p ON s.id = p.sequence_id
                            JOIN protein_go_term_annotation pgo ON p.id = pgo.protein_id
                            JOIN go_terms gt ON pgo.go_id = gt.go_id
                   """)
        self.go_annotations = {}

        with self.engine.connect() as connection:
            for row in connection.execute(sql):
                entry = {
                    "sequence": row.sequence,
                    "go_id": row.go_id,
                    "category": row.category,
                    "evidence_code": row.evidence_code,
                    "go_description": row.go_term_description,
                    "protein_id": row.protein_id,
                    "organism": row.organism,
                    "gene_name": row.gene_name,
                }
                self.go_annotations.setdefault(row.sequence_id, []).append(entry)

    def post_process_results(self):
        if not os.path.exists(self.raw_results_path):
            self.logger.warning("No raw results found for post-processing.")
            return

        self.logger.info("🔍 Starting post-processing of raw GO annotations.")
        start_total = time.perf_counter()

        df = pd.read_csv(self.raw_results_path)

        start_reliability = time.perf_counter()
        if self.distance_metric == "cosine":
            df["reliability_index"] = 1 - df["distance"]
        elif self.distance_metric == "euclidean":
            df["reliability_index"] = 0.5 / (0.5 + df["distance"])
        end_reliability = time.perf_counter()

        start_alignment = time.perf_counter()
        unique_pairs = df[["sequence_query", "sequence_reference"]].drop_duplicates()
        with ProcessPoolExecutor(max_workers=self.conf.get("store_workers", 4)) as executor:
            metrics_list = list(executor.map(compute_metrics, unique_pairs.to_dict("records")))
        metrics_df = pd.DataFrame(metrics_list)
        df = df.merge(metrics_df, on=["sequence_query", "sequence_reference"], how="left")
        end_alignment = time.perf_counter()

        df = df.drop(columns=["sequence_query", "sequence_reference"], errors="ignore")

        def is_ancestor(go_dag, parent, child):
            """True si `parent` es ancestro de `child` en la ontología GO."""
            return child in go_dag and parent in go_dag[child].get_all_parents()

        rows = []
        for (_, _), group in df.groupby(["accession", "model_name"]):
            for go_id, go_group in group.groupby("go_id"):
                all_go_ids = go_group["go_id"].unique()

                leaf_entries = []
                for i, row in go_group.iterrows():
                    term = row["go_id"]
                    is_leaf = not any(
                        term != other and is_ancestor(self.go, term, other)
                        for other in all_go_ids
                    )
                    if is_leaf:
                        leaf_entries.append(row)

                rows.extend(leaf_entries)

        # Nuevo DataFrame con solo términos hoja por (accession, go_id, model_name)
        df = pd.DataFrame(rows)
        df = df.sort_values(by=["accession", "go_id", "model_name", "reliability_index"],
                            ascending=[True, True, True, False])

        write_mode = "a" if os.path.exists(self.results_path) else "w"
        df.to_csv(self.results_path, mode=write_mode, index=False, header=(write_mode == "w"))

        if self.topgo_enabled:
            df_topgo = (
                df.groupby("accession")["go_id"]
                .apply(lambda x: ", ".join(x))
                .reset_index()
            )
            with open(self.topgo_path, "a") as f:
                df_topgo.to_csv(f, sep="\t", index=False, header=False)

        end_total = time.perf_counter()
        total_alignment_time = metrics_df["alignment_time"].sum() if "alignment_time" in metrics_df else None

        if total_alignment_time is not None:
            self.logger.info(
                f"✅ Post-processing finished: total={end_total - start_total:.2f}s | "
                f"reliability={end_reliability - start_reliability:.2f}s | "
                f"alignment={end_alignment - start_alignment:.2f}s | "
                f"alignment_total_time={total_alignment_time:.2f}s"
            )
        else:
            self.logger.info(
                f"✅ Post-processing finished: total={end_total - start_total:.2f}s | "
                f"reliability={end_reliability - start_reliability:.2f}s | "
                f"alignment={end_alignment - start_alignment:.2f}s"
            )
