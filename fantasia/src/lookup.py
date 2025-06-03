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
import traceback
from concurrent.futures import ProcessPoolExecutor

from protein_information_system.tasks.base import BaseTaskInitializer

import numpy as np
import pandas as pd
from goatools.base import get_godag
from protein_information_system.sql.model.entities.sequence.sequence import Sequence

from sqlalchemy import text
import h5py
from protein_information_system.sql.model.entities.embedding.sequence_embedding import SequenceEmbeddingType, \
    SequenceEmbedding
from protein_information_system.sql.model.entities.protein.protein import Protein

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
                base_module_path = "protein_information_system.operation.embedding.proccess.sequence"
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

        go_annotations = self.go_annotations
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
                if seq_id not in go_annotations:
                    continue

                annotations = go_annotations[seq_id]
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
        Generates non-redundant sequence clusters using MMseqs2.

        This method replaces the CD-HIT clustering with MMseqs2, using only the
        sequence IDs from the database and the HDF5 file as identifiers.
        """
        import tempfile
        import subprocess

        try:
            identity = self.conf.get("redundancy_filter", 0)
            coverage = self.conf.get("alignment_coverage", 0)
            threads = self.conf.get("threads", 12)

            with tempfile.TemporaryDirectory() as tmpdir:
                fasta_path = os.path.join(tmpdir, "redundancy.fasta")
                db_path = os.path.join(tmpdir, "seqDB")
                clu_path = os.path.join(tmpdir, "mmseqs_clu")
                tmp_path = os.path.join(tmpdir, "mmseqs_tmp")
                tsv_path = os.path.join(tmpdir, "clusters.tsv")

                self.logger.info("📄 Generating FASTA for MMseqs2 clustering...")
                with open(fasta_path, "w") as fasta:
                    # DB sequences
                    with self.engine.connect() as conn:
                        seqs = conn.execute(text("SELECT id, sequence FROM sequence")).fetchall()
                        for seq_id, seq in seqs:
                            fasta.write(f">{seq_id}\n{seq}\n")
                    # HDF5 sequences
                    with h5py.File(self.embeddings_path, "r") as h5file:
                        for accession, group in h5file.items():
                            if "sequence" in group:
                                sequence = group["sequence"][()].decode("utf-8")
                                clean_id = accession.removeprefix("accession_")
                                fasta.write(f">{clean_id}\n{sequence}\n")

                self.logger.info(f"⚙️ Running MMseqs2 (id={identity}, cov={coverage}, threads={threads})...")
                subprocess.run(["mmseqs", "createdb", fasta_path, db_path], check=True)
                subprocess.run([
                    "mmseqs", "cluster", db_path, clu_path, tmp_path,
                    "--min-seq-id", str(identity),
                    "--cov-mode", "1", "-c", str(coverage),
                    "--threads", str(threads)
                ], check=True)
                subprocess.run(["mmseqs", "createtsv", db_path, db_path, clu_path, tsv_path], check=True)

                # Cargar resultados
                import pandas as pd
                df = pd.read_csv(tsv_path, sep="\t", names=["cluster", "identifier"])
                self.clusters = df
                self.clusters_by_id = df.set_index("identifier")
                self.clusters_by_cluster = df.groupby("cluster")["identifier"].apply(set).to_dict()

                self.logger.info(f"✅ {len(self.clusters_by_cluster)} clusters loaded from MMseqs2.")

        except Exception as e:
            self.logger.error(f"❌ Error running MMseqs2 clustering: {e}")
            raise

    def retrieve_cluster_members(self, accession: str) -> set:
        """
        Retrieve the set of cluster members associated with a given accession.

        This method looks up the cluster ID corresponding to the provided accession
        and returns the set of all sequence identifiers that belong to the same cluster.

        Parameters
        ----------
        accession : str
            Accession string corresponding to a protein or sequence included in clustering.

        Returns
        -------
        set
            Set of accession strings belonging to the same cluster as the input accession.
            Returns an empty set if the accession is not found in any cluster.

        Logs
        ----
        - A warning if the accession is not found in the clustering result.

        Examples
        --------
        >>> members = lookup.retrieve_cluster_members("P12345")
        """
        try:
            cluster_id = self.clusters_by_id.loc[accession, "cluster"]
            members = self.clusters_by_cluster.get(cluster_id, set())
            return {m for m in members if m.isdigit()}
        except KeyError:
            self.logger.warning(f"Accession '{accession}' not found in clusters.")
            return set()

    def lookup_table_into_memory(self):
        """
        Loads sequence embeddings from the database into memory for each enabled embedding model.

        This method constructs a lookup table per model by retrieving embeddings from the database.
        It applies optional filtering by taxonomy (inclusion or exclusion lists), with support
        for hierarchical filtering (i.e., inclusion of descendant taxa via the NCBI taxonomy tree).
        """
        try:
            self.logger.info("🔄 Starting lookup table construction: loading embeddings into memory per model...")

            self.lookup_tables = {}
            limit_execution = self.conf.get("limit_execution")
            get_descendants = self.conf.get("get_descendants", False)

            def expand_tax_ids(key):
                ids = self.conf.get(key, [])
                if not isinstance(ids, list):
                    self.logger.warning(f"Expected list for '{key}', got {type(ids)}. Forcing empty list.")
                    return []

                clean_ids = [int(tid) for tid in ids if str(tid).isdigit()]

                if get_descendants and clean_ids:
                    expanded = get_descendant_ids(clean_ids)  # devuelve ints
                    return [str(tid) for tid in expanded]

                return [str(tid) for tid in clean_ids]

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
        """
        Post-processes raw GO annotation results after embedding-based lookup.

        This method computes reliability indices, filters redundant GO terms using
        ontology hierarchy, deduplicates identical GO annotations by keeping only
        the one with the highest reliability index, and computes sequence alignment
        metrics only for the final annotations.

        It also calculates collapsed term support and outputs clean results in
        CSV and optional TopGO-compatible formats.

        Notes
        -----
        - The column `support_count` indicates how many neighbors contributed
          the same GO term to a given entry (accession, model_name, go_id).
        - Collapsed GO terms are recorded in `collapsed_terms` and quantified
          in `collapsed_support` and `n_collapsed_terms`.
        - Alignment metrics are computed only after deduplication.
        """
        if not os.path.exists(self.raw_results_path):
            self.logger.warning("No raw results found for post-processing.")
            return

        self.logger.info("🔍 Starting post-processing of raw GO annotations.")
        start_total = time.perf_counter()

        df = pd.read_csv(self.raw_results_path)

        # Compute reliability index based on the selected distance metric
        start_reliability = time.perf_counter()
        if self.distance_metric == "cosine":
            df["reliability_index"] = 1 - df["distance"]
        elif self.distance_metric == "euclidean":
            df["reliability_index"] = 0.5 / (0.5 + df["distance"])
        end_reliability = time.perf_counter()

        # Cache for GO ancestry lookups
        ancestor_cache = {}

        def is_ancestor(go_dag, parent, child):
            key = (parent, child)
            if key in ancestor_cache:
                return ancestor_cache[key]
            result = child in go_dag and parent in go_dag[child].get_all_parents()
            ancestor_cache[key] = result
            return result

        # Count how many times each GO term appears per accession/model
        df["support_count"] = df.groupby(["accession", "model_name", "go_id"])["go_id"].transform("count")

        rows = []
        for (_, _), group in df.groupby(["accession", "model_name"]):
            all_go_ids = group["go_id"].unique().tolist()
            support_map = group.drop_duplicates(subset=["go_id"])[["go_id", "support_count"]].set_index("go_id")[
                "support_count"].to_dict()

            leaf_terms = []
            collapsed_terms = {}

            # Identify leaf terms and collapse ancestors
            for go_id in all_go_ids:
                is_leaf = not any(
                    go_id != other and is_ancestor(self.go, go_id, other)
                    for other in all_go_ids
                )
                if is_leaf:
                    leaf_terms.append(go_id)
                else:
                    for lt in all_go_ids:
                        if go_id != lt and is_ancestor(self.go, go_id, lt):
                            collapsed_terms.setdefault(lt, {"collapsed_support": 0, "terms": set()})
                            collapsed_terms[lt]["collapsed_support"] += support_map.get(go_id, 1)
                            collapsed_terms[lt]["terms"].add(go_id)

            # For each leaf term, keep only the annotation with highest reliability
            for go_id in leaf_terms:
                subset = group[group["go_id"] == go_id].copy()
                info = collapsed_terms.get(go_id, {})
                subset["collapsed_support"] = info.get("collapsed_support", 0)
                subset["n_collapsed_terms"] = len(info.get("terms", []))
                subset["collapsed_terms"] = ", ".join(sorted(info.get("terms", []))) if info.get("terms") else ""
                rows.append(subset.sort_values("reliability_index", ascending=False).iloc[0])

        df = pd.DataFrame(rows)

        # Compute alignment metrics for unique query-reference pairs after filtering
        start_alignment = time.perf_counter()
        unique_pairs = df[["sequence_query", "sequence_reference"]].drop_duplicates()
        with ProcessPoolExecutor(max_workers=self.conf.get("store_workers", 4)) as executor:
            metrics_list = list(executor.map(compute_metrics, unique_pairs.to_dict("records")))
        metrics_df = pd.DataFrame(metrics_list)
        df = df.merge(metrics_df, on=["sequence_query", "sequence_reference"], how="left")
        end_alignment = time.perf_counter()

        # Drop raw sequences to reduce output size
        df = df.drop(columns=["sequence_query", "sequence_reference"], errors="ignore")

        df = df.sort_values(by=["accession", "go_id", "model_name", "reliability_index"],
                            ascending=[True, True, True, False])

        # Round numeric columns for cleaner output
        columns_to_round = ["distance", "identity", "similarity", "alignment_score", "gaps_percentage",
                            "reliability_index"]
        for col in columns_to_round:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: round(x, 4))

        def extract_gene_name(g):
            try:
                val = eval(g)
                if (
                        isinstance(g, str)
                        and g.startswith("[{")
                        and isinstance(val, list)
                        and len(val) > 0
                        and "Name" in val[0]
                ):
                    return val[0]["Name"]
            except Exception:
                return None
            return None

        if "gene_name" in df.columns:
            df["gene_name"] = df["gene_name"].apply(extract_gene_name)

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
