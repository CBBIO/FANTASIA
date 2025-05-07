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

from protein_metamorphisms_is.tasks.base import BaseTaskInitializer

import numpy as np
import pandas as pd
from goatools.base import get_godag
from protein_metamorphisms_is.sql.model.entities.sequence.sequence import Sequence

from sqlalchemy import text
import h5py
from protein_metamorphisms_is.sql.model.entities.embedding.sequence_embedding import SequenceEmbeddingType, \
    SequenceEmbedding
from protein_metamorphisms_is.sql.model.entities.protein.protein import Protein

from fantasia.src.helpers.helpers import run_needle_from_strings


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
        Metric used to compute similarity between embeddings ("euclidean" or "cosine").
    types : dict
        Metadata and modules for each enabled embedding model.
    go : GODag
        Gene Ontology DAG loaded via goatools.
    """

    def __init__(self, conf: dict, current_date: str):
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
        import pprint

        # Dentro de __init__
        self.logger.info("📘 Parámetros de configuración cargados:")
        formatted_conf = pprint.pformat(self.conf, indent=2, width=120, compact=False)
        self.logger.info("\n" + formatted_conf)

        self.current_date = current_date
        self.logger.info("Initializing EmbeddingLookUp...")

        # Paths
        self.experiment_path = os.path.join(
            os.path.expanduser(self.conf["experiment"]["base_directory"]),
            f"{self.conf['experiment']['prefix']}_{current_date}"
        )
        if self.conf["input"].get("embeddings_path"):
            self.embeddings_path = self.conf["input"]["embeddings_path"]
        else:
            self.embeddings_path = os.path.join(self.experiment_path, "embeddings.h5")
        self.raw_results_path = os.path.join(self.experiment_path, "raw_results.csv")
        self.results_path = os.path.join(self.experiment_path, "results.csv")
        self.topgo_path = os.path.join(self.experiment_path, "results_topgo.tsv")

        # Limits and options
        self.limit_per_entry = self.conf["runtime"].get("limit_per_entry", 200)
        self.topgo_enabled = self.conf["output"].get("topgo", False)

        # Load model configurations
        self.fetch_models_info()

        # Optional redundancy filtering
        redundancy_filter_threshold = self.conf["runtime"].get("redundancy_filter", 0)
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
        """
        Executes the full lookup and annotation pipeline.

        This method performs the following steps:
        1. Preloads GO annotations from the database.
        2. Loads reference embeddings into memory.
        3. Reads query embeddings from the HDF5 file.
        4. Processes each embedding batch to compute distances and transfer GO terms.
        5. Stores the predicted annotations and triggers post-processing (e.g., metrics).

        Raises
        ------
        FileNotFoundError
            If the HDF5 input file containing embeddings is not found.
        Exception
            For any unexpected error during annotation or data processing.
        """
        self.logger.info("Starting embedding-based GO annotation process.")

        self.logger.info("Preloading GO annotations from the database.")
        self.preload_annotations()

        self.logger.info("Loading reference embeddings into memory.")
        self.lookup_table_into_memory()

        self.logger.info(f"Processing query embeddings from HDF5: {self.embeddings_path}")
        try:

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
                batch_size = self.conf["runtime"].get("batch_size", 4)

                for i in range(0, len(tasks), batch_size):
                    batch = tasks[i:i + batch_size]
                    annotations = self.process(batch)
                    self.store_entry(annotations)
                    total_batches += 1
                    self.logger.info(
                        f"Processed batch {total_batches} for model '{model_key}' with {len(batch)} entries."
                    )

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
        embedding module and stores metadata in the `self.types` dictionary.

        Raises
        ------
        Exception
            If the database query fails or if importing a model module fails.

        Notes
        -----
        - `self.types` stores per-model metadata: task name, ID, module, thresholds, and batch size.
        - ⚠ This method should ideally be moved to a shared base class to avoid code duplication.
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
        Reads sequence embeddings from an HDF5 file and enqueues batches of lookup tasks.

        This method reads each accession from the HDF5 file, retrieves all available embedding types
        associated with the sequence, and groups them into batches based on a configured batch size.
        Each batch is then published for downstream annotation processing.

        The structure of each task includes:
        - `accession`: Protein or chain identifier.
        - `sequence`: Amino acid sequence.
        - `embedding`: Embedding vector as a NumPy array.
        - `embedding_type_id`: Identifier for the embedding model.
        - `model_name`: Name of the embedding model.
        - `distance_threshold`: Threshold used for similarity filtering.

        Raises
        ------
        FileNotFoundError
            If the input HDF5 file with embeddings does not exist.
        Exception
            For any unexpected error during batching or task publishing.
        """
        try:
            self.logger.info(f"📂 Reading embeddings from HDF5: {self.embeddings_path}")

            if not os.path.exists(self.embeddings_path):
                raise FileNotFoundError(
                    f"The HDF5 file '{self.embeddings_path}' does not exist.\n"
                    "Make sure the embeddings have been generated or check the configured path."
                )

            batch_size = self.conf.get("batch_size", 4)
            batch = []
            total_batches = 0

            with h5py.File(self.embeddings_path, "r") as h5file:
                for accession, group in h5file.items():
                    if "sequence" not in group:
                        self.logger.warning(f"⚠️ Missing sequence for accession '{accession}'. Skipping.")
                        continue

                    sequence = group["sequence"][()].decode("utf-8")

                    for item_name, item_group in group.items():
                        if not item_name.startswith("type_") or "embedding" not in item_group:
                            continue

                        model_key = item_name.replace("type_", "")
                        if model_key not in self.types:
                            self.logger.warning(
                                f"⚠️ Model '{model_key}' not recognized for accession '{accession}'. Skipping."
                            )
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

                        if len(batch) == batch_size:
                            self.publish_task(batch)
                            total_batches += 1
                            self.logger.info(f"📦 Published batch {total_batches} with {batch_size} tasks.")
                            batch = []

            if batch:
                self.publish_task(batch)
                total_batches += 1
                self.logger.info(f"📦 Published final batch {total_batches} with {len(batch)} tasks.")

            self.logger.info(f"✅ Enqueued a total of {total_batches} batches for processing.")

        except OSError:
            self.logger.error(
                f"❌ Could not read HDF5 file: '{self.embeddings_path}'. "
                "The input file is required when running in lookup-only mode."
            )
            raise
        except Exception:
            import traceback
            self.logger.error(f"❌ Unexpected error during enqueue:\n{traceback.format_exc()}")
            raise

    def process(self, task_data: list[dict]) -> list[dict]:
        """
        Process a batch of query embeddings and return GO term annotation transfers.
        """
        import torch
        import numpy as np
        from scipy.spatial.distance import cdist

        # --- Paso 1: Preparación general ---
        task = task_data[0]
        model_id = task["embedding_type_id"]
        model_name = task["model_name"]
        threshold = task["distance_threshold"]
        use_gpu = self.conf["runtime"].get("use_gpu", True)
        limit = self.conf["runtime"].get("limit_per_entry", 1000)
        redundancy = self.conf.get("redundancy_filter", 0)

        # --- Paso 2: Lookup table ---
        lookup = self.lookup_tables.get(model_id)
        if lookup is None:
            self.logger.warning(f"No lookup table for embedding_type_id {model_id}. Skipping batch.")
            return []

        embeddings = np.stack([np.array(t["embedding"]) for t in task_data])
        accessions = [t["accession"].removeprefix("accession_") for t in task_data]
        sequences = {t["accession"].removeprefix("accession_"): t["sequence"] for t in task_data}

        # --- Paso 3: Mapeo secuencia→proteína (si no existe ya) ---
        if not hasattr(self, "sequence_to_protein"):
            from collections import defaultdict
            rows = self.session.execute(text("SELECT id, sequence_id FROM protein")).fetchall()
            self.sequence_to_protein = defaultdict(set)
            for row in rows:
                self.sequence_to_protein[row.sequence_id].add(str(row.id))

        # --- Paso 4: Cálculo de distancias ---
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

        # --- Paso 5: Cálculo de clusters redundantes ---
        redundant_ids = {}
        if redundancy > 0 and hasattr(self, "clusters_by_id"):
            for acc in accessions:
                redundant_ids[acc] = self.retrieve_cluster_members(acc)

        # --- Paso 6: Transferencia de anotaciones ---
        go_terms = []
        total_transfers = 0
        total_neighbors = 0
        lookup_seq_ids = np.array(lookup["ids"])

        for i, accession in enumerate(accessions):
            distances = dist_matrix[i]
            seq_ids = lookup_seq_ids

            # Aplicar filtro de redundancia si es necesario
            if redundancy > 0 and accession in redundant_ids:
                mask = np.array([
                    not any(
                        pid in redundant_ids[accession]
                        for pid in self.sequence_to_protein.get(sid, set())
                    )
                    for sid in seq_ids
                ])
                distances = distances[mask]
                seq_ids = seq_ids[mask]

            if len(distances) == 0:
                continue

            sorted_idx = np.argsort(distances)
            selected_idx = sorted_idx[distances[sorted_idx] <= threshold][:limit]
            total_neighbors += len(selected_idx)

            for idx in selected_idx:
                sid = seq_ids[idx]
                if sid not in self.go_annotations:
                    continue

                for ann in self.go_annotations[sid]:
                    total_transfers += 1
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

    def store_entry(self, annotations: list[dict]) -> None:
        """
        Store a batch of GO term annotation results into a raw CSV file.

        If the file already exists, new results are appended without overwriting previous content.
        The method assumes that all annotations in the batch share the same structure, including fields like:
        - 'accession': str, protein accession.
        - 'go_id': str, Gene Ontology term identifier.
        - 'sequence_query': str, query sequence.
        - 'sequence_reference': str, reference sequence.
        - 'distance': float, embedding similarity metric.
        - Other metadata fields (category, evidence_code, gene_name, etc.).

        Parameters
        ----------
        annotations : list of dict
            The GO annotation predictions to store.

        Raises
        ------
        Exception
            If an error occurs during DataFrame creation or CSV writing.
        """
        if not annotations:
            self.logger.info("No valid GO terms to store.")
            return

        try:
            df = pd.DataFrame(annotations)
            write_mode = "a" if os.path.exists(self.raw_results_path) else "w"
            include_header = write_mode == "w"

            df.to_csv(
                self.raw_results_path,
                mode=write_mode,
                index=False,
                header=include_header
            )
            self.logger.info(f"📥 Stored {len(df)} GO annotations to {self.raw_results_path}.")

        except Exception as e:
            self.logger.error(f"❌ Error writing raw results: {e}")
            raise

    def generate_clusters(self):
        """
        Generates non-redundant sequence clusters using MMseqs2.

        This method creates a temporary FASTA file combining sequences from:
        - Proteins in the database
        - Query embeddings from the HDF5 file

        Then, it runs MMseqs2 to cluster these sequences based on the configured identity
        and coverage thresholds. The resulting clusters are stored in memory in two mappings:
        - `self.clusters_by_id`: maps each sequence ID to its cluster.
        - `self.clusters_by_cluster`: maps each cluster to the set of sequence IDs it contains.

        Configuration requirements
        --------------------------
        - `redundancy_filter`: float (e.g., 0.9)
        - `alignment_coverage`: int
        - `threads`: int

        Raises
        ------
        Exception
            If MMseqs2 execution fails or required files cannot be created.
        """
        import tempfile
        import subprocess

        try:
            identity = self.conf["filters"].get("redundancy_filter", 0)
            coverage = self.conf["filters"].get("alignment_coverage", 0)
            threads = self.conf["runtime"].get("threads", 12)

            with tempfile.TemporaryDirectory() as tmpdir:
                fasta_path = os.path.join(tmpdir, "redundancy.fasta")
                db_path = os.path.join(tmpdir, "seqDB")
                clu_path = os.path.join(tmpdir, "mmseqs_clu")
                tmp_path = os.path.join(tmpdir, "mmseqs_tmp")
                tsv_path = os.path.join(tmpdir, "clusters.tsv")

                self.logger.info("🔬 Generating input FASTA file for MMseqs2 clustering...")
                with open(fasta_path, "w") as fasta:
                    with self.engine.connect() as conn:
                        rows = conn.execute(text("SELECT id, sequence_id FROM protein")).fetchall()
                        for row in rows:
                            seq = conn.execute(
                                text("SELECT sequence FROM sequence WHERE id = :sid"),
                                {"sid": row.sequence_id}
                            ).scalar()
                            if seq:
                                fasta.write(f">{row.id}\n{seq}\n")

                    with h5py.File(self.embeddings_path, "r") as h5file:
                        for accession, group in h5file.items():
                            if "sequence" in group:
                                seq = group["sequence"][()].decode("utf-8")
                                fasta.write(f">{accession.removeprefix('accession_')}\n{seq}\n")

                self.logger.info(f"🚀 Running MMseqs2 (identity={identity}, coverage={coverage}, threads={threads})...")
                subprocess.run(["mmseqs", "createdb", fasta_path, db_path], check=True)
                subprocess.run([
                    "mmseqs", "linclust", db_path, clu_path, tmp_path,
                    "--min-seq-id", str(identity),
                    "--cov-mode", "1", "-c", str(coverage),
                    "--threads", str(threads)
                ], check=True)
                subprocess.run(["mmseqs", "createtsv", db_path, db_path, clu_path, tsv_path], check=True)

                # Read the TSV and build cluster maps
                import pandas as pd
                df = pd.read_csv(tsv_path, sep="\t", names=["cluster", "identifier"])
                self.clusters_by_id = df.set_index("identifier")
                self.clusters_by_cluster = df.groupby("cluster")["identifier"].apply(set).to_dict()

                self.logger.info(f"✅ Loaded {len(self.clusters_by_cluster)} clusters from MMseqs2.")

        except Exception as e:
            self.logger.error(f"❌ Error while generating MMseqs2 clusters: {e}")
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
        >>> print("Cluster size:", len(members))
        """
        try:
            cluster_id = self.clusters_by_id.loc[accession, "cluster"]
            members = self.clusters_by_cluster.get(cluster_id, set())
            return {m for m in members}
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
            limit_execution = 100
            get_descendants = self.conf["runtime"].get("get_descendants", False)

            exclude_taxon_ids = self.conf.get("filters", {}).get("taxonomy_ids_to_exclude", [])
            include_taxon_ids = self.conf.get("filters", {}).get("taxonomy_ids_included_exclusively", [])

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
        """
        Final processing step after annotation to compute alignment metrics, filter redundant GO terms,
        collapse parent-child relations in GO, and write final output files.

        This method:
        - Computes reliability index (RI) based on the distance metric.
        - Aligns query/reference sequences using pairwise global alignment (Needleman-Wunsch).
        - Collapses ancestor GO terms when more specific child terms exist.
        - Aggregates support count and saves final results to CSV.
        - Optionally generates a TopGO-compatible TSV file.

        Notes
        -----
        - Distance → Reliability Index (RI):
            - Cosine: RI = 1 - distance
            - Euclidean: RI = 0.5 / (0.5 + distance)
        - Pairwise alignments are parallelized using `ProcessPoolExecutor`.
        - Collapsed GO terms are listed in additional columns with support counts.

        Raises
        ------
        Exception
            If errors occur during alignment computation or file writing.
        """
