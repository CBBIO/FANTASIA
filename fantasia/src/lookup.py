"""
EmbeddingLookUp Module
=======================

This module defines the `EmbeddingLookUp` class, which enables functional annotation of proteins
based on embedding similarity.

Given a set of query embeddings stored in HDF5 format, the class computes distances to reference
embeddings stored in a database, retrieves associated GO term annotations, and stores the results
in standard formats (CSV and optionally TopGO-compatible TSV). It also supports redundancy filtering
via MMseqs2 and flexible integration with custom embedding models.

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

import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor

from protein_information_system.tasks.base import BaseTaskInitializer

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist

from goatools.base import get_godag
from protein_information_system.sql.model.entities.sequence.sequence import Sequence

from sqlalchemy import text
import h5py
from protein_information_system.sql.model.entities.embedding.sequence_embedding import SequenceEmbeddingType, \
    SequenceEmbedding
from protein_information_system.sql.model.entities.protein.protein import Protein

from fantasia.src.helpers.helpers import get_descendant_ids, compute_metrics


class EmbeddingLookUp(BaseTaskInitializer):
    """
    EmbeddingLookUp performs GO annotation transfer using embedding similarity.

    Given a set of sequence embeddings in HDF5 format, it compares them against reference embeddings
    stored in a database, retrieves GO annotations from similar sequences, and stores the results in
    CSV and optionally TopGO format. The process includes:

    - configurable filtering by taxonomy,
    - redundancy-aware neighbor selection via MMseqs2 clusters,
    - support for multiple embedding models with per-model distance thresholds,
    - distance computation using GPU (via PyTorch) or CPU,
    - optional sequence alignment postprocessing to compute identity/similarity.

    Parameters
    ----------
    conf : dict
        Configuration dictionary defining paths, thresholds, model settings, and processing options.
    current_date : str
        Timestamp used to version output files.

    Notes
    -----
    - Supports cosine or euclidean distance metrics.
    - Redundancy filtering uses MMseqs2 clustering based on sequence identity and coverage.
    - GO annotations are preloaded from the relational database and filtered by taxonomy if requested.
    """

    def __init__(self, conf, current_date):
        """
        Prepares internal configuration, output paths, GO DAG, and optional MMseqs2 clustering.
        """

        super().__init__(conf)

        self.types = None

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
        """
        Main execution method for the GO annotation pipeline (layer-aware).

        Soporta:
          - NUEVO: /accession_*/type_*/layer_*/embedding (+ attr 'shape' en layer_*)
          - ANTIGUO: /accession_*/type_*/embedding
        """
        self.logger.info("Starting embedding-based GO annotation process.")

        self.load_model_definitions()
        self.logger.info("Loading reference embeddings into memory.")
        self.lookup_table_into_memory()
        self.logger.info("Preloading GO annotations from the database.")
        self.preload_annotations()

        self.logger.info(f"Processing query embeddings from HDF5: {self.embeddings_path}")
        try:
            batch_size = self.conf.get("batch_size", 1)
            batches_by_model = {}
            total_batches = 0

            if not os.path.exists(self.embeddings_path):
                raise FileNotFoundError(
                    f"HDF5 file not found: {self.embeddings_path}. "
                    f"Ensure embeddings have been generated prior to annotation."
                )

            import h5py
            with h5py.File(self.embeddings_path, "r") as h5file:
                for accession, group in h5file.items():
                    if "sequence" not in group:
                        self.logger.warning(f"Sequence missing for accession '{accession}'. Skipping.")
                        continue

                    sequence = group["sequence"][()].decode("utf-8")

                    # Recorremos los tipos disponibles
                    for type_key, type_grp in group.items():
                        if not type_key.startswith("type_"):
                            continue

                        try:
                            model_id = int(type_key.replace("type_", ""))
                        except Exception:
                            self.logger.warning(f"Malformed type group '{type_key}'. Skipping.")
                            continue

                        # Resolver modelo por ID (cargado en load_model_definitions)
                        model_info = next((info for info in self.types.values() if info["id"] == model_id), None)
                        if model_info is None:
                            self.logger.warning(f"No model config found for embedding type ID {model_id}, skipping.")
                            continue
                        model_key = model_info["task_name"]

                        # --- NUEVO: leer por capas si existen ---
                        layer_keys = [k for k in type_grp.keys() if k.startswith("layer_")]

                        if layer_keys:
                            for lk in sorted(layer_keys, key=lambda x: int(x.split("_", 1)[1])):
                                layer_grp = type_grp[lk]
                                if "embedding" not in layer_grp:
                                    continue
                                try:
                                    layer_index = int(lk.split("_", 1)[1])
                                except Exception:
                                    self.logger.warning(f"Malformed layer group '{lk}' under {type_key}. Skipping.")
                                    continue

                                embedding = layer_grp["embedding"][:]
                                task_data = {
                                    "accession": accession,
                                    "sequence": sequence,
                                    "embedding": embedding,
                                    "embedding_type_id": model_info["id"],
                                    "model_name": model_key,
                                    "distance_threshold": model_info["distance_threshold"],
                                    "layer_index": layer_index,  # ← trazabilidad de la capa del QUERY
                                }
                                batches_by_model.setdefault(model_key, []).append(task_data)
                        else:
                            # Compatibilidad con formato antiguo (sin capas)
                            if "embedding" not in type_grp:
                                continue
                            embedding = type_grp["embedding"][:]
                            task_data = {
                                "accession": accession,
                                "sequence": sequence,
                                "embedding": embedding,
                                "embedding_type_id": model_info["id"],
                                "model_name": model_key,
                                "distance_threshold": model_info["distance_threshold"],
                                # sin layer_index en formato antiguo
                            }
                            batches_by_model.setdefault(model_key, []).append(task_data)

            # Procesar por modelo en lotes
            for model_key, tasks in batches_by_model.items():
                for i in range(0, len(tasks), batch_size):
                    batch = tasks[i:i + batch_size]
                    annotations = self.process(batch)  # process ya es layer-aware
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

    def load_model_definitions(self):
        """
        Initializes `self.types` by matching embedding types from the database with
        those defined in the configuration.

        Only models present in both sources and marked as `enabled` are included.
        Each entry contains model ID, model name, task name, distance threshold, and batch size.

        Logs warnings for models missing in config or explicitly disabled.
        """

        self.types = {}

        try:
            db_models = self.session.query(SequenceEmbeddingType).all()
        except Exception as e:
            self.logger.error(f"Failed to query SequenceEmbeddingType table: {e}")
            raise

        config_models = self.conf.get("embedding", {}).get("models", {})

        for db_model in db_models:
            task_name = db_model.name  # usamos el 'name' (no 'task_name') de la BD
            config_models = self.conf.get("embedding", {}).get("models", {})

            # Si el modelo está definido en la config (por nombre), lo usamos
            matched_name = next((k for k in config_models if k.lower() == task_name.lower()), None)
            if matched_name is None:
                self.logger.warning(f"Model '{task_name}' exists in DB but not in config — skipping.")
                continue

            config = config_models[matched_name]
            if not config.get("enabled", True):
                self.logger.info(f"Model '{matched_name}' is disabled in config — skipping.")
                continue

            self.types[matched_name] = {
                "id": db_model.id,
                "model_name": db_model.model_name,
                "task_name": matched_name,
                "distance_threshold": config.get("distance_threshold"),
                "batch_size": config.get("batch_size"),
            }

        self.logger.info(f"Loaded {len(self.types)} model(s) from DB + config: {list(self.types.keys())}")

    def process(self, task_data):
        """
        Processes a batch of query embeddings (layer-aware).

        Cada entrada de task_data puede incluir 'layer_index' (int). Este método
        conserva ese índice y lo añade a cada transferencia GO resultante.
        """
        try:
            if not task_data:
                self.logger.warning("No task data provided for lookup. Skipping batch.")
                return []

            # --- Metadatos del batch (se asume mismo embedding_type_id) ---
            model_id = task_data[0]["embedding_type_id"]
            model_name = task_data[0]["model_name"]
            threshold = task_data[0].get("distance_threshold", self.conf.get("distance_threshold"))
            use_gpu = self.conf.get("use_gpu", True)
            limit = int(self.conf.get("limit_per_entry", 1000))

            # Tabla de referencia para este tipo de embedding
            lookup = self.lookup_tables.get(model_id)
            if lookup is None:
                self.logger.warning(f"No lookup table for embedding_type_id {model_id}. Skipping batch.")
                return []

            # --- Preparación de consultas ---
            embeddings = np.stack([np.asarray(t["embedding"]) for t in task_data])
            accessions = [t["accession"].removeprefix("accession_") for t in task_data]
            sequences = {t["accession"].removeprefix("accession_"): t["sequence"] for t in task_data}
            layer_indices = [t.get("layer_index") for t in task_data]  # puede contener None si formato antiguo

            # --- Distancias (GPU/CPU) ---
            if use_gpu:
                queries = torch.tensor(embeddings, dtype=torch.float32).cuda()
                targets = torch.tensor(lookup["embeddings"], dtype=torch.float32).cuda()

                if self.distance_metric == "euclidean":
                    q2 = (queries ** 2).sum(dim=1, keepdim=True)
                    t2 = (targets ** 2).sum(dim=1).unsqueeze(0)
                    d2 = q2 + t2 - 2 * (queries @ targets.T)
                    dist_matrix = torch.sqrt(torch.clamp(d2, min=0.0)).cpu().numpy()

                elif self.distance_metric == "cosine":
                    qn = torch.nn.functional.normalize(queries, p=2, dim=1)
                    tn = torch.nn.functional.normalize(targets, p=2, dim=1)
                    dist_matrix = (1 - (qn @ tn.T)).cpu().numpy()

                else:
                    raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
            else:
                dist_matrix = cdist(embeddings, lookup["embeddings"], metric=self.distance_metric)

            # --- Redundancia opcional ---
            redundancy = int(self.conf.get("redundancy_filter", 0))
            redundant_ids = {}
            if redundancy > 0:
                for acc in accessions:
                    redundant_ids[acc] = self.retrieve_cluster_members(acc)

            go_annotations = self.go_annotations
            go_terms = []
            total_transfers = 0
            total_neighbors = 0

            # --- Selección de vecinos y transferencia ---
            ids_ref = lookup["ids"]
            for i, accession in enumerate(accessions):
                distances_all = dist_matrix[i]
                ids_all = ids_ref

                if redundancy > 0 and accession in redundant_ids:
                    mask = ~np.isin(ids_all.astype(str), list(redundant_ids[accession]))
                    distances = distances_all[mask]
                    seq_ids = ids_all[mask]
                else:
                    distances = distances_all
                    seq_ids = ids_all

                if distances.size == 0:
                    continue

                order = np.argsort(distances)
                if not threshold:
                    selected = order[:limit]
                else:
                    selected = order[distances[order] <= float(threshold)][:limit]

                total_neighbors += len(selected)
                li = layer_indices[i]  # puede ser None si viene del formato antiguo

                for idx in selected:
                    ref_id = seq_ids[idx]
                    anns = go_annotations.get(ref_id)
                    if not anns:
                        continue

                    d = float(distances[idx])
                    for ann in anns:
                        go_terms.append({
                            "accession": accession,
                            "sequence_query": sequences[accession],
                            "sequence_reference": ann["sequence"],
                            "go_id": ann["go_id"],
                            "category": ann["category"],
                            "evidence_code": ann["evidence_code"],
                            "go_description": ann["go_description"],
                            "distance": d,
                            "model_name": model_name,
                            "embedding_type_id": model_id,
                            "layer_index": li,  # ← trazabilidad de la capa del QUERY
                            "protein_id": ann["protein_id"],
                            "organism": ann["organism"],
                            "gene_name": ann["gene_name"],
                        })
                        total_transfers += 1

            self.logger.info(
                f"✅ Batch processed ({len(accessions)} queries; layers: "
                f"{sorted({li for li in layer_indices if li is not None}) or ['legacy']}). "
                f"{total_neighbors} neighbors, {total_transfers} GO transfers."
            )
            return go_terms

        except Exception as e:
            self.logger.error(f"Error during lookup process: {e}\n{traceback.format_exc()}")
            raise

    def store_entry(self, annotations):
        """
        Guarda resultados RAW *compactados* para reducir tamaño y acelerar el post-procesado.

        - Colapsa por (accession, model_name, embedding_type_id, layer_index, go_id, ...).
        - 'support_count' queda ya calculado (nº de vecinos que respaldan el GO en el batch).
        - Conserva el vecino de menor 'distance' como representante (para alineamiento, etc.).
        - Si hay 'layer_index', escribe por capa en: raw_results_layer_{L}.csv
          (mejor IO y lectura selectiva después). Si no, usa raw_results.csv (legacy).

        Formato guardado (cabecera base):
          accession,model_name,embedding_type_id,layer_index,go_id,category,evidence_code,
          go_description,distance,sequence_query,sequence_reference,protein_id,organism,
          gene_name,support_count
        """
        if not annotations:
            self.logger.info("No valid GO terms to store.")
            return

        try:
            df = pd.DataFrame(annotations)
            # Asegura 'layer_index' por compatibilidad
            if "layer_index" not in df.columns:
                df["layer_index"] = None

            # --- COMPACTA antes de guardar ---
            df_compact = self._compact_annotations_df(df)

            # ¿Escritura por capa?
            write_layered = True  # activa escritura por capa por defecto
            if write_layered and "layer_index" in df_compact.columns and df_compact["layer_index"].notna().any():
                for layer_val, chunk in df_compact.groupby("layer_index", dropna=False):
                    if pd.isna(layer_val):
                        # legacy (sin capa): manda a raw_results.csv
                        out_path = self.raw_results_path
                    else:
                        out_path = os.path.join(self.experiment_path, f"raw_results_layer_{int(layer_val)}.csv")

                    write_header = not os.path.exists(out_path)
                    chunk.to_csv(out_path, mode="a", index=False, header=write_header)
                    self.logger.info(f"Stored {len(chunk)} compact rows → {os.path.basename(out_path)}")
            else:
                out_path = self.raw_results_path
                write_header = not os.path.exists(out_path)
                df_compact.to_csv(out_path, mode="a", index=False, header=write_header)
                self.logger.info(f"Stored {len(df_compact)} compact rows → {os.path.basename(out_path)}")

        except Exception as e:
            self.logger.error(f"Error writing compact raw results: {e}")
            raise

    def generate_clusters(self):
        """
        Generates non-redundant sequence clusters using MMseqs2.

        Combines protein sequences from the database and the HDF5 file into a temporary FASTA file,
        then runs MMseqs2 clustering based on identity and coverage thresholds. The resulting cluster
        assignments are stored in the following attributes:

        - `self.clusters`: raw cluster assignment as a DataFrame.
        - `self.clusters_by_id`: mapping from sequence ID to cluster ID.
        - `self.clusters_by_cluster`: mapping from cluster ID to set of sequence IDs.

        Configuration parameters:
        - `redundancy_filter` → identity threshold.
        - `alignment_coverage` → coverage threshold.
        - `threads` → number of threads for MMseqs2.

        Raises
        ------
        Exception
            If MMseqs2 fails or any step in the clustering pipeline encounters an error.
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
        Returns the set of sequence IDs that belong to the same MMseqs2 cluster as the given sequence.

        Parameters
        ----------
        accession : str
            Sequence ID used in the clustering (must match identifier used in FASTA header).

        Returns
        -------
        set of str
            Set of sequence IDs in the same cluster. Returns an empty set if not found.
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
        Loads sequence embeddings into memory to build lookup tables for each enabled model.

        Embeddings are retrieved from the database and filtered by optional taxonomy inclusion
        or exclusion lists. The result is stored in `self.lookup_tables`, keyed by model ID.

        Supports hierarchical filtering by NCBI taxonomy if `get_descendants` is enabled.

        Configuration parameters:
        - `taxonomy_ids_to_exclude`: list of taxonomy IDs to exclude.
        - `taxonomy_ids_included_exclusively`: list of taxonomy IDs to include.
        - `limit_execution`: optional SQL limit.
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
            self.exclude_taxon_ids = [str(tid) for tid in exclude_taxon_ids or []]
            self.include_taxon_ids = [str(tid) for tid in include_taxon_ids or []]

            if self.exclude_taxon_ids and self.include_taxon_ids:
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
        """
        Preloads GO annotations from the database and stores them in `self.go_annotations`.

        Annotations are grouped by sequence ID and filtered using `self.exclude_taxon_ids`.
        Each annotation includes GO ID, evidence code, category, and description.
        """

        sql = text("""
                   SELECT s.id           AS sequence_id,
                          s.sequence,
                          pgo.go_id,
                          gt.category,
                          gt.description AS go_term_description,
                          pgo.evidence_code,
                          p.id           AS protein_id,
                          p.organism,
                          p.taxonomy_id,
                          p.gene_name
                   FROM sequence s
                            JOIN protein p ON s.id = p.sequence_id
                            JOIN protein_go_term_annotation pgo ON p.id = pgo.protein_id
                            JOIN go_terms gt ON pgo.go_id = gt.go_id
                   """)
        self.go_annotations = {}

        with self.engine.connect() as connection:
            for row in connection.execute(sql):
                if row.taxonomy_id not in self.exclude_taxon_ids:
                    entry = {
                        "sequence": row.sequence,
                        "go_id": row.go_id,
                        "category": row.category,
                        "evidence_code": row.evidence_code,
                        "go_description": row.go_term_description,
                        "protein_id": row.protein_id,
                        "organism": row.organism,
                        "taxonomy_id": row.taxonomy_id,
                        "gene_name": row.gene_name,
                    }
                    self.go_annotations.setdefault(row.sequence_id, []).append(entry)

    # --- Helpers de metadatos ---------------------------------------------
    def _model_threshold_map(self) -> dict:
        """
        Devuelve un mapa {model_name -> distance_threshold} a partir de self.types.
        'model_name' aquí es el 'task_name' usado en los outputs.
        """
        try:
            return {info["task_name"]: info.get("distance_threshold") for info in self.types.values()}
        except Exception:
            return {}

    def _add_metadata_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Añade metadatos a las salidas:
          - distance_metric (constante de la ejecución)
          - distance_threshold (por fila, según model_name)
          - (opcional) elimina sequence_query/sequence_reference si keep_sequences=False
        """
        df = df.copy()
        df["distance_metric"] = self.distance_metric

        thr_map = self._model_threshold_map()
        df["distance_threshold"] = df["model_name"].map(thr_map).astype(object)

        keep_seq = (self.conf.get("postprocess", {}) or {}).get("keep_sequences", False)
        if not keep_seq:
            df = df.drop(columns=["sequence_query", "sequence_reference"], errors="ignore")

        return df

    # --- Normalización y utilidades ----------------------------------------
    def _safe_max(self, s: pd.Series) -> float:
        """Devuelve el máximo positivo o NaN si no existe."""
        if s is None or s.empty:
            return float("nan")
        m = pd.to_numeric(s, errors="coerce").max()
        return m if pd.notnull(m) and m > 0 else float("nan")

    def _normalize_by_accession(self, df: pd.DataFrame, col: str) -> pd.Series:
        """Normaliza una columna por accesión dividiendo por su máximo positivo."""

        def norm(group: pd.Series) -> pd.Series:
            m = self._safe_max(group)
            if not pd.notnull(m) or m == 0:
                return pd.Series(0.0, index=group.index)
            return group.fillna(0.0) / m

        return df.groupby("accession")[col].transform(norm)

    # --- SCORE (con layer_support) -----------------------------------------
    def _compute_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula 'score' compuesto por fila.

        Componentes:
          - reliability_index: derivado de distance (según el métrico)
          - support_count_norm: recuento de vecinos por término (normalizado por accesión)
          - collapsed_support_norm: apoyo ontológico de ancestros (normalizado por accesión)
          - model_consistency: acuerdo entre modelos (model_support / n_models_total)
          - alignment_norm: media de identidad y similitud en [0,1]
          - layer_support_norm: fracción de capas del mismo modelo que apoyan el término
                                (#capas que predicen el GO / #capas totales del modelo para esa accesión)

        Pesos (override en conf['postprocess']['weights']):
          reliability_index=0.50, support_count_norm=0.20, collapsed_support_norm=0.15,
          model_consistency=0.10, alignment_norm=0.05, layer_support_norm=0.05
        """
        df = df.copy()

        # Consistencia entre modelos
        model_counts = (
            df.groupby(["accession", "go_id"])["model_name"]
            .nunique().rename("model_support").reset_index()
        )
        total_models = (
            df.groupby("accession")["model_name"]
            .nunique().rename("n_models_total").reset_index()
        )
        df = df.merge(model_counts, on=["accession", "go_id"], how="left") \
            .merge(total_models, on="accession", how="left")
        df["model_consistency"] = (df["model_support"] / df["n_models_total"]).fillna(0.0)

        # NUEVO: layer_support (por modelo)
        n_layers_model = (
            df.groupby(["accession", "model_name"])["layer_index"]
            .nunique().rename("n_layers_model").reset_index()
        )
        layer_support = (
            df.groupby(["accession", "go_id", "model_name"])["layer_index"]
            .nunique().rename("layer_support").reset_index()
        )
        df = df.merge(layer_support, on=["accession", "go_id", "model_name"], how="left") \
            .merge(n_layers_model, on=["accession", "model_name"], how="left")
        df["layer_support_norm"] = (df["layer_support"] / df["n_layers_model"]).fillna(0.0)

        # Normalizaciones por accesión
        for c in ("support_count", "collapsed_support"):
            if c not in df.columns:
                df[c] = 0.0
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c + "_norm"] = self._normalize_by_accession(df, c)

        # Alineamiento
        for mcol in ("identity", "similarity"):
            if mcol not in df.columns:
                df[mcol] = 0.0
            df[mcol] = pd.to_numeric(df[mcol], errors="coerce").fillna(0.0)
        df["alignment_norm"] = 0.5 * (df["identity"] / 100.0) + 0.5 * (df["similarity"] / 100.0)

        # Pesos
        wconf = (self.conf.get("postprocess", {}) or {}).get("weights", {})
        w_RI = float(wconf.get("reliability_index", 0.50))
        w_SC = float(wconf.get("support_count_norm", 0.20))
        w_CS = float(wconf.get("collapsed_support_norm", 0.15))
        w_MC = float(wconf.get("model_consistency", 0.10))
        w_AL = float(wconf.get("alignment_norm", 0.05))
        w_LS = float(wconf.get("layer_support_norm", 0.05))  # NUEVO
        df["reliability_index"] = pd.to_numeric(df["reliability_index"], errors="coerce").fillna(0.0)

        df["score"] = (
                w_RI * df["reliability_index"] +
                w_SC * df["support_count_norm"] +
                w_CS * df["collapsed_support_norm"] +
                w_MC * df["model_consistency"] +
                w_AL * df["alignment_norm"] +
                w_LS * df["layer_support_norm"]
        ).astype(float)

        return df

    # --- Colapso al mejor modelo+layer --------------------------------------
    def _collapse_best_model_layer(self, df_scored: pd.DataFrame) -> pd.DataFrame:
        """
        Para cada 'accession', selecciona el (model_name, layer_index) con mayor 'score_global'
        (máximo 'score' observado en ese combo). Dentro de ese combo elimina duplicados de GO
        quedándose con la fila de mayor 'score'. Añade 'best_model', 'best_layer' y 'score_global'.
        """
        best_combo = (
            df_scored.groupby(["accession", "model_name", "layer_index"], dropna=False)["score"]
            .max().rename("score_global").reset_index()
        )

        best_by_acc = (
            best_combo.sort_values(["accession", "score_global"], ascending=[True, False])
            .groupby("accession", as_index=False)
            .first()
            .rename(columns={"model_name": "best_model", "layer_index": "best_layer"})
        )

        df_best = df_scored.merge(best_by_acc, on="accession", how="inner")
        mask = (df_best["model_name"] == df_best["best_model"]) & (df_best["layer_index"] == df_best["best_layer"])
        df_best = df_best[mask].copy()

        df_best = (
            df_best.sort_values(["accession", "go_id", "score"], ascending=[True, True, False])
            .drop_duplicates(subset=["accession", "go_id"], keep="first")
            .reset_index(drop=True)
        )

        # Anexar score_global sin duplicar columnas
        df_best = df_best.merge(best_by_acc[["accession", "score_global"]], on="accession", how="left")

        return df_best

    # --- Post-procesado completo --------------------------------------------
    def post_process_results(self):
        """
        Post-processes transferred GO annotations and produces final outputs.

        This version is designed to work efficiently with COMPACT RAW files written by
        `store_entry` (one row per (accession, model, layer, GO) + precomputed `support_count`).
        If `support_count` is absent (legacy RAW), it will be computed on the fly.

        Pipeline
        --------
        1) Load RAW:
           - Prefer layered files: raw_results_layer_*.csv (faster IO and selective reads).
           - Fallback to legacy: raw_results.csv.
           - Ensure `layer_index` exists.
        2) Compute `reliability_index` from `distance` according to `self.distance_metric`:
           - cosine    → 1 - distance
           - euclidean → 0.5 / (0.5 + distance)
           - default   → 1 / (1 + distance)
        3) Ensure `support_count`:
           - Keep the pre-aggregated column if already present (from compact RAW).
           - Otherwise, compute group count per (accession, model_name, layer_index, go_id).
        4) Reduce to GO leaf terms per (accession, model, layer) using the GO DAG:
           - Identify leaf terms (terms that are not parents of any other term in the group).
           - For each leaf, compute:
             * `collapsed_support`: sum of supports of ancestor terms present in the group
             * `n_collapsed_terms`: number of such ancestors
             * `collapsed_terms`: comma-separated GO IDs of those ancestors
           - For each leaf, keep the row with the highest `reliability_index`.
        5) (Optional) Pairwise alignment metrics:
           - If `sequence_query` and `sequence_reference` are present, compute
             identity / similarity / alignment_score / gaps / lengths.
        6) Add metadata and tidy:
           - Add `distance_metric` and per-model `distance_threshold`.
           - Optionally drop sequences (controlled by conf['postprocess']['keep_sequences']).
           - Round numeric fields and sort for readability.
           - Clean legacy `gene_name` shapes if needed.
        7) Scoring:
           - Compute composite `score` via `_compute_score` (includes `layer_support_norm`
             if you enabled it in the config).
           - Write `results_scored.csv` (all leaf terms per model/layer with scores).
        8) Collapse to best (model, layer) per protein:
           - For each accession, compute `score_global` = max score within each (model, layer).
           - Pick the (model, layer) with highest `score_global`.
           - Within that winning combo, drop duplicate `go_id` keeping the highest `score`.
           - Write `results_collapsed.csv` and copy it as `results.csv` (compatibility alias).
        9) TopGO exports (score-based):
           - `topgo/scored/layer_*/<model>/<category>.tsv`
           - `topgo/scored/all_layers/<model>/<category>.tsv`
           - `topgo/collapsed_best/<category>.tsv`

        Outputs
        -------
        - results_scored.csv     : detailed leaf-level predictions with `score`
        - results_collapsed.csv  : unique per protein from the best (model, layer), with `best_model`,
                                   `best_layer`, and `score_global`
        - results.csv            : alias of results_collapsed.csv (legacy compatibility)
        - topgo/...              : score-based TSV files for enrichment
        """
        import glob, time, shutil

        start_total = time.perf_counter()

        def log_stage(name, t0):
            self.logger.info(f"⏱ {name} → {time.perf_counter() - t0:.2f}s")

        # 1) Load RAW (prefer layered files)
        t0 = time.perf_counter()
        layer_files = sorted(glob.glob(os.path.join(self.experiment_path, "raw_results_layer_*.csv")))
        df_raw = None
        if layer_files:
            dfs = []
            for path in layer_files:
                g = pd.read_csv(path)
                if "layer_index" not in g.columns:
                    try:
                        li = int(os.path.splitext(os.path.basename(path))[0].split("_")[-1])
                    except Exception:
                        li = None
                    g["layer_index"] = li
                dfs.append(g)
            df_raw = pd.concat(dfs, ignore_index=True) if dfs else None
            if df_raw is not None:
                self.logger.info(f"Loaded {len(df_raw)} rows from layered RAW files.")
        if df_raw is None:
            if not os.path.exists(self.raw_results_path):
                self.logger.warning("No raw results found for post-processing.")
                return
            df_raw = pd.read_csv(self.raw_results_path)
            self.logger.info(f"Loaded {len(df_raw)} rows from {os.path.basename(self.raw_results_path)}")
        if df_raw.empty:
            self.logger.warning("Raw results file is empty. Nothing to post-process.")
            return
        log_stage("Load RAW", t0)

        # 2) Ensure layer_index
        t0 = time.perf_counter()
        if "layer_index" not in df_raw.columns:
            df_raw["layer_index"] = None
        log_stage("Ensure layer_index", t0)

        # 3) reliability_index from distance
        t0 = time.perf_counter()
        if self.distance_metric == "cosine":
            df_raw["reliability_index"] = 1 - df_raw["distance"]
        elif self.distance_metric == "euclidean":
            df_raw["reliability_index"] = 0.5 / (0.5 + df_raw["distance"])
        else:
            df_raw["reliability_index"] = 1.0 / (1.0 + df_raw["distance"])
        log_stage("Compute reliability_index", t0)

        # 4) Ensure / compute support_count
        t0 = time.perf_counter()
        if "support_count" not in df_raw.columns:
            df_raw["support_count"] = (
                df_raw.groupby(["accession", "model_name", "layer_index", "go_id"])["go_id"].transform("count")
            )
        else:
            df_raw["support_count"] = pd.to_numeric(df_raw["support_count"], errors="coerce").fillna(0).astype(int)
        log_stage("Compute/keep support_count", t0)

        # 5) Build GO parents cache
        t0 = time.perf_counter()
        unique_go_ids = pd.unique(df_raw["go_id"])
        parents_cache = {}

        def get_parents_cached(term):
            if term in parents_cache:
                return parents_cache[term]
            try:
                if term in self.go:
                    parents = self.go[term].get_all_parents()
                    parents_cache[term] = set(parents) if not isinstance(parents, set) else parents
                else:
                    parents_cache[term] = set()
            except Exception:
                parents_cache[term] = set()
            return parents_cache[term]

        for t in unique_go_ids:
            _ = get_parents_cached(t)
        log_stage("Build GO parents cache", t0)

        # 6) Reduce to leaf terms + collapsed_support  (FAST VERSION)
        t0 = time.perf_counter()
        rows = []

        # Ensure go_id is string (robust keys for parents_cache)
        df_raw["go_id"] = df_raw["go_id"].astype(str)

        for (acc, model, layer), group in df_raw.groupby(["accession", "model_name", "layer_index"], dropna=False):
            # Unique GO terms in this (accession, model, layer)
            group_go = pd.unique(group["go_id"])
            if len(group_go) == 0:
                continue
            group_go_set = set(group_go)

            # 1) FAST: best row per GO (highest reliability_index) once
            #    (much faster than subsetting for each GO later)
            best_per_go = (
                group.sort_values("reliability_index", ascending=False)
                .drop_duplicates(subset=["go_id"], keep="first")
                .copy()
            )

            # 2) Leaf detection using parents union
            #    parents_union = union of parents for all terms in the group
            parents_union = set()
            # Build ancestors_in_group only for terms in group (use cached parents)
            ancestors_in_group = {}
            for gid in group_go:
                parents = self.go[gid].get_all_parents() if gid in self.go else []
                # store as set (intersect with group terms)
                pg = set(parents) & group_go_set
                ancestors_in_group[gid] = pg
                # union for leaf detection
                parents_union.update(pg)

            # Leafs = terms that are NOT any other's parent within the group
            leaf_terms = [gid for gid in group_go if gid not in parents_union]
            if not leaf_terms:
                continue

            # 3) Keep only best rows for leaf GO terms
            leaf_best = best_per_go[best_per_go["go_id"].isin(leaf_terms)].copy()
            if leaf_best.empty:
                continue

            # 4) Build support map once (support_count already pre-aggregated or enforced above)
            #    Use best_per_go (1 per GO) to avoid duplicates in the index
            support_map = (
                best_per_go[["go_id", "support_count"]]
                .drop_duplicates("go_id")
                .set_index("go_id")["support_count"]
                .to_dict()
            )

            # 5) Vectorized mapping of collapsed metrics for each leaf term
            def _sum_support_ancestors(gid: str) -> int:
                return int(sum(support_map.get(a, 0) for a in ancestors_in_group.get(gid, set())))

            def _count_ancestors(gid: str) -> int:
                return int(len(ancestors_in_group.get(gid, set())))

            def _list_ancestors(gid: str) -> str:
                anc = ancestors_in_group.get(gid, set())
                return ", ".join(sorted(anc)) if anc else ""

            leaf_best["collapsed_support"] = leaf_best["go_id"].map(_sum_support_ancestors)
            leaf_best["n_collapsed_terms"] = leaf_best["go_id"].map(_count_ancestors)
            leaf_best["collapsed_terms"] = leaf_best["go_id"].map(_list_ancestors)

            # 6) Collect rows
            #    (already the best per GO + leaf-only + collapsed fields)
            rows.append(leaf_best)

        # Concatenate all groups
        df_out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=df_raw.columns.tolist() + [
            "collapsed_support", "n_collapsed_terms", "collapsed_terms"
        ])
        log_stage("Reduce to leaves + collapsed_support", t0)

        # 7) Optional pairwise alignment metrics
        t0 = time.perf_counter()
        unique_pairs = (
            df_out[["sequence_query", "sequence_reference"]].drop_duplicates()
            if {"sequence_query", "sequence_reference"} <= set(df_out.columns) else pd.DataFrame()
        )
        if not unique_pairs.empty:
            with ProcessPoolExecutor(max_workers=self.conf.get("store_workers", 4)) as ex:
                metrics_list = list(ex.map(compute_metrics, unique_pairs.to_dict("records")))
            metrics_df = pd.DataFrame(metrics_list)
            df_out = df_out.merge(metrics_df, on=["sequence_query", "sequence_reference"], how="left")
        log_stage("Compute pairwise metrics", t0)

        # 8) Metadata + cleanup
        t0 = time.perf_counter()
        df_out = self._add_metadata_columns(df_out)
        for col in ["distance", "identity", "similarity", "alignment_score", "gaps_percentage", "reliability_index"]:
            if col in df_out.columns:
                df_out[col] = pd.to_numeric(df_out[col], errors="coerce")
                df_out[col] = df_out[col].apply(lambda x: round(x, 4) if pd.notnull(x) else x)
        df_out = df_out.sort_values(
            by=["accession", "layer_index", "go_id", "model_name", "reliability_index"],
            ascending=[True, True, True, True, False]
        )

        # clean legacy gene_name lists like "[{...}]"
        if "gene_name" in df_out.columns:
            def _extract_gene_name(g):
                try:
                    val = eval(g)
                    if (isinstance(g, str) and g.startswith("[{") and isinstance(val, list)
                            and len(val) > 0 and "Name" in val[0]):
                        return val[0]["Name"]
                except Exception:
                    return None
                return None

            df_out["gene_name"] = df_out["gene_name"].apply(_extract_gene_name)
        log_stage("Add metadata + sort & round + clean gene_name", t0)

        # 9) SCORE + results_scored.csv
        t0 = time.perf_counter()
        df_scored = self._compute_score(df_out)
        results_scored_path = os.path.join(self.experiment_path, "results_scored.csv")
        df_scored.to_csv(results_scored_path, index=False)
        self.logger.info(f"📝 Wrote {len(df_scored)} rows → results_scored.csv")
        log_stage("Compute SCORE + write results_scored.csv", t0)

        # 10) Collapse
        t0 = time.perf_counter()
        df_collapsed = self._collapse_best_overall(df_scored)  # ← nuevo colapso global
        results_collapsed_path = os.path.join(self.experiment_path, "results_collapsed.csv")
        df_collapsed.to_csv(results_collapsed_path, index=False)
        shutil.copyfile(results_collapsed_path, self.results_path)  # legacy alias
        self.logger.info(f"📝 Wrote {len(df_collapsed)} rows → results_collapsed.csv (aliased to results.csv)")
        log_stage("Collapse best model+layer + write results_collapsed.csv + alias", t0)

        # 11) TopGO exports (score-based)
        t0 = time.perf_counter()
        if self.topgo_enabled:
            self.logger.info("📁 Generating TopGO outputs (using SCORE)...")
            base_dir = os.path.join(self.experiment_path, "topgo")
            os.makedirs(base_dir, exist_ok=True)

            def build_topgo(df_group):
                return (
                    df_group.groupby(["accession", "go_id"], as_index=False)["score"]
                    .max().rename(columns={"score": "score"})
                    .loc[:, ["accession", "go_id", "score"]]
                )

            # detailed per layer/model/category
            for (layer, model_name, category), g in df_scored.groupby(
                    ["layer_index", "model_name", "category"], dropna=False
            ):
                layer_tag = f"layer_{layer}" if layer is not None else "legacy"
                out_dir = os.path.join(base_dir, "scored", layer_tag, model_name)
                os.makedirs(out_dir, exist_ok=True)
                build_topgo(g).to_csv(os.path.join(out_dir, f"{category}.tsv"),
                                      sep="\t", index=False, header=False)

            # all layers per model/category
            combined_dir = os.path.join(base_dir, "scored", "all_layers")
            for (model_name, category), g in df_scored.groupby(["model_name", "category"]):
                out_dir = os.path.join(combined_dir, model_name)
                os.makedirs(out_dir, exist_ok=True)
                build_topgo(g).to_csv(os.path.join(out_dir, f"{category}.tsv"),
                                      sep="\t", index=False, header=False)

            # collapsed final set by category
            collapsed_dir = os.path.join(base_dir, "collapsed_best")
            os.makedirs(collapsed_dir, exist_ok=True)
            for category, g in df_collapsed.groupby("category"):
                build_topgo(g).to_csv(os.path.join(collapsed_dir, f"{category}.tsv"),
                                      sep="\t", index=False, header=False)

            self.logger.info("📝 TopGO outputs generated.")
        log_stage("Generate TopGO outputs", t0)

        self.logger.info(f"✅ Post-processing finished in {time.perf_counter() - start_total:.2f}s")

    def _compact_annotations_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compact raw annotations *within the batch* to reduce disk size and speed up post-processing.

        Grouping keys (NOTE: we DO NOT use 'go_description' as a key; it's redundant with 'go_id'):
          ('accession', 'model_name', 'embedding_type_id', 'layer_index',
           'go_id', 'category', 'evidence_code')

        For each group:
          - support_count := number of neighbors contributing that GO in the group
          - distance := minimal distance in the group (best neighbor)
          - Representative row := the row with minimal 'distance' (ties -> keep first)
          - Keep the representative's metadata (sequence_reference, protein_id, organism, gene_name, go_description, ...)

        Returns a compact DataFrame with one row per group + 'support_count' and the representative's fields.
        """
        if df.empty:
            return df

        df = df.copy()

        # --- Ensure expected columns exist
        expected = [
            "accession", "model_name", "embedding_type_id", "layer_index",
            "go_id", "category", "evidence_code", "go_description",
            "distance", "sequence_query", "sequence_reference",
            "protein_id", "organism", "gene_name"
        ]
        for c in expected:
            if c not in df.columns:
                df[c] = None

        # --- Coerce dtypes to avoid object/float collisions on keys
        # Numeric/nullable ints
        df["embedding_type_id"] = pd.to_numeric(df["embedding_type_id"], errors="coerce").astype("Int64")
        df["layer_index"] = pd.to_numeric(df["layer_index"], errors="coerce").astype("Int64")
        df["distance"] = pd.to_numeric(df["distance"], errors="coerce")

        # String-like keys
        df["accession"] = df["accession"].astype(str)
        df["model_name"] = df["model_name"].astype(str)
        df["go_id"] = df["go_id"].astype(str)
        df["category"] = df["category"].astype(str)
        df["evidence_code"] = df["evidence_code"].astype(str)

        # Non-key text fields (we keep them from the representative)
        # Force them to str to avoid surprises later (NaN -> "nan" is fine for metadata)
        for txt in ("go_description", "sequence_query", "sequence_reference", "organism", "gene_name"):
            df[txt] = df[txt].astype(str)

        # --- Grouping keys (exclude go_description!)
        keys = [
            "accession", "model_name", "embedding_type_id", "layer_index",
            "go_id", "category", "evidence_code"
        ]

        # support_count per group (size is robust vs NaN)
        df["support_count"] = df.groupby(keys, dropna=False)["go_id"].transform("size")

        # find the representative rows: those with minimal distance within each group
        min_distance = df.groupby(keys, dropna=False)["distance"].transform("min")
        df["__is_best"] = df["distance"].eq(min_distance)

        best = (
            df[df["__is_best"]]
            .sort_values(keys + ["distance"])
            .drop_duplicates(subset=keys, keep="first")
            .copy()
        )

        # Clean temp flag
        best.drop(columns=["__is_best"], inplace=True, errors="ignore")

        # Reorder columns: keep a consistent, compact header
        ordered = [
            "accession", "model_name", "embedding_type_id", "layer_index",
            "go_id", "category", "evidence_code", "go_description",
            "distance", "sequence_query", "sequence_reference",
            "protein_id", "organism", "gene_name", "support_count"
        ]
        rest = [c for c in best.columns if c not in ordered]
        return best[ordered + rest]

    def _collapse_best_overall(self, df_scored: pd.DataFrame) -> pd.DataFrame:
        """
        Colapsa ignorando modelo y capa:
          - Para cada (accession, go_id), conserva la fila con mayor 'score'
          - Define 'best_model', 'best_layer' y 'score_global' a partir de esa fila ganadora
        """
        df = df_scored.copy()
        df = (
            df.sort_values(["accession", "go_id", "score"], ascending=[True, True, False])
            .drop_duplicates(subset=["accession", "go_id"], keep="first")
            .reset_index(drop=True)
        )
        df["best_model"] = df["model_name"]
        df["best_layer"] = df["layer_index"]
        df["score_global"] = df["score"]
        return df
