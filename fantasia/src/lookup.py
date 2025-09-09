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
import pandas as pd
from protein_information_system.tasks.gpu import GPUTaskInitializer
from goatools.base import get_godag
from protein_information_system.sql.model.entities.sequence.sequence import Sequence

from sqlalchemy import text
import h5py
from protein_information_system.sql.model.entities.embedding.sequence_embedding import SequenceEmbeddingType, \
    SequenceEmbedding
from protein_information_system.sql.model.entities.protein.protein import Protein

from fantasia.src.helpers.helpers import get_descendant_ids, compute_metrics

import glob
import time
import shutil


class EmbeddingLookUp(GPUTaskInitializer):
    """
    GO annotation transfer via embedding similarity.

    This component reads query embeddings (HDF5) and compares them against reference
    embeddings stored in a vector-aware relational database. For the closest reference
    sequences, it retrieves GO annotations and writes results to CSV (and optionally
    TopGO-ready TSV). It supports:

      • Taxonomy-based filtering (include/exclude, with optional descendant expansion)
      • Redundancy-aware neighbor selection (MMseqs2 clusters)
      • Multiple embedding models with per-model distance thresholds
      • Distance computation on GPU (PyTorch) or CPU
      • Optional pairwise alignment post-processing (identity/similarity)

    Parameters
    ----------
    conf : dict
        Configuration including paths, thresholds, model settings, and processing options.
    current_date : str
        Timestamp suffix used to version output artifacts.

    Notes
    -----
    - Supported distance metrics: 'euclidean' and 'cosine' (default: 'euclidean').
    - Redundancy filtering is based on MMseqs2 identity/coverage thresholds.
    - GO annotations are preloaded once and can be filtered by taxonomy.
    """

    def __init__(self, conf, current_date):
        """
        Initialize configuration, paths, GO DAG, optional MMseqs2 clustering,
        and lazy reference lookup infrastructure.
        """
        super().__init__(conf)

        # --- NEW: adaptar referencias desde conf['lookup'] ---------------------
        lk = self.conf.get("lookup", {}) or {}

        # Copiar opciones directas si no estaban ya en raíz
        for k in ("use_gpu", "batch_size", "limit_per_entry", "topgo", "lookup_cache_max"):
            if k not in self.conf and k in lk:
                self.conf[k] = lk[k]

        # distance_metric: permitir lookup.distance_metric además de embedding.distance_metric
        if "distance_metric" in lk:
            emb = self.conf.setdefault("embedding", {})
            emb.setdefault("distance_metric", lk["distance_metric"])

        # Redundancia: mapear lookup.redundancy -> llaves planas esperadas
        r = lk.get("redundancy") or {}
        if "redundancy_filter" not in self.conf and "identity" in r:
            self.conf["redundancy_filter"] = r["identity"]
        if "alignment_coverage" not in self.conf and "coverage" in r:
            self.conf["alignment_coverage"] = r["coverage"]
        if "threads" not in self.conf and "threads" in r:
            self.conf["threads"] = r["threads"]

        # Taxonomía: mapear lookup.taxonomy -> llaves planas esperadas
        t = lk.get("taxonomy") or {}
        if "exclude" in t and not self.conf.get("taxonomy_ids_to_exclude"):
            self.conf["taxonomy_ids_to_exclude"] = t["exclude"]
        if "include_only" in t and not self.conf.get("taxonomy_ids_included_exclusively"):
            self.conf["taxonomy_ids_included_exclusively"] = t["include_only"]
        if "get_descendants" in t and self.conf.get("get_descendants") in (None, ""):
            self.conf["get_descendants"] = t["get_descendants"]

        # ----------------------------------------------------------------------

        self.types = None
        self.current_date = current_date
        self.logger.info("EmbeddingLookUp: initializing component…")

        # ---- Paths ---------------------------------------------------------
        self.experiment_path = self.conf.get("experiment_path")
        self.embeddings_path = self.conf.get("embeddings_path") or os.path.join(self.experiment_path, "embeddings.h5")
        self.raw_results_path = os.path.join(self.experiment_path, "raw_results.csv")
        self.results_path = os.path.join(self.experiment_path, "results.csv")
        self.topgo_path = os.path.join(self.experiment_path, "results_topgo.tsv")

        # ---- Limits & options ----------------------------------------------
        self.limit_per_entry = self.conf.get("limit_per_entry", 200)
        self.topgo_enabled = self.conf.get("topgo", False)
        self.batch_size = self.conf.get("batch_size", 1)

        # ---- Optional redundancy filtering --------------------------------
        redundancy_filter_threshold = self.conf.get("redundancy_filter", 0)
        if redundancy_filter_threshold > 0:
            self.logger.info(
                "Redundancy filter enabled (MMseqs2): identity>=%.3f, coverage>=%.3f, threads=%s",
                float(self.conf.get("redundancy_filter", 0)),
                float(self.conf.get("alignment_coverage", 0)),
                int(self.conf.get("threads", 12)),
            )
            self.generate_clusters()
        else:
            self.logger.info("Redundancy filter disabled.")

        # ---- GO ontology ---------------------------------------------------
        self.go = get_godag("go-basic.obo", optional_attrs="relationship")
        self.logger.info("GO DAG loaded (go-basic.obo).")

        # ---- Distance metric -----------------------------------------------
        self.distance_metric = self.conf.get("embedding", {}).get("distance_metric", "cosine")
        if self.distance_metric not in ("euclidean", "cosine"):
            self.logger.warning(
                "Unsupported distance metric '%s'; falling back to 'cosine'.", self.distance_metric
            )
            self.distance_metric = "cosine"
        self.logger.info("Distance metric: %s", self.distance_metric)

        # ---- Taxonomy filters (integers; optional descendant expansion) ----
        def _expand_tax_ids(ids):
            ids = ids or []
            clean = [int(t) for t in ids if str(t).isdigit()]
            if self.conf.get("get_descendants", False) and clean:
                return [int(t) for t in get_descendant_ids(clean)]
            return clean

        self.exclude_taxon_ids = _expand_tax_ids(self.conf.get("taxonomy_ids_to_exclude"))
        self.include_taxon_ids = _expand_tax_ids(self.conf.get("taxonomy_ids_included_exclusively"))
        self.logger.info(
            "Taxonomy filters initialized | exclude=%s | include=%s | expand_descendants=%s",
            self.exclude_taxon_ids or "[]",
            self.include_taxon_ids or "[]",
            bool(self.conf.get("get_descendants", False)),
        )

        # ---- Lazy reference lookup cache ----------------------------------
        # key: (model_id: int, layer_index: Optional[int]) -> {"ids": np.ndarray, "embeddings": np.ndarray, "layers": np.ndarray}
        self._lookup_cache: dict[tuple[int, int | None], dict] = {}
        self._lookup_cache_max = int(self.conf.get("lookup_cache_max", 4))
        self.logger.info(
            "Reference lookup will be loaded lazily per (model, layer) with an in-memory cache of max %d entry(ies).",
            self._lookup_cache_max,
        )

        # ---- Load model definitions & preload GO annotations ---------------
        self.load_model_definitions()
        self.logger.info("Loaded %d model definitions from DB+config: %s",
                         len(self.types or {}), list(self.types.keys()) if self.types else [])
        self.logger.info("Preloading GO annotations from the database…")
        self.preload_annotations()
        self.logger.info("GO annotations cached: %d sequences with annotations.",
                         len(getattr(self, "go_annotations", {})))

        self.logger.info("EmbeddingLookUp initialization completed successfully.")

    def enqueue(self):
        """
        Encola tareas por lotes homogéneos **(modelo, capa)**.

        - Recorre el HDF5 una sola vez (acceso perezoso).
        - Mantiene buffers por clave (embedding_type_id, layer_index).
        - Cuando un buffer alcanza batch_size -> publica el lote y limpia el buffer.
        - Al final, publica los restos de cada buffer.
        - Compatibilidad: si no hay capas, layer_index=None.
        """
        import h5py
        from collections import defaultdict

        self.logger.info("Starting embedding-based GO annotation process.")
        self.logger.info(f"Processing query embeddings from HDF5: {self.embeddings_path}")

        if not os.path.exists(self.embeddings_path):
            raise FileNotFoundError(
                f"HDF5 file not found: {self.embeddings_path}. "
                f"Ensure embeddings have been generated prior to annotation."
            )

        batch_size = int(self.batch_size)
        total_entries = 0
        total_batches = 0

        # buffers[(embedding_type_id, layer_index)] = [task_data, ...]
        buffers = defaultdict(list)

        # Mapa rápido id->info y id->nombre (para evitar búsquedas lineales por accesión)
        # self.types tiene keys por task_name, con info {'id', 'task_name', 'distance_threshold', ...}
        by_id = {info["id"]: info for info in self.types.values()}

        def flush(key):
            """Publica el buffer de 'key' en lotes de batch_size."""
            nonlocal total_batches
            buf = buffers[key]
            if not buf:
                return
            for i in range(0, len(buf), batch_size):
                chunk = buf[i:i + batch_size]
                # Todos comparten embedding_type_id y layer_index por construcción
                model_id = chunk[0]["embedding_type_id"]
                layer_index = chunk[0].get("layer_index")
                payload = {
                    "model_id": model_id,
                    "layer_index": layer_index,
                    "tasks": chunk,
                }
                model_type = chunk[0]["model_name"]  # p.ej. "esm2", "prott5", etc.
                self.publish_task(payload, model_type=model_type)
                total_batches += 1
            buffers[key].clear()

        try:
            with h5py.File(self.embeddings_path, "r") as h5file:
                for accession, group in h5file.items():
                    if "sequence" not in group:
                        self.logger.warning(f"Sequence missing for accession '{accession}'. Skipping.")
                        continue

                    # Recorremos tipos (type_<id>)
                    for type_key, type_grp in group.items():
                        if not type_key.startswith("type_"):
                            continue

                        try:
                            model_id = int(type_key.split("_", 1)[1])
                        except Exception:
                            self.logger.warning(f"Malformed type group '{type_key}'. Skipping.")
                            continue

                        model_info = by_id.get(model_id)
                        if model_info is None:
                            # Modelo no habilitado en config o no cargado en self.types
                            self.logger.info(f"Model id {model_id} not enabled/loaded — skipping.")
                            continue

                        model_name = model_info["task_name"]
                        distance_threshold = model_info.get("distance_threshold")

                        # ¿Hay capas?
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

                                task = {
                                    "h5_path": self.embeddings_path,  # para abrir el archivo en process
                                    "h5_group": f"{accession}/{type_key}/{lk}",  # ruta interna hasta layer_*
                                    "embedding_type_id": model_id,
                                    "model_name": model_name,
                                    "distance_threshold": distance_threshold,
                                    "layer_index": layer_index,
                                }

                                key = (model_id, layer_index)
                                buffers[key].append(task)
                                total_entries += 1

                                # Flush si alcanzamos batch_size
                                if len(buffers[key]) >= batch_size:
                                    flush(key)
            # Publica los restos
            for key in list(buffers.keys()):
                flush(key)

            self.logger.info(
                f"📮 Enqueued {total_entries} queries in {total_batches} homogeneous batches "
                f"(grouped by model & layer; batch_size={batch_size})."
            )

        except Exception as e:
            self.logger.error(f"Unexpected error during enqueue: {e}", exc_info=True)
            raise

    def load_model_definitions(self):
        """
        Initialize `self.types` by matching DB embedding types with configuration.
        Only models present in both and marked as enabled are kept.

        Logs, once per model (no duplicates):
          - model name and DB id
          - distance_threshold (config)
          - enabled_layers (config)
          - available_layers_in_h5 (scan of HDF5)
          - effective_layers (intersection or ALL if config is not restricting)
        """
        self.types = {}

        try:
            db_models = self.session.query(SequenceEmbeddingType).all()
        except Exception as e:
            self.logger.error("Failed to query SequenceEmbeddingType table: %s", e)
            raise

        cfg_models = self.conf.get("embedding", {}).get("models", {})

        # 1) Build self.types
        for db_model in db_models:
            task_name = db_model.name  # DB display name
            matched_name = next((k for k in cfg_models if k.lower() == task_name.lower()), None)
            if matched_name is None:
                self.logger.warning("Model '%s' exists in DB but not in config — skipping.", task_name)
                continue

            cfg = cfg_models[matched_name]
            if not cfg.get("enabled", True):
                self.logger.info("Model '%s' is disabled in config — skipping.", matched_name)
                continue

            self.types[matched_name] = {
                "id": db_model.id,
                "model_name": db_model.model_name,
                "task_name": matched_name,
                "distance_threshold": cfg.get("distance_threshold"),
                "batch_size": cfg.get("batch_size"),
                "enabled_layers": cfg.get("enabled_layers"),  # may be None/absent
            }

        # 2) Detailed, single-pass logging (prevents duplicates)
        if not self.types:
            self.logger.warning("No enabled models found after matching DB and config.")
            return

        # Sort for deterministic output
        for name in sorted(self.types.keys(), key=str.lower):
            info = self.types[name]
            model_id = info["id"]
            threshold = info.get("distance_threshold")
            enabled_layers = info.get("enabled_layers")

            # HDF5 scan (may be empty if file or groups are missing)
            available_layers = self._h5_available_layers(model_id)

            # Effective layers = intersection if config restricts; else "ALL"
            if enabled_layers and isinstance(enabled_layers, (list, tuple)):
                eff = sorted(set(enabled_layers) & set(available_layers)) if available_layers else sorted(
                    set(enabled_layers))
                effective_layers = eff
            else:
                effective_layers = "ALL" if available_layers else "[]"

            self.logger.info(
                "Model '%s' (id=%s): threshold=%s | enabled_layers=%s | available_layers_in_h5=%s | effective_layers=%s",
                name, model_id, threshold if threshold is not None else "None",
                enabled_layers if enabled_layers else "ALL",
                available_layers if available_layers else "[]",
                effective_layers
            )

        self.logger.info("Loaded %d model(s) from DB+config: %s",
                         len(self.types), list(sorted(self.types.keys(), key=str.lower)))

    def process(self, payload: dict) -> list[dict]:
        """
        Process a homogeneous batch of queries for a single (model_id, layer_index),
        returning *compact neighbor hits* only (no GO expansion, no sequences).

        Input payload schema
        --------------------
        payload : dict
            {
              "model_id": int,
              "layer_index": int | None,
              "tasks": list[dict]
                # Each task is EITHER:
                #   a) lightweight (recommended):
                #      {"h5_path": str, "h5_group": str, "model_name": str, "distance_threshold": float|None, "layer_index": int|None}
                #   b) legacy (discouraged due to payload size):
                #      {"embedding": np.ndarray, "accession": str, "sequence": str, "model_name": str, "distance_threshold": float|None, "layer_index": None}
            }

        Output (compact) schema
        -----------------------
        Returns a list of *hits* with one record per selected neighbor:
            {
              "accession": str,              # query identifier (without 'accession_' prefix)
              "ref_sequence_id": int,        # DB sequence.id of the reference neighbor
              "distance": float,             # distance to the neighbor (metric per self.distance_metric)
              "model_name": str,             # logical model key (e.g., "esm2", "prott5")
              "embedding_type_id": int,      # embedding type id from DB
              "layer_index": int | None      # layer shared by the batch or None for legacy
            }

        Rationale
        ---------
        - This method intentionally does NOT expand neighbors into GO annotations.
          It avoids building large payloads (neighbors × GO terms), keeping messages
          small and robust even when `limit_per_entry` > 1.
        - GO expansion is deferred to `store_entry()` using `self.go_annotations`
          (preloaded from the database), and sequences are read lazily only if needed
          by configuration.

        Notes
        -----
        - GPU distance is used when `conf['use_gpu']` is True (default), otherwise CPU (SciPy).
        - Redundancy filtering (MMseqs2 clusters) is applied when configured.
        - `limit_per_entry` and optional `distance_threshold` control neighbor selection.

        Returns
        -------
        list[dict]
            Compact hits. May be empty if no neighbors match.
        """
        import time
        from collections import defaultdict
        import numpy as np
        import h5py
        from scipy.spatial.distance import cdist
        import torch

        t_start = time.perf_counter()

        try:
            # --- Validate payload ------------------------------------------------
            if not isinstance(payload, dict):
                self.logger.error("process(payload): expected dict with keys ['model_id','layer_index','tasks'].")
                return []

            model_id = payload.get("model_id")
            layer_index_batch = payload.get("layer_index")
            batch = payload.get("tasks") or []

            if model_id is None or not isinstance(batch, list) or not batch:
                self.logger.error("process(payload): invalid or empty payload.")
                return []

            # Optional metadata from tasks/conf
            model_name = next((t.get("model_name") for t in batch if "model_name" in t), None)
            threshold = next((t.get("distance_threshold") for t in batch if "distance_threshold" in t),
                             self.conf.get("distance_threshold"))
            use_gpu = bool(self.conf.get("use_gpu", True))
            limit = int(self.conf.get("limit_per_entry", 1000))

            self.logger.info(
                "Batch start | model_id=%s (%s) | layer_index=%s | tasks=%d | metric=%s | threshold=%s | limit=%d | gpu=%s",
                model_id, model_name or "unknown", layer_index_batch, len(batch),
                self.distance_metric, threshold if threshold is not None else "None", limit, use_gpu
            )

            # --- Reference lookup (lazy, cached) ---------------------------------
            lookup = self._get_lookup_for_batch(model_id, layer_index_batch)
            if not lookup:
                self.logger.warning(
                    "process(payload): no reference lookup for model_id=%s, layer_index=%s — skipping.",
                    model_id, layer_index_batch
                )
                return []

            # --- Materialize query embeddings ------------------------------------
            embeddings_list = []
            accessions_list = []

            # Group tasks by file to minimize HDF5 opens
            by_h5 = defaultdict(list)
            for t in batch:
                if "h5_path" in t and "h5_group" in t:
                    by_h5[t["h5_path"]].append(t)
                else:
                    by_h5[None].append(t)

            for h5_path, items in by_h5.items():
                with h5py.File(h5_path, "r") as h5:
                    for t in items:
                        grp_path = t["h5_group"]  # e.g., "accession_X/type_1/layer_16" or ".../type_1" (legacy)
                        # Load only the embedding; skip sequences here to keep IO minimal
                        emb = h5[grp_path]["embedding"][:]
                        embeddings_list.append(np.asarray(emb))
                        # Accession (top-level node name)
                        acc_node = grp_path.split("/", 1)[0]  # "accession_X"
                        acc = acc_node.removeprefix("accession_")
                        accessions_list.append(acc)

            if not embeddings_list:
                self.logger.warning("process(payload): no query embeddings materialized — skipping batch.")
                return []

            embeddings = np.stack(embeddings_list)  # (N, D)
            accessions = accessions_list
            layer_indices = [layer_index_batch] * len(accessions)

            self.logger.info(
                "Queries materialized | N=%d | dim=%s",
                len(embeddings), tuple(embeddings.shape[1:])
            )

            # --- Distance computation (GPU/CPU) -----------------------------------
            t_dist = time.perf_counter()
            if use_gpu:
                queries = torch.tensor(embeddings, dtype=torch.float32).cuda(non_blocking=True)
                targets = torch.tensor(lookup["embeddings"], dtype=torch.float32).cuda(non_blocking=True)

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

                self.logger.info(
                    "Distances computed on GPU | queries=%s | refs=%s | elapsed=%.2fs",
                    tuple(queries.shape), tuple(targets.shape), time.perf_counter() - t_dist
                )
            else:
                dist_matrix = cdist(embeddings, lookup["embeddings"], metric=self.distance_metric)
                self.logger.info(
                    "Distances computed on CPU | queries=%s | refs=%s | elapsed=%.2fs",
                    embeddings.shape, lookup["embeddings"].shape, time.perf_counter() - t_dist
                )

            # --- Optional redundancy filter --------------------------------------
            redundancy = int(self.conf.get("redundancy_filter", 0))
            redundant_ids: dict[str, set] = {}
            if redundancy > 0:
                for acc in accessions:
                    redundant_ids[acc] = self.retrieve_cluster_members(acc)
                self.logger.info(
                    "Redundancy filter active | threshold=%.3f | accessions_with_clusters=%d",
                    float(redundancy), sum(1 for acc in accessions if redundant_ids.get(acc))
                )

            # --- Neighbor selection → COMPACT HITS -------------------------------
            hits: list[dict] = []
            total_neighbors = 0
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
                if threshold is None or threshold == 0:
                    selected = order[:limit]
                else:
                    selected = order[distances[order] <= float(threshold)][:limit]

                total_neighbors += len(selected)
                li = layer_indices[i]

                for idx in selected:
                    ref_id = int(seq_ids[idx])
                    d = float(distances[idx])
                    hits.append({
                        "accession": accession,
                        "ref_sequence_id": ref_id,
                        "distance": d,
                        "model_name": model_name,
                        "embedding_type_id": model_id,
                        "layer_index": li,
                    })

            elapsed = time.perf_counter() - t_start
            avg_neighbors = (total_neighbors / len(accessions)) if accessions else 0.0
            self.logger.info(
                "Batch done | queries=%d | layer=%s | neighbors=%d (avg=%.2f) | hits=%d | elapsed=%.2fs",
                len(accessions), layer_index_batch, total_neighbors, avg_neighbors, len(hits), elapsed
            )

            return hits

        except Exception as e:
            import traceback
            self.logger.error("process(payload) failed: %s\n%s", e, traceback.format_exc())
            raise

    def store_entry(self, annotations_or_hits: list[dict]) -> None:
        """
        Persist results to disk. Accepts EITHER:
          a) legacy expanded annotations (full rows), OR
          b) compact neighbor hits (new flow, with 'ref_sequence_id').

        Behavior
        --------
        - If input are *hits*, expand them to GO annotations using `self.go_annotations`
          (preloaded DB cache keyed by reference sequence_id).
        - Query and reference sequences are included ONLY when
          `conf['postprocess']['keep_sequences']` is True (lazy HDF5 read).
        - Compact the expanded data within the batch using `_compact_annotations_df`
          to reduce on-disk size and accelerate downstream post-processing.
        - Write to layered raw CSVs when `layer_index` is present:
            <experiment_path>/raw_results_layer_{L}.csv
          or to <experiment_path>/raw_results.csv for legacy/no-layer.

        Input schemas
        -------------
        Legacy expanded annotations (kept for backward compatibility):
            {
              "accession": str,
              "sequence_query": str | None,
              "sequence_reference": str | None,
              "go_id": str,
              "category": str,
              "evidence_code": str,
              "go_description": str,
              "distance": float,
              "model_name": str,
              "embedding_type_id": int,
              "layer_index": int | None,
              "protein_id": int,
              "organism": str,
              "gene_name": str | None,
              ...
            }

        Compact neighbor hits (preferred, output of process()):
            {
              "accession": str,
              "ref_sequence_id": int,
              "distance": float,
              "model_name": str,
              "embedding_type_id": int,
              "layer_index": int | None
            }

        Notes
        -----
        - Expansion fan-out is (hits × GO terms) but it happens server-side where memory
          and I/O are controlled; queue messages remain small.
        - The compact step is idempotent and safe to apply even if upstream has already
          reduced duplicates.

        Raises
        ------
        Exception
            Propagates unexpected IO or DataFrame errors after logging.
        """
        import os
        import pandas as pd

        items = annotations_or_hits or []
        if not items:
            self.logger.info("store_entry: no annotations/hits to persist.")
            return

        try:
            os.makedirs(self.experiment_path, exist_ok=True)

            # Detect compact-hits mode
            is_hits = items and isinstance(items[0], dict) and ("ref_sequence_id" in items[0])

            if is_hits:
                keep_seq = (self.conf.get("postprocess", {}) or {}).get("keep_sequences", False)
                h5 = None

                # Lazy HDF5 open only if we actually plan to read sequences
                if keep_seq and os.path.exists(self.embeddings_path):
                    import h5py
                    try:
                        h5 = h5py.File(self.embeddings_path, "r")
                    except Exception as e:
                        self.logger.warning("store_entry: could not open HDF5 for sequence read: %s", e)
                        h5 = None

                expanded_rows = []
                for hit in items:
                    acc = hit["accession"]
                    ref_id = int(hit["ref_sequence_id"])
                    d = float(hit["distance"])
                    model_name = hit["model_name"]
                    model_id = int(hit["embedding_type_id"])
                    li = hit.get("layer_index")

                    anns = self.go_annotations.get(ref_id, [])
                    if not anns:
                        continue

                    # Optionally load query sequence (once per hit) if keep_seq is enabled
                    seq_query = None
                    if keep_seq and h5 is not None:
                        acc_node = f"accession_{acc}"
                        try:
                            if acc_node in h5 and "sequence" in h5[acc_node]:
                                raw_seq = h5[acc_node]["sequence"][()]
                                seq_query = raw_seq.decode("utf-8") if hasattr(raw_seq, "decode") else str(raw_seq)
                        except Exception:
                            seq_query = None

                    for ann in anns:
                        expanded_rows.append({
                            "accession": acc,
                            "sequence_query": seq_query if keep_seq else None,
                            "sequence_reference": ann.get("sequence") if keep_seq else None,
                            "go_id": ann["go_id"],
                            "category": ann["category"],
                            "evidence_code": ann["evidence_code"],
                            "go_description": ann["go_description"],
                            "distance": d,
                            "model_name": model_name,
                            "embedding_type_id": model_id,
                            "layer_index": li,
                            "protein_id": ann["protein_id"],
                            "organism": ann["organism"],
                            "gene_name": ann["gene_name"],
                        })

                # Ensure we close HDF5 handle if opened
                if h5 is not None:
                    try:
                        h5.close()
                    except Exception:
                        pass

                if not expanded_rows:
                    self.logger.info("store_entry: expanded rows empty; nothing to write.")
                    return

                df = pd.DataFrame(expanded_rows)

            # Compact within-batch to cut duplicates and shrink output
            df_compact = self._compact_annotations_df(df)
            if df_compact.empty:
                self.logger.info("store_entry: compact DataFrame is empty; nothing to write.")
                return

            # Layer-aware writing strategy
            write_layered = df_compact["layer_index"].notna().any()
            if write_layered:
                total_rows = 0
                for layer_val, chunk in df_compact.groupby("layer_index", dropna=False):
                    if pd.isna(layer_val):
                        out_path = self.raw_results_path  # legacy bucket for null layer
                    else:
                        out_path = os.path.join(self.experiment_path, f"raw_results_layer_{int(layer_val)}.csv")

                    write_header = not os.path.exists(out_path)
                    chunk.to_csv(out_path, mode="a", index=False, header=write_header)
                    total_rows += len(chunk)

                    self.logger.info(
                        "store_entry: wrote %d compact rows → %s (header=%s)",
                        len(chunk), os.path.basename(out_path), write_header
                    )

                layers = sorted({int(layer_val) for layer_val in df_compact['layer_index'].dropna().unique()}) or [
                    "legacy"]
                self.logger.info(f"store_entry: layered write completed | layers={layers} | total_rows={total_rows}")

            else:
                out_path = self.raw_results_path
                write_header = not os.path.exists(out_path)
                df_compact.to_csv(out_path, mode="a", index=False, header=write_header)

                self.logger.info(
                    "store_entry: wrote %d compact rows → %s (header=%s)",
                    len(df_compact), os.path.basename(out_path), write_header
                )

        except Exception as e:
            self.logger.error("store_entry failed: %s", e, exc_info=True)
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

    # --- Metadata helpers -----------------------------------------------------
    def _model_threshold_map(self) -> dict:
        """
        Build a mapping {task_name -> distance_threshold} from `self.types`.

        Notes
        -----
        `task_name` here refers to the model key used across outputs (not the DB model_name).
        """
        try:
            return {info["task_name"]: info.get("distance_threshold") for info in self.types.values()}
        except Exception:
            return {}

    def _add_metadata_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Append run metadata to the output DataFrame.

        Adds
        ----
        - distance_metric : str
            The distance metric used in the run.
        - distance_threshold : float | None
            Per-row threshold resolved from the model's task_name.

        Side effects
        ------------
        If `conf['postprocess']['keep_sequences']` is False, drops
        'sequence_query' and 'sequence_reference' for leaner outputs.
        """
        df = df.copy()
        df["distance_metric"] = self.distance_metric

        thr_map = self._model_threshold_map()
        df["distance_threshold"] = df["model_name"].map(thr_map).astype(object)

        keep_seq = (self.conf.get("postprocess", {}) or {}).get("keep_sequences", False)
        if not keep_seq:
            df = df.drop(columns=["sequence_query", "sequence_reference"], errors="ignore")

        return df

    # --- Normalization utilities ----------------------------------------------
    def _safe_max(self, s: pd.Series) -> float:
        """
        Return the maximum positive value in the series, or NaN if none exists.
        """
        if s is None or s.empty:
            return float("nan")
        m = pd.to_numeric(s, errors="coerce").max()
        return m if pd.notnull(m) and m > 0 else float("nan")

    def _normalize_by_accession(self, df: pd.DataFrame, col: str) -> pd.Series:
        """
        Normalize a numeric column by 'accession', dividing by the positive max
        within each accession group. Returns 0 where no positive max exists.
        """

        def norm(group: pd.Series) -> pd.Series:
            m = self._safe_max(group)
            if not pd.notnull(m) or m == 0:
                return pd.Series(0.0, index=group.index)
            return group.fillna(0.0) / m

        return df.groupby("accession")[col].transform(norm)

    # --- SCORE (with layer_support) -------------------------------------------
    def _compute_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute a per-row composite 'score' from multiple normalized components.

        Components
        ----------
        reliability_index : float
            Derived from distance according to the selected metric.
        support_count_norm : float
            Normalized neighbor count supporting the GO term per accession.
        collapsed_support_norm : float
            Normalized ontological support from ancestors present in the group.
        model_consistency : float
            Agreement across models: model_support / n_models_total.
        alignment_norm : float
            0.5 * identity/100 + 0.5 * similarity/100 (if available, else 0).
        layer_support_norm : float
            Fraction of layers (of the same model) that support the term:
            (#layers predicting the GO / #layers available for that model & accession).

        Weights (override via conf['postprocess']['weights'])
        -----------------------------------------------------
        reliability_index=0.50, support_count_norm=0.20, collapsed_support_norm=0.15,
        model_consistency=0.10, alignment_norm=0.05, layer_support_norm=0.05
        """
        df = df.copy()

        # Cross-model consistency
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

        # Layer support (per model)
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

        # Normalizations by accession
        for c in ("support_count", "collapsed_support"):
            if c not in df.columns:
                df[c] = 0.0
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c + "_norm"] = self._normalize_by_accession(df, c)

        # Alignment (if present)
        for mcol in ("identity", "similarity"):
            if mcol not in df.columns:
                df[mcol] = 0.0
            df[mcol] = pd.to_numeric(df[mcol], errors="coerce").fillna(0.0)
        df["alignment_norm"] = 0.5 * (df["identity"] / 100.0) + 0.5 * (df["similarity"] / 100.0)

        # Weights
        wconf = (self.conf.get("postprocess", {}) or {}).get("weights", {})
        w_RI = float(wconf.get("reliability_index", 0.50))
        w_SC = float(wconf.get("support_count_norm", 0.20))
        w_CS = float(wconf.get("collapsed_support_norm", 0.15))
        w_MC = float(wconf.get("model_consistency", 0.10))
        w_AL = float(wconf.get("alignment_norm", 0.05))
        w_LS = float(wconf.get("layer_support_norm", 0.05))
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

    # --- Collapse to best (model, layer) --------------------------------------
    def _collapse_best_model_layer(self, df_scored: pd.DataFrame) -> pd.DataFrame:
        """
        For each accession, select the (model_name, layer_index) pair with the
        highest 'score_global' (max observed 'score' in that pair). Within the
        winning pair, drop duplicate GO terms keeping the row with highest 'score'.
        Adds 'best_model', 'best_layer', and 'score_global'.
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

        df_best = df_best.merge(best_by_acc[["accession", "score_global"]], on="accession", how="left")
        return df_best

    # --- Full post-processing pipeline ----------------------------------------
    def post_processing(self) -> None:
        """
        Post-process transferred GO annotations and produce final outputs.

        Designed to work efficiently with compact RAW created by `store_entry`
        (one row per (accession, model, layer, GO) + precomputed `support_count`).
        Falls back to legacy RAW if needed.

        Pipeline
        --------
        1) Load RAW:
           - Prefer layered files: raw_results_layer_*.csv (faster IO, selective reads).
           - Fallback: raw_results.csv (legacy).
           - Ensure `layer_index`.
        2) Compute `reliability_index` from `distance`:
           - cosine    → 1 - distance
           - euclidean → 0.5 / (0.5 + distance)
           - default   → 1 / (1 + distance)
        3) Ensure `support_count` (keep if present; else compute per group).
        4) Reduce to leaf GO terms per (accession, model, layer) using the GO DAG and
           compute collapsed support from ancestors in the group.
        5) (Optional) Pairwise alignment metrics if sequences are available.
        6) Append metadata, round numerics, sort for readability; clean legacy gene_name.
        7) Score with `_compute_score`; write `results_scored.csv`.
        8) Collapse to best (model, layer) per accession; write `results_collapsed.csv`
           and copy to `results.csv` (compatibility alias).
        9) Export TopGO TSVs (score-based) if enabled.
        """

        start_total = time.perf_counter()

        def log_stage(name: str, t0: float) -> None:
            self.logger.info("%s completed in %.2fs", name, time.perf_counter() - t0)

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
                self.logger.info("Loaded %d rows from layered RAW files.", len(df_raw))
        if df_raw is None:
            if not os.path.exists(self.raw_results_path):
                self.logger.warning("No RAW results found for post-processing.")
                return
            df_raw = pd.read_csv(self.raw_results_path)
            self.logger.info("Loaded %d rows from %s.", len(df_raw), os.path.basename(self.raw_results_path))
        if df_raw.empty:
            self.logger.warning("RAW results are empty. Nothing to post-process.")
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
        log_stage("Compute or keep support_count", t0)

        # 5) Build GO parents cache
        t0 = time.perf_counter()
        unique_go_ids = pd.unique(df_raw["go_id"])
        parents_cache: dict[str, set] = {}

        def get_parents_cached(term: str) -> set:
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

        # 6) Reduce to leaf terms + collapsed_support (fast)
        t0 = time.perf_counter()
        rows = []

        # Ensure go_id is string to match GO DAG keys
        df_raw["go_id"] = df_raw["go_id"].astype(str)

        for ((_acc, _model, _layer)), group in df_raw.groupby(["accession", "model_name", "layer_index"], dropna=False):
            group_go = pd.unique(group["go_id"])
            if len(group_go) == 0:
                continue
            group_go_set = set(group_go)

            # Best row per GO (highest reliability_index)
            best_per_go = (
                group.sort_values("reliability_index", ascending=False)
                .drop_duplicates(subset=["go_id"], keep="first")
                .copy()
            )

            # Compute ancestor sets only once per GO in this group
            parents_union = set()
            ancestors_in_group: dict[str, set] = {}
            for gid in group_go:
                parents = self.go[gid].get_all_parents() if gid in self.go else []
                pg = set(parents) & group_go_set
                ancestors_in_group[gid] = pg
                parents_union.update(pg)

            # Leafs are those not acting as parent of any other term in the group
            leaf_terms = [gid for gid in group_go if gid not in parents_union]
            if not leaf_terms:
                continue

            leaf_best = best_per_go[best_per_go["go_id"].isin(leaf_terms)].copy()
            if leaf_best.empty:
                continue

            # Map support_count from best_per_go (1 per GO)
            support_map = (
                best_per_go[["go_id", "support_count"]]
                .drop_duplicates("go_id")
                .set_index("go_id")["support_count"]
                .to_dict()
            )

            def _sum_support_ancestors(gid: str, _support_map=support_map, _anc=ancestors_in_group) -> int:
                return int(sum(_support_map.get(a, 0) for a in _anc.get(gid, set())))

            def _count_ancestors(gid: str, _anc=ancestors_in_group) -> int:
                return int(len(_anc.get(gid, set())))

            def _list_ancestors(gid: str, _anc=ancestors_in_group) -> str:
                anc = _anc.get(gid, set())
                return ", ".join(sorted(anc)) if anc else ""

            leaf_best["collapsed_support"] = leaf_best["go_id"].map(_sum_support_ancestors)
            leaf_best["n_collapsed_terms"] = leaf_best["go_id"].map(_count_ancestors)
            leaf_best["collapsed_terms"] = leaf_best["go_id"].map(_list_ancestors)

            rows.append(leaf_best)

        df_out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
            columns=df_raw.columns.tolist() + ["collapsed_support", "n_collapsed_terms", "collapsed_terms"]
        )
        log_stage("Reduce to leaves + collapsed_support", t0)

        # 7) Optional pairwise alignment metrics
        t0 = time.perf_counter()
        unique_pairs = (
            df_out[["sequence_query", "sequence_reference"]].drop_duplicates()
            if {"sequence_query", "sequence_reference"} <= set(df_out.columns) else pd.DataFrame()
        )
        if not unique_pairs.empty:
            from concurrent.futures import ProcessPoolExecutor
            metrics_list = []
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

        # Clean legacy gene_name shapes like "[{...}]"
        # Clean legacy gene_name shapes like "[{...}]"
        if "gene_name" in df_out.columns:
            def _extract_gene_name(g):
                try:
                    import ast
                    val = ast.literal_eval(g)
                    if (
                            isinstance(g, str) and
                            g.startswith("[{") and
                            isinstance(val, list) and
                            len(val) > 0 and
                            isinstance(val[0], dict) and
                            "Name" in val[0]
                    ):
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
        self.logger.info("Wrote %d rows → results_scored.csv", len(df_scored))
        log_stage("Compute SCORE + write results_scored.csv", t0)

        # 10) Collapse
        t0 = time.perf_counter()
        df_collapsed = self._collapse_best_overall(df_scored)
        results_collapsed_path = os.path.join(self.experiment_path, "results_collapsed.csv")
        df_collapsed.to_csv(results_collapsed_path, index=False)
        shutil.copyfile(results_collapsed_path, self.results_path)  # legacy alias
        self.logger.info("Wrote %d rows → results_collapsed.csv (aliased to results.csv)", len(df_collapsed))
        log_stage("Collapse best model+layer + write results_collapsed.csv + alias", t0)

        # 11) TopGO exports (score-based)
        t0 = time.perf_counter()
        if self.topgo_enabled:
            self.logger.info("Generating TopGO outputs (score-based)…")
            base_dir = os.path.join(self.experiment_path, "topgo")
            os.makedirs(base_dir, exist_ok=True)

            def build_topgo(df_group: pd.DataFrame) -> pd.DataFrame:
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

            self.logger.info("TopGO outputs generated.")
        log_stage("Generate TopGO outputs", t0)

        self.logger.info("Post-processing finished in %.2fs", time.perf_counter() - start_total)

    def _compact_annotations_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compact raw annotations *within the batch* to reduce disk size and speed up post-processing.

        Grouping keys (we DO NOT use 'go_description' as a key; it's redundant with 'go_id'):
            ('accession', 'model_name', 'embedding_type_id', 'layer_index',
             'go_id', 'category', 'evidence_code')

        For each group
        --------------
        - support_count := number of neighbors contributing that GO in the group
        - distance      := minimal distance in the group (best neighbor)
        - representative row := the row with minimal 'distance' (ties → first)
        - keep representative metadata (sequence_reference, protein_id, organism, gene_name, go_description, ...)

        Returns
        -------
        DataFrame with one row per group + 'support_count' and representative fields.
        """
        if df.empty:
            return df

        df = df.copy()

        expected = [
            "accession", "model_name", "embedding_type_id", "layer_index",
            "go_id", "category", "evidence_code", "go_description",
            "distance", "sequence_query", "sequence_reference",
            "protein_id", "organism", "gene_name"
        ]
        for c in expected:
            if c not in df.columns:
                df[c] = None

        # Dtypes for robust grouping/ordering
        df["embedding_type_id"] = pd.to_numeric(df["embedding_type_id"], errors="coerce").astype("Int64")
        df["layer_index"] = pd.to_numeric(df["layer_index"], errors="coerce").astype("Int64")
        df["distance"] = pd.to_numeric(df["distance"], errors="coerce")

        for col in ("accession", "model_name", "go_id", "category", "evidence_code"):
            df[col] = df[col].astype(str)
        for txt in ("go_description", "sequence_query", "sequence_reference", "organism", "gene_name"):
            df[txt] = df[txt].astype(str)

        keys = [
            "accession", "model_name", "embedding_type_id", "layer_index",
            "go_id", "category", "evidence_code"
        ]

        df["support_count"] = df.groupby(keys, dropna=False)["go_id"].transform("size")

        min_distance = df.groupby(keys, dropna=False)["distance"].transform("min")
        df["__is_best"] = df["distance"].eq(min_distance)

        best = (
            df[df["__is_best"]]
            .sort_values(keys + ["distance"])
            .drop_duplicates(subset=keys, keep="first")
            .copy()
        )

        best.drop(columns=["__is_best"], inplace=True, errors="ignore")

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
        Collapse predictions ignoring model and layer:
        keep, for each (accession, go_id), the row with the highest 'score'.
        Then expose 'best_model', 'best_layer', and 'score_global' from that winning row.
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

    def _get_lookup_for_batch(self, model_id: int, layer_index: int | None) -> dict | None:
        """
        Lazily build (and cache) the reference lookup table for a given (model_id, layer_index).

        Applies taxonomy filters and optional SQL LIMIT as configured.

        Returns
        -------
        dict | None
            {"ids": np.ndarray, "embeddings": np.ndarray, "layers": np.ndarray}
            or None if no rows match.
        """
        import numpy as np

        key = (int(model_id), None if layer_index is None else int(layer_index))
        if key in self._lookup_cache:
            return self._lookup_cache[key]

        def _as_str_list(xs):
            return [str(t) for t in (xs or [])]

        # usar los enteros preparados en __init__
        exclude_taxon_ids = _as_str_list(
            getattr(self, "exclude_taxon_ids", self.conf.get("taxonomy_ids_to_exclude", [])))
        include_taxon_ids = _as_str_list(
            getattr(self, "include_taxon_ids", self.conf.get("taxonomy_ids_included_exclusively", [])))

        limit_execution = self.conf.get("limit_execution")

        q = (
            self.session
            .query(
                Sequence.id,  # 0
                SequenceEmbedding.embedding,  # 1 (pgvector -> .to_numpy())
                SequenceEmbedding.layer_index  # 2
            )
            .join(Sequence, Sequence.id == SequenceEmbedding.sequence_id)
            .join(Protein, Sequence.id == Protein.sequence_id)
            .filter(SequenceEmbedding.embedding_type_id == key[0])
        )

        if key[1] is None:
            q = q.filter(SequenceEmbedding.layer_index.is_(None))
        else:
            q = q.filter(SequenceEmbedding.layer_index == key[1])

        if exclude_taxon_ids:
            q = q.filter(~Protein.taxonomy_id.in_(exclude_taxon_ids))
        if include_taxon_ids:
            q = q.filter(Protein.taxonomy_id.in_(include_taxon_ids))

        if isinstance(limit_execution, int) and limit_execution > 0:
            self.logger.info(
                "SQL LIMIT applied: %d for lookup(model_id=%s, layer_index=%s)",
                limit_execution, key[0], key[1]
            )
            q = q.limit(limit_execution)

        rows = q.all()
        if not rows:
            self.logger.warning("Empty lookup for model_id=%s, layer_index=%s.", key[0], key[1])
            return None

        ids = np.fromiter((r[0] for r in rows), dtype=int, count=len(rows))
        layers = np.fromiter((r[2] for r in rows), dtype=np.int64, count=len(rows))
        embeddings = np.vstack([r[1].to_numpy() for r in rows])

        lookup = {"ids": ids, "embeddings": embeddings, "layers": layers}

        self._lookup_cache[key] = lookup
        if len(self._lookup_cache) > self._lookup_cache_max:
            old_key = next(iter(self._lookup_cache.keys()))
            if old_key != key:
                self._lookup_cache.pop(old_key, None)

        self.logger.info(
            "Lookup loaded for model_id=%s, layer_index=%s | rows=%d | shape=%s",
            key[0], key[1], len(ids), embeddings.shape
        )
        return lookup

    def _h5_available_layers(self, model_id: int) -> list[int]:
        """
        Inspect the HDF5 file and return the sorted list of layer indices
        available under `type_{model_id}` across accessions.
        """
        import h5py
        layers = set()
        if not os.path.exists(self.embeddings_path):
            return []
        with h5py.File(self.embeddings_path, "r") as h5:
            for _, group in h5.items():
                type_key = f"type_{model_id}"
                if type_key not in group:
                    continue
                for k in group[type_key].keys():
                    if k.startswith("layer_"):
                        try:
                            layers.add(int(k.split("_", 1)[1]))
                        except Exception:
                            continue
        return sorted(layers)

    def load_model(self, model_type):
        return

    def unload_model(self, model_type):
        return
