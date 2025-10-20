"""
Embedding LookUp Module
========================================================================

Overview
--------
This module implements the :class:`EmbeddingLookUp` component for transferring Gene Ontology (GO)
annotations to query protein sequences based on vector similarity between sequence embeddings.

Given an HDF5 file of query embeddings, the component:
  1) Retrieves reference embeddings from a vector-aware relational database.
  2) Computes distances (GPU or CPU) to the references.
  3) Selects nearest neighbors under configurable thresholds and redundancy filters.
  4) Expands neighbors into GO-term annotations (preloaded from the DB).
  5) Persists results per query/model/layer and exports TopGO-ready TSV files.
  6) Optionally performs alignment-based post-processing to derive identity/similarity metrics.

Key features
------------
- Taxonomy filters (include/exclude lists; optional descendant expansion).
- Redundancy-aware neighbor selection using MMseqs2 clusters.
- Support for multiple embedding models with per-model thresholds and layer control.
- Distance computation on GPU (PyTorch) or CPU (SciPy).
- Post-processing pipeline (Polars/Pandas) with scoring and summary aggregation.
- TopGO exports by model/layer and ensemble across models.

Inputs
------
- Query embeddings in HDF5, organized by accession → type_{model_id} → layer_{k} → ``embedding``.
- Reference embeddings and GO annotations stored in the database.

Outputs
-------
- Hierarchical CSVs under ``<experiment_path>/raw_results/{model}/layer_{k}/<accession>.csv``.
- Global summary at ``<experiment_path>/summary.csv``.
- TopGO TSVs under ``<experiment_path>/topgo/{model}/layer_{k}/`` and ensemble under ``topgo/ensemble/``.
- A combined FASTA of all seen sequences at ``<experiment_path>/<sequences_fasta|sequences.fasta>``.

Configuration (selected keys)
-----------------------------
- ``experiment_path`` (str): Base directory for inputs/outputs.
- ``embeddings_path`` (str): Path to HDF5 with query embeddings.
- ``batch_size`` (int), ``limit_per_entry`` (int), ``precision`` (int).
- ``lookup`` (dict):
    - ``use_gpu`` (bool), ``batch_size`` (int), ``limit_per_entry`` (int), ``topgo`` (bool), ``lookup_cache_max`` (int)
    - ``distance_metric`` ({"euclidean","cosine"})
    - ``redundancy``: ``identity`` (float), ``coverage`` (float), ``threads`` (int)
    - ``taxonomy``: ``exclude`` (list[int]), ``include_only`` (list[int]), ``get_descendants`` (bool)
- ``embedding.models`` (dict per logical model name):
    - ``enabled`` (bool), ``distance_threshold`` (float|None),
      ``batch_size`` (int|None), ``layer_index`` (list[int]|None).
- ``postprocess`` (dict): ``keep_sequences`` (bool), ``store_workers`` (int),
  and ``summary`` spec with metric aggregation and weights.
- ``limit_execution`` (int|None): Optional SQL LIMIT for reference lookup.

Dependencies
------------
Relies on SQLAlchemy models for sequences, proteins and embeddings; GO terms are loaded with
``goatools``; distances can be computed with PyTorch or SciPy; aggregation uses Polars/Pandas.
MMseqs2 is required for redundancy clustering.

Reference
---------
Inspired by *GoPredSim* (Rostlab). See: https://github.com/Rostlab/goPredSim

"""

# --- Standard library ---
import os
import re
import subprocess
import tempfile
import time
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# --- Third-party libraries ---
import h5py
import numpy as np
import pandas as pd
import polars as pl
import torch
from goatools.base import get_godag
from scipy.spatial.distance import cdist
from sqlalchemy import text

# --- Project-specific imports ---
from fantasia.src.helpers.helpers import compute_metrics, get_descendant_ids
from protein_information_system.sql.model.entities.embedding.sequence_embedding import (
    SequenceEmbedding,
    SequenceEmbeddingType,
)
from protein_information_system.sql.model.entities.protein.protein import Protein
from protein_information_system.sql.model.entities.sequence.sequence import Sequence
from protein_information_system.tasks.gpu import GPUTaskInitializer


class EmbeddingLookUp(GPUTaskInitializer):
    """
    GO annotation transfer via embedding similarity.

    This component reads query embeddings (HDF5) and compares them against reference
    embeddings stored in a vector-aware relational database. For the closest reference
    sequences, it retrieves GO annotations and writes results to CSV (and optionally
    TopGO-ready TSV).

    Features
    --------
    - Taxonomy-based filtering (include/exclude, optional descendant expansion).
    - Redundancy-aware neighbor selection (MMseqs2 clusters).
    - Multiple embedding models with per-model distance thresholds and layer control.
    - Distance computation on GPU (PyTorch) or CPU (SciPy).
    - Optional pairwise alignment post-processing (identity/similarity).

    Parameters
    ----------
    conf : dict
        Runtime configuration including paths, thresholds, model settings, and processing options.
    current_date : str
        Timestamp-like suffix to version output artifacts.

    Notes
    -----
    - Supported distance metrics: ``"euclidean"`` and ``"cosine"`` (default: **"cosine"**).
    - Redundancy filtering uses MMseqs2 identity/coverage thresholds when enabled.
    - GO annotations are preloaded once from the database and may be constrained by taxonomy filters.
    """

    def __init__(self, conf, current_date):
        """Initialize the embedding-based GO annotation component.

        This constructor:
          - Normalizes configuration entries.
          - Initializes paths for embeddings, results, and optional exports.
          - Loads the Gene Ontology DAG.
          - Optionally generates non-redundant sequence clusters (via MMseqs2).
          - Sets up taxonomy filters with optional descendant expansion.
          - Prepares lazy reference lookups per (model, layer) using an in-memory cache.
          - Loads model definitions and preloads GO annotations.
          - Prepares internal structures for sequence indexing.

        Args:
            conf (dict): Run configuration. Relevant keys include:
                - experiment_path (str): Base folder for inputs/outputs.
                - embeddings_path (str): Path to the query HDF5 file
                  (defaults to ``<experiment_path>/embeddings.h5``).
                - lookup (dict): Optional nested section. If present, its fields are mapped
                  to the flat config (e.g., ``lookup.distance_metric``, ``lookup.batch_size``,
                  ``lookup.redundancy.identity``/``coverage``, taxonomy include/exclude, etc.).
                - embedding.distance_metric (str): ``"cosine"`` (default) or ``"euclidean"``.
                - batch_size (int): Maximum number of queries per batch.
                - limit_per_entry (int): Maximum neighbors retained per query.
                - topgo (bool): Whether to produce TopGO-ready exports later.
                - redundancy_filter (float): Identity threshold for MMseqs2 clustering;
                  requires ``alignment_coverage`` and ``threads``.
                - taxonomy_ids_to_exclude (list[int]): Optional taxonomy IDs to exclude.
                - taxonomy_ids_included_exclusively (list[int]): Optional allow-list.
                - get_descendants (bool): If True, expands provided taxonomy IDs to all
                  descendants before filtering.
                - lookup_cache_max (int): Max distinct (model, layer) lookups cached in RAM.
            current_date (str): Timestamp suffix for versioning artifacts.

        Side Effects:
            - Logs configuration and derived options.
            - Loads GO DAG from ``go-basic.obo`` with ``relationship`` attributes.
            - Initializes taxonomy filters (optionally expanded via descendants).
            - Optionally generates MMseqs2 clusters and caches cluster mappings.
            - Loads model definitions and caches GO annotations from the DB.
            - Prepares sequence indexers (for global FASTA later).

        Raises:
            Exception: Propagates failures when loading DB metadata, GO DAG, or
                building redundancy clusters.

        Notes:
            Unsupported ``distance_metric`` values are coerced to ``"cosine"`` with a warning.
        """

        super().__init__(conf)

        # --- Adapt references from conf['lookup'] ------------------------------
        lk = self.conf.get("lookup", {}) or {}

        # Copy direct options if not already in root
        for k in ("use_gpu", "batch_size", "limit_per_entry", "topgo", "lookup_cache_max"):
            if k not in self.conf and k in lk:
                self.conf[k] = lk[k]

        # Distance metric: allow lookup.distance_metric as well as embedding.distance_metric
        if "distance_metric" in lk:
            emb = self.conf.setdefault("embedding", {})
            emb.setdefault("distance_metric", lk["distance_metric"])

        # Redundancy: map lookup.redundancy -> flat keys
        r = lk.get("redundancy") or {}
        if "redundancy_filter" not in self.conf and "identity" in r:
            self.conf["redundancy_filter"] = r["identity"]
        if "alignment_coverage" not in self.conf and "coverage" in r:
            self.conf["alignment_coverage"] = r["coverage"]
        if "threads" not in self.conf and "threads" in r:
            self.conf["threads"] = r["threads"]

        # Taxonomy: map lookup.taxonomy -> flat keys
        t = lk.get("taxonomy") or {}
        if "exclude" in t and not self.conf.get("taxonomy_ids_to_exclude"):
            self.conf["taxonomy_ids_to_exclude"] = t["exclude"]
        if "include_only" in t and not self.conf.get("taxonomy_ids_included_exclusively"):
            self.conf["taxonomy_ids_included_exclusively"] = t["include_only"]
        if "get_descendants" in t and self.conf.get("get_descendants") in (None, ""):
            self.conf["get_descendants"] = t["get_descendants"]

        # -----------------------------------------------------------------------

        self.types = None
        self.current_date = current_date
        self.logger.info("EmbeddingLookUp: initializing component...")

        # ---- Paths ------------------------------------------------------------
        self.experiment_path = self.conf.get("experiment_path")
        self.embeddings_path = self.conf.get("embeddings_path") or os.path.join(self.experiment_path, "embeddings.h5")
        self.raw_results_path = os.path.join(self.experiment_path, "raw_results.csv")
        self.results_path = os.path.join(self.experiment_path, "results.csv")
        self.topgo_path = os.path.join(self.experiment_path, "results_topgo.tsv")

        # ---- Limits & options -------------------------------------------------
        self.limit_per_entry = self.conf.get("limit_per_entry", 200)
        self.topgo_enabled = self.conf.get("topgo", False)
        self.batch_size = self.conf.get("batch_size", 1)

        # ---- Optional redundancy filtering -----------------------------------
        redundancy_filter_threshold = self.conf.get("redundancy_filter", 0)
        if redundancy_filter_threshold > 0:
            self.logger.info(
                "Redundancy filter enabled (MMseqs2): identity >= %.3f, coverage >= %.3f, threads = %s",
                float(self.conf.get("redundancy_filter", 0)),
                float(self.conf.get("alignment_coverage", 0)),
                int(self.conf.get("threads", 12)),
            )
            self.generate_clusters()
        else:
            self.logger.info("Redundancy filter disabled.")

        # ---- GO ontology ------------------------------------------------------
        self.go = get_godag("go-basic.obo", optional_attrs="relationship")
        self.logger.info("GO DAG loaded successfully from go-basic.obo.")

        # ---- Distance metric --------------------------------------------------
        self.distance_metric = self.conf.get("embedding", {}).get("distance_metric", "cosine")
        if self.distance_metric not in ("euclidean", "cosine"):
            self.logger.warning(
                "Unsupported distance metric '%s'; falling back to 'cosine'.", self.distance_metric
            )
            self.distance_metric = "cosine"
        self.logger.info("Distance metric set to: %s", self.distance_metric)

        # ---- Taxonomy filters (integers; optional descendant expansion) -------
        def _expand_tax_ids(ids):
            ids = ids or []
            clean = [int(t) for t in ids if str(t).isdigit()]
            if self.conf.get("get_descendants", False) and clean:
                return [int(t) for t in get_descendant_ids(clean)]
            return clean

        self.exclude_taxon_ids = _expand_tax_ids(self.conf.get("taxonomy_ids_to_exclude"))
        self.include_taxon_ids = _expand_tax_ids(self.conf.get("taxonomy_ids_included_exclusively"))
        self.logger.info(
            "Taxonomy filters initialized | exclude = %s | include = %s | expand_descendants = %s",
            self.exclude_taxon_ids or "[]",
            self.include_taxon_ids or "[]",
            bool(self.conf.get("get_descendants", False)),
        )

        # ---- Lazy reference lookup cache -------------------------------------
        # key: (model_id: int, layer_index: Optional[int]) -> {"ids": np.ndarray, "embeddings": np.ndarray, "layers": np.ndarray}
        self._lookup_cache: dict[tuple[int, int | None], dict] = {}
        self._lookup_cache_max = int(self.conf.get("lookup_cache_max", 4))
        self.logger.info(
            "Reference lookup will be loaded lazily per (model, layer) with an in-memory cache of max %d entry(ies).",
            self._lookup_cache_max,
        )

        # ---- Load model definitions & preload GO annotations -----------------
        self.load_model_definitions()
        self.logger.info("Loaded %d model definitions from DB+config: %s",
                         len(self.types or {}), list(self.types.keys()) if self.types else [])
        self.logger.info("Preloading GO annotations from the database...")
        self.preload_annotations()
        self.logger.info("GO annotations cached: %d sequences with annotations.",
                         len(getattr(self, "go_annotations", {})))

        self.logger.info("EmbeddingLookUp initialization completed successfully.")

        # ---- Sequence indexing structures ------------------------------------
        self._q_seq_to_idx: dict[str, int] = {}
        self._q_idx_to_seq: list[str] = []
        self._r_seq_to_idx: dict[str, int] = {}
        self._r_idx_to_seq: list[str] = []

        # FASTA path (configurable via conf['sequences_fasta'])
        self.sequences_fasta_path = os.path.join(
            self.experiment_path,
            self.conf.get("sequences_fasta", "sequences.fasta")
        )

        # Numerical precision for results
        self.precision = self.conf.get("precision", 4)

    def enqueue(self):
        """Scan the query HDF5 file and publish homogeneous batches of embeddings.

        This method iterates over accessions in the query HDF5, groups items by
        (embedding_type_id, layer_index), and publishes batches up to ``batch_size``.
        Each task contains lightweight HDF5 pointers (``h5_path`` + ``h5_group``),
        the resolved model metadata, and the per-model distance threshold.

        Behavior:
            * Skips accessions missing the ``sequence`` dataset.
            * Supports both layered embeddings (``.../type_<id>/layer_<k>``)
              and legacy embeddings without layers (layer_index=None).
            * Emits one payload per group of tasks with the following schema:
                ``{"model_id": int, "layer_index": Optional[int], "tasks": [ ... ]}``.

        Logging:
            - Reports the number of queries encountered.
            - Reports the number of published batches (based on group size and ``batch_size``).
            - Warns when encountering malformed groups or missing sequences.

        Raises:
            FileNotFoundError: If the configured HDF5 file does not exist.
            Exception: Propagates any unexpected errors during processing.
        """

        self.logger.info("Starting query enqueue process for embedding-based GO annotation.")
        self.logger.info("Loading query embeddings from HDF5 file: %s", self.embeddings_path)

        if not os.path.exists(self.embeddings_path):
            raise FileNotFoundError(
                f"HDF5 file not found: {self.embeddings_path}. "
                "Ensure embeddings have been generated prior to running annotation."
            )

        batch_size = int(self.batch_size)
        total_entries = 0
        total_batches = 0

        # Buffers: map (embedding_type_id, layer_index) -> list of task dicts
        buffers = defaultdict(list)

        # Quick lookup by model_id (avoids repeated lookups by accession)
        # self.types: keys = task_name, values = {"id", "task_name", "distance_threshold", ...}
        by_id = {info["id"]: info for info in self.types.values()}

        def flush(key):
            """Publish buffered tasks for a given (model_id, layer_index)."""
            nonlocal total_batches
            buf = buffers[key]
            if not buf:
                return
            for i in range(0, len(buf), batch_size):
                chunk = buf[i:i + batch_size]
                # All tasks in the chunk share model_id and layer_index
                model_id = chunk[0]["embedding_type_id"]
                layer_index = chunk[0].get("layer_index")
                payload = {
                    "model_id": model_id,
                    "layer_index": layer_index,
                    "tasks": chunk,
                }
                model_type = chunk[0]["model_name"]  # e.g. "esm2", "prott5", etc.
                self.publish_task(payload, model_type=model_type)
                total_batches += 1
            buffers[key].clear()

        try:
            with h5py.File(self.embeddings_path, "r") as h5file:
                for accession, group in h5file.items():
                    if "sequence" not in group:
                        self.logger.warning("Missing sequence dataset for accession '%s'. Skipping.", accession)
                        continue

                    # Iterate over type_* groups
                    for type_key, type_grp in group.items():
                        if not type_key.startswith("type_"):
                            continue

                        try:
                            model_id = int(type_key.split("_", 1)[1])
                        except Exception:
                            self.logger.warning("Malformed type group '%s'. Skipping.", type_key)
                            continue

                        model_info = by_id.get(model_id)
                        if model_info is None:
                            # Model not enabled in config or not loaded into self.types
                            self.logger.info("Model id %s not enabled or not loaded. Skipping.", model_id)
                            continue

                        model_name = model_info["task_name"]
                        distance_threshold = model_info.get("distance_threshold")

                        # Check for layers
                        layer_keys = [k for k in type_grp.keys() if k.startswith("layer_")]

                        if layer_keys:
                            for lk in sorted(layer_keys, key=lambda x: int(x.split("_", 1)[1])):
                                layer_grp = type_grp[lk]
                                if "embedding" not in layer_grp:
                                    continue
                                try:
                                    layer_index = int(lk.split("_", 1)[1])
                                except Exception:
                                    self.logger.warning(
                                        "Malformed layer group '%s' under type group '%s'. Skipping.",
                                        lk, type_key
                                    )
                                    continue

                                # Layer filtering based on config
                                enabled_layers = model_info.get("enabled_layers")
                                if enabled_layers and isinstance(enabled_layers, (list, tuple)):
                                    if layer_index not in enabled_layers:
                                        self.logger.debug(
                                            "Skipping layer %d for model %s "
                                            "(not in enabled_layers=%s).",
                                            layer_index, model_name, enabled_layers
                                        )
                                        continue

                                task = {
                                    "h5_path": self.embeddings_path,  # to reopen file in workers
                                    "h5_group": f"{accession}/{type_key}/{lk}",  # internal HDF5 path to embedding
                                    "embedding_type_id": model_id,
                                    "model_name": model_name,
                                    "distance_threshold": distance_threshold,
                                    "layer_index": layer_index,
                                }

                                key = (model_id, layer_index)
                                buffers[key].append(task)
                                total_entries += 1

                                # Flush immediately if buffer reached batch_size
                                if len(buffers[key]) >= batch_size:
                                    flush(key)

            # Flush remaining tasks
            for key in list(buffers.keys()):
                flush(key)

            self.logger.info(
                "Enqueued %d queries into %d homogeneous batches "
                "(grouped by model & layer; batch_size=%d).",
                total_entries, total_batches, batch_size
            )

        except Exception as e:
            self.logger.error("Unexpected error during enqueue: %s", e, exc_info=True)
            raise

    def load_model_definitions(self):
        """Initialize `self.types` by matching DB embedding types with configuration.

        Behavior:
            - Queries available embedding types from the database.
            - Matches DB models with those defined in the configuration.
            - Keeps only models that appear in both sources and are marked as enabled.
            - For each model, inspects available layers in the HDF5 file (if present)
              and determines effective layers to be used.

        Logging:
            - Skips if a DB model is missing in config or is disabled.
            - Warns if no enabled models remain after matching.
            - For each enabled model, logs:
                * model name and DB id
                * distance_threshold (from config)
                * enabled_layers (from config, or ALL if unrestricted)
                * available_layers_in_h5 (discovered by scanning the HDF5 file)
                * effective_layers (intersection if restricted, else ALL)

        Raises:
            Exception: If database query fails.
        """
        self.types = {}

        try:
            db_models = self.session.query(SequenceEmbeddingType).all()
        except Exception as e:
            self.logger.error("Failed to query SequenceEmbeddingType table: %s", e)
            raise

        cfg_models = self.conf.get("embedding", {}).get("models", {})

        # 1) Build self.types with models present in both DB and config
        for db_model in db_models:
            task_name = db_model.name  # Display name in DB
            matched_name = next((k for k in cfg_models if k.lower() == task_name.lower()), None)
            if matched_name is None:
                self.logger.warning("Model '%s' exists in DB but not in config. Skipping.", task_name)
                continue

            cfg = cfg_models[matched_name]
            if not cfg.get("enabled", True):
                self.logger.info("Model '%s' is disabled in config. Skipping.", matched_name)
                continue

            self.types[matched_name] = {
                "id": db_model.id,
                "model_name": db_model.model_name,
                "task_name": matched_name,
                "distance_threshold": cfg.get("distance_threshold"),
                "batch_size": cfg.get("batch_size"),
                "enabled_layers": cfg.get("layer_index"),  # may be None / not specified
            }

        # 2) Logging with single pass to avoid duplicates
        if not self.types:
            self.logger.warning("No enabled models found after matching DB and config.")
            return

        # Sort by model name for consistent logging
        for name in sorted(self.types.keys(), key=str.lower):
            info = self.types[name]
            model_id = info["id"]
            threshold = info.get("distance_threshold")
            enabled_layers = info.get("enabled_layers")

            # Scan available layers in HDF5 (may return empty if file/groups are missing)
            available_layers = self._h5_available_layers(model_id)

            # Determine effective layers
            if enabled_layers and isinstance(enabled_layers, (list, tuple)):
                if available_layers:
                    effective_layers = sorted(set(enabled_layers) & set(available_layers))
                else:
                    effective_layers = sorted(set(enabled_layers))
            else:
                effective_layers = "ALL" if available_layers else "[]"

            self.logger.info(
                "Model '%s' (id=%s): threshold=%s | enabled_layers=%s | available_layers_in_h5=%s | effective_layers=%s",
                name, model_id,
                threshold if threshold is not None else "None",
                enabled_layers if enabled_layers else "ALL",
                available_layers if available_layers else "[]",
                effective_layers
            )

        self.logger.info("Successfully loaded %d model(s) from DB+config: %s",
                         len(self.types), list(sorted(self.types.keys(), key=str.lower)))

    def process(self, payload: dict) -> list[dict]:
        """
        Compute nearest neighbors for a homogeneous (model_id, layer) batch.

        This method:

        - Loads query embeddings from HDF5 (or directly from payload in legacy mode).
        - Computes distances against the cached reference matrix for the given
          (model_id, layer_index).
        - Optionally applies redundancy filtering (exclude cluster members).
        - Selects nearest neighbors based on distance threshold and per-entry limit.
        - Returns a compact list of neighbor hits.

        Expected Payload
        ----------------
        ::

            {
              "model_id": int,
              "layer_index": Optional[int],
              "tasks": [
                {
                  "h5_path": str,
                  "h5_group": str,
                  "model_name": str,
                  "distance_threshold": Optional[float],
                  "layer_index": Optional[int]
                },
                # Legacy form (discouraged):
                {"embedding": np.ndarray, "accession": str, ...}
              ]
            }

        Distance computation
        --------------------
        - Uses GPU (PyTorch) if ``conf['use_gpu']`` is True.
        - Otherwise falls back to CPU (SciPy ``cdist``).
        - ``self.distance_metric`` must be either ``"cosine"`` or ``"euclidean"``.

        Redundancy
        ----------
        - If MMseqs2 clustering is configured, neighbors belonging to the same
          redundancy cluster as the query accession are excluded.

        Neighbor selection
        ------------------
        - Results are sorted by ascending distance.
        - Filtered by per-model ``distance_threshold`` (if provided).
        - Truncated to ``limit_per_entry``.

        Returns
        -------
        list[dict]
            One record per selected reference neighbor, with fields:

            - accession (str)
            - ref_sequence_id (int)
            - distance (float)
            - model_name (str)
            - embedding_type_id (int)
            - layer_index (Optional[int])

        Notes
        -----
        GO expansion and sequence retrieval are deferred to :meth:`store_entry`
        to keep payloads minimal.
        """

        t_start = time.perf_counter()

        try:
            # --- Validate payload ------------------------------------------------
            if not isinstance(payload, dict):
                self.logger.error("process(): expected a dict with keys ['model_id','layer_index','tasks'].")
                return []

            model_id = payload.get("model_id")
            layer_index_batch = payload.get("layer_index")
            batch = payload.get("tasks") or []

            if model_id is None or not isinstance(batch, list) or not batch:
                self.logger.error("process(): invalid or empty payload.")
                return []

            # Optional metadata
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
                    "process(): no reference lookup found for model_id=%s, layer_index=%s. Skipping batch.",
                    model_id, layer_index_batch
                )
                return []

            # --- Materialize query embeddings ------------------------------------
            embeddings_list = []
            accessions_list = []

            # Group tasks by file to minimize HDF5 open/close operations
            by_h5 = defaultdict(list)
            for t in batch:
                if "h5_path" in t and "h5_group" in t:
                    by_h5[t["h5_path"]].append(t)
                else:
                    by_h5[None].append(t)  # Legacy payload

            for h5_path, items in by_h5.items():
                with h5py.File(h5_path, "r") as h5:
                    for t in items:
                        grp_path = t["h5_group"]  # e.g., "accession_X/type_1/layer_16"
                        emb = h5[grp_path]["embedding"][:]  # Load embedding only
                        embeddings_list.append(np.asarray(emb))
                        # Extract accession from top-level node
                        acc_node = grp_path.split("/", 1)[0]  # "accession_X"
                        acc = acc_node.removeprefix("accession_")
                        accessions_list.append(acc)

            if not embeddings_list:
                self.logger.warning("process(): no query embeddings materialized. Skipping batch.")
                return []

            embeddings = np.stack(embeddings_list)  # shape (N, D)
            accessions = accessions_list
            layer_indices = [layer_index_batch] * len(accessions)

            self.logger.info(
                "Queries materialized | N=%d | embedding_dim=%s",
                len(embeddings), tuple(embeddings.shape[1:])
            )

            # --- Distance computation (GPU/CPU) ----------------------------------
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
            redundancy = float(self.conf.get("redundancy_filter", 0))
            redundant_ids: dict[str, set] = {}
            if redundancy > 0:
                for acc in accessions:
                    redundant_ids[acc] = self.retrieve_cluster_members(acc)
                self.logger.info(
                    "Redundancy filter applied | threshold=%.3f | accessions_with_clusters=%d",
                    float(redundancy), sum(1 for acc in accessions if redundant_ids.get(acc))
                )

            # --- Neighbor selection → compact hits -------------------------------
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
                if threshold not in (None, 0):
                    order = order[distances[order] <= float(threshold)]

                selected = order[:limit]
                total_neighbors += len(selected)
                li = layer_indices[i]

                for idx in selected:
                    hits.append({
                        "accession": accession,
                        "ref_sequence_id": int(seq_ids[idx]),
                        "distance": float(distances[idx]),
                        "model_name": model_name,
                        "embedding_type_id": model_id,
                        "layer_index": li,
                    })

            elapsed = time.perf_counter() - t_start
            avg_neighbors = (total_neighbors / len(accessions)) if accessions else 0.0
            self.logger.info(
                "Batch complete | queries=%d | layer=%s | neighbors=%d (avg=%.2f) | hits=%d | elapsed=%.2fs",
                len(accessions), layer_index_batch, total_neighbors, avg_neighbors, len(hits), elapsed
            )

            return hits

        except Exception as e:

            self.logger.error("process() failed: %s\n%s", e, traceback.format_exc())
            raise

    def store_entry(self, annotations_or_hits: list[dict]) -> None:
        """Persist per-(model, layer) results and update the global FASTA index.

        Input:
            Either
              (a) compact neighbor *hits* produced by :meth:`process` (preferred), or
              (b) legacy, already-expanded annotation rows.

        Pipeline:
            1) If input are compact hits, expand them into per-GO rows using
               preloaded ``self.go_annotations``; lazily read the query sequence
               from HDF5 (once per accession) and fetch the reference sequence from
               the cache.
            2) Cast types with Polars, compute a reliability index from distance
               (metric-dependent), and optionally compute pairwise alignment metrics
               (identity, similarity, etc.) when both sequences are available.
            3) Append execution metadata (distance metric and per-model threshold),
               then write CSV shards hierarchically under:
               ``raw_results/{model_name}/layer_{k or 'legacy'}/{accession}.csv``.
            4) Update a global FASTA containing all queries (``>Q{idx}``) and
               references (``>R{idx}``) with stable indices for downstream tools.

        Configuration:
            - ``postprocess.keep_sequences`` (bool): keep or drop raw sequences in CSV.
            - ``precision`` (int): float formatting precision for CSV output.

        Raises:
            Exception: Propagates unexpected errors after logging context.
            Partial writes may remain for accessions processed before the failure.
        """

        items = annotations_or_hits or []
        if not items:
            self.logger.info("store_entry: no annotations or hits to persist.")
            return

        try:
            os.makedirs(self.experiment_path, exist_ok=True)

            # ------------------------------------------------------------------
            # 1) Determine input type: compact hits (preferred) or legacy rows
            # ------------------------------------------------------------------
            is_hits = (
                    isinstance(items, list) and
                    items and
                    isinstance(items[0], dict) and
                    "ref_sequence_id" in items[0]
            )

            # Config options
            keep_sequences = bool((self.conf.get("postprocess", {}) or {}).get("keep_sequences", False))
            store_workers = int(self.conf.get("store_workers", 4))
            h5 = None  # used for query sequence retrieval when hits are provided

            def _sanitize(name: str) -> str:
                """Normalize a string into a filesystem-safe tag."""
                name = str(name or "").strip().lower()
                name = re.sub(r"\s+", "_", name)
                name = re.sub(r"[^a-z0-9._-]", "_", name)
                return name or "model"

            if is_hits:
                # --------------------------------------------------------------
                # 1.a) Expand hits into annotation rows
                # --------------------------------------------------------------
                expanded_rows: list[dict] = []

                # Prepare HDF5 for query sequence retrieval
                if os.path.exists(self.embeddings_path):
                    try:
                        h5 = h5py.File(self.embeddings_path, "r")
                    except Exception as e:
                        self.logger.warning("store_entry: could not open HDF5 for sequence read: %s", e)
                        h5 = None
                else:
                    self.logger.warning("store_entry: embeddings HDF5 not found at %s", self.embeddings_path)

                # Local cache to avoid redundant HDF5 reads
                qseq_cache: dict[str, str | None] = {}

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

                    # Query sequence (cached or from HDF5)
                    seq_query = qseq_cache.get(acc)
                    if seq_query is None:
                        seq_query = None
                        if h5 is not None:
                            acc_node = f"accession_{acc}"
                            try:
                                if acc_node in h5 and "sequence" in h5[acc_node]:
                                    raw_seq = h5[acc_node]["sequence"][()]
                                    seq_query = (
                                        raw_seq.decode("utf-8") if hasattr(raw_seq, "decode") else str(raw_seq)
                                    )
                            except Exception as e:
                                self.logger.debug("store_entry: sequence read failed for %s: %s", acc, e)
                                seq_query = None
                        qseq_cache[acc] = seq_query  # cache even if None

                    q_idx = self._index_sequence(seq_query, "Q") if seq_query else None

                    for ann in anns:
                        seq_ref = ann.get("sequence")  # from preload_annotations()
                        r_idx = self._index_sequence(seq_ref, "R") if seq_ref else None

                        expanded_rows.append({
                            "accession": acc,
                            "model_name": model_name,
                            "embedding_type_id": model_id,
                            "layer_index": li,
                            "distance": d,
                            "go_id": ann["go_id"],
                            "category": ann["category"],
                            "evidence_code": ann["evidence_code"],
                            "go_description": ann["go_description"],
                            "protein_id": ann["protein_id"],
                            "organism": ann["organism"],
                            "gene_name": ann["gene_name"],
                            "query_idx": q_idx,
                            "ref_idx": r_idx,
                            "query_len": len(seq_query) if seq_query else None,
                            "ref_len": len(seq_ref) if seq_ref else None,
                            "sequence_query": seq_query,
                            "sequence_reference": seq_ref,
                        })

                if h5 is not None:
                    try:
                        h5.close()
                    except Exception:
                        pass

                if not expanded_rows:
                    self.logger.info("store_entry: no rows produced after hit expansion.")
                    return

                df = pd.DataFrame(expanded_rows)

            else:
                # --------------------------------------------------------------
                # 1.b) Legacy: rows are already expanded
                # --------------------------------------------------------------
                df = pd.DataFrame(items)
                if not df.empty:
                    for c in ("sequence_query", "sequence_reference"):
                        if c not in df.columns:
                            df[c] = None

                    q_idx_list, r_idx_list, q_len, r_len = [], [], [], []
                    for q, r in zip(df["sequence_query"], df["sequence_reference"]):
                        qi = self._index_sequence(q, "Q") if isinstance(q, str) and q else None
                        ri = self._index_sequence(r, "R") if isinstance(r, str) and r else None
                        q_idx_list.append(qi)
                        r_idx_list.append(ri)
                        q_len.append(len(q) if isinstance(q, str) else None)
                        r_len.append(len(r) if isinstance(r, str) else None)

                    df["query_idx"] = q_idx_list
                    df["ref_idx"] = r_idx_list
                    df["query_len"] = q_len
                    df["ref_len"] = r_len

            if df.empty:
                self.logger.info("store_entry: dataframe is empty after processing.")
                return

            # ----------------------------------------------------------------------
            # 2) Polars: type casting, reliability index, and alignment metrics
            # ----------------------------------------------------------------------
            pl_df = pl.from_pandas(df).with_columns(
                pl.col("distance").cast(pl.Float64, strict=False),
                pl.col("layer_index").cast(pl.Int64, strict=False),
                pl.col("embedding_type_id").cast(pl.Int64, strict=False),
                pl.col("query_idx").cast(pl.Int64, strict=False),
                pl.col("ref_idx").cast(pl.Int64, strict=False),
                pl.col("query_len").cast(pl.Int64, strict=False),
                pl.col("ref_len").cast(pl.Int64, strict=False),
            )

            # Reliability index
            if self.distance_metric == "cosine":
                pl_df = pl_df.with_columns((1 - pl.col("distance")).alias("reliability_index"))
            elif self.distance_metric == "euclidean":
                pl_df = pl_df.with_columns((0.5 / (0.5 + pl.col("distance"))).alias("reliability_index"))
            else:
                pl_df = pl_df.with_columns((1.0 / (1.0 + pl.col("distance"))).alias("reliability_index"))

            # Alignment metrics if sequences available
            have_seq_cols = set(pl_df.columns) >= {"sequence_query", "sequence_reference"}
            if have_seq_cols:
                pairs = (
                    pl_df.select(["sequence_query", "sequence_reference", "model_name", "layer_index"])
                    .drop_nulls(subset=["sequence_query", "sequence_reference"])
                    .unique()
                    .to_dicts()
                )
                metrics_list = []
                if pairs:
                    with ProcessPoolExecutor(max_workers=store_workers) as ex:
                        metrics_list = list(ex.map(compute_metrics, pairs))
                if metrics_list:
                    met = pl.DataFrame(metrics_list)
                    merge_cols = ["sequence_query", "sequence_reference"]
                    if "model_name" in met.columns:
                        merge_cols.append("model_name")
                    if "layer_index" in met.columns:
                        merge_cols.append("layer_index")
                    pl_df = pl_df.join(met, on=merge_cols, how="left")

            # Robust casting of alignment metric columns
            for c in (
                    "identity", "similarity", "alignment_score", "gaps_percentage",
                    "identity_sw", "similarity_sw", "alignment_score_sw", "gaps_percentage_sw",
                    "alignment_length", "alignment_length_sw",
            ):
                if c in pl_df.columns:
                    pl_df = pl_df.with_columns(pl.col(c).cast(pl.Float64, strict=False))

            # Drop sequences unless explicitly requested
            if not keep_sequences:
                pl_df = pl_df.drop([c for c in ("sequence_query", "sequence_reference") if c in pl_df.columns])

            # Add execution metadata
            df = pl_df.to_pandas()
            df = self._add_metadata_columns(df)

            # ----------------------------------------------------------------------
            # 3) Hierarchical CSV output: per accession, model, and layer
            # ----------------------------------------------------------------------
            total_rows = 0
            grouped = df.groupby(["accession", "layer_index", "model_name"], dropna=False)

            for (accession, layer_val, model_name), chunk in grouped:
                model_tag = _sanitize(model_name)
                acc_tag = _sanitize(accession)

                if pd.isna(layer_val):
                    dir_out = os.path.join(self.experiment_path, "raw_results", model_tag, "legacy")
                else:
                    dir_out = os.path.join(self.experiment_path, "raw_results", model_tag, f"layer_{int(layer_val)}")

                os.makedirs(dir_out, exist_ok=True)
                out_path = os.path.join(dir_out, f"{acc_tag}.csv")

                write_header = not os.path.exists(out_path)
                float_fmt = f"%.{self.precision}f"
                chunk.to_csv(
                    out_path,
                    mode="a",
                    index=False,
                    header=write_header,
                    float_format=float_fmt
                )
                total_rows += len(chunk)

                self.logger.info("store_entry: wrote %d rows → %s", len(chunk), out_path)

            models = sorted({str(m) for m in df["model_name"].unique()}) if "model_name" in df else []
            layers = (
                sorted({int(layer) for layer in df["layer_index"].dropna().unique()})
                if "layer_index" in df
                else ["legacy"]
            )

            self.logger.info(
                "store_entry: write completed | models=%s | layers=%s | total_rows=%d",
                models or ["-"], layers or ["legacy"], total_rows
            )

            # ----------------------------------------------------------------------
            # 4) Update the global FASTA index (Q*/R*)
            # ----------------------------------------------------------------------
            try:
                fasta_path = self._flush_sequences_fasta()
                self.logger.info("store_entry: sequences FASTA updated at %s", fasta_path)
            except Exception as e:
                self.logger.warning("store_entry: could not update sequences FASTA: %s", e)

        except Exception as e:
            self.logger.error("store_entry failed: %s", e, exc_info=True)
            raise

    def generate_clusters(self):
        """
        Generate non-redundant sequence clusters using MMseqs2.

        Steps:
            1. Collect all protein sequences from the database and the embeddings HDF5.
            2. Write them into a temporary FASTA file.
            3. Run MMseqs2 `createdb`, `cluster`, and `createtsv` with the configured thresholds.
            4. Load the resulting cluster assignments into in-memory structures.

        Outputs:
            - self.clusters: pandas DataFrame of raw cluster assignments.
            - self.clusters_by_id: pandas DataFrame indexed by sequence ID → cluster ID.
            - self.clusters_by_cluster: dict mapping cluster ID → set of sequence IDs.

        Configuration:
            - redundancy_filter (float): sequence identity threshold.
            - alignment_coverage (float): alignment coverage threshold.
            - threads (int): number of threads to use.

        Raises:
            Exception: If MMseqs2 fails or the clustering pipeline encounters an error.
        """

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

                # --------------------------------------------------------------
                # 1) Write all sequences (DB + HDF5) into FASTA
                # --------------------------------------------------------------
                self.logger.info("📄 Preparing FASTA file for MMseqs2 clustering...")
                with open(fasta_path, "w") as fasta:
                    # Database sequences
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

                # --------------------------------------------------------------
                # 2) Run MMseqs2 clustering
                # --------------------------------------------------------------
                self.logger.info(
                    "⚙️ Running MMseqs2 clustering (identity=%.3f, coverage=%.3f, threads=%d)...",
                    float(identity), float(coverage), int(threads)
                )
                subprocess.run(["mmseqs", "createdb", fasta_path, db_path], check=True)
                subprocess.run([
                    "mmseqs", "cluster", db_path, clu_path, tmp_path,
                    "--min-seq-id", str(identity),
                    "--cov-mode", "1", "-c", str(coverage),
                    "--threads", str(threads)
                ], check=True)
                subprocess.run(["mmseqs", "createtsv", db_path, db_path, clu_path, tsv_path], check=True)

                # --------------------------------------------------------------
                # 3) Load clustering results
                # --------------------------------------------------------------
                df = pd.read_csv(tsv_path, sep="\t", names=["cluster", "identifier"])
                self.clusters = df
                self.clusters_by_id = df.set_index("identifier")
                self.clusters_by_cluster = df.groupby("cluster")["identifier"].apply(set).to_dict()

                # Save clusters to experiment folder
                out_path = os.path.join(self.experiment_path, "clusters.tsv")
                df.to_csv(out_path, sep="\t", index=False)

                self.logger.info("✅ MMseqs2 clustering completed: %d clusters written to %s",
                                 len(self.clusters_by_cluster), out_path)

        except Exception as e:
            self.logger.error("❌ MMseqs2 clustering failed: %s", e, exc_info=True)
            raise

    def retrieve_cluster_members(self, accession: str) -> set:
        """
        Retrieve all sequence IDs belonging to the same MMseqs2 cluster as the given accession.

        Parameters
        ----------
        accession : str
            Sequence ID used in clustering (must match the identifier in the FASTA header).

        Returns
        -------
        set of str
            Set of sequence IDs in the same cluster.
            Returns an empty set if the accession is not found or has no cluster members.
        """
        try:
            cluster_id = self.clusters_by_id.loc[accession, "cluster"]
            members = self.clusters_by_cluster.get(cluster_id, set())
            clean_members = {m for m in members if m.isdigit()}  # keep only numeric IDs
            self.logger.debug(
                "Cluster lookup | accession=%s | cluster_id=%s | members=%d",
                accession, cluster_id, len(clean_members)
            )
            return clean_members
        except KeyError:
            self.logger.warning("retrieve_cluster_members: accession '%s' not found in clusters.", accession)
            return set()

    def preload_annotations(self):
        """
        Preload GO annotations from the database into memory.

        Behavior:
            - Queries `sequence`, `protein`, `protein_go_term_annotation`, and `go_terms`.
            - Groups annotations by sequence ID.
            - Skips entries whose taxonomy_id is in `self.exclude_taxon_ids`.

        Each stored annotation includes:
            - sequence (str)
            - go_id (str)
            - category (str)
            - evidence_code (str)
            - go_description (str)
            - protein_id (int)
            - organism (str)
            - taxonomy_id (int)
            - gene_name (str)

        Results are stored in:
            self.go_annotations : dict[int, list[dict]]
                Mapping from sequence_id → list of annotation dicts.
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

        self.logger.info(
            "Preloaded GO annotations: %d sequences.",
            len(self.go_annotations)
        )

    # --- Metadata helpers -----------------------------------------------------
    def _model_threshold_map(self) -> dict:
        """
        Build a mapping {task_name -> distance_threshold} from `self.types`.

        Notes
        -----
        - `task_name` refers to the config/lookup key used in outputs (not the DB `model_name`).
        - If `self.types` is missing or malformed, an empty dict is returned.
        """
        try:
            mapping = {
                info["task_name"]: info.get("distance_threshold")
                for info in (self.types or {}).values()
            }
            self.logger.debug(
                "_model_threshold_map built: %d entries → %s",
                len(mapping), list(mapping.keys())
            )
            return mapping
        except Exception as e:
            self.logger.error("_model_threshold_map failed: %s", e, exc_info=True)
            return {}

    def _add_metadata_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Append run metadata to the output DataFrame.

        Adds
        ----
        - distance_metric : str
            The distance metric used in the run (e.g., "cosine" or "euclidean").
        - distance_threshold : float | None
            Threshold retrieved from the per-model configuration (`task_name` → threshold).

        Side effects
        ------------
        - If ``conf['postprocess']['keep_sequences']`` is False,
          the columns ``sequence_query`` and ``sequence_reference`` are dropped
          to produce leaner outputs.

        Returns
        -------
        pd.DataFrame
            A copy of the input DataFrame with metadata columns appended and
            sequences optionally removed.
        """
        df = df.copy()

        # Attach global run metric
        df["distance_metric"] = self.distance_metric

        # Attach per-model threshold from config mapping
        thr_map = self._model_threshold_map()
        df["distance_threshold"] = df["model_name"].map(thr_map).astype(object)

        # Optionally drop sequence columns
        keep_seq = (self.conf.get("postprocess", {}) or {}).get("keep_sequences", False)
        if not keep_seq:
            df = df.drop(columns=["sequence_query", "sequence_reference"], errors="ignore")
            self.logger.debug("_add_metadata_columns: sequence columns dropped.")
        else:
            self.logger.debug("_add_metadata_columns: sequence columns preserved.")

        self.logger.debug(
            "_add_metadata_columns: metadata attached (distance_metric=%s, thresholds=%s)",
            self.distance_metric, {k: v for k, v in thr_map.items()}
        )

        return df

    # --- Normalization utilities ----------------------------------------------
    def _safe_max(self, s: pd.Series) -> float:
        """
        Safely compute the maximum positive value in a pandas Series.

        Behavior
        --------
        - Non-numeric values are coerced to NaN.
        - If the series is empty, None, or has no positive values, returns NaN.
        - Only strictly positive values (> 0) are considered valid maxima.

        Parameters
        ----------
        s : pd.Series
            Input series to evaluate.

        Returns
        -------
        float
            Maximum positive value if present, otherwise NaN.
        """
        if s is None or s.empty:
            self.logger.debug("_safe_max: received empty or None series → returning NaN")
            return float("nan")

        m = pd.to_numeric(s, errors="coerce").max()
        if pd.notnull(m) and m > 0:
            self.logger.debug("_safe_max: max positive value found = %s", m)
            return m

        self.logger.debug("_safe_max: no positive values found → returning NaN")
        return float("nan")

    def _normalize_by_accession(self, df: pd.DataFrame, col: str) -> pd.Series:
        """
        Normalize a numeric column by 'accession'.

        Behavior
        --------
        - Groups the DataFrame by `accession`.
        - For each group, divides the values in `col` by the group's maximum
          positive value (computed via `_safe_max`).
        - If a group has no positive maximum, all values in that group are set to 0.
        - NaNs in the original column are replaced with 0 before normalization.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame containing at least the columns `accession` and `col`.
        col : str
            The name of the numeric column to normalize.

        Returns
        -------
        pd.Series
            Normalized values, aligned with the original DataFrame index.
        """

        def norm(group: pd.Series) -> pd.Series:
            """Normalize a single accession group."""
            m = self._safe_max(group)
            if not pd.notnull(m) or m == 0:
                self.logger.debug(
                    "_normalize_by_accession: accession=%s → no positive max, set to 0",
                    group.name
                )
                return pd.Series(0.0, index=group.index)
            self.logger.debug(
                "_normalize_by_accession: accession=%s → max=%s",
                group.name, m
            )
            return group.fillna(0.0) / m

        return df.groupby("accession")[col].transform(norm)

    # --- Collapse to best (model, layer) --------------------------------------
    def _collapse_best_model_layer(self, df_scored: pd.DataFrame) -> pd.DataFrame:
        """
        Collapse predictions to the best (model, layer) per accession.

        Behavior
        --------
        1. For each (accession, model_name, layer_index) group, compute the maximum
           score observed → `score_global`.
        2. For each accession, select the single (model, layer) pair with the
           highest `score_global`. These are labeled as `best_model` and `best_layer`.
        3. Within the winning pair, drop duplicate GO terms, keeping only the row
           with the highest `score` for that (accession, go_id).
        4. Attach the per-accession `score_global` for reference.

        Parameters
        ----------
        df_scored : pd.DataFrame
            Input DataFrame with at least the columns:
            ['accession', 'model_name', 'layer_index', 'go_id', 'score'].

        Returns
        -------
        pd.DataFrame
            Filtered DataFrame with only the best (model, layer) per accession,
            unique GO terms per accession, and additional columns:
            - best_model
            - best_layer
            - score_global
        """

        # Step 1: max score per (accession, model, layer)
        best_combo = (
            df_scored.groupby(["accession", "model_name", "layer_index"], dropna=False)["score"]
            .max().rename("score_global").reset_index()
        )

        # Step 2: select best (model, layer) per accession
        best_by_acc = (
            best_combo.sort_values(["accession", "score_global"], ascending=[True, False])
            .groupby("accession", as_index=False)
            .first()
            .rename(columns={"model_name": "best_model", "layer_index": "best_layer"})
        )
        self.logger.debug(
            "_collapse_best_model_layer: selected best model/layer for %d accessions",
            best_by_acc.shape[0]
        )

        # Step 3: filter df_scored down to rows from the winning (model, layer)
        df_best = df_scored.merge(best_by_acc, on="accession", how="inner")
        mask = (
                (df_best["model_name"] == df_best["best_model"]) &
                (df_best["layer_index"] == df_best["best_layer"])
        )

        df_best = df_best[mask].copy()

        # Step 4: remove duplicate GO terms, keep highest score
        df_best = (
            df_best.sort_values(["accession", "go_id", "score"], ascending=[True, True, False])
            .drop_duplicates(subset=["accession", "go_id"], keep="first")
            .reset_index(drop=True)
        )

        # Attach score_global
        df_best = df_best.merge(best_by_acc[["accession", "score_global"]], on="accession", how="left")

        self.logger.debug(
            "_collapse_best_model_layer: collapsed to %d rows across %d accessions",
            df_best.shape[0], df_best["accession"].nunique()
        )

        return df_best

    # --- Full post-processing pipeline ----------------------------------------
    def post_processing(self) -> str:
        """
        Aggregate per-accession results, compute weighted scores, and write a global summary.

        Workflow
        --------
        1. Locate all CSV shards under ``raw_results/**`` across models and layers.
        2. For each accession:
            - Concatenate all shards belonging to it.
            - Compute per-GO aggregation metrics defined in the configuration.
            - Apply weighting scheme (if provided) to compute per-metric contributions
              and a global ``final_score``.
            - Collect associated proteins and counts.
            - Write results incrementally to a global ``summary.csv``.
        3. Trigger downstream exports:
            - Per model/layer TopGO files: ``topgo/{model}/layer_{k}/{category}.topgo``.
            - Ensemble TopGO files: ``topgo/ensemble/{category}.topgo``.

        Configuration (conf['postprocess']['summary'])
        ----------------------------------------------
        - metrics : dict
            {column_name: [aggregation_fn]}.
            Aggregation functions: "mean", "max", "min" (others ignored).
        - aliases : dict
            Optional renaming for metrics used in weighting.
        - include_counts : bool
            If True, add a normalized neighbor/support count metric.
        - weights : dict
            Raw weights per metric, alias, or metric+agg combination.
            Normalized to form ``final_score`` and weighted columns prefixed by ``weighted_prefix``.
        - weighted_prefix : str
            Prefix for weighted metric outputs (default: "w\\_").

        Output
        ------
        - ``<experiment_path>/summary.csv`` (incremental write).
        - TopGO export files under ``topgo/``.

        Returns
        -------
        str
            Absolute path to the written ``summary.csv``.
            Returns an empty string if no raw input files were found.
        """

        base_dir = Path(self.experiment_path) / "raw_results"
        paths = sorted(base_dir.glob("**/*.csv"))
        if not paths:
            self.logger.info("post_processing: no raw_results/*.csv found under %s", base_dir)
            return ""

        # --- Load configuration -------------------------------------------------
        spec = (self.conf.get("postprocess", {}) or {}).get("summary", {}) or {}
        metrics: dict = spec.get("metrics", {})
        aliases: dict = spec.get("aliases", {})
        include_counts: bool = bool(spec.get("include_counts", True))
        weights_spec: dict = spec.get("weights", {}) or {}
        weighted_prefix: str = str(spec.get("weighted_prefix", "w_"))

        # k parameter (limit of neighbors per query)
        k = int(getattr(
            self, "limit_per_entry",
            self.conf.get("limit_per_entry", (self.conf.get("lookup", {}) or {}).get("limit_per_entry", 1))
        ) or 1)
        if k <= 0:
            k = 1

        # --- Helpers -------------------------------------------------------------
        def _norm_fun(f: str) -> str:
            """Normalize function aliases (e.g., 'avg' -> 'mean')."""
            f = (f or "").lower()
            return "mean" if f == "avg" else f

        def _weight_for(metric: str, fun: str) -> tuple[float, bool]:
            """
            Resolve raw weight for a metric-function combination.
            Returns (weight, is_active).
            """
            alias = aliases.get(metric, metric)
            fun = _norm_fun(fun)

            out_key = f"{fun}_{alias}"
            if out_key in weights_spec and isinstance(weights_spec[out_key], (int, float)):
                return float(weights_spec[out_key]), True

            for key in (metric, alias):
                if key in weights_spec:
                    val = weights_spec[key]
                    if isinstance(val, (int, float)):
                        return float(val), True
                    if isinstance(val, dict):
                        for kfun, v in val.items():
                            if _norm_fun(kfun) == fun and isinstance(v, (int, float)):
                                return float(v), True
            return 0.0, False

        summary_path = base_dir.parent / "summary.csv"

        # --- Group files by accession -------------------------------------------
        files_by_accession = defaultdict(list)
        for p in paths:
            accession = p.stem
            files_by_accession[accession].append(p)

        self.logger.info("post_processing: processing %d accessions", len(files_by_accession))

        # --- Process accession by accession -------------------------------------
        for i, (accession, flist) in enumerate(files_by_accession.items()):
            dfs = [pl.read_csv(f, ignore_errors=True) for f in flist]
            df = pl.concat(dfs, how="vertical_relaxed")

            if df.is_empty():
                self.logger.debug("post_processing: accession %s has empty data, skipping", accession)
                continue

            # Per-term info (GO count and proteins)
            per_term = (
                df.group_by(["accession", "go_id"])
                .agg([
                    pl.len().alias("term_count"),
                    pl.col("protein_id").cast(pl.Utf8, strict=False).unique().alias("proteins_list"),
                ])
            )

            group_keys = ["accession", "go_id", "model_name", "layer_index"]

            # --- Build aggregation items ----------------------------------------
            agg_items = []
            base_names = []

            for col, funs in metrics.items():
                alias = aliases.get(col, col)
                for f in funs:
                    f = _norm_fun(f)
                    if f == "min":
                        expr = pl.col(col).min()
                    elif f == "max":
                        expr = pl.col(col).max()
                    elif f == "mean":
                        expr = pl.col(col).mean()
                    else:
                        continue
                    name = f"{f}_{alias}"
                    w_raw, apply = _weight_for(col, f)
                    agg_items.append({"name": name, "expr": expr, "w_raw": w_raw, "weighted": apply})
                    base_names.append(name)

            # Add normalized count if configured
            if include_counts:
                cnt_expr = (pl.len().cast(pl.Float64) / pl.lit(float(k)))
                cw = weights_spec.get("count", None)
                apply_c = isinstance(cw, (int, float))
                agg_items.append({"name": "count", "expr": cnt_expr,
                                  "w_raw": float(cw) if apply_c else 0.0, "weighted": apply_c})
                base_names.append("count")

            # Normalize weights
            total_w = sum(
                i["w_raw"] for i in agg_items if i["weighted"]
            )

            def norm(w, total_w=total_w):
                return (w / total_w) if total_w > 0 else 0.0

            have_weights = total_w > 0.0

            # Build aggregation expressions
            agg_exprs = []
            score_terms = []
            out_cols = []

            for item in agg_items:
                agg_exprs.append(item["expr"].alias(item["name"]))
                out_cols.append(item["name"])
                if have_weights and item["weighted"]:
                    w_norm = norm(item["w_raw"])
                    wname = f"{weighted_prefix}{item['name']}"
                    contrib = (item["expr"].fill_null(0.0) * pl.lit(w_norm)).round(self.precision)
                    agg_exprs.append(contrib.alias(wname))
                    out_cols.append(wname)
                    score_terms.append(contrib)

            if score_terms:
                final_expr = score_terms[0]
                for e in score_terms[1:]:
                    final_expr = final_expr + e
                agg_exprs.append(final_expr.alias("final_score"))
                out_cols.append("final_score")

            stats = df.group_by(group_keys).agg(agg_exprs)

            # Pivot to wide format
            def _stack_one(colname: str, stats=stats) -> pl.DataFrame:
                return stats.select(
                    pl.col("accession"),
                    pl.col("go_id"),
                    pl.concat_str([
                        pl.lit(colname), pl.lit("_"),
                        pl.col("model_name"), pl.lit("_L"),
                        pl.coalesce([pl.col("layer_index").cast(pl.Utf8), pl.lit("legacy")])
                    ]).alias("col"),
                    pl.col(colname).alias("value"),
                )

            frames = [
                _stack_one(c).with_columns(pl.col("value").cast(pl.Float32))
                for c in out_cols
            ]
            long = pl.concat(frames, how="vertical")
            wide = long.pivot(values="value", index=["accession", "go_id"], columns="col")

            out_df = (
                per_term
                .join(wide, on=["accession", "go_id"], how="left")
                .with_columns(pl.col("proteins_list").list.join("|").alias("proteins"))
                .drop("proteins_list")
                .sort(["accession", "go_id"])
            )

            # --- Incremental write to summary.csv -------------------------------
            if not summary_path.exists() and i == 0:
                out_df.write_csv(summary_path)
            else:
                with open(summary_path, "a") as f:
                    out_df.write_csv(f, include_header=False)

            self.logger.info(
                "post_processing: accession=%s → %d rows appended to summary",
                accession, out_df.height
            )

        self.logger.info("✅ post_processing completed. Summary written to %s", summary_path)

        # Trigger downstream exports
        self.export_topgo()
        self.export_topgo_ensemble()

        return str(summary_path)

    def _get_lookup_for_batch(self, model_id: int, layer_index: int | None) -> dict | None:
        """
        Lazily build (and cache) the reference lookup table for a given (model_id, layer_index).

        Applies taxonomy filters and optional SQL LIMIT as configured.

        Parameters
        ----------
        model_id : int
            Identifier of the embedding model in the DB.
        layer_index : int | None
            Specific layer to query (None for models without layers).

        Returns
        -------
        dict | None
            {
                "ids": np.ndarray[int],         # sequence IDs
                "embeddings": np.ndarray[float],# stacked embeddings
                "layers": np.ndarray[int],      # layer indices
            }
            or None if no rows match.
        """

        # Normalize key
        key = (int(model_id), None if layer_index is None else int(layer_index))

        # Serve from cache if available
        if key in self._lookup_cache:
            self.logger.debug("_get_lookup_for_batch: cache hit for %s", key)
            return self._lookup_cache[key]

        def _as_str_list(xs):
            return [str(t) for t in (xs or [])]

        # Taxonomy filters (already pre-normalized in __init__)
        exclude_taxon_ids = _as_str_list(
            getattr(self, "exclude_taxon_ids", self.conf.get("taxonomy_ids_to_exclude", [])))
        include_taxon_ids = _as_str_list(
            getattr(self, "include_taxon_ids", self.conf.get("taxonomy_ids_included_exclusively", [])))

        limit_execution = self.conf.get("limit_execution")

        # --- SQL query ---
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

        # Layer filtering
        if key[1] is None:
            q = q.filter(SequenceEmbedding.layer_index.is_(None))
        else:
            q = q.filter(SequenceEmbedding.layer_index == key[1])

        # Taxonomy filters
        if exclude_taxon_ids:
            q = q.filter(~Protein.taxonomy_id.in_(exclude_taxon_ids))
        if include_taxon_ids:
            q = q.filter(Protein.taxonomy_id.in_(include_taxon_ids))

        # Optional LIMIT
        if isinstance(limit_execution, int) and limit_execution > 0:
            self.logger.info(
                "SQL LIMIT applied: %d for lookup(model_id=%s, layer_index=%s)",
                limit_execution, key[0], key[1]
            )
            q = q.limit(limit_execution)

        # --- Execute ---
        rows = q.all()
        if not rows:
            self.logger.warning("Empty lookup for model_id=%s, layer_index=%s", key[0], key[1])
            return None

        try:
            seen = {}
            for r in rows:
                seq_id = int(r[0])
                if seq_id not in seen:
                    seen[seq_id] = (r[1].to_numpy(), int(r[2]) if r[2] is not None else -1)

            ids = np.fromiter(seen.keys(), dtype=int, count=len(seen))
            layers = np.fromiter((v[1] for v in seen.values()), dtype=np.int64, count=len(seen))
            embeddings = np.vstack([v[0] for v in seen.values()])
        except Exception as e:
            self.logger.error("Failed to build numpy arrays for lookup(%s): %s", key, e, exc_info=True)
            return None

        lookup = {"ids": ids, "embeddings": embeddings, "layers": layers}

        # Cache with eviction policy
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
        Inspect the embeddings HDF5 file and collect the set of available layer indices
        for a given model.

        Behavior
        --------
        - Opens the HDF5 file at `self.embeddings_path`.
        - Iterates over all accessions stored as groups.
        - For each accession, inspects the subgroup `type_{model_id}` if present.
        - Collects all keys matching `layer_*`, parsing the suffix as an integer.
        - Returns the sorted unique list of layer indices.

        Parameters
        ----------
        model_id : int
            The numeric identifier of the embedding model.

        Returns
        -------
        list[int]
            Sorted list of available layer indices.
            Returns an empty list if the HDF5 file does not exist or no layers are found.
        """

        layers: set[int] = set()

        if not os.path.exists(self.embeddings_path):
            self.logger.debug(
                "_h5_available_layers: embeddings file not found at %s", self.embeddings_path
            )
            return []

        try:
            with h5py.File(self.embeddings_path, "r") as h5:
                type_key = f"type_{model_id}"
                for accession, group in h5.items():
                    if type_key not in group:
                        continue
                    for k in group[type_key].keys():
                        if k.startswith("layer_"):
                            try:
                                layers.add(int(k.split("_", 1)[1]))
                            except Exception:
                                self.logger.debug(
                                    "_h5_available_layers: failed to parse layer index "
                                    "for model_id=%s in accession=%s key=%s",
                                    model_id, accession, k
                                )
        except Exception as e:
            self.logger.error(
                "_h5_available_layers: error inspecting HDF5 for model_id=%s → %s",
                model_id, e, exc_info=True
            )
            return []

        result = sorted(layers)
        self.logger.debug(
            "_h5_available_layers: model_id=%s → %d layers found: %s",
            model_id, len(result), result
        )
        return result

    def load_model(self, model_type):
        """Placeholder: load a model into memory if required."""
        return

    def unload_model(self, model_type):
        """Placeholder: unload a model from memory if required."""
        return

    def _index_sequence(self, seq: str, which: str) -> int | None:
        """
        Return the stable index for a given sequence.

        Parameters
        ----------
        seq : str
            The sequence string.
        which : str
            Either 'Q' for queries/targets or 'R' for references.

        Returns
        -------
        int | None
            Stable index assigned to the sequence, or None if input is invalid.
        """
        if not seq:
            return None
        if which == 'Q':
            if seq in self._q_seq_to_idx:
                return self._q_seq_to_idx[seq]
            idx = len(self._q_idx_to_seq)
            self._q_seq_to_idx[seq] = idx
            self._q_idx_to_seq.append(seq)
            return idx
        elif which == 'R':
            if seq in self._r_seq_to_idx:
                return self._r_seq_to_idx[seq]
            idx = len(self._r_idx_to_seq)
            self._r_seq_to_idx[seq] = idx
            self._r_idx_to_seq.append(seq)
            return idx
        else:
            return None

    def _flush_sequences_fasta(self) -> str:
        """
        Write a global FASTA file containing all sequences seen so far.

        Sequences are indexed deterministically:
        - Queries/targets: >Q{idx}
        - References: >R{idx}

        Returns
        -------
        str
            Path to the written FASTA file.
        """
        os.makedirs(self.experiment_path, exist_ok=True)
        path = self.sequences_fasta_path
        with open(path, "w") as f:
            for i, s in enumerate(self._q_idx_to_seq):
                f.write(f">Q{i}\n{s}\n")
            for i, s in enumerate(self._r_idx_to_seq):
                f.write(f">R{i}\n{s}\n")
        self.logger.info(
            "Sequences FASTA written → %s | queries(Q)=%d | references(R)=%d",
            path, len(self._q_idx_to_seq), len(self._r_idx_to_seq)
        )
        return path

    # Inside EmbeddingLookUp

    def export_topgo(self):
        """
        Export TopGO-compatible TSV files:
        topgo/{model}/layer_{k}/{category}.topgo

        Columns: accession, go_term, reliability_index
        """

        base_dir = Path(self.experiment_path) / "raw_results"
        paths = sorted(base_dir.glob("**/*.csv"))
        if not paths:
            self.logger.info("export_topgo: no raw_results found in %s", base_dir)
            return

        for p in paths:
            df = pd.read_csv(p)
            if df.empty or "reliability_index" not in df:
                continue

            # Ensure required columns exist
            cols = ["accession", "go_id", "reliability_index", "category", "model_name", "layer_index"]
            for c in cols:
                if c not in df.columns:
                    df[c] = None

            # Write per-category files
            grouped = df.groupby(["model_name", "layer_index", "category"])
            for (model, layer, cat), chunk in grouped:
                if chunk.empty:
                    continue

                out_dir = Path(self.experiment_path) / "topgo" / str(
                    model) / f"layer_{layer if pd.notna(layer) else 'legacy'}"
                os.makedirs(out_dir, exist_ok=True)
                out_path = out_dir / f"{cat}.topgo"

                chunk_out = chunk[["accession", "go_id", "reliability_index"]].drop_duplicates()
                chunk_out = chunk_out.rename(columns={"go_id": "go_term"})
                chunk_out.to_csv(out_path, sep="\t", index=False, header=False,
                                 mode="a")

                self.logger.info("export_topgo: wrote %d rows → %s", len(chunk_out), out_path)

    def export_topgo_ensemble(self):
        """
        Export ensemble TopGO TSV files by collapsing across models/layers.

        Keeps the best reliability_index per (accession, go_id, category) and writes:
        topgo/ensemble/{category}.topgo
        """

        base_dir = Path(self.experiment_path) / "raw_results"
        paths = sorted(base_dir.glob("**/*.csv"))
        if not paths:
            self.logger.info("export_topgo_ensemble: no raw_results/*.csv found")
            return

        dfs = [pd.read_csv(p) for p in paths if p.stat().st_size > 0]
        if not dfs:
            return
        df = pd.concat(dfs, ignore_index=True)

        # Ensure required columns exist
        cols = ["accession", "go_id", "reliability_index", "category"]
        for c in cols:
            if c not in df.columns:
                self.logger.warning("export_topgo_ensemble: missing column %s", c)
                return

        # Select best row per (accession, go_id, category)
        df_best = (
            df.sort_values("reliability_index", ascending=False)
            .groupby(["accession", "go_id", "category"], as_index=False)
            .first()
        )

        # Write per-category ensemble files
        out_dir = Path(self.experiment_path) / "topgo" / "ensemble"
        os.makedirs(out_dir, exist_ok=True)

        for cat, chunk in df_best.groupby("category"):
            out_path = out_dir / f"{cat}.topgo"
            chunk_out = chunk[["accession", "go_id", "reliability_index"]].rename(columns={"go_id": "go_term"})
            chunk_out.to_csv(out_path, sep="\t", index=False, header=False)
            self.logger.info("export_topgo_ensemble: wrote %d rows → %s", len(chunk_out), out_path)
