"""
Embedding Lookup Module
=======================

Overview
--------
The ``EmbeddingLookup`` module implements an embedding-based workflow for
**Gene Ontology (GO) annotation transfer**. Given a collection of query protein
embeddings stored in an HDF5 file, the component executes the following
operations:

1. Retrieves reference embeddings from a vector-enabled relational database.
2. Computes pairwise distances between query and reference embeddings
   using either GPU (PyTorch) or CPU (SciPy).
3. Selects nearest neighbours according to configurable distance thresholds and
   optional redundancy filtering.
4. Expands selected neighbours into GO annotations, which are preloaded from
   the database.
5. Writes per-query results to structured CSV files and optionally generates
   TopGO-compatible TSV files.
6. (Optional) Performs alignment-based post-processing, computing identity and
   similarity metrics for each query–reference pair.

The module is optimised for large-scale protein annotation experiments, supports
multiple embedding models with heterogeneous configurations, and provides a
complete post-processing and scoring pipeline.

Key Features
------------
- **Taxonomy-aware filtering** supporting exclude-lists, include-lists, and
  optional descendant expansion.
- **Redundancy filtering with MMseqs2**, preventing annotation transfer from
  redundant reference sequences.
- **Multi-model support**, enabling per-model configuration of thresholds,
  layers, and batch sizes.
- **GPU-accelerated distance computation** via PyTorch, with SciPy CPU fallback.
- **Comprehensive post-processing pipeline** using Polars/Pandas, including
  scoring, aggregation, dynamic GO-term selection, and cross-model comparison.
- **TopGO export** for downstream functional enrichment analysis.

Inputs
------
The module expects the following inputs:

- **Query embeddings (HDF5)** organised as::

      <accession>/
          type_<model_id>/
              layer_<k>/
                  embedding
          sequence
          taxonomy

- **Reference embeddings (SQL database)** retrieved via SQLAlchemy, stored as
  vector types and joined to protein metadata.

- **GO annotations (SQL database)** preloaded once during initialisation.

Outputs
-------
The workflow generates:

- **Per-query annotation files**::

      <experiment_path>/raw_results/<model_name>/layer_<k>/<accession>.csv

- **A global summary file**::

      <experiment_path>/summary.csv

- **TopGO-ready TSV directories**::

      <experiment_path>/topgo/<model_name>/layer_<k>/
      <experiment_path>/topgo/ensemble/

- **A combined FASTA** containing all sequences observed during lookup::

      <experiment_path>/<sequences_fasta | sequences.fasta>

Configuration (Selected Keys)
-----------------------------

Top-level Keys
~~~~~~~~~~~~~~
``experiment_path`` (str)
    Base directory for all inputs and outputs.

``embeddings_path`` (str)
    Path to the HDF5 file containing query embeddings.

``batch_size`` (int)
    Maximum number of queries per processing batch.

``limit_per_entry`` (int)
    Maximum number of neighbours retained per query.

``precision`` (int)
    Floating-point precision used in exported files.

``limit_execution`` (int | None)
    Optional SQL ``LIMIT`` applied when retrieving reference embeddings.

Lookup Section
~~~~~~~~~~~~~~
``lookup.use_gpu`` (bool)
    Enables GPU-accelerated distance computation.

``lookup.batch_size`` (int)
    Overrides global ``batch_size`` for lookup operations.

``lookup.limit_per_entry`` (int)
    Overrides global ``limit_per_entry``.

``lookup.topgo`` (bool)
    Enables TopGO-compatible output generation.

``lookup.lookup_cache_max`` (int)
    Maximum number of cached lookup matrices.

``lookup.distance_metric`` (``"cosine"`` or ``"euclidean"``)
    Distance metric used during neighbour selection.

Lookup Redundancy Subsection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``lookup.redundancy.identity`` (float)
    Minimum sequence identity threshold for MMseqs2 clustering.

``lookup.redundancy.coverage`` (float)
    Minimum alignment coverage for cluster membership.

``lookup.redundancy.mmseqs_threads`` (int)
    Number of MMseqs2 threads.

Lookup Taxonomy Subsection
^^^^^^^^^^^^^^^^^^^^^^^^^^
``lookup.taxonomy.exclude`` (list[int])
    Taxonomy identifiers to exclude from reference lookup.

``lookup.taxonomy.include_only`` (list[int])
    Restricts lookup to these taxonomy identifiers exclusively.

``lookup.taxonomy.get_descendants`` (bool)
    Expands provided taxonomy identifiers into their descendant sets.

Embedding Model Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Each logical model under ``embedding.models`` supports:

``enabled`` (bool)
    Enables or disables the model for the current run.

``distance_threshold`` (float | None)
    Maximum distance allowed during neighbour selection.

``batch_size`` (int | None)
    Optional model-specific batch size.

``layer_index`` (list[int] | None)
    Specific embedding layers to process for this model.

Post-processing Section
~~~~~~~~~~~~~~~~~~~~~~~
``postprocess.keep_sequences`` (bool)
    Retains raw sequences in intermediate outputs.

``postprocess.store_workers`` (int)
    Number of worker processes used for alignment metrics.

Post-processing Summary Subsection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``postprocess.summary.metrics`` (dict)
    Mapping from metric names to aggregation functions
    (``mean``, ``max``, ``min``).

``postprocess.summary.aliases`` (dict)
    Optional renaming of metric identifiers.

``postprocess.summary.include_counts`` (bool)
    Whether to include neighbour counts normalised by ``k``.

``postprocess.summary.weights`` (dict)
    Weighting scheme applied to aggregated metrics.

``postprocess.summary.weighted_prefix`` (str)
    Prefix for weighted output columns.

Dependencies
------------
This module depends on:

- **SQLAlchemy** for sequence, protein, embedding, and GO-annotation retrieval.
- **goatools** for loading the Gene Ontology DAG.
- **PyTorch** (GPU) and **SciPy** (CPU) for distance computations.
- **Polars** and **Pandas** for scoring, aggregation, and summary generation.
- **MMseqs2** for sequence clustering and redundancy filtering.

Reference
---------
The design of this workflow is inspired by *GoPredSim* (Rostlab):

  https://github.com/Rostlab/goPredSim

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

# ======================================================================
# Standard Library
# ======================================================================
import os
import re
import time
import tempfile
import traceback
import subprocess
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

# ======================================================================
# Third-Party Libraries
# ======================================================================
import h5py
import numpy as np
import pandas as pd
import polars as pl
import torch

from scipy.spatial.distance import cdist
from sqlalchemy import text
from goatools.base import get_godag

# ======================================================================
# Project-Specific Imports
# ======================================================================
from fantasia.src.helpers.helpers import (
    compute_metrics,
    compute_taxonomy,
    get_descendant_ids,
)

from protein_information_system.sql.model.entities.embedding.sequence_embedding import (
    SequenceEmbedding,
    SequenceEmbeddingType,
)

from protein_information_system.sql.model.entities.protein.protein import (
    Protein,
)

from protein_information_system.sql.model.entities.sequence.sequence import (
    Sequence,
)

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
    """

    def __init__(self, conf, current_date):
        super().__init__(conf)

        self.current_date = current_date
        self.logger.info("EmbeddingLookUp: initializing component...")

        # ==========================================================================
        # CONFIG SECTIONS — EXACT MATCH TO NEW config.yaml
        # ==========================================================================

        # lookup root
        self.lookup_cfg = self.conf.get("lookup", {}) or {}

        # lookup.compute
        self.compute_cfg = self.lookup_cfg.get("compute", {}) or {}

        # lookup.nn
        self.nn_cfg = self.lookup_cfg.get("nn", {}) or {}

        # lookup.filters
        self.filters_cfg = self.lookup_cfg.get("filters", {}) or {}
        self.taxonomy_cfg = self.filters_cfg.get("taxonomy", {}) or {}
        self.redundancy_cfg = self.filters_cfg.get("redundancy", {}) or {}

        # embedding.models
        self.embedding_cfg = self.conf.get("embedding", {}) or {}
        self.embedding_models_cfg = self.embedding_cfg.get("models", {}) or {}

        # ==========================================================================
        # PATHS
        # ==========================================================================
        self.experiment_path = self.conf["experiment_path"]

        self.embeddings_path = self.conf.get(
            "embeddings_path",
            os.path.join(self.experiment_path, "embeddings.h5"),
        )

        self.raw_results_path = os.path.join(self.experiment_path, "raw_results.csv")
        self.results_path = os.path.join(self.experiment_path, "results.csv")
        self.topgo_path = os.path.join(self.experiment_path, "results_topgo.tsv")

        # ==========================================================================
        # LOOKUP PARAMETERS (NEW YAML STRUCTURE)
        # ==========================================================================
        self.limit_per_entry = int(self.nn_cfg.get("limit_per_entry", 200))
        self.batch_size = int(self.compute_cfg.get("batch_size", 50))
        self.use_gpu = bool(self.compute_cfg.get("use_gpu", True))

        # Normalize metric directly here (no static method needed)
        metric_raw = self.compute_cfg.get("distance_metric", "cosine")
        self.distance_metric = (
            metric_raw if metric_raw in ("cosine", "euclidean") else "cosine"
        )

        self.topgo_enabled = bool(
            (self.lookup_cfg.get("output", {}) or {}).get("topgo", False)
        )

        self.lookup_cache_max = int(self.nn_cfg.get("cache_max", 1))

        # ==========================================================================
        # REDUNDANCY FILTER (MMseqs2)
        # ==========================================================================
        self.redundancy_filter = float(self.redundancy_cfg.get("identity", 0))
        self.alignment_coverage = float(self.redundancy_cfg.get("coverage", 0.7))
        self.mmseqs_threads = int(self.redundancy_cfg.get("threads", 10))

        if self.redundancy_filter > 0:
            self.logger.info(
                "Redundancy filter enabled: identity >= %.3f, coverage >= %.3f",
                self.redundancy_filter,
                self.alignment_coverage,
            )
            self.generate_clusters()
        else:
            self.logger.info("Redundancy filtering disabled.")

        # ==========================================================================
        # TAXONOMY FILTERS
        # ==========================================================================
        raw_exclude = self.taxonomy_cfg.get("exclude", [])
        raw_include = self.taxonomy_cfg.get("include_only", [])
        expand = bool(self.taxonomy_cfg.get("get_descendants", False))

        self.exclude_taxon_ids = self._expand_taxonomy_ids(raw_exclude, expand)
        self.include_taxon_ids = self._expand_taxonomy_ids(raw_include, expand)

        self.logger.info(
            "Taxonomy filters → exclude=%s include=%s expand=%s",
            self.exclude_taxon_ids, self.include_taxon_ids, expand
        )

        # ==========================================================================
        # LOOKUP CACHE
        # ==========================================================================
        self._lookup_cache = {}

        # ==========================================================================
        # LOAD MODELS + GO ANNOTATIONS
        # ==========================================================================
        self.load_model_definitions()
        self.preload_reference_data()

        # ==========================================================================
        # FASTA + PRECISION
        # ==========================================================================
        self.sequences_fasta_path = os.path.join(
            self.experiment_path,
            self.conf.get("sequences_fasta", "sequences.fasta"),
        )
        self.precision = int(self.conf.get("precision", 4))

        self.logger.info("EmbeddingLookUp initialisation completed successfully.")

    @staticmethod
    def _expand_taxonomy_ids(values, expand: bool):
        if not values:
            return []
        clean = [int(v) for v in values if str(v).isdigit()]
        if expand and clean:
            return list(get_descendant_ids(clean))
        return clean

    def enqueue(self):
        """
        Scan the query HDF5 file and publish homogeneous batches of tasks.

        This method implements a functional-style pipeline:
            HDF5 → accessions → embedding types → layers → tasks → batches → publish.
        """

        self.logger.info("Starting enqueue process for embedding-based GO annotation.")
        self.logger.info("Loading query embeddings from: %s", self.embeddings_path)

        if not os.path.exists(self.embeddings_path):
            raise FileNotFoundError(
                f"HDF5 file not found: {self.embeddings_path}. "
                "Ensure embeddings have been generated prior to running annotation."
            )

        batch_size = int(self.batch_size)
        total_entries = 0
        total_batches = 0
        buffers = defaultdict(list)

        # Quick lookup by model ID
        by_id = {info["id"]: info for info in self.types.values()}

        try:
            with h5py.File(self.embeddings_path, "r") as h5file:

                for accession, group in h5file.items():
                    if "sequence" not in group:
                        self.logger.warning("Missing sequence for accession '%s'. Skipping.", accession)
                        continue

                    # Iterate through type_* groups
                    for model_id, type_group in self._iter_type_groups(group):
                        model_info = by_id.get(model_id)
                        if model_info is None:
                            self.logger.debug("Model id %s not active/enabled. Skipping.", model_id)
                            continue

                        model_name = model_info["task_name"]
                        distance_threshold = model_info.get("distance_threshold")
                        enabled_layers = model_info.get("enabled_layers")

                        # Iterate through layers (layered or legacy)
                        for layer_index, layer_group in self._iter_layer_groups(type_group):
                            # Apply layer filtering from config
                            if enabled_layers and layer_index is not None:
                                if layer_index not in enabled_layers:
                                    continue

                            if "embedding" not in layer_group:
                                continue

                            # Build task
                            task = self._build_task(
                                accession=accession,
                                model_id=model_id,
                                model_name=model_name,
                                distance_threshold=distance_threshold,
                                layer_index=layer_index,
                                h5_path=self.embeddings_path,
                                h5_group=f"{accession}/type_{model_id}"
                                         + (f"/layer_{layer_index}" if layer_index is not None else "")
                            )

                            key = (model_id, layer_index)
                            buffers[key].append(task)
                            total_entries += 1

                            # Flush as soon as we reach batch size
                            if len(buffers[key]) >= batch_size:
                                total_batches += self._flush_buffer(
                                    key, buffers, batch_size
                                )

                # Final flush for all buffers
                for key in list(buffers.keys()):
                    total_batches += self._flush_buffer(key, buffers, batch_size)

        except Exception:
            self.logger.error("Unexpected error during enqueue.", exc_info=True)
            raise

        self.logger.info(
            "Enqueued %d tasks into %d batches (batch_size=%d).",
            total_entries, total_batches, batch_size
        )

    @staticmethod
    def _iter_type_groups(group):
        """Yield (model_id, type_group) pairs for keys like 'type_7'."""
        for key, type_grp in group.items():
            if not key.startswith("type_"):
                continue
            try:
                model_id = int(key.split("_", 1)[1])
            except Exception:
                continue
            yield model_id, type_grp

    @staticmethod
    def _iter_layer_groups(type_group):
        """
        Yield (layer_index, layer_group) pairs.
        If no layers exist, yield a legacy embedding: (None, type_group).
        """
        layer_keys = [k for k in type_group.keys() if k.startswith("layer_")]

        if not layer_keys:
            raise ValueError(f"Model group has no layers: {type_group.name}")

        for lk in sorted(layer_keys, key=lambda x: int(x.split("_", 1)[1])):
            try:
                layer_index = int(lk.split("_", 1)[1])
            except Exception:
                continue
            yield layer_index, type_group[lk]

    @staticmethod
    def _build_task(
            accession,
            model_id,
            model_name,
            distance_threshold,
            layer_index,
            h5_path,
            h5_group,
    ):
        """Return a normalized task dictionary."""
        return {
            "h5_path": h5_path,
            "h5_group": h5_group,
            "embedding_type_id": model_id,
            "model_name": model_name,
            "distance_threshold": distance_threshold,
            "layer_index": layer_index,
            "accession": accession,
        }

    def _flush_buffer(self, key, buffers, batch_size):
        """
        Publish buffered tasks for a (model_id, layer_index) key.
        Returns the number of published batches.
        """
        buf = buffers[key]
        if not buf:
            return 0

        batches = 0
        for i in range(0, len(buf), batch_size):
            chunk = buf[i:i + batch_size]
            model_id = chunk[0]["embedding_type_id"]
            layer_index = chunk[0]["layer_index"]
            model_name = chunk[0]["model_name"]

            payload = {
                "model_id": model_id,
                "layer_index": layer_index,
                "tasks": chunk,
            }

            self.publish_task(payload, model_type=model_name)
            batches += 1

        buffers[key].clear()
        return batches

    def load_model_definitions(self):
        """
        Load embedding model definitions by matching DB embedding types with the new config.

        Key principles:
        - No HDF5 inspection occurs here.
        - Layer selection is defined purely by config:
            * If layer_index is provided → use exactly those.
            * If not provided → treat as "ALL layers allowed", enqueue will discover them.
        - Only models defined both in DB and config AND enabled are kept.
        """

        self.types = {}

        try:
            db_models = self.session.query(SequenceEmbeddingType).all()
        except Exception as exc:
            self.logger.error("Failed to query SequenceEmbeddingType: %s", exc)
            raise

        cfg_models = self.embedding_models_cfg  # from the new YAML layout

        for db_model in db_models:
            task_name = db_model.name
            matched_name = next(
                (k for k in cfg_models if k.lower() == task_name.lower()),
                None
            )

            if matched_name is None:
                self.logger.debug(
                    "Model '%s' exists in DB but not in config. Skipping.",
                    task_name
                )
                continue

            cfg = cfg_models[matched_name]

            if not cfg.get("enabled", True):
                self.logger.info(
                    "Model '%s' disabled by config. Skipping.", matched_name
                )
                continue

            enabled_layers = cfg.get("layer_index")  # list or None

            self.types[matched_name] = {
                "id": db_model.id,
                "model_name": db_model.model_name,
                "task_name": matched_name,
                "distance_threshold": cfg.get("distance_threshold"),
                "batch_size": cfg.get("batch_size"),
                "enabled_layers": enabled_layers,  # None = unlimited
            }

            self.logger.info(
                "Model '%s' (id=%s): threshold=%s | enabled_layers=%s",
                matched_name, db_model.id,
                cfg.get("distance_threshold"),
                enabled_layers if enabled_layers else "ALL",
            )

        if not self.types:
            self.logger.warning("No enabled models found after matching DB + config.")
            return

        self.logger.info(
            "Loaded %d enabled model(s): %s",
            len(self.types),
            list(self.types.keys())
        )

    def process(self, payload: dict) -> list[dict]:
        """
        Modular version of the lookup pipeline.
        Performs:
            1) payload validation
            2) loading query embeddings from HDF5
            3) retrieving reference embeddings (cached)
            4) computing distances (GPU/CPU)
            5) redundancy lookup
            6) selecting nearest neighbors (minimal hits)
        """

        t_start = time.perf_counter()

        # ----------------------------------------------------------
        # (1) Validate payload & extract normalized parameters
        # ----------------------------------------------------------
        try:
            (
                model_id,
                layer_index,
                tasks,
                model_name,
                threshold,
                limit,
                use_gpu,
            ) = self._validate_payload(payload)
        except Exception as e:
            self.logger.error(f"process(): invalid payload → {e}")
            return []

        # ----------------------------------------------------------
        # (2) Materialize query embeddings from HDF5
        # ----------------------------------------------------------
        queries, accessions, taxonomies = self._materialize_queries(tasks)

        if queries is None or not len(accessions):
            self.logger.warning("process(): no query embeddings loaded.")
            return []

        # ----------------------------------------------------------
        # (3) Load reference embedding bank
        # ----------------------------------------------------------
        refbank = self._get_reference_embeddings(model_id, layer_index)
        if not refbank:
            self.logger.warning(f"process(): reference lookup empty for model={model_id}")
            return []

        ref_ids = refbank["ids"]
        ref_emb = refbank["vectors"]

        # ----------------------------------------------------------
        # (4) Compute distance matrix (N × M)
        # ----------------------------------------------------------
        dist_matrix = self._compute_distances(
            queries,
            ref_emb,
            metric=self.distance_metric,
            use_gpu=use_gpu,
            logger=self.logger,
        )

        # ----------------------------------------------------------
        # (5) Redundancy filtering (optional)
        # ----------------------------------------------------------
        redundant = self._get_redundancy_members(accessions)

        # ----------------------------------------------------------
        # (6) Select neighbors (threshold + limit + redundancy)
        # ----------------------------------------------------------
        hits = self._select_neighbors(
            dist_matrix=dist_matrix,
            accessions=accessions,
            taxonomies=taxonomies,
            ref_ids=ref_ids,
            model_id=model_id,
            model_name=model_name,
            layer_index=layer_index,
            threshold=threshold,
            limit=limit,
            redundant_map=redundant,
        )

        # ----------------------------------------------------------
        # Logging
        # ----------------------------------------------------------
        elapsed = time.perf_counter() - t_start
        self.logger.info(
            f"process(): queries={len(accessions)} | hits={len(hits)} | "
            f"model={model_name} | layer={layer_index} | time={elapsed:.2f}s"
        )
        return hits

    def _get_reference_embeddings(self, model_id: int, layer_index: int | None) -> dict | None:
        """
        Load reference embeddings for (model_id, layer_index) into RAM in fp16.
        Uses streaming (yield_per) to avoid huge RAM spikes.
        """

        self.logger.info(
            f"Loading reference embeddings from database → model={model_id}, layer={layer_index}"
        )

        key = (int(model_id), None if layer_index is None else int(layer_index))

        # ----------------------------------------------------------
        # Cache hit
        # ----------------------------------------------------------
        if key in self._lookup_cache:
            self.logger.debug(f"reference cache hit → {key}")
            return self._lookup_cache[key]

        # ----------------------------------------------------------
        # SQL query
        # ----------------------------------------------------------
        q = (
            self.session.query(
                SequenceEmbedding.sequence_id,
                SequenceEmbedding.embedding,
                SequenceEmbedding.layer_index,
            )
            .filter(SequenceEmbedding.embedding_type_id == model_id)
        )

        if layer_index is None:
            q = q.filter(SequenceEmbedding.layer_index.is_(None))
        else:
            q = q.filter(SequenceEmbedding.layer_index == layer_index)

        limit = self.conf.get("limit_execution")
        if isinstance(limit, int) and limit > 0:
            self.logger.info(f"SQL LIMIT={limit} for model={model_id}, layer={layer_index}")
            q = q.limit(limit)

        # ----------------------------------------------------------
        # STREAMING iteration (CRITICAL)
        # ----------------------------------------------------------
        rows = q.yield_per(10_000)

        ids = []
        layers = []
        emb_list = []

        for seq_id, pgvec, lyr in rows:
            ids.append(int(seq_id))
            layers.append(int(lyr) if lyr is not None else -1)

            vec = pgvec.to_numpy().astype(np.float16, copy=False)
            emb_list.append(vec)

        # ----------------------------------------------------------
        # Handle empty result
        # ----------------------------------------------------------
        if not emb_list:
            self.logger.warning(f"No reference embeddings for model={model_id}, layer={layer_index}")
            return None

        # ----------------------------------------------------------
        # Stack to NumPy
        # ----------------------------------------------------------
        try:
            vectors = np.vstack(emb_list)
        except Exception as e:
            self.logger.error(f"Failed stacking fp16 vectors → {e}")
            return None

        # Free temp list early
        del emb_list

        ids = np.asarray(ids, dtype=np.int64)
        layers = np.asarray(layers, dtype=np.int64)

        ref = {
            "ids": ids,
            "vectors": vectors,
            "layers": layers,
        }

        # ----------------------------------------------------------
        # Cache + eviction
        # ----------------------------------------------------------
        self._lookup_cache[key] = ref
        if len(self._lookup_cache) > self.lookup_cache_max:
            old = next(iter(self._lookup_cache))
            if old != key:
                self._lookup_cache.pop(old, None)

        # ----------------------------------------------------------
        # Logging
        # ----------------------------------------------------------
        n, dim = vectors.shape
        mem = vectors.nbytes / 1024 ** 2

        self.logger.info(
            f"Reference embeddings loaded → model={model_id}, layer={layer_index} | "
            f"{n} rows, dim={dim}, fp16 RAM={mem:.2f} MB"
        )

        return ref

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

    def _validate_payload(self, payload: dict):
        """
        Validate the structure of the incoming batch payload and extract
        normalized parameters for the processing pipeline.
        """

        if not isinstance(payload, dict):
            raise ValueError("process(): payload must be a dict.")

        model_id = payload.get("model_id")
        layer_index = payload.get("layer_index")
        tasks = payload.get("tasks") or []

        if model_id is None or not tasks:
            raise ValueError("process(): invalid payload (missing model_id or empty tasks).")

        # --------------------------------------------------
        # Model name (from tasks, already normalized upstream)
        # --------------------------------------------------
        model_name = next(
            (t.get("model_name") for t in tasks if t.get("model_name") is not None),
            None
        )

        # --------------------------------------------------
        # Distance threshold
        # Priority:
        #   1) task-level (per model, injected at enqueue)
        #   2) None (no global fallback exists in YAML)
        # --------------------------------------------------
        threshold = next(
            (t.get("distance_threshold") for t in tasks if t.get("distance_threshold") is not None),
            None
        )

        # --------------------------------------------------
        # limit_per_entry → lookup.nn.limit_per_entry
        # --------------------------------------------------
        limit = int(
            (self.nn_cfg or {}).get("limit_per_entry", 10)
        )

        # --------------------------------------------------
        # use_gpu → lookup.compute.use_gpu
        # --------------------------------------------------
        use_gpu = bool(
            (self.compute_cfg or {}).get("use_gpu", True)
        )

        return model_id, layer_index, tasks, model_name, threshold, limit, use_gpu

    @staticmethod
    def _materialize_queries(tasks: list[dict]):
        """
        Load all query embeddings and taxonomy from HDF5.
        """

        by_h5 = defaultdict(list)
        for t in tasks:
            by_h5[t["h5_path"]].append(t)

        embeddings_list = []
        accessions = []
        taxonomies = []

        for h5_path, subset in by_h5.items():
            with h5py.File(h5_path, "r") as h5:
                for t in subset:
                    grp = t["h5_group"]
                    acc_name = grp.split("/", 1)[0]
                    acc = acc_name.removeprefix("accession_")

                    # embedding
                    emb = h5[grp]["embedding"][:]
                    embeddings_list.append(np.asarray(emb))
                    accessions.append(acc)

                    # taxonomy (query protein)
                    tax_id = h5[acc_name]["taxonomy"][()]
                    taxonomies.append(int(tax_id))

        if not embeddings_list:
            return None, [], []

        embeddings = np.stack(embeddings_list)
        return embeddings, accessions, taxonomies

    @staticmethod
    def _compute_distances(queries, references, metric: str, use_gpu: bool, logger=None):
        """
        Compute full N × M distance matrix between queries and references.

        Parameters
        ----------
        queries : np.ndarray (N × D)
        references : np.ndarray (M × D)
        metric : {"cosine", "euclidean"}
        use_gpu : bool
        """

        t0 = time.perf_counter()

        if use_gpu:
            import torch

            q = torch.tensor(queries, dtype=torch.float32).cuda(non_blocking=True)
            r = torch.tensor(references, dtype=torch.float32).cuda(non_blocking=True)

            if metric == "euclidean":
                q2 = (q ** 2).sum(dim=1, keepdim=True)
                r2 = (r ** 2).sum(dim=1).unsqueeze(0)
                d2 = q2 + r2 - 2 * (q @ r.T)
                dist = torch.sqrt(torch.clamp(d2, min=0.0)).cpu().numpy()

            elif metric == "cosine":
                qn = torch.nn.functional.normalize(q, p=2, dim=1)
                rn = torch.nn.functional.normalize(r, p=2, dim=1)
                dist = (1 - (qn @ rn.T)).cpu().numpy()

            else:
                raise ValueError(f"Unsupported distance metric: {metric}")

            if logger:
                logger.info(
                    "Distances computed on GPU | N=%d M=%d | elapsed=%.2f s",
                    q.shape[0], r.shape[0], time.perf_counter() - t0,
                )

            return dist

        # CPU fallback (SciPy)
        from scipy.spatial.distance import cdist
        dist = cdist(queries, references, metric=metric)

        if logger:
            logger.info(
                "Distances computed on CPU | N=%d M=%d | elapsed=%.2f s",
                queries.shape[0], references.shape[0], time.perf_counter() - t0,
            )

        return dist

    def _get_redundancy_members(self, accessions: list[str]) -> dict[str, set]:
        """
        Return a mapping accession → set of cluster members to exclude.

        If redundancy filtering is disabled, returns empty dict.
        """
        threshold = float(self.conf.get("redundancy_filter", 0))
        if threshold <= 0:
            return {}

        redundant = {}
        for acc in accessions:
            redundant[acc] = set(self.retrieve_cluster_members(acc))
        return redundant

    def _select_neighbors(
            self,
            dist_matrix: np.ndarray,
            accessions: list[str],
            taxonomies: list[int],
            ref_ids: np.ndarray,
            model_id: int,
            model_name: str,
            layer_index: int | None,
            threshold: float | None,
            limit: int,
            redundant_map: dict[str, set],
    ) -> list[dict]:
        """
        Convert each query row into a compact list of neighbor hits.

        Applies:
          - ascending sort
          - threshold filtering
          - per-entry limit
          - redundancy filtering (exclude cluster members)

        Additionally propagates:
          - query_taxonomy_id (from HDF5)
        """

        hits: list[dict] = []
        N = len(accessions)

        for i in range(N):
            acc = accessions[i]
            query_taxonomy_id = taxonomies[i]

            d_all = dist_matrix[i]
            ids_all = ref_ids

            # --------------------------------------------------
            # Redundancy filtering
            # --------------------------------------------------
            if acc in redundant_map and redundant_map[acc]:
                mask = ~np.isin(ids_all.astype(str), list(redundant_map[acc]))
                d = d_all[mask]
                ids = ids_all[mask]
            else:
                d = d_all
                ids = ids_all

            if d.size == 0:
                continue

            # --------------------------------------------------
            # Sort by ascending distance
            # --------------------------------------------------
            order = np.argsort(d)

            # --------------------------------------------------
            # Distance threshold filtering
            # --------------------------------------------------
            if threshold not in (None, 0):
                thr = float(threshold)
                order = order[d[order] <= thr]

            # --------------------------------------------------
            # Per-entry limit
            # --------------------------------------------------
            order = order[:limit]

            # --------------------------------------------------
            # Emit hits
            # --------------------------------------------------
            for idx in order:
                hits.append({
                    "accession": acc,
                    "ref_sequence_id": int(ids[idx]),
                    "distance": float(d[idx]),
                    "model_name": model_name,
                    "embedding_type_id": model_id,
                    "layer_index": layer_index,
                    "query_taxonomy_id": query_taxonomy_id,
                })

        return hits

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
            threads = self.mmseqs_threads

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

    def preload_reference_data(self) -> None:
        """
        Preload GO annotations and reference sequences into memory.

        - go_annotations: sequence_id -> list[GO annotation dicts]
        - ref_sequences: sequence_id -> protein sequence
        """
        self.logger.info("Preloading GO annotations and reference sequences…")

        self.go_annotations = {}
        self.ref_sequences = {}

        query = """
                SELECT s.id           AS sequence_id, \
                       s.sequence     AS sequence, \
                       pgo.go_id      AS go_id, \
                       gt.category    AS category, \
                       gt.description AS go_description, \
                       pgo.evidence_code, \
                       p.id           AS protein_id, \
                       p.organism, \
                       p.taxonomy_id, \
                       p.gene_name
                FROM sequence s
                         JOIN protein p ON s.id = p.sequence_id
                         JOIN protein_go_term_annotation pgo ON p.id = pgo.protein_id
                         JOIN go_terms gt ON pgo.go_id = gt.go_id \
                """

        with self.engine.begin() as conn:
            result = conn.execute(text(query))

            for row in result:
                sid = int(row.sequence_id)

                # 1) Store reference sequence (only once)
                if sid not in self.ref_sequences:
                    self.ref_sequences[sid] = row.sequence

                # 2) Store GO annotation
                ann = {
                    "go_id": row.go_id,
                    "category": row.category,
                    "go_description": row.go_description,
                    "evidence_code": row.evidence_code,
                    "protein_id": row.protein_id,
                    "organism": row.organism,
                    "taxonomy_id": row.taxonomy_id,
                    "gene_name": row.gene_name,
                }

                self.go_annotations.setdefault(sid, []).append(ann)

        self.logger.info(
            "Preload completed: %d sequences | %d annotated sequences",
            len(self.ref_sequences),
            len(self.go_annotations)
        )

    def store_entry(self, annotations_or_hits: list[dict]) -> None:
        """
        Persist results for a single (model, layer) lookup run.

        This method assumes that all incoming records belong to the same
        embedding model and layer. Results are written as one CSV file per
        query accession.

        Accepted input formats
        ----------------------
        - Compact neighbor hits produced by `process()`:
            Each item contains a `ref_sequence_id` and distance information.
        - Legacy already-expanded rows (kept for backward compatibility).

        Processing steps
        ----------------
        1) Expand neighbor hits into GO-level rows using cached GO annotations
           and preloaded reference sequences.
        2) Compute alignment metrics (identity, similarity, etc.) if both
           query and reference sequences are available.
        3) Compute a reliability index from the embedding distance.
        4) Optionally apply dynamic GO-level selection per accession and
           GO category (P/F/C), controlled by configuration.
        5) Write one CSV file per accession under `raw_results/`.

        Notes
        -----
        - Dynamic selection is applied *per accession and per GO category*
          after all metrics have been computed.
        - No fragmentation by model or layer is performed at write time;
          homogeneity is guaranteed upstream.
        """

        items = annotations_or_hits or []
        if not items:
            self.logger.info("store_entry: no annotations or hits to persist.")
            return

        try:
            os.makedirs(self.experiment_path, exist_ok=True)

            # ------------------------------------------------------------------
            # Detect input type
            # ------------------------------------------------------------------
            is_hits = (
                    isinstance(items, list)
                    and items
                    and isinstance(items[0], dict)
                    and "ref_sequence_id" in items[0]
            )

            keep_sequences = bool(
                (self.conf.get("postprocess", {}) or {}).get("keep_sequences", False)
            )
            store_workers = int(self.conf.get("store_workers", 4))

            # ------------------------------------------------------------------
            # 1) Expand hits → GO-level rows
            # ------------------------------------------------------------------
            if is_hits:
                expanded_rows: list[dict] = []

                # Open HDF5 only to read query sequences
                h5 = None
                if os.path.exists(self.embeddings_path):
                    try:
                        h5 = h5py.File(self.embeddings_path, "r")
                    except Exception as e:
                        self.logger.warning("store_entry: could not open HDF5: %s", e)

                qseq_cache: dict[str, str | None] = {}

                for hit in items:
                    acc = hit["accession"]
                    ref_id = int(hit["ref_sequence_id"])
                    dist = float(hit["distance"])

                    model_name = hit["model_name"]
                    model_id = int(hit["embedding_type_id"])
                    layer_index = hit.get("layer_index")

                    anns = self.go_annotations.get(ref_id, [])
                    if not anns:
                        continue

                    # Query sequence (cached per accession)
                    if acc not in qseq_cache:
                        seq_q = None
                        if h5 is not None:
                            try:
                                acc_node = f"accession_{acc}"
                                if acc_node in h5 and "sequence" in h5[acc_node]:
                                    raw = h5[acc_node]["sequence"][()]
                                    seq_q = (
                                        raw.decode("utf-8")
                                        if isinstance(raw, (bytes, bytearray))
                                        else str(raw)
                                    )
                            except Exception:
                                seq_q = None
                        qseq_cache[acc] = seq_q

                    seq_query = qseq_cache[acc]
                    seq_ref = self.ref_sequences.get(ref_id)

                    for ann in anns:
                        tax = compute_taxonomy(
                            hit.get("query_taxonomy_id"),
                            ann["taxonomy_id"],
                        )

                        expanded_rows.append({
                            # Context
                            "accession": acc,
                            "protein_id": ann["protein_id"],
                            "ref_sequence_id": ref_id,  # ← ESTA ES LA CLAVE
                            "model_name": model_name,
                            "embedding_type_id": model_id,
                            "layer_index": layer_index,

                            # Distance
                            "distance": dist,

                            # GO annotation
                            "go_id": ann["go_id"],
                            "category": ann["category"],
                            "evidence_code": ann["evidence_code"],
                            "go_description": ann["go_description"],
                            "organism": ann["organism"],
                            "gene_name": ann["gene_name"],
                            "taxonomy_id": ann["taxonomy_id"],

                            # Sequences (used only for post-processing)
                            "sequence_query": seq_query,
                            "sequence_reference": seq_ref,

                            # Taxonomy metrics
                            "taxonomy_query": hit.get("query_taxonomy_id"),
                            "taxonomy_ref": ann["taxonomy_id"],
                            "tax_distance": tax["distance"],
                            "tax_relation": tax["relation"],
                            "tax_lca": tax["lca"],
                            "tax_common_ancestors": tax["common_ancestors"],
                        })

                if h5 is not None:
                    h5.close()

                if not expanded_rows:
                    self.logger.info("store_entry: no rows after hit expansion.")
                    return

                df = pd.DataFrame(expanded_rows)

            else:
                # Legacy already-expanded rows (kept for compatibility)
                df = pd.DataFrame(items)
                if df.empty:
                    return

            # ------------------------------------------------------------------
            # 2) Polars: reliability index and alignment metrics
            # ------------------------------------------------------------------
            pl_df = pl.from_pandas(df)

            # Reliability index
            if self.distance_metric == "cosine":
                pl_df = pl_df.with_columns(
                    (1.0 - pl.col("distance")).alias("reliability_index")
                )
            elif self.distance_metric == "euclidean":
                pl_df = pl_df.with_columns(
                    (0.5 / (0.5 + pl.col("distance"))).alias("reliability_index")
                )
            else:
                pl_df = pl_df.with_columns(
                    (1.0 / (1.0 + pl.col("distance"))).alias("reliability_index")
                )

            # ------------------------------------------------------------------
            # Alignment metrics (only if sequences exist)
            # ------------------------------------------------------------------
            have_seqs = {"sequence_query", "sequence_reference"} <= set(pl_df.columns)

            if have_seqs:
                pairs_df = (
                    pl_df
                    .select(["sequence_query", "sequence_reference"])
                    .drop_nulls(subset=["sequence_query", "sequence_reference"])
                    .unique()
                )

                if pairs_df.height > 0:
                    pairs = pairs_df.to_dicts()

                    with ProcessPoolExecutor(max_workers=store_workers) as ex:
                        metrics = list(ex.map(compute_metrics, pairs))

                    met_df = pl.DataFrame(metrics)

                    pl_df = pl_df.join(
                        met_df,
                        on=["sequence_query", "sequence_reference"],
                        how="left",
                    )

            # Drop sequences unless explicitly requested
            if not keep_sequences:
                pl_df = pl_df.drop(
                    [c for c in ("sequence_query", "sequence_reference") if c in pl_df.columns]
                )

            # ------------------------------------------------------------------
            # Column ordering (semantic and stable)
            # ------------------------------------------------------------------
            preferred_order = [
                # Context
                "accession",
                "protein_id",
                "ref_sequence_id",
                "model_name",
                "embedding_type_id",
                "layer_index",

                # Scores
                "reliability_index",
                "distance",

                # GO annotation
                "go_id",
                "category",
                "evidence_code",
                "go_description",
                "organism",
                "gene_name",

                # Taxonomy
                "taxonomy_query",
                "taxonomy_ref",
                "tax_distance",
                "tax_relation",
                "tax_lca",
                "tax_common_ancestors",
            ]

            alignment_cols = [
                c for c in pl_df.columns
                if c.startswith(("identity", "similarity", "alignment_", "gaps_", "length_"))
            ]

            sequence_cols = [
                c for c in ("sequence_query", "sequence_reference")
                if c in pl_df.columns
            ]

            final_order = (
                    [c for c in preferred_order if c in pl_df.columns]
                    + alignment_cols
                    + sequence_cols
            )

            pl_df = pl_df.select(final_order)

            # ------------------------------------------------------------------
            # 2.5) Dynamic GO-level selection per accession and category
            # ------------------------------------------------------------------
            sel_cfg = (self.conf.get("lookup", {}) or {}).get("selection", {}) or {}
            selection_enabled = bool(sel_cfg.get("enabled", False))

            if selection_enabled:
                df_sel = pl_df.to_pandas()
                selected_chunks = []

                for accession, df_acc in df_sel.groupby("accession"):
                    for category in ("P", "F", "C"):
                        df_cat = df_acc[df_acc["category"] == category]
                        if df_cat.empty:
                            continue

                        try:
                            df_out = self.dynamic_select_hits_per_accession(
                                df_cat=df_cat,
                                category=category,
                            )
                            if not df_out.empty:
                                selected_chunks.append(df_out)
                        except KeyError:
                            # Category not configured → silently ignore
                            continue

                if not selected_chunks:
                    self.logger.info("store_entry: dynamic selection produced no rows.")
                    return

                pl_df = pl.from_pandas(
                    pd.concat(selected_chunks, ignore_index=True)
                )

            # ------------------------------------------------------------------
            # 3) Hierarchical CSV output
            # ------------------------------------------------------------------
            grouped = pl_df.group_by(
                ["accession", "layer_index", "model_name"],
                maintain_order=True
            )

            for (accession, layer_val, model_name), chunk in grouped:
                model_tag = re.sub(r"[^a-z0-9._-]", "_", str(model_name).lower())
                acc_tag = re.sub(r"[^a-z0-9._-]", "_", str(accession).lower())

                if layer_val is None:
                    out_dir = Path(self.experiment_path) / "raw_results" / model_tag / "legacy"
                else:
                    out_dir = Path(self.experiment_path) / "raw_results" / model_tag / f"layer_{int(layer_val)}"

                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{acc_tag}.csv"

                write_header = not out_path.exists()

                chunk.write_csv(
                    out_path,
                    include_header=write_header,
                    float_precision=self.precision,
                )

                self.logger.info("store_entry: wrote %d rows → %s", chunk.height, out_path)

        except Exception as e:
            self.logger.error("store_entry failed: %s", e, exc_info=True)
            raise

    def dynamic_select_hits_per_accession(self, df_cat, category):
        """
        Dynamic selection for one category (P/F/C) of one accession.
        Works directly at GO-level (df_cat retains go_id, category, etc.)
        """

        sel_cfg = self.conf["lookup"]["selection"]
        thr = sel_cfg["thresholds"]

        if category not in sel_cfg["categories"]:
            raise KeyError(f"Category '{category}' missing in config under selection.categories")

        cat_cfg = sel_cfg["categories"][category]

        K = cat_cfg["K_good"]
        interm_mult = cat_cfg.get("intermediate_multiplier", 1)
        weak_mult = cat_cfg.get("weak_multiplier", 1)

        # -----------------------------
        # Partition by identity levels
        # -----------------------------
        strong = df_cat[
            (df_cat["identity"] >= thr["strong"]["min_identity"]) &
            (df_cat["identity"] <= thr["strong"]["max_identity"])
            ]

        intermediate = df_cat[
            (df_cat["identity"] >= thr["intermediate"]["min_identity"]) &
            (df_cat["identity"] < thr["intermediate"]["max_identity"])
            ]

        weak = df_cat[
            (df_cat["identity"] >= thr["weak"]["min_identity"]) &
            (df_cat["identity"] < thr["weak"]["max_identity"])
            ]

        # -----------------------------
        # 1) Strong first
        # -----------------------------
        if len(strong) >= K:
            return strong.nsmallest(K, "distance")

        selected = [strong]
        remaining = K - len(strong)

        # -----------------------------
        # 2) Intermediate
        # -----------------------------
        take_inter = min(remaining * interm_mult, len(intermediate))
        if take_inter > 0:
            sel_inter = intermediate.nsmallest(take_inter, "distance")
            selected.append(sel_inter)
            remaining -= len(sel_inter)

        if remaining <= 0:
            return pd.concat(selected, ignore_index=True)

        # -----------------------------
        # 3) Weak
        # -----------------------------
        take_weak = min(remaining * weak_mult, len(weak))
        if take_weak > 0:
            sel_weak = weak.nsmallest(take_weak, "distance")
            selected.append(sel_weak)

        out = pd.concat(selected, ignore_index=True)

        # Upper bound safeguard
        max_hits_allowed = K * max(interm_mult, weak_mult)
        if len(out) > max_hits_allowed:
            out = out.nsmallest(max_hits_allowed, "distance")

        return out


    def load_model(self, model_type):
        """Placeholder: load a model into memory if required."""
        return

    def unload_model(self, model_type):
        """Placeholder: unload a model from memory if required."""
        return