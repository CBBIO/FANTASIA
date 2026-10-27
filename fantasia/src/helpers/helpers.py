# helpers.py
from __future__ import annotations

import logging
import os
import subprocess
import urllib.parse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import parasail
import requests
from ete3 import NCBITaxa
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from tqdm import tqdm

__all__ = [
    "download_embeddings",
    "load_dump_to_db",
    "reference_has_embeddings",
    "parse_unknown_args",
    "compute_metrics",
    "run_smith_waterman_from_strings",
    "run_needle_from_strings",
    "get_descendant_ids",
]


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _get_logger(logger: Optional[logging.Logger] = None) -> logging.Logger:
    """Return the provided logger or a module-scoped default."""
    return logger if logger is not None else logging.getLogger(__name__)


def _db_url(db_config: Mapping[str, Any]) -> URL:
    """
    Build the SQLAlchemy URL for the reference database.

    Uses psycopg2, the driver shipped with protein-information-system: a bare
    ``postgresql://`` URL selects psycopg (v3) since SQLAlchemy 2.1. ``URL.create``
    also escapes credentials containing characters such as ``@`` or ``/``.
    """
    return URL.create(
        drivername="postgresql+psycopg2",
        username=str(db_config["DB_USERNAME"]),
        password=str(db_config["DB_PASSWORD"]),
        host=str(db_config["DB_HOST"]),
        port=int(db_config["DB_PORT"]),
        database=str(db_config["DB_NAME"]),
    )


def _coerce_bool(value: Any) -> bool:
    """Coerce common truthy / falsy CLI tokens into a boolean."""
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    return s in {"1", "true", "t", "yes", "y", "on"}


# ---------------------------------------------------------------------------
# Network / IO
# ---------------------------------------------------------------------------

def local_path_from_url(url: str) -> Optional[Path]:
    """
    Return the local path referenced by ``url``, or None for remote URLs.

    Accepts ``file://`` URLs (``file:///data/dump.backup``) and plain filesystem
    paths, so a dump already present on shared storage is used in place instead
    of being copied into ``base_directory``.
    """
    parsed = urllib.parse.urlparse(str(url))
    if parsed.scheme == "file":
        return Path(urllib.parse.unquote(parsed.path)).expanduser().resolve()
    if parsed.scheme == "":
        return Path(str(url)).expanduser().resolve()
    return None


def download_embeddings(
        url: str,
        tar_path: str | Path,
        *,
        overwrite: bool = False,
        chunk_size: int = 1024 * 1024,
        timeout: int = 60,
        logger: Optional[logging.Logger] = None,
) -> Path:
    """
    Download an embeddings TAR file from a URL with a progress bar.

    ``file://`` URLs and plain local paths are not copied: the referenced file is
    returned as is.

    Parameters
    ----------
    url : str
        Source URL for the embeddings archive.
    tar_path : str | pathlib.Path
        Destination path (file). Parent folders are created if missing.
    overwrite : bool, optional
        Whether to overwrite an existing file. Defaults to False.
    chunk_size : int, optional
        Streaming chunk size in bytes. Defaults to 1 MiB.
    timeout : int, optional
        Per-request timeout (seconds). Defaults to 60.
    logger : logging.Logger, optional
        Logger instance. If None, uses a module-scoped logger.

    Returns
    -------
    pathlib.Path
        Absolute path to the downloaded file.

    Raises
    ------
    requests.HTTPError
        If the HTTP request fails.
    OSError
        If the file cannot be written.
    """
    log = _get_logger(logger)
    local = local_path_from_url(url)
    if local is not None:
        if not local.exists():
            raise FileNotFoundError(f"Local embeddings file not found: {local}")
        log.info("Using local embeddings file: %s (no download).", local)
        return local

    dest = Path(tar_path).expanduser().resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not overwrite:
        log.info("Embeddings file already present: %s (skipping download).", dest)
        return dest

    log.info("Downloading embeddings from %s → %s", url, dest)

    # Download to a temporary name and rename on success, so an interrupted or
    # concurrent download never leaves a truncated file at ``dest`` that a later
    # run would take as complete.
    partial = dest.with_name(f"{dest.name}.{os.getpid()}.part")
    with requests.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0)) or None

        with open(partial, "wb") as fh, tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc="Downloading",
                disable=total is None,  # only show if we know the size
        ) as bar:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                fh.write(chunk)
                if total:
                    bar.update(len(chunk))

    size = partial.stat().st_size
    if total is not None and size != total:
        partial.unlink()
        raise OSError(f"Incomplete download of {url}: {size} of {total} bytes")
    os.replace(partial, dest)

    log.info("Embeddings downloaded successfully: %s", dest)
    return dest


def load_dump_to_db(
        dump_path: str | Path,
        db_config: Mapping[str, Any],
        *,
        drop_and_init: bool = True,
        logger: Optional[logging.Logger] = None,
) -> bool:
    """
    Load a PostgreSQL custom/TAR dump into a database via `pg_restore`.

    This function can (optionally) reset the target schema and ensure the
    `vector` extension is available before restoring.

    Parameters
    ----------
    dump_path : str | pathlib.Path
        Path to the database dump (e.g., `.dump` or `.tar`).
    db_config : Mapping[str, Any]
        Must provide the keys: DB_USERNAME, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME.
    drop_and_init : bool, optional
        If True, resets `public` schema and ensures `vector` extension. Defaults to True.
    logger : logging.Logger, optional
        Logger instance.

    Returns
    -------
    bool
        True if the restore finished successfully; False otherwise.

    Notes
    -----
    * Requires `pg_restore` to be available on PATH.
    * Uses `PGPASSWORD` environment variable to authenticate non-interactively.
    """
    log = _get_logger(logger)
    dump = Path(dump_path).expanduser().resolve()
    if not dump.exists():
        log.error("Dump file not found: %s", dump)
        return False

    required = {"DB_USERNAME", "DB_PASSWORD", "DB_HOST", "DB_PORT", "DB_NAME"}
    missing = required.difference(db_config.keys())
    if missing:
        log.error("Missing DB config keys: %s", ", ".join(sorted(missing)))
        return False

    url = _db_url(db_config)

    if drop_and_init:
        log.info("Resetting schema and ensuring extensions…")
        engine = create_engine(url, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execution_options(isolation_level="AUTOCOMMIT")
            # Reset schema and ensure pgvector (extension name is typically lowercase 'vector')
            conn.execute(text("DROP SCHEMA IF EXISTS public CASCADE;"))
            conn.execute(text("CREATE SCHEMA public;"))
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        log.info("Schema prepared and 'vector' extension ensured.")

    log.info("Restoring dump into the database via pg_restore…")
    env: Dict[str, str] = {**os.environ, "PGPASSWORD": str(db_config["DB_PASSWORD"])}

    cmd = [
        "pg_restore",
        "--verbose",
        "--no-owner",
        "-U", str(db_config["DB_USERNAME"]),
        "-h", str(db_config["DB_HOST"]),
        "-p", str(db_config["DB_PORT"]),
        "-d", str(db_config["DB_NAME"]),
        str(dump),
    ]
    log.debug("Executing: %s", " ".join(cmd))

    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode == 0:
        log.info("Database dump loaded successfully.")
        if proc.stdout:
            log.debug(proc.stdout)
        return True

    stderr = (proc.stderr or "").strip()
    errors = _pg_restore_errors(stderr)
    if errors and all(_is_ignorable_restore_error(e) for e in errors):
        log.warning(
            "pg_restore exited with %s, but only reported settings unknown to this server "
            "(dump created by a newer PostgreSQL). The data was restored: %s",
            proc.returncode, "; ".join(errors),
        )
        return True

    log.error("pg_restore failed (exit %s).", proc.returncode)
    if stderr:
        log.error(stderr)
    if "unsupported version" in stderr:
        log.error(
            "The dump was created by a newer pg_dump than the pg_restore in use. "
            "The reference dumps require PostgreSQL 18 client tools."
        )
    return False


# Settings emitted by newer pg_dump versions that older servers do not know.
# Failing to SET them does not affect the restored data.
_IGNORABLE_RESTORE_SETTINGS = ("transaction_timeout",)


def _pg_restore_errors(stderr: str) -> List[str]:
    """Extract the ``pg_restore: error:`` messages from pg_restore stderr."""
    errors = []
    for line in stderr.splitlines():
        line = line.strip()
        if line.startswith("pg_restore: error:"):
            errors.append(line)
    return errors


def _is_ignorable_restore_error(message: str) -> bool:
    """Return True for errors caused only by unknown session settings."""
    return "unrecognized configuration parameter" in message and any(
        f'"{name}"' in message for name in _IGNORABLE_RESTORE_SETTINGS
    )


def reference_has_embeddings(db_config: Mapping[str, Any]) -> bool:
    """
    Return True if the reference database holds at least one embedding.

    A missing ``sequence_embeddings`` table (database never initialized) counts
    as empty.
    """
    engine = create_engine(_db_url(db_config), pool_pre_ping=True)
    try:
        with engine.connect() as conn:
            if conn.execute(text("SELECT to_regclass('public.sequence_embeddings')")).scalar() is None:
                return False
            return conn.execute(text("SELECT EXISTS (SELECT 1 FROM sequence_embeddings)")).scalar()
    finally:
        engine.dispose()


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def parse_unknown_args(unknown_args: Iterable[str]) -> Dict[str, Any]:
    """
    Convert a flat list of unknown CLI args into a dict of `--key value` pairs.

    Behavior
    --------
    * Tokens that start with `--` are treated as keys.
    * If the next token exists and does NOT start with `--`, it is used as value.
    * Otherwise the key is treated as a boolean flag with value `True`.
    * Boolean-like values are NOT coerced here (kept as strings) to avoid
      surprising conversions; the caller may coerce later if desired.

    Parameters
    ----------
    unknown_args : Iterable[str]
        Typically the list returned by argparse's `parse_known_args()[1]`.

    Returns
    -------
    dict
        Mapping of keys (without `--`) to values (str or True).
    """
    result: Dict[str, Any] = {}
    args = list(unknown_args)
    i = 0
    while i < len(args):
        token = args[i]
        if token.startswith("--"):
            key = token[2:]
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                result[key] = args[i + 1]
                i += 2
                continue
            result[key] = True
        i += 1
    return result


# ---------------------------------------------------------------------------
# Alignment & metrics
# ---------------------------------------------------------------------------

def _empty_alignment_metrics(len1: int, len2: int) -> Dict[str, Optional[float]]:
    """
    Metrics for a pair that cannot be aligned because a sequence is empty.

    Parasail rejects empty inputs, so the alignment is skipped. Metrics are
    reported as missing (None) rather than 0.0 so that they cannot be mistaken
    for a real alignment with no identity. Callers report the number of skipped
    pairs once; this logs each pair at DEBUG level only.
    """
    logging.getLogger(__name__).debug("Empty sequence (lengths %d / %d); alignment skipped.", len1, len2)
    return {
        "identity_count": None,
        "alignment_length": None,
        "identity_percentage": None,
        "similarity_percentage": None,
        "gaps_percentage": None,
        "alignment_score": None,
    }


def run_smith_waterman_from_strings(
        seq1: str,
        seq2: str,
        *,
        gap_open: int = 10,
        gap_extend: int = 1,
) -> Dict[str, float]:
    """
    Compute local alignment (Smith–Waterman) between two sequences using Parasail.

    Parameters
    ----------
    seq1, seq2 : str
        Amino-acid sequences.
    gap_open : int, optional
        Gap opening penalty. Defaults to 10.
    gap_extend : int, optional
        Gap extension penalty. Defaults to 1.

    Returns
    -------
    dict
        Metrics with keys:
          - identity_count
          - alignment_length
          - identity_percentage
          - similarity_percentage
          - gaps_percentage
          - alignment_score
    """
    if not seq1 or not seq2:
        return _empty_alignment_metrics(len(seq1 or ""), len(seq2 or ""))

    result = parasail.sw_trace_striped_32(seq1, seq2, gap_open, gap_extend, parasail.blosum62)

    aligned_q = result.traceback.query
    aligned_r = result.traceback.ref
    comp_line = result.traceback.comp

    aln_len = len(aligned_q)
    if aln_len == 0:
        return {
            "identity_count": 0.0,
            "alignment_length": 0.0,
            "identity_percentage": 0.0,
            "similarity_percentage": 0.0,
            "gaps_percentage": 0.0,
            "alignment_score": float(result.score),
        }

    matches = sum(a == b for a, b in zip(aligned_q, aligned_r) if a != "-" and b != "-")
    similarity = sum(c in "|:" for c in comp_line)
    gaps = aligned_q.count("-") + aligned_r.count("-")

    return {
        "identity_count": float(matches),
        "alignment_length": float(aln_len),
        "identity_percentage": matches / aln_len,
        "similarity_percentage": similarity / aln_len,
        "gaps_percentage": gaps / aln_len,
        "alignment_score": float(result.score),
    }


def run_needle_from_strings(
        seq1: str,
        seq2: str,
        *,
        gap_open: int = 10,
        gap_extend: int = 1,
) -> Dict[str, float]:
    """
    Compute global alignment (Needleman–Wunsch) between two sequences using Parasail.

    Parameters
    ----------
    seq1, seq2 : str
        Amino-acid sequences.
    gap_open : int, optional
        Gap opening penalty. Defaults to 10.
    gap_extend : int, optional
        Gap extension penalty. Defaults to 1.

    Returns
    -------
    dict
        Metrics with keys:
          - identity_count
          - alignment_length
          - identity_percentage
          - similarity_percentage
          - gaps_percentage
          - alignment_score
    """
    if not seq1 or not seq2:
        return _empty_alignment_metrics(len(seq1 or ""), len(seq2 or ""))

    result = parasail.nw_trace_striped_32(seq1, seq2, gap_open, gap_extend, parasail.blosum62)

    aligned_q = result.traceback.query
    aligned_r = result.traceback.ref
    comp_line = result.traceback.comp

    aln_len = len(aligned_q)
    if aln_len == 0:
        return {
            "identity_count": 0.0,
            "alignment_length": 0.0,
            "identity_percentage": 0.0,
            "similarity_percentage": 0.0,
            "gaps_percentage": 0.0,
            "alignment_score": float(result.score),
        }

    matches = sum(a == b for a, b in zip(aligned_q, aligned_r) if a != "-" and b != "-")
    similarity = sum(c in "|:" for c in comp_line)
    gaps = aligned_q.count("-") + aligned_r.count("-")

    return {
        "identity_count": float(matches),
        "alignment_length": float(aln_len),
        "identity_percentage": matches / aln_len,
        "similarity_percentage": similarity / aln_len,
        "gaps_percentage": gaps / aln_len,
        "alignment_score": float(result.score),
    }


def compute_metrics(row: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Compute both global (Needleman–Wunsch) and local (Smith–Waterman) alignment metrics
    for a pair of sequences, returning a merged record.

    Parameters
    ----------
    row : Mapping[str, Any]
        Must contain keys: "sequence_query" and "sequence_reference".

    Returns
    -------
    dict
        Combined metrics with the following keys:
          - sequence_query, sequence_reference
          - identity, similarity, alignment_score, gaps_percentage, alignment_length
          - identity_sw, similarity_sw, alignment_score_sw, gaps_percentage_sw, alignment_length_sw
          - length_query, length_reference
    """
    seq1 = str(row.get("sequence_query", "") or "")
    seq2 = str(row.get("sequence_reference", "") or "")

    nw = run_needle_from_strings(seq1, seq2)
    sw = run_smith_waterman_from_strings(seq1, seq2)

    return {
        "sequence_query": seq1,
        "sequence_reference": seq2,

        # Global (Needleman–Wunsch) — kept for backward compatibility
        "identity": nw["identity_percentage"],
        "similarity": nw.get("similarity_percentage"),
        "alignment_score": nw["alignment_score"],
        "gaps_percentage": nw.get("gaps_percentage"),
        "alignment_length": nw["alignment_length"],

        # Lengths
        "length_query": len(seq1),
        "length_reference": len(seq2),

        # Local (Smith–Waterman)
        "identity_sw": sw["identity_percentage"],
        "similarity_sw": sw.get("similarity_percentage"),
        "alignment_score_sw": sw["alignment_score"],
        "gaps_percentage_sw": sw.get("gaps_percentage"),
        "alignment_length_sw": sw["alignment_length"],
    }


# ---------------------------------------------------------------------------
# Taxonomy helpers
# ---------------------------------------------------------------------------

def get_descendant_ids(
        parent_ids: Iterable[int | str],
        *,
        include_parents: bool = True,
        ncbi: Optional[NCBITaxa] = None,
        logger: Optional[logging.Logger] = None,
) -> List[int]:
    """
    Resolve descendant NCBI Taxonomy IDs for a set of parent taxa.

    Parameters
    ----------
    parent_ids : Iterable[int | str]
        Parent taxonomy identifiers (numeric). Strings are accepted and coerced to int.
    include_parents : bool, optional
        If True, include the original parent IDs in the result. Defaults to True.
    ncbi : ete3.NCBITaxa, optional
        Reusable NCBITaxa instance. If None, a new one is constructed.
    logger : logging.Logger, optional
        Logger instance.

    Returns
    -------
    list[int]
        Unique taxonomy IDs (descendants ± parents) with no duplicates.

    Notes
    -----
    * ETE's NCBITaxa may trigger a local database download/update on first use.
    """
    log = _get_logger(logger)
    ids: List[int] = []
    for t in parent_ids:
        s = str(t).strip()
        if s.isdigit():
            ids.append(int(s))
        else:
            log.warning("Ignoring non-numeric taxonomy id: %r", t)

    if not ids:
        return []

    ncbi_obj = ncbi or NCBITaxa()
    out: set[int] = set()
    for taxon in ids:
        try:
            desc = ncbi_obj.get_descendant_taxa(taxon, intermediate_nodes=True)
            out.update(int(x) for x in desc)
        except Exception as e:
            log.warning("Failed to resolve descendants for taxon %s: %s", taxon, e)

    if include_parents:
        out.update(ids)

    return sorted(out)
