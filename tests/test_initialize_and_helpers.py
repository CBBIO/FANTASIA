import importlib
import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from fantasia.src.helpers import helpers
from fantasia.src.helpers.parser import DEFAULT_CONFIG_PATH, build_parser

DB_CONF = {
    "DB_USERNAME": "usuario",
    "DB_PASSWORD": "clave",
    "DB_HOST": "localhost",
    "DB_PORT": 5432,
    "DB_NAME": "BioData",
}


# ---------------------------------------------------------------------------
# #60: empty sequences must not reach parasail
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fn", [helpers.run_needle_from_strings, helpers.run_smith_waterman_from_strings])
@pytest.mark.parametrize("seq1, seq2", [("", "MKV"), ("MKV", ""), ("", "")])
def test_alignment_with_empty_sequence_returns_missing_metrics(fn, seq1, seq2):
    result = fn(seq1, seq2)
    assert set(result) == {
        "identity_count", "alignment_length", "identity_percentage",
        "similarity_percentage", "gaps_percentage", "alignment_score",
    }
    assert all(value is None for value in result.values())


def test_compute_metrics_with_empty_reference_sequence():
    metrics = helpers.compute_metrics({"sequence_query": "MKVLA", "sequence_reference": None})
    assert metrics["identity"] is None
    assert metrics["identity_sw"] is None
    assert metrics["length_reference"] == 0


def test_alignment_of_identical_sequences_is_unchanged():
    result = helpers.run_needle_from_strings("MKVLA", "MKVLA")
    assert result["identity_percentage"] == 1.0


# ---------------------------------------------------------------------------
# #54: file:// URLs and local paths
# ---------------------------------------------------------------------------

def test_local_path_from_url(tmp_path):
    dump = tmp_path / "ref dump.backup"
    assert helpers.local_path_from_url(dump.as_uri()) == dump
    assert helpers.local_path_from_url(str(dump)) == dump
    assert helpers.local_path_from_url("https://zenodo.org/records/1/files/x.backup?download=1") is None


def test_download_embeddings_uses_local_file_in_place(tmp_path, monkeypatch):
    dump = tmp_path / "shared" / "ref.backup"
    dump.parent.mkdir()
    dump.write_bytes(b"PGDMP")
    monkeypatch.setattr(helpers.requests, "get", lambda *a, **k: pytest.fail("must not download"))

    result = helpers.download_embeddings(dump.as_uri(), tmp_path / "embeddings" / "ref.backup")

    assert result == dump
    assert not (tmp_path / "embeddings").exists()


class _FakeResponse:
    def __init__(self, chunks, content_length):
        self._chunks = chunks
        self.headers = {"content-length": str(content_length)}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        return iter(self._chunks)


def test_download_is_atomic(tmp_path, monkeypatch):
    monkeypatch.setattr(helpers.requests, "get", lambda *a, **k: _FakeResponse([b"PG", b"DMP"], 5))
    dest = tmp_path / "ref.backup"

    assert helpers.download_embeddings("https://example.org/ref.backup", dest) == dest

    assert dest.read_bytes() == b"PGDMP"
    assert list(tmp_path.iterdir()) == [dest]


def test_truncated_download_leaves_no_file(tmp_path, monkeypatch):
    monkeypatch.setattr(helpers.requests, "get", lambda *a, **k: _FakeResponse([b"PG"], 5))
    dest = tmp_path / "ref.backup"

    with pytest.raises(OSError, match="Incomplete download"):
        helpers.download_embeddings("https://example.org/ref.backup", dest)

    # Nothing a later run could mistake for a complete dump.
    assert list(tmp_path.iterdir()) == []


def test_db_url_escapes_credentials():
    url = helpers._db_url({**DB_CONF, "DB_PASSWORD": "p@ss/w:rd"})
    assert url.drivername == "postgresql+psycopg2"
    assert url.password == "p@ss/w:rd"
    assert url.host == "localhost"
    assert "p%40ss%2Fw%3Ard" in url.render_as_string(hide_password=False)


def test_download_embeddings_missing_local_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        helpers.download_embeddings((tmp_path / "missing.backup").as_uri(), tmp_path / "out.backup")


# ---------------------------------------------------------------------------
# #57 / #61: pg_restore outcome
# ---------------------------------------------------------------------------

def _fake_restore(monkeypatch, returncode, stderr):
    monkeypatch.setattr(
        helpers.subprocess, "run",
        lambda *a, **k: SimpleNamespace(returncode=returncode, stdout="", stderr=stderr),
    )


def _dump(tmp_path):
    dump = tmp_path / "ref.backup"
    dump.write_bytes(b"PGDMP")
    return dump


def test_load_dump_reports_unsupported_dump_version(tmp_path, monkeypatch, caplog):
    _fake_restore(monkeypatch, 1, "pg_restore: error: unsupported version (1.16) in file header\n")
    with caplog.at_level(logging.ERROR):
        ok = helpers.load_dump_to_db(_dump(tmp_path), DB_CONF, drop_and_init=False)
    assert ok is False
    assert "PostgreSQL 18 client" in caplog.text


def test_load_dump_tolerates_transaction_timeout_on_older_server(tmp_path, monkeypatch):
    stderr = (
        "pg_restore: connecting to database for restore\n"
        "pg_restore: error: could not execute query: ERROR:  "
        "unrecognized configuration parameter \"transaction_timeout\"\n"
        "Command was: SET transaction_timeout = 0;\n"
        "pg_restore: creating TABLE \"public.sequence\"\n"
        "pg_restore: warning: errors ignored on restore: 1\n"
    )
    _fake_restore(monkeypatch, 1, stderr)
    assert helpers.load_dump_to_db(_dump(tmp_path), DB_CONF, drop_and_init=False) is True


def test_load_dump_fails_on_other_errors(tmp_path, monkeypatch):
    stderr = (
        "pg_restore: error: could not execute query: ERROR:  "
        "unrecognized configuration parameter \"transaction_timeout\"\n"
        "pg_restore: error: could not execute query: ERROR:  type \"vector\" does not exist\n"
        "pg_restore: warning: errors ignored on restore: 2\n"
    )
    _fake_restore(monkeypatch, 1, stderr)
    assert helpers.load_dump_to_db(_dump(tmp_path), DB_CONF, drop_and_init=False) is False


# ---------------------------------------------------------------------------
# #46: resources resolved independently of the working directory
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("command", ["initialize", "run"])
def test_default_config_is_bundled_config(tmp_path, monkeypatch, command):
    monkeypatch.chdir(tmp_path)
    args, _ = build_parser().parse_known_args([command])
    assert args.config == DEFAULT_CONFIG_PATH
    assert Path(args.config).is_file()
    assert Path(args.config).parent.name == "fantasia"


def test_go_obo_is_resolved_from_installation(tmp_path, monkeypatch):
    lookup = pytest.importorskip("fantasia.src.lookup")  # needs torch
    monkeypatch.chdir(tmp_path)
    assert Path(lookup._resolve_go_obo_path({"base_directory": str(tmp_path)})) == lookup.BUNDLED_GO_OBO

    monkeypatch.setattr(lookup, "BUNDLED_GO_OBO", tmp_path / "missing.obo")
    assert lookup._resolve_go_obo_path({"base_directory": str(tmp_path / "base")}) == str(
        tmp_path / "base" / "go-basic.obo"
    )


# ---------------------------------------------------------------------------
# #57 / #61: fantasia initialize must exit non-zero when the restore fails
# ---------------------------------------------------------------------------

def _load_main(monkeypatch, load_dump_result, download_calls):
    def module(name, **attrs):
        mod = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(mod, key, value)
        monkeypatch.setitem(sys.modules, name, mod)
        return mod

    for name in (
        "protein_information_system",
        "protein_information_system.helpers",
        "protein_information_system.helpers.logger",
        "protein_information_system.helpers.config",
        "protein_information_system.helpers.services",
        "protein_information_system.sql",
        "protein_information_system.sql.model",
        "protein_information_system.sql.model.model",
    ):
        module(name)
    module("protein_information_system.helpers.logger.logger", setup_logger=lambda *a, **k: None)
    module("protein_information_system.helpers.config.yaml", read_yaml_config=lambda *a, **k: {})
    module("protein_information_system.helpers.services.services", check_services=lambda *a, **k: None)
    module("fantasia.src.embedder", SequenceEmbedder=object)
    module("fantasia.src.lookup", EmbeddingLookUp=object)

    def download(url, tar_path, **kwargs):
        download_calls.append((url, tar_path))
        return Path(tar_path)

    module(
        "fantasia.src.helpers.helpers",
        download_embeddings=download,
        load_dump_to_db=lambda *a, **k: load_dump_result,
        reference_has_embeddings=lambda *a, **k: False,
        parse_unknown_args=lambda *a, **k: {},
    )
    sys.modules.pop("fantasia.main", None)
    return importlib.import_module("fantasia.main")


def test_initialize_raises_when_restore_fails(monkeypatch, tmp_path):
    calls = []
    main = _load_main(monkeypatch, load_dump_result=False, download_calls=calls)
    conf = {"base_directory": str(tmp_path), "embeddings_url": "https://example.org/ref.backup?download=1"}

    with pytest.raises(RuntimeError, match="Could not load the reference dump"):
        main.initialize(conf)
    assert calls[0][1] == str(tmp_path / "embeddings" / "ref.backup")


def test_initialize_succeeds_when_restore_succeeds(monkeypatch, tmp_path):
    main = _load_main(monkeypatch, load_dump_result=True, download_calls=[])
    main.initialize({"base_directory": str(tmp_path), "embeddings_url": "https://example.org/ref.backup"})


def test_cli_initialize_exits_non_zero_on_failure(monkeypatch, tmp_path):
    main = _load_main(monkeypatch, load_dump_result=False, download_calls=[])
    conf = {"base_directory": str(tmp_path), "embeddings_url": "https://example.org/ref.backup",
            "log_path": str(tmp_path / "logs")}
    monkeypatch.setattr(main, "build_parser", lambda: SimpleNamespace(
        parse_known_args=lambda: (SimpleNamespace(command="initialize"), [])))
    monkeypatch.setattr(main, "load_and_merge_config", lambda args, unknown: dict(conf))
    monkeypatch.setattr(main, "setup_logger", lambda *a, **k: logging.getLogger("test"))

    with pytest.raises(SystemExit) as excinfo:
        main.main()
    assert excinfo.value.code == 1


def test_postgres_client_encoding_defaults_to_utf8(monkeypatch):
    import os
    # setenv first so monkeypatch restores the original state, even if the
    # variable was absent, after main sets it on import.
    monkeypatch.setenv("PGCLIENTENCODING", "placeholder")
    monkeypatch.delenv("PGCLIENTENCODING")
    _load_main(monkeypatch, load_dump_result=True, download_calls=[])
    assert os.environ["PGCLIENTENCODING"] == "UTF8"


def test_constants_path_falls_back_to_bundled_file(monkeypatch, tmp_path):
    main = _load_main(monkeypatch, load_dump_result=True, download_calls=[])
    monkeypatch.chdir(tmp_path)
    bundled = main.resolve_constants_path("./fantasia/constants.yaml")
    assert Path(bundled).is_file()
    assert Path(bundled).parent.name == "fantasia"
    custom = tmp_path / "my_constants.yaml"
    custom.write_text("{}")
    assert main.resolve_constants_path(str(custom)) == str(custom)
