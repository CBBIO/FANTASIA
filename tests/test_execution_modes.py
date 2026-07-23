import importlib
import sys
import types
from types import SimpleNamespace

import pytest


def _install_import_stubs(monkeypatch):
    def module(name, **attrs):
        mod = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(mod, key, value)
        monkeypatch.setitem(sys.modules, name, mod)
        return mod

    module("protein_information_system")
    module("protein_information_system.helpers")
    module("protein_information_system.helpers.logger")
    module(
        "protein_information_system.helpers.logger.logger",
        setup_logger=lambda *args, **kwargs: None,
    )
    module("protein_information_system.helpers.config")
    module(
        "protein_information_system.helpers.config.yaml",
        read_yaml_config=lambda *args, **kwargs: {},
    )
    module("protein_information_system.helpers.services")
    module(
        "protein_information_system.helpers.services.services",
        check_services=lambda *args, **kwargs: None,
    )
    module("protein_information_system.sql")
    module("protein_information_system.sql.model")
    module("protein_information_system.sql.model.model")
    module(
        "fantasia.src.embedder",
        SequenceEmbedder=object,
    )
    module(
        "fantasia.src.helpers.helpers",
        download_embeddings=lambda *args, **kwargs: None,
        load_dump_to_db=lambda *args, **kwargs: None,
        parse_unknown_args=lambda *args, **kwargs: {},
    )
    module(
        "fantasia.src.lookup",
        EmbeddingLookUp=object,
    )
    module(
        "fantasia.src.helpers.parser",
        build_parser=lambda *args, **kwargs: None,
    )


def _load_main(monkeypatch):
    _install_import_stubs(monkeypatch)
    sys.modules.pop("fantasia.main", None)
    return importlib.import_module("fantasia.main")


def test_embedding_only_runs_embedding_and_skips_lookup(monkeypatch, tmp_path):
    main = _load_main(monkeypatch)
    calls = []

    monkeypatch.setattr(
        main,
        "setup_experiment_directories",
        lambda conf, timestamp: {**conf, "experiment_path": str(tmp_path)},
    )

    class Embedder:
        def __init__(self, conf, timestamp):
            self.conf = conf

        def start(self):
            calls.append("embedding")
            (tmp_path / "embeddings.h5").touch()

    class Lookup:
        def __init__(self, conf, timestamp):
            pass

        def start(self):
            calls.append("lookup")

    monkeypatch.setattr(main, "SequenceEmbedder", Embedder)
    monkeypatch.setattr(main, "EmbeddingLookUp", Lookup)

    main.run_pipeline({"input": "query.fasta", "only_embedding": True, "only_lookup": False})

    assert calls == ["embedding"]


def test_lookup_only_runs_lookup_and_skips_embedding(monkeypatch, tmp_path):
    main = _load_main(monkeypatch)
    calls = []

    monkeypatch.setattr(
        main,
        "setup_experiment_directories",
        lambda conf, timestamp: {**conf, "experiment_path": str(tmp_path)},
    )

    class Embedder:
        def __init__(self, conf, timestamp):
            pass

        def start(self):
            calls.append("embedding")

    class Lookup:
        def __init__(self, conf, timestamp):
            calls.append(("embeddings_path", conf["embeddings_path"]))

        def start(self):
            calls.append("lookup")

    monkeypatch.setattr(main, "SequenceEmbedder", Embedder)
    monkeypatch.setattr(main, "EmbeddingLookUp", Lookup)

    main.run_pipeline({"input": "existing.h5", "only_embedding": False, "only_lookup": True})

    assert calls == [("embeddings_path", "existing.h5"), "lookup"]


def test_default_runs_embedding_then_lookup(monkeypatch, tmp_path):
    main = _load_main(monkeypatch)
    calls = []

    monkeypatch.setattr(
        main,
        "setup_experiment_directories",
        lambda conf, timestamp: {**conf, "experiment_path": str(tmp_path)},
    )

    class Embedder:
        def __init__(self, conf, timestamp):
            self.conf = conf

        def start(self):
            calls.append("embedding")
            (tmp_path / "embeddings.h5").touch()

    class Lookup:
        def __init__(self, conf, timestamp):
            calls.append(("embeddings_path", conf["embeddings_path"]))

        def start(self):
            calls.append("lookup")

    monkeypatch.setattr(main, "SequenceEmbedder", Embedder)
    monkeypatch.setattr(main, "EmbeddingLookUp", Lookup)

    main.run_pipeline({"input": "query.fasta", "only_embedding": False, "only_lookup": False})

    assert calls == [
        "embedding",
        ("embeddings_path", str(tmp_path / "embeddings.h5")),
        "lookup",
    ]


@pytest.mark.parametrize(
    "yaml_config",
    [
        {"lookup": {"taxonomy": {"get_descendants": True}}},
        {"taxonomy": {"get_descendants": True}},
    ],
)
def test_get_descendants_true_is_rejected_at_all_supported_paths(
    monkeypatch, yaml_config
):
    main = _load_main(monkeypatch)
    monkeypatch.setattr(main, "read_yaml_config", lambda _path: yaml_config)

    with pytest.raises(ValueError, match="deprecated and disabled"):
        main.load_and_merge_config(
            SimpleNamespace(command="run", config="unused.yaml"), []
        )


def test_get_descendants_is_forced_false_in_resolved_config(monkeypatch):
    main = _load_main(monkeypatch)
    yaml_config = {
        "lookup": {
            "taxonomy": {
                "exclude": ["10090"],
                "get_descendants": False,
            }
        }
    }
    monkeypatch.setattr(main, "read_yaml_config", lambda _path: yaml_config)

    resolved = main.load_and_merge_config(
        SimpleNamespace(command="run", config="unused.yaml"), []
    )

    assert resolved["get_descendants"] is False
    assert resolved["lookup"]["taxonomy"]["get_descendants"] is False


def test_embedding_cli_overrides_map_to_canonical_yaml_keys(monkeypatch):
    main = _load_main(monkeypatch)
    monkeypatch.setattr(
        main,
        "read_yaml_config",
        lambda _path: {"embedding": {}, "lookup": {"taxonomy": {}}},
    )

    resolved = main.load_and_merge_config(
        SimpleNamespace(
            command="run",
            config="unused.yaml",
            device="cpu",
            length_filter=0,
            sequence_queue_package=25,
        ),
        [],
    )

    assert resolved["embedding"]["device"] == "cpu"
    assert resolved["embedding"]["max_sequence_length"] == 0
    assert resolved["embedding"]["queue_batch_size"] == 25
