import gzip
import importlib
import sys
import types


def _load_embedder(monkeypatch):
    module = types.ModuleType(
        "protein_information_system.operation.embedding.sequence_embedding"
    )
    module.SequenceEmbeddingManager = object
    monkeypatch.setitem(
        sys.modules,
        "protein_information_system.operation.embedding.sequence_embedding",
        module,
    )
    sys.modules.pop("fantasia.src.embedder", None)
    return importlib.import_module("fantasia.src.embedder").SequenceEmbedder


def test_plain_and_gzip_fasta_inputs_are_equivalent(monkeypatch, tmp_path):
    embedder = _load_embedder(monkeypatch)
    fasta_text = ">protein_1 description\nACDEFG\n>protein_2\nMNPQRS\n"
    plain_path = tmp_path / "proteins.fasta"
    gzip_path = tmp_path / "proteins.fasta.gz"
    plain_path.write_text(fasta_text, encoding="utf-8")
    with gzip.open(gzip_path, "wt", encoding="utf-8") as handle:
        handle.write(fasta_text)

    plain_records = embedder._parse_fasta_robust(None, str(plain_path))
    gzip_records = embedder._parse_fasta_robust(None, str(gzip_path))

    assert plain_records == gzip_records
    assert [(record.id, record.seq) for record in gzip_records] == [
        ("protein_1", "ACDEFG"),
        ("protein_2", "MNPQRS"),
    ]
