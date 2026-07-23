import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fantasia.src.embedder import SequenceEmbedder


class _Logger:
    def info(self, *_args, **_kwargs):
        pass


class _FakeModule:
    model_argument = None
    tokenizer_argument = None

    @classmethod
    def load_model(cls, model_name, _conf):
        cls.model_argument = model_name
        return "model"

    @classmethod
    def load_tokenizer(cls, model_name):
        cls.tokenizer_argument = model_name
        return "tokenizer"


class ModelRevisionEnforcementTests(unittest.TestCase):
    def _embedder(self, model_type, config, module=_FakeModule):
        embedder = object.__new__(SequenceEmbedder)
        embedder.conf = {"embedding": {"models": {model_type: config}}}
        embedder.types = {
            model_type: {"module": module, "model_name": config["repository"]}
        }
        embedder.model_instances = {}
        embedder.tokenizer_instances = {}
        embedder.logger = _Logger()
        return embedder

    def test_model_and_tokenizer_receive_pinned_snapshot_path(self):
        revision = "a" * 40
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = Path(tmp) / revision
            snapshot.mkdir()
            embedder = self._embedder(
                "Prot-T5",
                {"repository": "example/model", "revision": revision},
            )
            with patch(
                "huggingface_hub.snapshot_download", return_value=str(snapshot)
            ) as download:
                embedder.load_model("Prot-T5")

            download.assert_called_once_with(
                repo_id="example/model", revision=revision
            )
            self.assertEqual(_FakeModule.model_argument, str(snapshot.resolve()))
            self.assertEqual(_FakeModule.tokenizer_argument, str(snapshot.resolve()))
            self.assertEqual(embedder.model_instances["Prot-T5"], "model")

    def test_legacy_config_without_provenance_keys_uses_supported_defaults(self):
        revision = "973be27c52ee6474de9c945952a8008aeb2a1a73"
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = Path(tmp) / revision
            snapshot.mkdir()
            embedder = self._embedder("Prot-T5", {"repository": "placeholder"})
            embedder.conf["embedding"]["models"]["Prot-T5"] = {"enabled": True}
            with patch(
                "huggingface_hub.snapshot_download", return_value=str(snapshot)
            ) as download:
                embedder.load_model("Prot-T5")
            download.assert_called_once_with(
                repo_id="Rostlab/prot_t5_xl_uniref50", revision=revision
            )

    def test_esmc_weight_checksum_is_verified(self):
        import esm.pretrained

        revision = "b" * 40
        payload = b"pinned-esmc-weight"
        expected = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = Path(tmp) / revision
            weight_dir = snapshot / "data" / "weights"
            weight_dir.mkdir(parents=True)
            (weight_dir / "weights.pth").write_bytes(payload)

            class FakeESMCModule(_FakeModule):
                @classmethod
                def load_model(cls, model_name, _conf):
                    self.assertEqual(esm.pretrained.data_root("esmc-600"), snapshot)
                    return "esmc-model"

            embedder = self._embedder(
                "ESM3c",
                {
                    "repository": "example/esmc",
                    "revision": revision,
                    "serialization": "weights.pth",
                    "weights_sha256": expected,
                },
                module=FakeESMCModule,
            )
            with patch(
                "huggingface_hub.snapshot_download", return_value=str(snapshot)
            ):
                embedder.load_model("ESM3c")
            self.assertEqual(embedder.model_instances["ESM3c"], "esmc-model")


if __name__ == "__main__":
    unittest.main()
