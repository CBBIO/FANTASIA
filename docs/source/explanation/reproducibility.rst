Reproducibility
===============

Retain for every published run:

- FANTASIA release and Git commit;
- input file and SHA-256 checksum;
- saved ``experiment_config.yaml`` and logs;
- reference Zenodo record/version and local dump checksum;
- model identifier, requested and resolved immutable revision, serialization,
  weight/config/tokenizer checksums, and layer;
- Python, package, PyTorch, CUDA, driver, operating-system, and GPU versions;
- output checksums and any post-processing commands;
- random seeds used by downstream analyses.

FANTASIA automatically saves the resolved run configuration, logs, and
``model_provenance.yaml``. The provenance file records the configured repository
and immutable revision for every supported model, requested layers, enabled
state, and relevant installed package versions. ESM3c also includes its known
serialization filename and weight checksum.

FANTASIA enforces each recorded revision by resolving the immutable Hugging
Face snapshot before passing its local path to the model and tokenizer loaders.
For ESM3c, the pinned snapshot is selected explicitly and the serialized weight
file is verified against its recorded SHA-256 before loading. Input,
reference-database, and output checksums and detailed hardware information must
still be recorded externally.
