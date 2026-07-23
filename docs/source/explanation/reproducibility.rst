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

The recorded revision is an audit record. Current upstream loaders do not all
accept or enforce a ``revision`` argument, so verify the resolved cache snapshot
when exact byte-for-byte reproducibility is required. Input, reference-database,
and output checksums and detailed hardware information must still be recorded
externally.
