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

FANTASIA automatically saves the resolved run configuration and logs. It does
not currently guarantee automatic recording of upstream model commit hashes,
weight checksums, input checksums, reference checksums, or complete hardware
and package manifests; record these externally. Multiple revisions and weight
formats can coexist in a local Hugging Face cache, so the cache directory alone
is not reliable provenance.
