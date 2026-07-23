Requirements
============

Verified software requirements
------------------------------

- Linux (the supported repository, Docker, and HPC workflows are Linux based).
- Python ``>=3.12,<3.13``.
- PostgreSQL 16 with pgvector; the Compose service uses
  ``pgvector/pgvector:0.7.0-pg16``.
- RabbitMQ; the Compose service uses ``rabbitmq:3.13-management-alpine``.
- ``pg_restore`` from PostgreSQL client 16 for reference restoration.
- Poetry for repository installation. A PyPI package exists, but repository
  installation is the documented path because configs and constants are
  required.
- MMseqs2 only when redundancy masking is enabled.

Hardware and storage
--------------------

GPU execution is the packaged default for embedding and lookup. CPU execution
is supported but requires both ``embedding.device: cpu`` and
``lookup.use_gpu: false``.

Use these conservative starting recommendations for one model at a time:

.. list-table::
   :header-rows: 1

   * - Workload
     - Free disk
     - RAM
     - GPU VRAM
   * - 20-protein test
     - 30 GB
     - 16 GB
     - 12 GB
   * - Proteome, final-layer reference
     - 100 GB
     - 32 GB
     - 16 GB minimum; 24 GB recommended
   * - Multilayer/all-model work
     - 200 GB
     - 64 GB
     - 24 GB recommended

These values are operational guidance rather than guaranteed minima. The
final-layer and multilayer downloads are approximately 3.1 and 17.1 GB, but
restored PostgreSQL data, model caches, embeddings and results require much
more space than the compressed archive. Uncapped long sequences may exceed the
listed VRAM. See `Resource requirements <../performance/resource_requirements.rst>`_.

Network access
--------------

Initialization downloads the selected Zenodo dump. The first use of a model
may download its weights from its upstream model repository. For offline runs,
pre-populate and preserve both resources.
