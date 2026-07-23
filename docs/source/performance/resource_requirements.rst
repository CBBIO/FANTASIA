Resource requirements
=====================

Resource requirements are workload-specific. Use the following conservative
starting points for one model at a time:

.. list-table::
   :header-rows: 1
   :widths: 34 18 18 30

   * - Workload
     - Free disk
     - System RAM
     - GPU VRAM
   * - Bundled 20-protein test
     - 30 GB
     - 16 GB
     - 12 GB
   * - Proteome with final-layer reference
     - 100 GB
     - 32 GB
     - 16 GB minimum; 24 GB recommended
   * - Multilayer or broad all-model work
     - 200 GB
     - 64 GB
     - 24 GB recommended

These are operational recommendations, not guaranteed hardware minima.
Embedding cost grows with model, sequence number and length, and model batch
size. In particular, uncapped long proteins may require more than the listed
VRAM. Lookup cost grows with query count, reference rows, embedding dimension,
selected layers and lookup batch size.

The compressed final-layer and multilayer reference downloads are about 3.1
and 17.1 GB, respectively. The restored PostgreSQL database is larger, and the
same filesystem may also hold container layers, caches for several PLMs, query
embeddings, per-protein raw CSVs and consolidated outputs. Therefore, do not
use compressed download size as the disk allocation.

Embedding and lookup run sequentially, so their transient allocations do not
normally peak at the same time. For GPU OOM during embedding, lower the active
model ``batch_size``; for lookup OOM, lower ``lookup.batch_size``. Processing
additional models/layers sequentially mainly increases wall time and stored
output, whereas running concurrent jobs multiplies RAM, VRAM and disk needs.
CPU execution needs no VRAM but is substantially slower and still needs the
listed RAM/disk headroom. Validate the 20-sequence config before extrapolating
to a proteome.
