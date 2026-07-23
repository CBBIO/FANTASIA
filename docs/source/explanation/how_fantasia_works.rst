How FANTASIA works
==================

Full FANTASIA has two sequential computational stages. The embedder parses
protein FASTA, dispatches model tasks through RabbitMQ/PIS workers, and stores
per-query, model, and layer vectors in HDF5. Lookup loads compatible reference
vectors and experimental GO annotations from PostgreSQL, restricts eligible
references, computes cosine or Euclidean distances on CPU/GPU, retains the
nearest candidates, expands their annotations, computes optional sequence
alignments, and writes raw and summarized outputs.

PostgreSQL/pgvector is persistent reference storage; distance kernels run in
the application on materialized arrays. Embedding and lookup do not overlap
inside one run.
