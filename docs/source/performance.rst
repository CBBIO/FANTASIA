Performance
===========

Overview
--------

FANTASIA is being optimized for high-throughput functional annotation via embedding-based similarity. The current implementation is GPU-accelerated and fully batched, enabling efficient distance computations between query and reference embeddings. The system leverages matrix operations on GPU to scale across large proteomes and multiple embedding models.

The lookup phase has been designed to minimize memory transfers and CPU-GPU bottlenecks. Its computational complexity is dominated by indexing overhead and GO term transfer, which are generally lighter than the embedding generation step. All critical operations are vectorized, and bottlenecks have been addressed to ensure robust performance across species and configurations.
Execution Benchmark

-------------------

For benchmarking purposes, FANTASIA was executed on a dataset comprising all protein sequences from *Mus musculus* (87,492 entries), using a single NVIDIA A100 GPU with CUDA 12.2 and 256 CPU cores (50 in use).

Embedding generation times were as follows:

+-------------------+-------------------+-------------------+
| Model             | Total Time        | Time per Sample   |
+===================+===================+===================+
| ESM               | 18 min 21 sec     | 12.59 ms/sample   |
+-------------------+-------------------+-------------------+
| ProSTT5           | 1 hr 51 min 37 sec| 76.55 ms/sample   |
+-------------------+-------------------+-------------------+
| ProtT5            | 2 hr 1 min 6 sec  | 83.05 ms/sample   |
+-------------------+-------------------+-------------------+


+-------------------+-------------------+-------------------+
| Operation         | Total Time        | Time per Sample   |
+===================+===================+===================+
| Lookup            | To be calculated  | <T5 embedding gen |
+-------------------+-------------------+-------------------+



