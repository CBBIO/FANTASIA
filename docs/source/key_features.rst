Key Features
------------

- **✅ Advanced Embedding Models**
  Supports protein language models: **ProtT5**, **ProstT5**, and **ESM2** for sequence representation.

- **🔍 Redundancy Filtering**
  Filters out homologous sequences using **CD-HIT**, allowing controlled redundancy levels through an adjustable threshold, ensuring reliable benchmarking and evaluation.

- **💾 Optimized Data Storage**
  Embeddings are stored in **HDF5 format** for input sequences, while similarity lookups are performed in a vector database (**pgvector in PostgreSQL**) for fast retrieval.

- **🚀 Efficient Similarity Lookup**
  Performs high-speed searches using **pgvector**, enabling accurate annotation based on embedding similarity.

- **🔬 Functional Annotation by Similarity**
  Assigns Gene Ontology (GO) terms to proteins based on **embedding space similarity**, leveraging pre-generated embeddings.
