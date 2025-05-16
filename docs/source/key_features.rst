Key Features
------------

- **✅ Advanced Embedding Models**
  Supports protein language models: **ProtT5**, **ProstT5**, and **ESM2** for sequence representation.

- **🔍 Redundancy Filtering**
  Filters out homologous sequences using **MMSeqs2**, allowing controlled redundancy levels through an adjustable
  threshold, ensuring reliable benchmarking and evaluation.

- **💾 Optimized Data Storage**
  Embeddings are stored in **HDF5 format** for input sequences. The reference table, however, is hosted in a **public
  relational PostgreSQL database** using **pgvector**.

- **🚀 Efficient Similarity Lookup**
  Performs high-speed searches using **in-memory computations**. Reference vectors are retrieved from a **PostgreSQL
  database with pgvector** for comparison.

- **🔬 Functional Annotation by Similarity**
  Assigns Gene Ontology (GO) terms to proteins based on **embedding space similarity**, leveraging pre-trained
  embeddings.
