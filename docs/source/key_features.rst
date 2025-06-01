Key Features
------------

- **✅ Advanced Embedding Models**
  Supports protein language models: **ProtT5**, **ProstT5**, and **ESM2** for sequence representation.

- **🔁 Redundancy Filtering**
  Reduces bias by removing highly similar sequences from the reference database using **MMSeqs2**.
  Configurable thresholds allow clustering proteins based on identity and coverage.
  This improves generalization by avoiding annotation from near-identical entries.

- **🌿 Taxonomy-Based Filtering**
  Enables exclusion or inclusion of specific taxa from the annotation reference set based on **NCBI Taxonomy IDs**.
  Supports descendant expansion for clade-level filtering. Essential for studies targeting particular lineages
  or excluding over-represented model organisms.

- **💾 Optimized Data Storage**
  Embeddings are stored in **HDF5 format** for input sequences. The reference table, however, is hosted in a **public
  relational PostgreSQL database** using **pgvector**.

- **🚀 Efficient Similarity Lookup**
  Performs high-speed searches using **in-memory computations**. Reference vectors are retrieved from a **PostgreSQL
  database with pgvector** for comparison.

- **🔬 Functional Annotation by Similarity**
  Assigns Gene Ontology (GO) terms to proteins based on **embedding space similarity**, leveraging pre-trained
  embeddings.
