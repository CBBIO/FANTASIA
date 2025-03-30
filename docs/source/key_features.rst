Key Features
------------

- **✅ Availability of different Embedding Models**

  Currently supports the protein language models: **ProtT5**, **ProstT5**, and **ESM2** for sequence representation.

- **🔍 Filtering by sequence similarity**

  Filters out sequences by sequence similarity using the standard **CD-HIT**, enabling redundancy levels through an adjustable threshold. This is relevant for reliable benchmarking and evaluation of the methods.

- **💾 Optimized Data Storage**

  Embeddings are stored in **HDF5 format** for input sequences, while similarity lookups are performed in a vector database (**pgvector in PostgreSQL**) for fast retrieval.

- **🚀 Efficient Similarity Lookup**

  Performs high-speed searches using **pgvector**, enabling accurate annotation based on embedding similarity. 

- **🔬 Functional Annotation by Similarity in the Embedding space**

  Assigns Gene Ontology (GO) terms (Molecular Function, Biological Process, and Cellular Component) to proteins based on **embedding space similarity**. Only the most specific term is transferred and only the CAFA's standards for Experimental evidence are transferred (EXP, IDA, IPI, IMP, IGI, IEP, TAS, IC).
