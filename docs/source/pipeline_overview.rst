Pipeline Overview
-----------------

0. **System Setup**

   Sets up the environment and dependencies. Pre-trained embeddings are downloaded automatically or local embeddings can be used.

   - **Core dependencies**:

     - **PostgreSQL**
       Stores protein metadata, Gene Ontology (GO) annotations, and reference embedding information.

     - **RabbitMQ**
       Handles distributed execution through parallel task queues.

     - **pgvector**
       PostgreSQL extension for efficient vector similarity indexing and retrieval.

1. **Embedding Generation**

   Computes deep protein embeddings using cutting-edge language models.

   - **Supported models**:

     - **ProtT5**
     - **ProstT5**
     - **ESM2**

   - **Batch processing**
     Embeddings are generated in configurable batches to ensure scalability and resource efficiency.

   - **Output format**
     Embeddings are stored in **HDF5 format**, enabling fast I/O and compatibility with large-scale downstream analysis.

2. **GO Term Lookup**

   Assigns Gene Ontology (GO) terms to query proteins via similarity search in the embedding space.

   - **Search strategy**
     Similarity is computed **in-memory** after retrieving reference embeddings from a **PostgreSQL database with pgvector**.

   - **Key parameters**:

     - **Max reference proteins**
       Sets the upper limit of reference proteins considered per query.

     - **Distance threshold (per model)**
       Controls the similarity cutoff for assigning annotations.

     - **Redundancy filtering**
       Uses **CD-HIT** to remove homologous sequences prior to lookup, with a customizable identity threshold.

   - **Annotation output**
     GO terms are assigned based on the most similar reference proteins in the embedding space.
