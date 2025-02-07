Pipeline Overview
-----------------

0. **System Setup**

   Automatically downloads pre-trained embeddings or allows the use of local embeddings.

   - **Required dependencies**:

     - **PostgreSQL**:
       Stores protein metadata, Gene Ontology annotations, and embedding information.
     - **RabbitMQ**:
       Manages parallel task queues for pipeline execution.
     - **pgvector**:
       PostgreSQL extension for vector-based similarity searches.

1. **Embedding Generation**
   Computes protein embeddings using state-of-the-art language models.

   - **Available models**:

     - **ProtT5**
     - **ProstT5**
     - **ESM2**

   - **Batch processing**: Uses a configurable batch size per model to optimize efficiency and scalability.

   Stores embeddings in **HDF5 format**, enabling fast and scalable access for downstream tasks.

2. **GO Term Lookup**

   Performs similarity-based searches using embeddings stored in a vector database (**pgvector**).

   - **Configuration Parameters**:

     - **Maximum number of reference proteins**:
       Limits the number of closest proteins considered per query.

     - **Distance threshold (per model)**:
       Defines the similarity cutoff for assigning functional annotations.

     - **Redundancy filter (per execution)**:
       Uses **CD-HIT** to exclude homologous sequences based on an adjustable identity threshold, ensuring robust benchmarking and reliable functional annotation.

   Annotates query proteins with associated Gene Ontology (GO) terms retrieved from the most similar reference proteins.
