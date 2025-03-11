.. _fantasia_methods:

FANTASIA: Functional Annotation Pipeline
========================================

FANTASIA is a reimplementation of the GOPredSim algorithm (Littmann et al. 2021) for functionally annotating full species proteomes based on Gene Ontology (GO) term transfer using protein embedding similarity. Our version improves scalability for full proteomes, simplifies installation, and provides a more intuitive command-line interface, making it more accessible and customizable.

FANTASIA Methodology
--------------------

The FANTASIA pipeline consists of two optional input/output filtering and formatting steps and three main steps:

1. **Input Preprocessing**
   - Optional removal of long sequences and sequence similarity filtering with CD-HIT (up to 50%).
   - Recommended for benchmarking but not for full proteome functional annotation.

2. **Embedding Computation**
   - Protein embeddings are computed per selected model and per sequence.
   - Supported models: ProtT5, ESM2, and ProstT5.
   - Batch processing is supported to optimize efficiency and scalability.
   - Embeddings are stored in HDF5 format for further analysis.

3. **Embedding Similarity Computation**
   - Computes the distance between each input sequence embedding and the reference vector database managed with PostgreSQL.
   - The reference database contains metadata, Gene Ontology annotations, sequences, and precomputed embeddings for the supported pLMs.
   - By default, the Euclidean distance is computed:

     .. math::
        d_e(n,m) = \sqrt{\sum_{i=1}^{e} (n_i - m_i)^2}

   - Cosine similarity can also be selected:

     .. math::
        d_c(n,m) = \frac{\sum_{i=1}^{e} n_i m_i}{\sqrt{\sum_{i=1}^{e} n_i^2} \sqrt{\sum_{i=1}^{e} m_i^2}}

   - This step is optimized with an SQL-based information retrieval system.

4. **GO Term Transfer**
   - Transfers GO terms from the k closest hits in the database.
   - Users can define a distance threshold per model.
   - By default, only the closest hit (k=1) is used.
   - GO terms are based on the GOA release from November 3, 2024.

5. **Output Description and Optional Formatting**
   - The output is a comma-separated file (CSV) with 10 columns:

     1. Sequence accession (header).
     2. GO term identifier.
     3. GO category (F: molecular function, P: biological process, C: cellular component).
     4. GOA evidence code for the reference annotation.
     5. GO term description.
     6. Embedding distance (Euclidean or cosine) between query and reference.
     7. Protein language model used for embedding generation.
     8. Reference protein ID with the closest embedding.
     9. Organism of the reference protein.
     10. Reliability index (RI):

        .. math::
           RI(g) = 1 - \sum_{i=1}^{k} \frac{l}{0.5 + d(q, n_i)}

        where k is the total number of hits, and l is the number of hits annotated with GO term g.

   - By default, FANTASIA also converts the standard output file to topGO’s GO enrichment input format to facilitate broader biological workflows. This feature can be deactivated by the user.

Optimization and Resource Usage
------------------------------

- Allows users to select between CPU or GPU execution, with CPU as the default.
- GPU acceleration reduces execution time without increasing memory usage.
- Memory usage is proportional to the number of sequences in the proteome and follows the equation:

  .. math::
     y = 4.4 + 0.0021 x

  where y is memory in gigabytes and x is the number of sequences in the proteome.

FANTASIA provides a scalable and efficient solution for functionally annotating proteomes based on protein embeddings, enabling its application in large-scale genomic studies.