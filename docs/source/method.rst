.. _methods:

Methods
=======

FANTASIA is a reimplementation of the GOPredSim algorithm [Littmann2021]_ for functionally annotating full proteomes based on GO term transference from protein embedding similarity. Building on the original method, our pipeline enhances usability by offering scalability to full proteomes, a more reliable installation process with minimized dependency conflicts, and an intuitive command-line interface that simplifies parameter customization for less advanced users.

.. figure:: _static/pipeline.png
   :alt: FANTASIA Pipeline Overview
   :align: center
   :width: 80%

Step 1: Setup and Input File Preprocessing
------------------------------------------

Optionally, long sequences can be removed, and sequence similarity filtering can be applied using CD-HIT [Fu2012CDHIT]_, with a minimum identity threshold of 50%. This filtering step allows users to exclude sequences that are highly similar to those in the annotated reference database.

Before computing distances between embeddings, the input dataset is concatenated with the reference table. Then, for each query embedding, a comparison is performed against the entire reference table. During this process, embeddings corresponding to sequences that belong to the same cluster as the query are excluded, ensuring that no matches (hits) are made with proteins that exceed the specified sequence identity threshold.

The use of these options depends on the intended application. They are recommended for benchmarking experiments but not for proteome functional annotation. For details on parameter selection for different applications, refer to the pipeline documentation.

Step 2: Embedding Computation
-----------------------------

FANTASIA computes the protein embeddings per model selected in the input and per sequence. The current version supports ProtT5 [Heinzinger2019ProtTrans]_, ESM2 [Lin2022ESM2]_, and ProstT5. Batch processing of input sequences is supported to optimize efficiency and scalability. Embeddings (and the corresponding sequences) are stored in HDF5 format for the user to use in other analyses.

Step 3: Embedding Similarity
----------------------------

FANTASIA then computes the distance between each input sequence embedding and the embeddings in the reference vector database [ZenodoRef]_ managed with PostgreSQL, allowing a fast retrieval of results. The reference database contains, for each reference protein, its metadata, its GO term annotations, its sequence, and its precomputed embeddings for the supported pLMs.

By default, Euclidean distance (:math:`d_e`) between embeddings :math:`n` and :math:`m` for a model with an embedding dimensionality :math:`k` is computed with the following formula:

.. math::
   d_e(n, m) = \sum_{i=1}^{k} (n_i - m_i)^2

where :math:`k` represents the number of dimensions in the embedding space, which varies depending on the selected protein language model: :math:`k = 1024` for ProtT5 and ProstT5, and :math:`k = 320` for ESM2.

Alternatively, cosine similarity (:math:`d_c`) can be selected as a parameter, using the formula:

.. math::
   d_c(n, m) = \frac{\sum_{i=1}^{k} n_i m_i}{\sqrt{\sum_{i=1}^{k} n_i^2} \cdot \sqrt{\sum_{i=1}^{k} m_i^2}}

This step is significantly accelerated by leveraging ``pgvector``, a PostgreSQL extension optimized for efficient similarity searches in high-dimensional embedding spaces [pgvector_git]_. By default, ``pgvector`` performs exact nearest neighbor search, ensuring perfect recall. However, it also supports approximate nearest neighbor search, which trades some recall for increased speed. Unlike typical indexes, adding an approximate index may yield slightly different query results. In our implementation, we use exact search to maximize accuracy, but we leave the option open for approximate search if faster retrieval is needed in future optimizations.

Step 4: GO Transfer
-------------------

FANTASIA then transfers GO terms from the :math:`k` closest hits in the database. Additionally, the user can define a distance threshold for each model that determines the maximum allowed distance between query and reference embeddings. By default, only the closest hit (:math:`k=1`) is used, regardless of its distance to the query embedding. GO terms of the current version correspond to the release from the 3rd of November 2024 (GOA2024).

Step 5: Output Description and Optional Formatting
--------------------------------------------------

The output of FANTASIA consists of a comma-separated file (CSV) with the following ten columns:

1. Sequence accession (header).
2. GO term identifier.
3. GO category (F: molecular function, P: biological process, C: cellular component).
4. GOA evidence code for the reference annotation.
5. GO term description.
6. Embedding distance (:math:`d_c` for cosine similarity or :math:`d_e` for Euclidean distance) between the query and the reference.
7. Protein language model used for the embedding generation.
8. ID of the reference protein bearing the closest embedding.
9. Organism the target reference protein belongs to.
10. Reliability index (RI).

Reliability Index (RI)
^^^^^^^^^^^^^^^^^^^^^^

The reliability index (RI) is a transformation of the distance into a similarity scale, making it easier to interpret the confidence in the functional annotation. This approach of scaling distance into a similarity metric follows principles previously established in Littmann et al. (2021) [Littmann2021]_. FANTASIA supports two distinct RI formulations, depending on the selected distance metric:

- If using the **direct similarity measure**, applied to cosine similarity (:math:`d_c`), RI is computed as:

  .. math::
     RI = 1 - d_c(q, n_i)

  where :math:`d_c(q, n_i)` represents the cosine distance between the query embedding :math:`q` and its closest reference :math:`n_i`.

- If using the **inverse similarity transformation**, applied to Euclidean distance (:math:`d_e`), RI is defined as:

  .. math::
     RI = \frac{0.5}{0.5 + d_e(q, n_i)}

  where lower Euclidean distances yield higher confidence scores.

While both formulations produce values ranging from 0 to 1, they are **not directly comparable**, as they capture confidence in different ways. Users should exercise caution when interpreting RI scores across different similarity metrics.

Additionally, **the Euclidean distance is not inherently comparable across different protein language models**, as it depends on the magnitude of the embedding vectors generated by each model. In contrast, the **cosine similarity metric** is more suitable for cross-model comparisons, as it primarily captures the relative orientation of embeddings rather than their absolute magnitude.

Filtering GO Terms
^^^^^^^^^^^^^^^^^^^^^^
To avoid duplicates and ensure that only the most reliable annotation is kept for each combination of protein accession (``accession``) and GO term (``go_id``), FANTASIA retains only the GO term with the highest reliability index (RI) for each unique pair. This step improves the precision of functional annotations by eliminating redundancies.

Identifying Parent Terms
^^^^^^^^^^^^^^^^^^^^^^
To enrich the functional analysis, FANTASIA identifies the parent GO terms associated with each annotated GO term. This is achieved using the ``goatools`` library [GOATools]_, which allows navigation through the Gene Ontology (GO) hierarchy. Parent terms provide a broader context for the biological functions, processes, or cellular components associated with the proteins, facilitating higher-level functional analyses.

TopGO Compatibility
^^^^^^^^^^^^^^^^^^^^^^
By default, FANTASIA also converts the standard output file into the input format required for ``topGO``'s GO enrichment analysis [Alexa2017topGO]_, facilitating its integration into broader biological workflows. This feature can be deactivated by the user if desired.




References
--------------------------------------------------

.. [Littmann2021] M. Littmann et al., "Embeddings from deep learning transfer GO annotations beyond homology," *Scientific Reports*, vol. 11, no. 1, p. 1160, 2021.

.. [Fu2012CDHIT] L. Fu et al., "CD-HIT: accelerated clustering for next-generation sequencing data," *Bioinformatics*, vol. 28, no. 23, pp. 3150-3152, 2012.

.. [Heinzinger2019ProtTrans] M. Heinzinger et al., "Modeling aspects of the language of life through transfer-learning protein sequences," *BMC Bioinformatics*, vol. 20, no. 1, p. 723, 2019.

.. [Lin2022ESM2] Z. Lin et al., "Evolutionary-scale prediction of atomic-level protein structure with a language model," *Science*, vol. 379, no. 6637, pp. 1123-1130, 2023.

.. [pgvector_git] PostgreSQL extension for high-dimensional similarity searches, available at: `GitHub <https://github.com/pgvector/pgvector>`_.

.. [Alexa2017topGO] A. Alexa and J. Rahnenfuhrer, "topGO: Enrichment analysis for gene ontology," *Bioconductor*, 2017.

.. [ZenodoRef] Reference database for FANTASIA, available at: `Zenodo <https://zenodo.org/records/14864851>`_.

.. [GOATools] goatools library for Gene Ontology analysis, available at: https://github.com/tanghaibao/goatools_.