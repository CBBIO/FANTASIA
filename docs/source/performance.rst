.. _performance_hpc:

Performance on HPC
==================

For detailed instructions on deploying FANTASIA in an HPC environment, please refer to the :ref:`HPC Deployment Guide <fantasia_hpc_deployment>`.

Input Data
----------

The input dataset used for this performance evaluation consists of all protein sequences from *Mus musculus* (house mouse) available in UniProt:

- **Source:** `UniProtKB REST API <https://rest.uniprot.org/uniprotkb/stream?download=true&format=fasta&query=%28%28taxonomy_id%3A10090%29%29>`_
- **Taxonomy:** *Mus musculus* (species-level dataset)
- **Total sequences:** 87,492

The dataset will be processed using FANTASIA on an HPC system to evaluate performance in terms of execution time, resource utilization, and scalability.

Execution Parameters
-------------------

General Settings
~~~~~~~~~~~~~~~~

- **Maximum number of worker threads for parallel processing:** ``max_workers``: ``50``
- **Reference tag used for lookup operations:** ``lookup_reference_tag``: ``GOA2022``
- **K-closest protein to consider for lookup:** ``limit_per_entry``: ``1``
- **Prefix for output file names:** ``fantasia_prefix``: ``uniprotkb_taxonomy_id_10090_2025_03_13``
- **Threshold for sequence length filtering:** ``length_filter``: ``5000000``
- **Threshold for redundancy filtering:** ``redundancy_filter``: ``0.95``
- **Number of sequences to package in each queue batch:** ``sequence_queue_package``: ``1024``
- **Delete queues after processing:** ``delete_queues``: ``True``

Embedding Configuration
~~~~~~~~~~~~~~~~~~~~~~~

- **Distance metric:** ``distance_metric``: ``"<->"``  (options: ``"<=>"`` for cosine or ``"<->"`` for Euclidean)
- **Models:**
  - **ESM:**
    - Enabled: ``True``
    - Distance threshold: ``1.5``
    - Batch size: ``256``
  - **Prost-T5:**
    - Enabled: ``True``
    - Distance threshold: ``1.5``
    - Batch size: ``256``
  - **Prot-T5:**
    - Enabled: ``True``
    - Distance threshold: ``3``
    - Batch size: ``256``

Functional Analysis
~~~~~~~~~~~~~~~~~~~

- **Enable Gene Ontology enrichment analysis using TopGO:** ``topgo``: ``True``

Hardware Configuration
----------------------

The execution was performed on an HPC node equipped with:

- **CPU:** 256 cores
- **Total RAM:** 100GB
- **GPU Model:** NVIDIA A100-SXM4-80GB
- **CUDA Version:** 12.2
- **Driver Version:** 535.230.02
- **Available GPUs:** 4
- **GPUs in use:** 1

Although more CPU cores were available, the execution was limited to **50 worker threads** for parallel processing.

Summary
-------

- **100 workers** allow parallel execution of queries.
- **No sequence length filtering** (value set extremely high).
- **MMseqs2 at 95% sequence identity** to remove redundancy.
- **Only proteins from GOA2022 are used as reference**.
- **Euclidean distance metric** is applied.
- **Batch size of 256** for all three embedding models.
- **Execution performed on NVIDIA A100 GPUs with CUDA 12.2**.
- **Only 1 GPU was used, despite 4 being available**.
- **256 CPU cores available, but only 50 were used**.
- **100GB of RAM available during execution**.

Execution Times
---------------

Embedding Generation
~~~~~~~~~~~~~~~~~~~~

The execution times for generating embeddings with different models are summarized below:

+-------------------+-------------------+-------------------+
| Model             | Total Time        | Time per Sample   |
+===================+===================+===================+
| ESM               | 18 min 21 sec     | 12.59 ms/sample   |
+-------------------+-------------------+-------------------+
| ProSTT5           | 1 hr 51 min 37 sec| 76.55 ms/sample   |
+-------------------+-------------------+-------------------+
| ProtT5            | 2 hr 1 min 6 sec  | 83.05 ms/sample   |
+-------------------+-------------------+-------------------+

Lookup Processing
~~~~~~~~~~~~~~~~~

Lookup operations were performed after embedding generation, with the following performance:

+-------------------+-------------------+-------------------+
| Operation         | Total Time        | Time per Sample   |
+===================+===================+===================+
| Lookup            | 9 hr 25 min 7 sec | 387.54 ms/sample  |
+-------------------+-------------------+-------------------+

Conclusions
-----------

- ESM is the fastest model, processing each sample in 12.59 ms.
- ProSTT5 and ProtT5 are significantly slower, with times of 76.55 ms and 83.05 ms per sample, respectively.
- Lookup is the bottleneck, taking 4.5 times longer than embedding generation at 387.54 ms per sample.
- Further optimization of lookup operations (e.g., parallelization improvements or better GPU utilization) could significantly reduce processing time.