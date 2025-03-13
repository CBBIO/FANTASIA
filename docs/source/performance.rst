.. _performance_hpc:

Performance on HPC
==================

For detailed instructions on deploying FANTASIA in an HPC environment, please refer to the :ref:`HPC Deployment Guide <fantasia_hpc_deployment>`.

## Input Data

The input dataset used for this performance evaluation consists of all protein sequences from *Mus musculus* (house mouse) available in UniProt:

- **Source:** [UniProtKB REST API](https://rest.uniprot.org/uniprotkb/stream?download=true&format=fasta&query=%28%28taxonomy_id%3A10090%29%29)
- **Taxonomy:** *Mus musculus* (species-level dataset)
- **Total sequences:** 87,492

The dataset will be processed using FANTASIA on an HPC system to evaluate performance in terms of execution time, resource utilization, and scalability.

## Execution Parameters

### General Settings

- **Maximum number of worker threads for parallel processing:** `max_workers`: `50`
- **Reference tag used for lookup operations:** `lookup_reference_tag`: `GOA2022`
- **K-closest protein to consider for lookup:** `limit_per_entry`: `1`
- **Prefix for output file names:** `fantasia_prefix`: `uniprotkb_taxonomy_id_10090_2025_03_13`
- **Threshold for sequence length filtering:** `length_filter`: `5000000`
- **Threshold for redundancy filtering:** `redundancy_filter`: `0.95`
- **Number of sequences to package in each queue batch:** `sequence_queue_package`: `1024`
- **Delete queues after processing:** `delete_queues`: `True`

### Embedding Configuration

- **Distance metric:** `distance_metric`: `"<->"`  (options: `"<=>"` for cosine or `"<->"` for Euclidean)
- **Models:**
  - **ESM:**
    - Enabled: `True`
    - Distance threshold: `1.5`
    - Batch size: `256`
  - **Prost-T5:**
    - Enabled: `True`
    - Distance threshold: `1.5`
    - Batch size: `256`
  - **Prot-T5:**
    - Enabled: `True`
    - Distance threshold: `3`
    - Batch size: `256`

### Functional Analysis

- **Enable Gene Ontology enrichment analysis using TopGO:** `topgo`: `True`

## Hardware Configuration

The execution was performed on an HPC node equipped with:

- **CPU:** 256 cores
- **Total RAM:** 100GB
- **GPU Model:** NVIDIA A100-SXM4-80GB
- **CUDA Version:** 12.2
- **Driver Version:** 535.230.02
- **Available GPUs:** 4
- **GPUs in use:** 1

Although more CPU cores were available, the execution was limited to **50 worker threads** for parallel processing.

## Summary

- *100 workers** allow parallel execution of queries.
- **No sequence length filtering** (value set extremely high).
- **CD-HIT at 95% sequence identity** to remove redundancy.
- **Only proteins from GOA2022 are used as reference**.
- **Euclidean distance metric** is applied.
- **Batch size of 256** for all three embedding models.
- **Execution performed on NVIDIA A100 GPUs with CUDA 12.2**.
- **Only 1 GPU was used, despite 4 being available**.
- **256 CPU cores available, but only 50 were used**.
- **100GB of RAM available during execution**.
