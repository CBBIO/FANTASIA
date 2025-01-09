FANTASIA
========
**Functional ANnoTAtion based on embedding space SImilArity**

**FANTASIA** is an advanced pipeline designed for annotating Gene Ontology (GO) terms in protein sequence files using state-of-the-art protein language models: **ProtT5**, **ProstT5**, and **ESM2**.

This pipeline accepts a proteome file as input (either the longest isoform or the full set of isoforms for all genes). It removes identical sequences using **CD-HIT**, optionally filters sequences based on their length, computes embeddings for all sequences, and stores them in an **HDF5 file**. The pipeline then queries a **vector database** to identify GO terms associated with similar proteins, based on these embeddings.

Pipeline Overview
-----------------
0. **System Setup**:

   By default, embeddings are downloaded and imported automatically during the pipeline execution. Alternatively, you can use the dedicated project: `protein-metamorphisms-is <https://github.com/CBBIO/protein-metamorphisms-is>`_ for robust embedding management and querying.

1. **Redundancy Filtering**:

   - Removes identical sequences using **CD-HIT**.
   - Optionally filters sequences based on length constraints.

2. **Embedding Generation**:

   - Computes embeddings for protein sequences using **ProtT5**, **ProstT5**, and/or **ESM2**.
   - Stores embeddings in an **HDF5 file**, organized by sequence accession IDs and embedding types.

3. **GO Term Lookup**:

   - Embeddings are compared for similarity using a **vector database**.
   - Retrieves GO terms associated with the most similar proteins.
   - Results include GO terms, distances, and metadata.

4. **Results**:

   - Annotations are saved in timestamped CSV files for reproducibility.

Acknowledgments
---------------
This pipeline is the result of a collaborative effort between **Ana Roja's lab** (Andalusian Center for Developmental Biology, CSIC) and **Rosa Fernández's lab** (Metazoa Phylogenomics Lab, Institute of Evolutionary Biology, CSIC-UPF). It exemplifies the power of synergy between research teams with diverse expertise. We extend our gratitude to **LifeHUB-CSIC** for inspiring this project and encouraging us to "think big."

Usage
-----
Refer to the `Setup Instructions <../deployment/setup_instructions.html>`_ for a detailed guide on configuring and executing the pipeline.

Citing FANTASIA
---------------
If you use **FANTASIA** in your research, please cite the following publications:

1. Martínez-Redondo, G. I., Barrios, I., Vázquez-Valls, M., Rojas, A. M., & Fernández, R. (2024).
   Illuminating the functional landscape of the dark proteome across the Animal Tree of Life.
   `https://doi.org/10.1101/2024.02.28.582465 <https://doi.org/10.1101/2024.02.28.582465>`_

2. Barrios-Núñez, I., Martínez-Redondo, G. I., Medina-Burgos, P., Cases, I., Fernández, R., & Rojas, A. M. (2024).
   Decoding proteome functional information in model organisms using protein language models.
   `https://doi.org/10.1101/2024.02.14.580341 <https://doi.org/10.1101/2024.02.14.580341>`_

Contact Information
-------------------
For inquiries, please contact the project team:

- Francisco Miguel Pérez Canales: fmpercan@upo.es (Developer)
- Gemma I. Martínez-Redondo: gemma.martinez@ibe.upf-csic.es
- Ana M. Rojas: a.rojas.m@csic.es
- Rosa Fernández: rosa.fernandez@ibe.upf-csic.es
