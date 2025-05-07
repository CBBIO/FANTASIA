[![PyPI - Version](https://img.shields.io/pypi/v/fantasia)](https://pypi.org/project/fantasia/)
[![Documentation Status](https://readthedocs.org/projects/fantasia/badge/?version=latest)](https://fantasia.readthedocs.io/en/latest/?badge=latest)
![Linting Status](https://github.com/CBBIO/fantasia/actions/workflows/test-lint.yml/badge.svg?branch=main)

![FANTASIA Logo](docs/source/_static/FANTASIA.png)


# FANTASIA

**Functional ANnoTAtion based on embedding space SImilArity**

FANTASIA is an advanced pipeline for the automatic functional annotation of protein sequences using state-of-the-art protein language models. It integrates deep learning embeddings and in-memory similarity searches, retrieving reference vectors from a PostgreSQL database with pgvector, to associate Gene Ontology (GO) terms with proteins.

For full documentation, visit [FANTASIA Documentation](https://fantasia.readthedocs.io/en/latest/).

## Key Features

- **✅ Available Embedding Models**  
  Supports protein language models: **ProtT5**, **ProstT5**, and **ESM2** for sequence representation.

- **🔍 Redundancy Filtering**  
  Filters out homologous sequences using **CD-HIT** in the lookup table, allowing controlled redundancy levels through an adjustable
  threshold, ensuring reliable benchmarking and evaluation.

- **💾 Optimized Data Storage**  
  Embeddings are stored in **HDF5 format** for input sequences. The reference table, however, is hosted in a **public
  relational PostgreSQL database** using **pgvector**.

- **🚀 Efficient Similarity Lookup**  
  Performs high-speed searches using **in-memory computations**. Reference vectors are retrieved from a PostgreSQL database with pgvector for comparison.

- **🔬 Functional Annotation by Similarity**  
  Assigns Gene Ontology (GO) terms to proteins based on **embedding space similarity**, using pre-trained
  embeddings.

## Pipeline Overview (Simplified)

1. **Embedding Generation**  
   Computes protein embeddings using deep learning models (**ProtT5**, **ProstT5**, and **ESM2**).

2. **GO Term Lookup**  
   Performs vector similarity searches using **in-memory computations** to assign Gene Ontology terms. Reference
   embeddings are retrieved from a **PostgreSQL database with pgvector**. Only experimental evidence codes are used for transfer.

## Acknowledgments

FANTASIA is the result of a collaborative effort between **Ana Rojas’ Lab (CBBIO)** (Andalusian Center for Developmental
Biology, CSIC) and **Rosa Fernández’s Lab** (Metazoa Phylogenomics Lab, Institute of Evolutionary Biology, CSIC-UPF).
This project demonstrates the synergy between research teams with diverse expertise.

This version of FANTASIA builds upon previous work from:

- [`Metazoa Phylogenomics Lab's FANTASIA`](https://github.com/MetazoaPhylogenomicsLab/FANTASIA)  
  The original implementation of FANTASIA for functional annotation.

- [`bio_embeddings`](https://github.com/sacdallago/bio_embeddings)  
  A state-of-the-art framework for generating protein sequence embeddings.

- [`GoPredSim`](https://github.com/Rostlab/goPredSim)  
  A similarity-based approach for Gene Ontology annotation.

- [`protein-metamorphisms-is`](https://github.com/CBBIO/protein-metamorphisms-is)  
  Serves as the **reference biological information system**, providing a robust data model and curated datasets for
  protein structural and functional analysis.

We also extend our gratitude to **LifeHUB-CSIC** for inspiring this initiative and fostering innovation in computational
biology.

## Citing FANTASIA

If you use **FANTASIA** in your research, please cite the following publications:

1. Martínez-Redondo, G. I., Barrios, I., Vázquez-Valls, M., Rojas, A. M., & Fernández, R. (2024).  
   *Illuminating the functional landscape of the dark proteome across the Animal Tree of Life.*  
   [DOI: 10.1101/2024.02.28.582465](https://doi.org/10.1101/2024.02.28.582465)

2. Barrios-Núñez, I., Martínez-Redondo, G. I., Medina-Burgos, P., Cases, I., Fernández, R., & Rojas, A. M. (2024).  
   *Decoding proteome functional information in model organisms using protein language models.*  
   [DOI: 10.1101/2024.02.14.580341](https://doi.org/10.1101/2024.02.14.580341)

---

### 👥 Project Team

#### 🔧 Technical Team

- **Francisco Miguel Pérez Canales**: [fmpercan@upo.es](mailto:fmpercan@upo.es)  
  *Author of the system’s engineering and technical implementation*
- **Francisco J. Ruiz Mota**: [fraruimot@alum.us.es](mailto:fraruimot@alum.us.es)  
  *Junior developer*

#### 🧬 Scientific Team & Original Authors of FANTASIA v1

- **Ana M. Rojas**: [a.rojas.m@csic.es](mailto:a.rojas.m@csic.es)
- **Gemma I. Martínez-Redondo**: [gemma.martinez@ibe.upf-csic.es](mailto:gemma.martinez@ibe.upf-csic.es)
- **Rosa Fernández**: [rosa.fernandez@ibe.upf-csic.es](mailto:rosa.fernandez@ibe.upf-csic.es)

---


# FANTASIA

**Functional ANnoTAtion based on embedding space SImilArity**

FANTASIA is an advanced pipeline for the automatic functional annotation of protein sequences using state-of-the-art protein language models. It integrates deep learning embeddings and in-memory similarity searches, retrieving reference vectors from a PostgreSQL database with pgvector, to associate Gene Ontology (GO) terms with proteins.

For full documentation, visit [FANTASIA Documentation](https://fantasia.readthedocs.io/en/latest/).

## Key Features

- **✅ Available Embedding Models**  
  Supports protein language models: **ProtT5**, **ProstT5**, and **ESM2** for sequence representation.

- **🔍 Redundancy Filtering**  
  Filters out homologous sequences using **CD-HIT** in the lookup table, allowing controlled redundancy levels through an adjustable
  threshold, ensuring reliable benchmarking and evaluation.

- **💾 Optimized Data Storage**  
  Embeddings are stored in **HDF5 format** for input sequences. The reference table, however, is hosted in a **public
  relational PostgreSQL database** using **pgvector**.

- **🚀 Efficient Similarity Lookup**  
  Performs high-speed searches using **in-memory computations**. Reference vectors are retrieved from a PostgreSQL database with pgvector for comparison.

- **🔬 Functional Annotation by Similarity**  
  Assigns Gene Ontology (GO) terms to proteins based on **embedding space similarity**, using pre-trained
  embeddings.

## Pipeline Overview (Simplified)

1. **Embedding Generation**  
   Computes protein embeddings using deep learning models (**ProtT5**, **ProstT5**, and **ESM2**).

2. **GO Term Lookup**  
   Performs vector similarity searches using **in-memory computations** to assign Gene Ontology terms. Reference
   embeddings are retrieved from a **PostgreSQL database with pgvector**. Only experimental evidence codes are used for transfer.

## Acknowledgments

FANTASIA is the result of a collaborative effort between **Ana Rojas’ Lab (CBBIO)** (Andalusian Center for Developmental
Biology, CSIC) and **Rosa Fernández’s Lab** (Metazoa Phylogenomics Lab, Institute of Evolutionary Biology, CSIC-UPF).
This project demonstrates the synergy between research teams with diverse expertise.

This version of FANTASIA builds upon previous work from:

- [`Metazoa Phylogenomics Lab's FANTASIA`](https://github.com/MetazoaPhylogenomicsLab/FANTASIA)  
  The original implementation of FANTASIA for functional annotation.

- [`bio_embeddings`](https://github.com/sacdallago/bio_embeddings)  
  A state-of-the-art framework for generating protein sequence embeddings.

- [`GoPredSim`](https://github.com/Rostlab/goPredSim)  
  A similarity-based approach for Gene Ontology annotation.

- [`protein-metamorphisms-is`](https://github.com/CBBIO/protein-metamorphisms-is)  
  Serves as the **reference biological information system**, providing a robust data model and curated datasets for
  protein structural and functional analysis.

We also extend our gratitude to **LifeHUB-CSIC** for inspiring this initiative and fostering innovation in computational
biology.

## Citing FANTASIA

If you use **FANTASIA** in your research, please cite the following publications:

1. Martínez-Redondo, G. I., Barrios, I., Vázquez-Valls, M., Rojas, A. M., & Fernández, R. (2024).  
   *Illuminating the functional landscape of the dark proteome across the Animal Tree of Life.*  
   [DOI: 10.1101/2024.02.28.582465](https://doi.org/10.1101/2024.02.28.582465)

2. Barrios-Núñez, I., Martínez-Redondo, G. I., Medina-Burgos, P., Cases, I., Fernández, R., & Rojas, A. M. (2024).  
   *Decoding proteome functional information in model organisms using protein language models.*  
   [DOI: 10.1101/2024.02.14.580341](https://doi.org/10.1101/2024.02.14.580341)

---

### 👥 Project Team

#### 🔧 Technical Team

- **Francisco Miguel Pérez Canales**: [fmpercan@upo.es](mailto:fmpercan@upo.es)  
  *Author of the system’s engineering and technical implementation*
- **Francisco J. Ruiz Mota**: [fraruimot@alum.us.es](mailto:fraruimot@alum.us.es)  
  *Junior developer*

#### 🧬 Scientific Team & Original Authors of FANTASIA v1

- **Ana M. Rojas**: [a.rojas.m@csic.es](mailto:a.rojas.m@csic.es)
- **Gemma I. Martínez-Redondo**: [gemma.martinez@ibe.upf-csic.es](mailto:gemma.martinez@ibe.upf-csic.es)
- **Rosa Fernández**: [rosa.fernandez@ibe.upf-csic.es](mailto:rosa.fernandez@ibe.upf-csic.es)

---
