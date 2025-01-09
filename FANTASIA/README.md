
---

# FANTASIA

![FANTASIA Logo](img/FANTASIA_logo.png)

FANTASIA (Functional ANnoTAtion based on embedding space SImilArity) is a pipeline for annotating Gene Ontology (GO) terms for protein sequences using advanced protein language models like **ProtT5**, **ProstT5**, and **ESM2**. This system automates complex workflows, from sequence processing to functional annotation, providing a scalable and efficient solution for protein structure and functionality analysis.

---

## Key Features

- **Redundancy Filtering**: Removes identical sequences with **CD-HIT** and optionally excludes sequences based on length constraints.
- **Embedding Generation**: Utilizes state-of-the-art models for protein sequence embeddings.
- **GO Term Lookup**: Matches embeddings with a vector database to retrieve associated GO terms.
- **Results**: Outputs annotations in timestamped CSV files for reproducibility.

---

## Installation

To install FANTASIA, ensure you have Python 3.8+ installed and use the following commands:

```bash
pip install protein-metamorphisms-is
```

---

## Quick Start

### Prerequisites

Ensure the **Information System** is properly configured before running FANTASIA. Detailed instructions are available in the [project documentation](../../../README.md).

### Running the Pipeline

Execute the following command, specifying the path to the configuration file:

```bash
python main.py --config <path_to_config.yaml>
```

### Pipeline Overview

1. **Redundancy Filtering**: Removes identical sequences and optionally filters sequences based on length.
2. **Embedding Generation**: Computes embeddings for sequences using supported models and stores them in HDF5 format.
3. **GO Term Lookup**: Queries a vector database to find and annotate similar proteins.
4. **Output**: Saves annotations in a structured CSV file.

---

## Documentation

For complete details on pipeline configuration, parameters, and deployment, visit the [FANTASIA Documentation](https://protein-metamorphisms-is.readthedocs.io/en/latest/pipelines/fantasia.html).

---

## Citation

If you use FANTASIA in your work, please cite the following:

1. Martínez-Redondo, G. I., Barrios, I., Vázquez-Valls, M., Rojas, A. M., & Fernández, R. (2024). Illuminating the functional landscape of the dark proteome across the Animal Tree of Life.  
   https://doi.org/10.1101/2024.02.28.582465.

2. Barrios-Núñez, I., Martínez-Redondo, G. I., Medina-Burgos, P., Cases, I., Fernández, R. & Rojas, A.M. (2024). Decoding proteome functional information in model organisms using protein language models.  
   https://doi.org/10.1101/2024.02.14.580341.

---

## Contact Information

- Francisco Miguel Pérez Canales: fmpercan@upo.es  
- Gemma I. Martínez-Redondo: gemma.martinez@ibe.upf-csic.es  
- Ana M. Rojas: a.rojas.m@csic.es  
- Rosa Fernández: rosa.fernandez@ibe.upf-csic.es  

