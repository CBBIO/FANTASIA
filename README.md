![FANTASIA Logo](docs/source/_static/FANTASIA.png)

[![PyPI - Version](https://img.shields.io/pypi/v/fantasia)](https://pypi.org/project/fantasia/)
[![Documentation Status](https://readthedocs.org/projects/fantasia/badge/?version=latest)](https://fantasia.readthedocs.io/en/latest/?badge=latest)
![Linting Status](https://github.com/CBBIO/fantasia/actions/workflows/test-lint.yml/badge.svg?branch=main)



# FANTASIA

**Functional ANnoTAtion based on embedding space SImilArity**

FANTASIA is an advanced pipeline for the automatic functional annotation of protein sequences using state-of-the-art protein language models. It integrates deep learning embeddings and in-memory similarity searches, retrieving reference vectors from a PostgreSQL database with pgvector, to associate Gene Ontology (GO) terms with proteins.

For full documentation, visit [FANTASIA Documentation](https://fantasia.readthedocs.io/en/latest/).



## Reference Datasets
Two packaged reference datasets are available; select one depending on your analysis needs:

- **Main Reference (last layer, default)**  
  Embeddings extracted only from the **final hidden layer** of each PLM.  
  Recommended for most annotation tasks (smaller, faster to load).  
  *Record*: https://zenodo.org/records/17167843

- **Multilayer Reference (intermediate + final layers)**  
  Embeddings extracted from **multiple hidden layers** (including intermediate and final).  
  Suitable for comparative and exploratory analyses requiring layer-wise representations.  
  *Record*: https://zenodo.org/records/17151847


## Key Features

- **✅ Available Embedding Models**  
  Supports protein language models: **ProtT5**, **ProstT5**, **ESM2** and **Ankh** for sequence representation.

- **🔍 Redundancy Filtering**  
  Filters out homologous sequences using **MMseqs2** in the lookup table, allowing controlled redundancy levels through an adjustable
  threshold, ensuring reliable benchmarking and evaluation.

- **💾 Optimized Data Storage**  
  Embeddings are stored in **HDF5 format** for input sequences. The reference table, however, is hosted in a **public
  relational PostgreSQL database** using **pgvector**.

- **🚀 Efficient Similarity Lookup**  
  High-throughput similarity search with a **hybrid approach**: reference embeddings are stored in a **PostgreSQL + pgvector** database and **fetched in batches to memory** to compute similarities at speed.

- **🧭 Global & Local Alignment of Hits**  
  Candidate hits from the reference table are **aligned both globally and locally** against the input protein for validation and scoring.

- **🧩 Multi-layer Embedding Support**  
  Optional support for **intermediate + final layers** to enable layer-wise analyses and improved exploration.

- **📦 Raw Outputs & Flexible Post-processing**  
  Exposes **raw result tables** for custom analyses and includes a **flexible post-processing & scoring system** that produces **TopGO-ready** files.  
  Performs high-speed searches using **in-memory computations**. Reference vectors are retrieved from a PostgreSQL database with pgvector for comparison.

- **🔬 Functional Annotation by Similarity**  
  Assigns Gene Ontology (GO) terms to proteins based on **embedding space similarity**, using pre-trained
  embeddings.

## Pipeline Overview (Simplified)

1. **Embedding Generation**  
   Computes protein embeddings using deep learning models (**ProtT5**, **ProstT5**, **ESM2** and **Ankh**).

2. **GO Term Lookup**  
   Performs vector similarity searches using **in-memory computations** to assign Gene Ontology terms. Reference
   embeddings are retrieved from a **PostgreSQL database with pgvector**. Only experimental evidence codes are used for transfer.

## 📚 Supported Embedding Models

| Name         | Model ID                                 | Params | Architecture      | Description                                                                 |
|--------------|-------------------------------------------|--------|-------------------|-----------------------------------------------------------------------------|
| **ESM-2**     | `facebook/esm2_t33_650M_UR50D`            | 650M   | Encoder (33L)     | Learns structure/function from UniRef50. No MSAs. Optimized for accuracy.  |
| **ProtT5**    | `Rostlab/prot_t5_xl_uniref50`             | 1.2B   | Encoder-Decoder   | Trained on UniRef50. Strong transfer for structure/function tasks.         |
| **ProstT5**   | `Rostlab/ProstT5`                         | 1.2B   | Multi-modal T5     | Learns 3Di structural states + function. Enhances contact/function tasks.  |
| **Ankh3-Large** | `ElnaggarLab/ankh3-large`              | 620M   | Encoder (T5-style)| Fast inference. Good semantic/structural representation.                   |
| **ESM3c**     | `esmc_600m`                               | 600M   | Encoder (36L)     | New gen. model trained on UniRef + MGnify + JGI. High precision & speed.   |


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

- [`protein-information-system`](https://github.com/CBBIO/protein-information-system)  
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


## License

FANTASIA is distributed under the terms of the [GNU Affero General Public License v3.0](LICENSE).


---

### 👥 Project Team

- **Ana M. Rojas**: [a.rojas.m@csic.es](mailto:a.rojas.m@csic.es)
- **Rosa Fernández**: [rosa.fernandez@ibe.upf-csic.es](mailto:rosa.fernandez@ibe.upf-csic.es)
- **Gemma I. Martínez-Redondo**: [gemma.martinez@ibe.upf-csic.es](mailto:gemma.martinez@ibe.upf-csic.es)
- **Francisco Miguel Pérez Canales**: [fmpercan@upo.es](mailto:fmpercan@upo.es)
- **Belén Carbonetto**: [belen.carbonetto.metazomics@gmail.com](mailto:belen.carbonetto.metazomics@gmail.com)
- **Francisco J. Ruiz Mota**: [fraruimot@alum.us.es](mailto:fraruimot@alum.us.es)  
- **Àlex Domínguez Rodríguez**: [alex.dominguezro@autonoma.cat](mailto:alex.dominguezro@autonoma.cat)

---

