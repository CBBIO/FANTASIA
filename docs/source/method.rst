============================
Definition of the Method
============================

## Overview

The benchmarking method used in this work is designed to evaluate the performance of functional annotation transfer based on protein sequence embeddings. The process systematically removes proteins from reference datasets to simulate annotation of novel sequences and assess the accuracy of prediction methods.

## Methodology

1. **Selection of Reference Dataset:**
   - The Gene Ontology (GO) annotated proteins are used as the primary dataset, referencing a UniProt 2024 mirror with corresponding GOA annotations.
   - Evidence codes from the **CAFA** ([CAFA Initiative](https://biofunctionprediction.org/cafa/)) and **GOA** ([GOA Database](https://www.ebi.ac.uk/GOA/downloads)) standards are considered. The allowed evidence codes used for annotation transfer include:

.. code-block:: yaml

    allowed_evidences:
     - EXP  # Inferred from Experiment.
     - IDA  # Inferred from Direct Assay.
     - IPI  # Inferred from Physical Interaction.
     - IMP  # Inferred from Mutant Phenotype.
     - IGI  # Inferred from Genetic Interaction.
     - IEP  # Inferred from Expression Pattern.
     - TAS  # Traceable Author Statement.
     - IC   # Inferred by Curator.

2. **Exclusion of Specific Proteins:**
   - Proteins from a selected organism, such as **Mus musculus**, are removed.
   - Identical sequences are also eliminated to avoid redundancy in annotation transfer.

3. **Generation of Protein Embeddings:**
   - Embeddings are generated using deep-learning-based models:
     - **ESM2** ([Hugging Face](https://huggingface.co/facebook/esm2_t36_3B_UR50D))
     - **ProtT5** ([Hugging Face](https://huggingface.co/Rostlab/prot_t5_xl_uniref50))
     - **ProstT5** ([Hugging Face](https://huggingface.co/Rostlab/ProstT5))

4. **Similarity-Based Annotation Transfer:**
   - Nearest neighbor search is performed in the embedding space.
   - The top matches are used to infer functional annotations.

5. **Evaluation Metrics:**
   - Performance is measured using:
   - Comparison against manually curated datasets.

## Implementation Details

The benchmarking pipeline is implemented using **GoPredSim** ([GitHub Repository](https://github.com/Rostlab/goPredSim/blob/master/file_utils.py)), ensuring compatibility with previously published methods. The full configuration parameters are provided in [`config.yaml`](https://github.com/CBBIO/protein-metamorphisms-is/blob/main/protein_metamorphisms_is/config/config.yaml).

This method allows the assessment of annotation accuracy and helps optimize parameters for better functional annotation predictions.

