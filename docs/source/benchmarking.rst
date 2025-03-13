.. _benchmarking:

==========================================
Benchmarking
==========================================

Objective
---------

This use case provides a basic implementation for **benchmarking execution** in functional protein annotation. The goal is to generate annotations for proteins **NOT** present in the reference sets, allowing performance comparisons between different methods in downstream analyses.

An example of this approach is published in **NARGAB** [NARGAB2024]_, where annotations were evaluated using various metrics.

The reference datasets consist of proteins annotated in the three **Gene Ontology (GO)** domains:

- **F**: Molecular Function
- **B**: Biological Process
- **C**: Cellular Component

The evidence codes considered follow the **CAFA** standards, as defined by the CAFA Initiative [CAFA]_ and the Gene Ontology Annotation Database (GOA) [GOA]_.

Step-by-Step Procedure
----------------------

1. **Extract a reference proteome**, e.g., **mouse (*Mus musculus*)**.
2. **Remove mouse sequences** from the reference dataset (both sequences and embeddings) to avoid model biases.
3. **Remove identical sequences** from the reference dataset (both sequences and embeddings) to minimize the impact of closely related species.
4. **Execute the analysis pipeline** and transfer annotations based on embedding similarity (**GOA2024**).
   - **GOA2022** is available for comparison with previously published analyses.

We use the default settings from **GoPredSim** [GoPredSim]_, as implemented in the GoPredSim repository, for consistency with the original methods, though many parameters can be adjusted.

Input Data
----------

Input data must be **protein sequences in FASTA format**, as described in the EMBOSS documentation [EMBOSS]_, concatenated into a single file.

Example of **FILENAME_test.fasta**:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: none

   >tr|A0A087WPB2|A0A087WPB2_MOUSE MyoD family inhibitor domain containing OS=Mus musculus OX=10090 GN=Mdfic PE=1 SV=1
   MSCAGEALAPGPAEQQCPVEAGGGRLGSPAHEACNEDNTEKDKRPATSGHTRCGLMRDQS
   ...
   >tr|A0A087WQA5|A0A087WQA5_MOUSE TAR DNA binding protein (Fragment) OS=Mus musculus OX=10090 GN=Tardbp PE=1 SV=1
   XDETDASSAVKVKRAVQKTSDLIVLGLPWKTTEQDLKDYFSTFGEVLMVQVKKDLKTGHS
   ...

**Note:** The validation of input files to ensure they contain **proteins and not DNA** is automatically handled within the pipeline.

Configuration Parameters
------------------------

Pipeline Configuration
^^^^^^^^^^^^^^^^^^^^^^

Below, copy the code to a **`benchmark_config.yaml`** file in a text editor. The full configuration file, including additional essential system parameters, can be found at [`protein-metamorphisms-is/config/config.yaml`](https://github.com/CBBIO/protein-metamorphisms-is/blob/main/protein_metamorphisms_is/config/config.yaml).

.. code-block:: yaml

   # Path to the input FASTA file containing protein sequences
   input: data_sample/FILENAME_test.fasta

   # Reference tag used for lookup operations.
   lookup_reference_tag: GOA2022  # "0" (enables GOA2024) | "GOA2024" (includes GOA2022)

   # Number of closest proteins to consider in the lookup.
   limit_per_entry: 1  # k=1 is used in the GoPredSim method.

   # Prefix for output file names.
   fantasia_prefix: FILENAME_test_Prot_100_1.2

   # Sequence length filtering threshold.
   length_filter: 5000000  # Refers to the length of the sequence in amino acids. A high value means no filtering.

   # Redundancy filtering threshold (removes identical sequences).
   redundancy_filter: 1  # "0" (no filtering) | "1-0.5" (100%-50% redundancy removal)

Description of Parameters:
^^^^^^^^^^^^^^^^^^^^^^^^^^

- **lookup_reference_tag**: Defines which reference database is used for annotation lookup. Allows switching between **GOA2022** and **GOA2024** [GOA]_ to assess differences in methods.
- **limit_per_entry**: Determines how many similar proteins are considered for annotation transfer. `k=1` follows **GoPredSim** [GoPredSim]_ but can be adjusted.
- **length_filter**: Set to a high value to avoid sequence length filtering by default. It can be adjusted to remove abnormally long proteins if needed. Our new implementation correctly handles sequences longer than 5K amino acids.
- **redundancy_filter**: Controls the removal of identical sequences to prevent biases in method comparisons. This is relevant to avoid biases.

Embedding Configuration
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   embedding:
     distance_metric: "<->"  # "<=>" (cosine) | "<->" (Euclidean, default)
     models:
       esm:
         enabled: True
         distance_threshold: 0
         batch_size: 32
       prost_t5:
         enabled: True
         distance_threshold: 0
         batch_size: 32
       prot_t5:
         enabled: True
         distance_threshold: 0
         batch_size: 32

**Model References:**
- **ESM2** [ESM2]_
- **ProtT5** [ProtT5]_
- **ProstT5** [ProstT5]_
- **CD-HIT** [CDHIT]_

Functional Analysis
^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   # Enable or disable file formatting for TOPGO downstream analyses
   topgo: True  # "True" (enabled) | "False" (disabled)

**Reference:** **TOPGO** [TOPGO]_

Results
-------

Two main output files are generated:

1. **FILENAME_test.csv** → Contains detailed information for each query protein.
2. **FILENAME_test.TOPGO.txt** → Contains annotations formatted for **TOPGO** software.

These results can be used to evaluate prediction accuracy and compare the performance of different methods.

References
----------

.. [NARGAB2024] Example of benchmarking approach published in **NARGAB**, available at: `DOI: 10.1093/nargab/lqae078 <https://doi.org/10.1093/nargab/lqae078>`_.

.. [CAFA] CAFA Initiative, available at: `https://biofunctionprediction.org/cafa/`.

.. [GOA] Gene Ontology Annotation Database (GOA), available at: `https://www.ebi.ac.uk/GOA/downloads`.

.. [GoPredSim] GoPredSim repository, available at: `https://github.com/Rostlab/goPredSim/blob/master/file_utils.py`.

.. [EMBOSS] EMBOSS documentation, available at: `http://emboss.open-bio.org/html/use/apas01.html`.

.. [ESM2] ESM2 model on Hugging Face, available at: `https://huggingface.co/facebook/esm2_t36_3B_UR50D`.

.. [ProtT5] ProtT5 model on Hugging Face, available at: `https://huggingface.co/Rostlab/prot_t5_xl_uniref50`.

.. [ProstT5] ProstT5 model on Hugging Face, available at: `https://huggingface.co/Rostlab/ProstT5`.

.. [CDHIT] CD-HIT tool, available at: `https://www.bioinformatics.org/cd-hit/`.

.. [TOPGO] TOPGO software, available at: `https://bioconductor.org/packages/release/bioc/html/topGO.html`.