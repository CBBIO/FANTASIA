.. _benchmarking:

==========================================
Benchmarking
==========================================

Objective
---------

This use case provides a basic implementation for **benchmarking execution** in functional protein annotation. The goal is to generate annotations for proteins **NOT** present in the reference sets, allowing performance comparisons between different methods in downstream analyses.

The rationale of this approach is published [1]_, where methods were evaluated using various metrics, the ProtT5 is the most appropiate for general annotation task.

The reference data are proteins annotated in any or all the three **Gene Ontology (GO)** domains:

- **F**: Molecular Function
- **B**: Biological Process
- **C**: Cellular Component

The evidence codes considered follow the **CAFA** standards, as defined by the CAFA community [2]_ and the Gene Ontology Annotation Database (GOA) [3]_.

Step-by-Step Procedure
----------------------

1. **Select a reference proteome**, e.g., **mouse (*Mus musculus*)**.
2. **Remove mouse sequences** from the reference dataset (both sequences and embeddings) to avoid model biases if possible OR
3. **Remove identical sequences** from the reference dataset (both sequences and embeddings) to minimize the impact of closely related species.
4. **Execute the analysis pipeline** and transfer annotations based on embedding similarity (**Current GOA**).
  
We use the default settings from **GoPredSim** [4]_, as implemented in the GoPredSim repository, for consistency with the original methods, though most parameters can be adjusted.

Input Data
----------

Input data must be **protein sequences in FASTA format**, as described here [5]_, concatenated into a single file.

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
   lookup_reference_tag: 0  

   # Number of closest proteins to consider in the lookup.
   limit_per_entry: 1  # k=1 is used in the GoPredSim method.

   # Prefix for output file names.
   fantasia_prefix: FILENAME_test_Prot_100_1.2

   # Sequence length filtering threshold.
   length_filter: 5000000  # Refers to the length of the sequence in amino acids. A high value means no filtering.

   # Redundancy filtering threshold  in the lookup table (removes identical sequences).
   redundancy_filter: 1  # "0" (no filtering) | "1-0.5" (100%-50% redundancy removal)

Description of Parameters:
^^^^^^^^^^^^^^^^^^^^^^^^^^

- **lookup_reference_tag**: Defines which reference database is used for annotation lookup. Allows switching between **GOA2022**, **GOA2024** [3]_, or current **GOA2025** to assess differences in methods. The lookuptable is generated from the current GOA since it is built from the UniProtKB API.
- **limit_per_entry**: Determines how many similar proteins are considered for annotation transfer. `k=1` follows **GoPredSim** [4]_ but can be adjusted.
- **length_filter**: Set to a high value to avoid sequence length filtering by default. It can be adjusted to remove abnormally long proteins if needed. Our new implementation correctly handles sequences longer than 5K amino acids.
- **redundancy_filter**: Controls the removal of identical sequences to prevent biases in method comparisons. This is relevant to avoid biases.

Embedding Configuration
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   embedding:
    device: cuda # "cpu" to launch GPU usage
     distance_metric: "euclidean"  # "cosine" (cosine) 
     models:
       esm:
         enabled: True # "False" to disable it
         distance_threshold: 3  #cannot be "0"
         batch_size: 32
       prost_t5:
         enabled: True # "False" to disable it
         distance_threshold: 3 #cannot be "0"
         batch_size: 32
       prot_t5:
         enabled: True # "False" to disable it
         distance_threshold: 3 #cannot be "0"
         batch_size: 32

**Model References:**
- **ESM2** [6]_
- **ProtT5** [7]_
- **ProstT5** [8]_
- **CD-HIT** [9]_

Functional Analysis
^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   # Enable or disable file formatting for TOPGO downstream analyses
   topgo: True  # "True" (enabled) | "False" (disabled)

**Reference:** **TOPGO** [10]_

Results
-------

Two main output files are generated:

1. **FILENAME_test.csv** → Contains detailed information for each query protein.
2. **FILENAME_test.TOPGO.txt** → Contains annotations formatted for **TOPGO** software for functional analyses.

These results can be used to evaluate prediction accuracy and compare the performance of different methods.

References
----------

.. [1] Our benchmark of different methods published in **NARGAB**, available at: DOI: 10.1093/nargab/lqae078 <https://doi.org/10.1093/nargab/lqae078>.
.. [2] CAFA Initiative, available at: `https://biofunctionprediction.org/cafa/`.
.. [3] Gene Ontology Annotation Database (GOA), available at: `https://www.ebi.ac.uk/GOA/downloads`.
.. [4] GoPredSim repository, available at: `https://github.com/Rostlab/goPredSim/blob/master/file_utils.py`.
.. [5] EMBOSS documentation, available at: `http://emboss.open-bio.org/html/use/apas01.html`.
.. [6] ESM2 model on Hugging Face, available at: `https://huggingface.co/facebook/esm2_t36_3B_UR50D`.
.. [7] ProtT5 model on Hugging Face, available at: `https://huggingface.co/Rostlab/prot_t5_xl_uniref50`.
.. [8] ProstT5 model on Hugging Face, available at: `https://huggingface.co/Rostlab/ProstT5`.
.. [9] CD-HIT tool, available at: `https://www.bioinformatics.org/cd-hit/`.
.. [10] TOPGO software, available at: `https://bioconductor.org/packages/release/bioc/html/topGO.html`.
