=========================
Benchmarking Use Case
=========================

Objective
---------

This use case provides a basic implementation for **benchmarking execution** in functional protein annotation. The goal is to generate annotations for proteins **NOT** present in the reference sets, allowing performance comparisons between different methods in downstream analyses.

An example of this approach is published in **NARGAB** (`DOI: 10.1093/nargab/lqae078 <https://doi.org/10.1093/nargab/lqae078>`_), where annotations were evaluated using various metrics.

The reference datasets consist of proteins annotated in the three **Gene Ontology (GO)** domains:

- **F**: Molecular Function
- **B**: Biological Process
- **C**: Cellular Component

The evidence codes considered follow the **CAFA** standards:

- **EXP, IDA, IPI, IMP, IGI, IEP, TAS, IC**

Step-by-Step Procedure
----------------------

1. **Extract a reference proteome**, e.g., **mouse (Mus musculus)**.
2. **Remove mouse sequences** from the reference dataset (both sequences and embeddings) to avoid model biases.
3. **Remove identical sequences** from the reference dataset (both sequences and embeddings) to minimize the impact of closely related species.
4. **Execute the analysis pipeline** and transfer annotations based on embedding similarity (**GOA2024**).
   - **GOA2022** is available for comparison with previously published analyses.

We use the default settings from **GoPredSim** for consistency with the original methods, though many parameters can be adjusted.

Input Data
----------

Input data must be **protein sequences in FASTA format**, concatenated into a single file.

Example of **FILENAME_test.fasta**:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: none

   >tr|A0A087WPB2|A0A087WPB2_MOUSE MyoD family inhibitor domain containing OS=Mus musculus OX=10090 GN=Mdfic PE=1 SV=1
   MSCAGEALAPGPAEQQCPVEAGGGRLGSPAHEACNEDNTEKDKRPATSGHTRCGLMRDQS
   ...
   >tr|A0A087WQA5|A0A087WQA5_MOUSE TAR DNA binding protein (Fragment) OS=Mus musculus OX=10090 GN=Tardbp PE=1 SV=1
   XDETDASSAVKVKRAVQKTSDLIVLGLPWKTTEQDLKDYFSTFGEVLMVQVKKDLKTGHS
   ...

**Note:** A validation step should be implemented to ensure that input files contain **proteins and not DNA**, avoiding execution errors in the pipeline.

Configuration Parameters
------------------------

Pipeline Configuration
^^^^^^^^^^^^^^^^^^^^^^

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
   length_filter: 5000000  # A high value means no filtering.

   # Redundancy filtering threshold (removes identical sequences).
   redundancy_filter: 1  # "0" (no filtering) | "1-0.5" (100%-50% redundancy removal)

**Parameter Justification:**

- **lookup_reference_tag:** Defines which reference database is used for annotation lookup. Allows switching between `GOA2022` and `GOA2024` to assess differences in methods.
- **limit_per_entry:** Determines how many similar proteins are considered for annotation transfer. `k=1` follows GoPredSim but can be adjusted.
- **length_filter:** Set to a high value to avoid sequence length filtering by default. It can be adjusted to remove abnormally long proteins if needed.
- **redundancy_filter:** Controls the removal of identical sequences to prevent biases in method comparisons.

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

**Parameter Justification:**

- **distance_metric:** Defines the distance function used to compare embeddings. Euclidean distance (`<->`) is the default, but cosine similarity (`<=>`) can be useful for normalized embeddings.
- **models:** The selected embedding models (`esm`, `prost_t5`, `prot_t5`) are enabled to extract protein representations.
- **distance_threshold:** This value determines the maximum allowed distance between query embeddings and reference entries. A lower threshold restricts matches to highly similar proteins, while `0` means no filtering.
- **batch_size:** Controls the number of sequences processed in parallel to optimize memory usage and computational efficiency.

Functional Analysis
^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   # Enable or disable file formatting for TOPGO downstream analyses
   topgo: true  # "true" (enabled) | "false" (disabled)

**Justification:**

- **topgo:** Enables generating files compatible with TOPGO, a tool used for functional enrichment analysis.

Results
-------

Two main output files are generated:

1. **FILENAME_test.csv** → Contains detailed information for each query protein.
2. **FILENAME_test.TOPGO.txt** → Contains annotations formatted for **TOPGO** software.

These results can be used to evaluate prediction accuracy and compare the performance of different methods.