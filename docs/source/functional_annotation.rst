.. _functional_annotation:

==========================================
Functional Annotation
==========================================

Objective
---------
This use case describes the **functional annotation process** in **FANTASIA**.
The goal is to predict **functional annotations for unknown sequences**, enabling their classification based on similarity to known protein functions.

FANTASIA leverages **embedding-based approaches** to transfer functional information from well-characterized proteins to unannotated sequences.
This method provides a reliable annotation strategy, especially for proteins with no clear homologs.

The annotation is performed using the three **Gene Ontology (GO)** domains:

- **F**: Molecular Function
- **B**: Biological Process
- **C**: Cellular Component

Annotations are assigned based on similarity to reference datasets following **CAFA** standards:

- **EXP, IDA, IPI, IMP, IGI, IEP, TAS, IC**

Functional Annotation Procedure
--------------------------------

1. **Input a set of unknown protein sequences**.
2. **Generate embeddings** for each sequence using **ESM, ProtT5, or other models**.
3. **Compare embeddings** against reference datasets with known functional annotations.
4. **Assign GO terms** to unknown sequences based on the closest matches.
5. **Export annotation results** for further analysis or integration into biological workflows.

Input Data
----------

The input must be **protein sequences in FASTA format**, concatenated into a single file.

Example of **FILENAME_query.fasta**:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

   >query1 Unknown protein sequence
   MVKFTASDLKQGERTSLP...
   >query2 Hypothetical protein
   MLFTGASDVKNQTWPAL...

**Note:** Ensure the input consists of amino acid sequences, not DNA.

Functional Annotation Configuration
-----------------------------------

Pipeline Configuration
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   # Path to the input FASTA file containing unknown protein sequences
   input: data_sample/FILENAME_query.fasta

   # Reference tag used for lookup operations.
   lookup_reference_tag: GOA2024  # Accepted values: "0" (no filtering) | "GOA2024" (excludes GOA2022)

   # Number of closest proteins to consider for annotation transfer.
   limit_per_entry: 5  # Default is 5, can be optimized.

   # Prefix for output file names.
   fantasia_prefix: FILENAME_query_annotated

Embedding Configuration
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   embedding:
     distance_metric: "<->"  # Options: "<=>" (cosine) | "<->" (Euclidean, default)
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

Functional Analysis
^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   # Enable or disable file formatting for TOPGO downstream analyses
   topgo: true  # Accepted values: "true" (enabled) | "false" (disabled)

Results
------------------

Two main output files are generated:

1. **FILENAME_query.csv** → Contains predicted annotations for each sequence.
2. **FILENAME_query.TOPGO.txt** → Contains annotations formatted for **TOPGO** software.

These results enable further downstream analysis, including enrichment studies and pathway predictions.
