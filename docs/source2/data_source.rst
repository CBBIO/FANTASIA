Data Source
============

FANTASIA **requires** two different data: 

- The **input** data (query), which is the user's query data. 
- The **reference**  (or lookup table) data used in FANTASIA. This is generated in the **Protein Information System (PIS)** at CBBIO [1]_, a structured and publicly available information system designed to facilitate large-scale functional annotation and protein-related analyses. 


Three different **reference**  datasets are available in FANTASIA  containing UniProt protein entries and their corresponding Gene Ontology (GO) annotations:

- The **GOA2022 dataset**: Derived from the goPredSim project, it contains over 2 million annotations, including those inferred electronically (IEA). It is kept for benchmarking purposes [2]_.
- **GOA2024 dataset**: Contains UniProt entries with embeddings generated using ProtT5 and ESM2 models. It stores all UniProt entries and their computationally predicted annotations corresponding to GOA2024 [3]_.
- **GOA2025 dataset**: This is the **DEFAULT** lookup table used in FANTASIA. It includes **only** annotations supported by experimental evidence (e.g., EXP, IDA, IPI, IMP, IGI, IEP, TAS, IC), extracted directly from the UniProt API in April 2025. This ensures a high-confidence annotation set for downstream benchmarking and interpretation for the latest data [4]_.


The system to create **reference tables** is accessible at [1]_, and it is based on a **relational databases** framework since it offers several advantages:


- **Enhanced Data Traceability**: Metadata integration enables provenance tracking and reproducibility.
- **Relational Algebra Support**: Structured data allows advanced queries and relationship-based analyses.
- **Scalability & Expandability**: The system can integrate new data sources (e.g., structural, evolutionary, or experimental).
- **Cross-Referencing with Structural Data**: Enables combined sequence- and structure-based functional analyses.



.. figure:: _static/PIS.png
   :alt: FANTASIA Pipeline Overview
   :align: center
   :width: 80%


References
^^^^^^^^^^

.. [1] Protein Information System (PIS) GitHub Repository, available at: `GitHub <https://github.com/frapercan/protein_information_system>`_.
.. [2] Reference database for GOA2022, available at: `Zenodo GOA2022 <https://zenodo.org/records/15095845>`_.
.. [3] Reference database for GOA2024, available at: `Zenodo GOA2024 <https://zenodo.org/records/14864851>`_.
.. [4] Reference database for GOA2025, available at: `Zenodo GOA2025 <https://zenodo.org/records/15261639>`_.
