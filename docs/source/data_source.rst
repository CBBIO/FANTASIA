.. _data_source:


Data Source
============

The data used in FANTASIA originates from the **Protein Information System (PIS)**, a structured and publicly available information system designed to facilitate large-scale functional annotation and protein-related analyses. The system is accessible at [PISGitHub]_.

Our reference embedding set consists of a large-scale extraction of all UniProt protein sequences, including their associated **Gene Ontology (GO)** terms. This approach not only provides candidate annotations but also incorporates additional metadata, offering several advantages:

- **Enhanced Data Traceability**: By integrating metadata, we can track the provenance of annotations and ensure reproducibility.
- **Relational Algebra Support**: The structured nature of the database allows advanced queries and relationship-based operations.
- **Scalability & Expandability**: The system enables seamless integration with additional sources, such as structural data, experimental annotations, and evolutionary insights.
- **Optimization for Efficient Retrieval**: The use of PostgreSQL-based vector searches ensures that even large-scale proteomes can be processed efficiently.
- **Cross-Referencing with Structural Data**: By incorporating structural information, we facilitate comparative analyses that leverage both sequence-based and structure-based functional annotations.

.. figure:: _static/PIS.png
   :alt: FANTASIA Pipeline Overview
   :align: center
   :width: 80%

The data used in FANTASIA is publicly available on Zenodo [ZenodoRef]_. The pipeline is designed to allow future updates to the dataset without disrupting its functionality. Researchers can access the database using the provided credentials (username and password) and perform SQL queries to inspect its contents. This ensures transparency and facilitates reproducibility in large-scale functional annotation studies.

References
^^^^^^^^^^

.. [PISGitHub] Protein Information System (PIS) GitHub Repository, available at: `GitHub <https://github.com/frapercan/protein_information_system>`_.

.. [ZenodoRef] Reference database for FANTASIA, available at: `Zenodo <https://zenodo.org/records/14864851>`_.