Data Sources
============

The data used in FANTASIA originates from the **Protein Information System (PIS)**, a structured and publicly available information system designed to facilitate large-scale functional annotation and protein-related analyses. The system is accessible at `PIS GitHub Repository <https://github.com/frapercan/protein_information_system>`_.

Our reference embedding set consists of a large-scale extraction of all UniProt protein sequences, including their associated **Gene Ontology (GO)** terms. This approach not only provides candidate annotations but also incorporates additional metadata, offering several advantages:

- **Enhanced Data Traceability**: By integrating metadata, we can track the provenance of annotations and ensure reproducibility.

- **Relational Algebra Support**: The structured nature of the database allows advanced queries and relationship-based operations.

- **Scalability & Expandability**: The system enables seamless integration with additional sources, such as structural data, experimental annotations, and evolutionary insights.

- **Optimization for Efficient Retrieval**: The use of PostgreSQL-based vector searches ensures that even large-scale proteomes can be processed efficiently.

- **Cross-Referencing with Structural Data**: By incorporating structural information, we facilitate comparative analyses that leverage both sequence-based and structure-based functional annotations.

The pipeline consists of two optional input and output filtering and formatting steps and three main steps: protein embeddings computation, embedding similarity with a functionally annotated embedding reference database, and GO terms transfer from closest reference sequences (Figure :ref:`fantasia_overview`).


.. figure:: images/PIS.png
   :alt: Protein Information System
   :align: center

   The Protein Information System (PIS) and its integration within FANTASIA.
