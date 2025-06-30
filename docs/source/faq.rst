Frequently Asked Questions (FAQ)
================================

How can I add other new embedding models?
-------------------------------------------------

FANTASIA is modular by design. To integrate a new protein language model such as Ankh:

1. Declare the model in the `constants.yaml` file under `sequence_embedding_types`.
2. Implement the corresponding Python module in:
   `protein_information_system/operation/embedding/proccess/sequence/new_model.py`
3. Register the model in your runtime configuration (`config.yaml`) using `embedding.types`.

📘 Full guide: `Adding Embedding Models <https://protein-information-system.readthedocs.io/en/latest/adding_embedding_models.html>`_

How is the functional lookup table generated and populated?
-----------------------------------------------------------

FANTASIA supports several pathways to build the functional lookup table, including:

- Automatic accession fetching from UniProt using search criteria.
- Manual import from CSV lists.
- Local annotation processing via custom TOPGO files.

Each mode is fully configurable.

📘 Full guide: `Lookup Table Generation <https://protein-information-system.readthedocs.io/en/latest/lookup_table_generation.html>`_
