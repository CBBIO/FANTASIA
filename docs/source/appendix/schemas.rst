Schemas
=======

This section describes the main database schema used by the Protein Information System (PIS).
It includes core entities, their attributes, and relationships.

.. contents::
   :local:
   :depth: 2

Accession
---------
Represents accession codes (e.g., UniProt identifiers) associated with proteins.

- **Primary Key**: ``code``
- **Fields**:

  - ``primary`` (boolean): whether this accession is the main one
  - ``tag`` (text): optional label
  - ``protein_id`` (FK → ``protein.id``)
  - ``created_at``, ``updated_at`` (timestamps)


GO Terms
--------
Controlled vocabulary of Gene Ontology (GO) terms.

- **Primary Key**: ``go_id``
- **Fields**:

  - ``category`` (BP, MF, CC)
  - ``description`` (text)

Protein
-------
Represents proteins and their metadata.

- **Primary Key**: ``id`` (text, e.g., UniProt ID)
- **Fields**:

  - ``sequence_id`` (FK → ``sequence.id``)
  - ``data_class``, ``molecule_type``
  - ``created_date``, ``sequence_update_date``, ``annotation_update_date``
  - ``description``, ``gene_name``, ``organism``, ``organelle``
  - ``taxonomy_id`` (text)
  - ``protein_existence`` (integer)
  - ``comments``, ``seqinfo``
  - ``disappeared`` (boolean)
  - ``created_at``, ``updated_at``

Protein–GO Term Annotation
--------------------------
Links proteins with GO terms and evidence codes.

- **Primary Key**: ``id``
- **Unique Constraint**: (``protein_id``, ``go_id``)
- **Fields**:

  - ``protein_id`` (FK → ``protein.id``)
  - ``go_id`` (FK → ``go_terms.go_id``)
  - ``evidence_code`` (text)

Sequence
--------
Stores raw sequences.

- **Primary Key**: ``id``
- **Fields**:

  - ``sequence`` (text, required)
  - ``sequence_hash`` (text, optional)

.. note::

   A single sequence may be linked to multiple proteins. This allows
   embeddings to be shared, and implies that a single embedding-level
   hit can expand into multiple protein-level results.


Sequence Embedding Type
-----------------------
Describes available embedding models.

- **Primary Key**: ``id``
- **Fields**:

  - ``name`` (unique, e.g., ProtT5, ESM2)
  - ``description``
  - ``task_name``
  - ``model_name``

Sequence Embeddings
-------------------
Stores embeddings per sequence, model, and layer.

- **Primary Key**: ``id``
- **Unique Constraint**: (``sequence_id``, ``embedding_type_id``, ``layer_index``)
- **Fields**:

  - ``sequence_id`` (FK → ``sequence.id``)
  - ``embedding_type_id`` (FK → ``sequence_embedding_type.id``)
  - ``layer_index`` (integer)
  - ``embedding`` (halfvec)
  - ``shape`` (integer[])
  - ``created_at``, ``updated_at``


Relationships
-------------
- One ``Protein`` → one ``Sequence``
- One ``Protein`` → many ``GO Terms`` (via ``Protein–GO Term Annotation``)
- One ``Sequence`` → many ``Embeddings`` (across types and layers)
- One ``Sequence`` → many ``Proteins`` (shared sequence reused by different proteins)


