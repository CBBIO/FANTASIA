Computational Environment
==========================

Two complementary tools
-----------------------

- **PIS (Protein Information System)** — controls the **reference** side: it builds, versions,
  and packages the lookup table (embeddings + annotations + metadata) using a relational model.
- **FANTASIA** — exploits those references to run the **method** (embedding of queries,
  similarity search, and post-processing), with knobs to tune models, layers, and evaluation.


Current deployment reality
--------------------------
There is **no centrally hosted PIS** instance at this time. Each operator is expected to:

- **Deploy PostgreSQL (with pgvector)** locally or on the cluster.
- **Deploy RabbitMQ** locally or on the cluster.
- **Run FANTASIA’s initializer** pointing to the chosen PIS dump (reference dataset).


Typical operator workflow (two paths)
-------------------------------------

.. rubric:: Path A — Use an existing reference dump

#. Start **PostgreSQL (pgvector enabled)** locally or on the cluster.
#. Run **FANTASIA’s initializer** pointing to the chosen reference dump
   (e.g., *UniProt2025*, *CAFA3*, *CAFA5*). This step **loads the dump into PostgreSQL**
   and makes the reference available for lookup.
#. Run experiments and collect results.

.. rubric:: Path B — Build a new reference with PIS

#. Deploy **PIS** and **ingest** data via one of its supported modes:

   - Direct **UniProt** queries (using UniProt filters).
   - **FASTA + TSV** inputs (common in challenges/pipelines).
   - A predefined **accession list** produced elsewhere.
#. **Compute embeddings** for the target models/layers and attach **GO annotations** and provenance.
#. **Package/export** the reference as a **PIS dump**.

.. note::

   Both FANTASIA and PIS are available as **PyPI** packages. Container images and
   **HPC scripts** are provided to automate DB startup, dump loading via the initializer,
   and the subsequent pipeline run.

.. important::

   There is **no centrally hosted PIS** at present. Each operator provisions PostgreSQL
   and loads the dump locally.



