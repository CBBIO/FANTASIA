Introduction
============

The challenge of functional annotation
--------------------------------------

Understanding protein function is fundamental to biology, yet a large fraction of the protein-coding
universe remains uncharacterized. This "dark proteome" is particularly pronounced in *non-model
organisms*, where experimental evidence is scarce and sequence-homology approaches often fail to
provide reliable annotations. Even in well-studied species, many proteins lack meaningful functional
information, limiting our ability to interpret genomes, analyze evolutionary processes, and connect
genotype to phenotype.

Beyond homology: the role of protein language models
----------------------------------------------------

Traditional annotation pipelines rely on homology transfer, which is powerful but inherently limited
by sequence divergence and evolutionary bias. Recent advances in **protein language models (PLMs)**
offer a complementary strategy: embeddings that capture structural and functional signals extending
beyond detectable homology. Embedding-based similarity enables zero-shot annotation transfer and
greater generalizability across taxa, addressing key challenges posed by the dark proteome.

From FANTASIA v1 to v4
-----------------------

The first release of **FANTASIA** demonstrated how PLMs can illuminate hidden biology by assigning
functions to millions of genes across the animal tree of life. However, that version was primarily
tailored for large-scale annotation, with limitations in deployment, scalability, and model
flexibility.

**FANTASIA** is a modular reimplementation that overcomes these issues by:

- Incorporating a **reference system built on a vector database** (see `Protein Information System`_),
  which ensures both *traceability* of the data used (embeddings, GO annotations, reference sets) and
  *flexibility* to update or adapt them as new models and evidence emerge.
- Supporting **multiple PLMs**, with an architecture that is easily extensible to incorporate new
  models as they become available.
- Allowing users to **select hidden layers** from the underlying PLMs, enabling fine-grained control
  over the embeddings used for annotation.
- Enabling **flexible benchmarking**, including compatibility with CAFA standards.
- Providing a streamlined and extensible **command-line interface**.

Validation and scope
--------------------

In this documentation we describe FANTASIA v4, its architecture and applications. We highlight two
complementary use cases:

- **Proteome-wide annotation**, extending functional coverage to proteins overlooked by homology-based
  tools.
- **Evaluation in CAFA3**, ensuring rigorous validation of accuracy and
  generalizability under community standards.


.. _Protein Information System: https://github.com/CBBIO/protein-information-system