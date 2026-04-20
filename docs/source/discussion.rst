Discussion
==========

.. contents::
   :local:
   :depth: 2

Contributors
------------

Francisco Miguel Pérez Canales
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FANTASIA is currently being evaluated in both model and non-model organisms
using semantic similarity procedures. By analyzing the recovery of annotations
after applying taxonomic and redundancy filters, we can assess the potential of
each protein language model (PLM) beyond the naïve evaluation performed on the
CAFA3 benchmark.

For model organisms, taxonomic and redundancy filters are systematically applied
to ensure that predictions are not trivially recovered from closely related
entries. In contrast, analyses on non-model organisms follow a different
strategy, where the focus lies on exploring novel functions without applying
such constraints, in order to maximize discovery potential.

To follow the trace of this research, it is important to stay up to date with
the recent joint publications from the **Metazoa Phylogenomics Lab, Institut de
Biologia Evolutiva (IBE-CSIC/UPF)** and the **Computational Biology and
Bioinformatics Group (CABD)**. These collaborations highlight advances in the
application of PLMs.

The inclusion of multiple hidden layers is expected to yield better overall
performance and provide deeper insights into how PLMs capture evolutive and functional
signals. This direction opens the door to more interpretable and robust
annotation strategies.

Modeling Hierarchical Levels of Complexity and the Tokenization Question
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In many current approaches, function is associated with the protein as the basic
unit. This is overly coarse: proteins consist of chains, which in turn comprise
domains with specific roles. Each level contributes its own semantics and
biological complexity. Ignoring this hierarchy is akin to interpreting a text
by analyzing only the entire book, without attending to chapters, paragraphs,
or sentences.

The analogy with NLP and LLMs is instructive. In natural language processing,
models evolved from treating whole words as atomic units to decomposing text
into more flexible and expressive tokens capable of capturing subwords,
characters, and even implicit semantic units. This shift enabled robust handling
of multiple languages, morphological variation, and contextual meaning.

Protein language modeling is still early in this evolution. The near-universal
assumption that a single amino acid corresponds to a single token seems natural
but may constrain the model’s ability to capture biological semantics. Just as
an isolated character rarely conveys meaning in a sentence, an isolated amino
acid often lacks clear functional value outside its structural and dynamical
context.

This raises a central question: **what tokenization alternatives can enrich PLMs?**

Bit-Level Encoding
""""""""""""""""""

Recent proposals in NLP suggest discarding traditional tokenization altogether
and encoding inputs directly at the bit level. This removes arbitrary
segmentation decisions and allows the model to learn hierarchical
representations from the rawest possible signal. In proteins, an analogous
approach would feed digitized sequence (and potentially structural) information
directly to the model, allowing unsupervised discovery of meaningful units.

A recent and directly relevant example in NLP is **H-Net**
(*Dynamic Chunking for End-to-End Hierarchical Sequence Modeling*),
which eliminates the tokenizer by operating on **raw bytes** and learns a
**dynamic hierarchical segmentation** end-to-end [HNet2025]_.

Block-Based Amino Acid Tokenization
"""""""""""""""""""""""""""""""""""

An intermediate path is tokenization via amino-acid **n-grams** or recurrent
**structural/functional motifs**. For example, grouping triplets or pentamers
that tend to form helices, sheets, or catalytic/interaction motifs. This may
better capture the mesoscopic level of protein semantics—not merely the “character”
(amino acid) but the local patterns that shape folding and function. Importantly,
standard single–amino acid tokens can be retained in parallel, yielding a
**multiscale, mixed vocabulary**.

Aggregation Beyond Mean-Pooling and Batching Considerations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A critical limitation in current PLM-based pipelines is the widespread reliance
on mean-pooling to consolidate per-residue representations into a single
sequence-level embedding. While computationally convenient, uniform averaging
tends to collapse the hierarchical and contextual structure of proteins,
diminishing signal from residues, motifs, or domains that are functionally
decisive. This shortcoming persists irrespective of whether pooling is applied
over 2D maps, higher-dimensional stacks across layers, or combinations of
layers and models.

To address this, future designs should consider aggregation mechanisms that
retain biological salience:

- **Hierarchical attention** that learns to weight residues, motifs, domains,
  and chains according to their functional relevance.
- **Adaptive pooling** strategies that modulate aggregation in response to
  salient local patterns (e.g., catalytic motifs or interface regions),
  rather than enforcing uniform contributions.
- **Multi-scale representations** that preserve parallel embeddings at multiple
  biological levels (residue → block/motif → domain → chain → protein),
  enabling downstream tasks to query the appropriate level of granularity
  instead of forcing an early collapse.
- **Graph neural networks (GNNs)** that explicitly model proteins as graphs
  (residues as nodes, contacts/interactions as edges), enabling aggregation
  schemes that respect structural connectivity and capture higher-order
  relationships beyond linear sequence context.


In parallel, we have observed practical challenges related to batching and its
interaction with attention mechanisms. Early experiments with larger batch sizes
produced unstable behavior, partly due to the propagation of attention maps not
preserving the original sequence length at the output stage. After correcting
this propagation to restore the native sequence dimensionality, we have found
no evidence of degraded encodings. Nevertheless, given the immaturity of the
field and the modest computational gains in our configuration, we currently
favor a **batch size of 1** to maximize stability and traceability of the
embeddings. This choice is pragmatic rather than prescriptive, and should be
revisited

Jane Doe
~~~~~~~~
[Placeholder for a new author. Brief context, main points, references.]

John Smith
~~~~~~~~~~
[Placeholder for a new author. Brief context, main points, references.]

Open-Source and Contributions
-----------------------------

This project remains entirely open-source, and contributions are welcome from
both specialists in functional annotation and technicians willing to support the
maintenance and improvement of the framework.

How to Contribute
~~~~~~~~~~~~~~~~~

- Add a new subsection under **Contributors** with your name as the title.
- Keep the tone technical and focused on discussion/interpretation.
- Cite datasets, code, and figures as needed.


References
----------

.. [HNet2025] Sukjun Hwang, Brandon Wang, and Albert Gu (2025).
   *Dynamic Chunking for End-to-End Hierarchical Sequence Modeling*.
   arXiv:2507.07955.
