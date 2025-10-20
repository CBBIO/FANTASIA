Abstract
========

Protein functional annotation is a central challenge in modern biology, yet many protein-coding
genes remain uncharacterized, particularly in non-model organisms. **FANTASIA**
(Functional ANnoTAtion based on embedding space SImilArity) addresses this gap by integrating
state-of-the-art protein language models for large-scale functional annotation.

In its first release (FANTASIA v1), the framework was applied to nearly 1,000 animal proteomes,
assigning functions to virtually all proteins, including up to 50% that had no annotation from
conventional homology-based methods. By enabling the discovery of novel gene functions,
FANTASIA advanced our understanding of molecular evolution and organismal biology.

This reimplementation has adopted software engineering practices to improve robustness and
reproducibility. In addition, it incorporates methodological advances common in current research,
such as taxonomy-based filtering, redundancy control, integration of more recent protein language
models, support for future model extensions, and the ability to exploit multiple hidden states
beyond the final embedding layer.
