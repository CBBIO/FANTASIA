Preparation
===========

The evaluation of FANTASIA on the CAFA3 benchmark is grounded in the official
datasets and methodology of the
`CAFA-evaluator <https://github.com/BioComputingUP/CAFA-evaluator>`_.
This framework was originally developed to provide a standardized protocol for
CAFA competitions, producing precision–recall curves, F-max scores and related
metrics. It remains the reference tool for reproducible evaluation.

Input dataset
-------------

The benchmark input is derived from the official CAFA3 ground truth:

- ``cafa3_gt.fasta`` — protein sequences that were unannotated at the time of
  the CAFA3 deadline and later received new annotations during the evaluation
  period.
- ``cafa3_gt.tsv`` — ground truth annotations collected post-deadline.

These files implement the “future knowledge” principle of CAFA, ensuring that
methods are scored against annotations unavailable at submission time.

Reference database in FANTASIA
------------------------------

In this experiment, FANTASIA was executed against an **updated UniProt-based
reference table** integrated into the Protein Information System (PIS). This
choice implies important differences compared to the original CAFA3 evaluation:

- The annotations that once served as *future knowledge* are now fully present
  in UniProt, and therefore available in the reference database.
- Other annotations may also have been corrected, merged or extended since the
  original CAFA3 period.
- As a consequence, the evaluation here is **retrospective**: FANTASIA predicts
  using knowledge that was inaccessible to CAFA3 participants, which may inflate
  coverage and alter precision.

Configuration of FANTASIA
-------------------------

In this experiment, FANTASIA did not recompute embeddings on the fly. Instead,
the embeddings were **calculated in a previous run** and reused here as input
for the lookup stage. These precomputed embeddings have been uploaded together
with the results to ensure reproducibility.

The main lookup settings were:

- **Distance metric**: cosine similarity.

- **Neighbors per entry**: ``k = 3``.

- **Redundancy filter**: disabled (``identity = 0``).

- **Additional metrics**:

  - Global identity (``id_g``)
  - Local identity (``id_l``)
  - Reliability index (``ri``)
- **Post-processing weights**:

  - Reliability index (max): 0.4
  - Global identity (max): 0.2
  - Local identity (max): 0.2
  - Count: 0.2
