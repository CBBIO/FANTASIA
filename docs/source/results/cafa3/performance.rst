Performance
===========

System information
------------------

The CAFA3 experiment was executed on a local workstation with the following
specifications:

- **CPU:** Intel Core i5-12400F (12 cores, 6 performance cores × 2 threads, up to 5.6 GHz)
- **RAM:** 32 GB total
- **GPU:** NVIDIA GeForce RTX 3060 (12 GB GDDR6)
- **Operating system:** Ubuntu 24.04.2 LTS (Noble Numbat), kernel 6.14.0-29-generic
- **Storage:** NVMe SSDs


Workload summary
----------------

- **Sequences to annotate:** 3,328 (from ``cafa3_gt.fasta``)
- **Models enabled:** 5 (ESM, ESM3c, Ankh3-Large, Prot-T5, Prost-T5)
- **Layers used:** last layer only (final embedding representation)

Embedding stage
---------------

During this stage, embeddings were generated for all five enabled models across
the benchmark set of ~3,300 sequences. The process lasted about 40 minutes from
start to finish, ensuring that each sequence was represented once per model.

This completed the embedding phase, after which the resulting representations
were passed on to the lookup and annotation steps.

Lookup stage
------------

The lookup phase started once embeddings were available and lasted about
four minutes. A total of 16,640 similarity queries were enqueued
and processed in 35 homogeneous batches (batch size 516, grouped by model and
layer). During this stage, the system also produced the consolidated FASTA file
(``sequences.fasta``) containing 3,328 queries and 24,992 reference entries.
Lookup completed once all queues were emptied.

Post-processing
---------------

Immediately afterwards, a post-processing step was applied to the 3,328
benchmark accessions. This phase lasted around two minutes and
included filtering, identity calculations and score weighting. It marked the
final step of the annotation workflow prior to evaluation.
