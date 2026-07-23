Supported models
================

.. list-table::
   :header-rows: 1

   * - Key
     - Model identifier
     - Config layer indices
     - Dimension
   * - ``ESM`` (ESM-2)
     - ``facebook/esm2_t33_650M_UR50D``
     - 0–33
     - 1280
   * - ``ESM3c``
     - ``esmc_600m`` (repository ``EvolutionaryScale/esmc-600m-2024-12``)
     - 0–35
     - 1152
   * - ``Ankh3-Large``
     - ``ElnaggarLab/ankh3-large``
     - 0–48
     - 1536
   * - ``Prot-T5``
     - ``Rostlab/prot_t5_xl_uniref50``
     - 0–24
     - 1024
   * - ``Prost-T5``
     - ``Rostlab/ProstT5``
     - 0–24
     - 1024

Index 0 is the final/output representation. The reference database must contain
the requested model/layer. CPU and CUDA loaders are available, but CPU runtime
can be impractical. Model-owned input limits can still cause failures when
FANTASIA truncation is disabled.

A model ID does not fix an immutable revision. Record the resolved commit,
serialization, checksums, packages, and run config; see
:doc:`/explanation/reproducibility`.
