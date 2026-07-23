Supported models
================

The shipped configuration records the following repository and immutable
revision for each supported model. Every run copies this information to
``model_provenance.yaml``.

.. list-table::
   :header-rows: 1
   :widths: 14 29 35 12 10

   * - Key
     - Model repository
     - Recorded revision
     - Config layers
     - Dimension
   * - ``ESM`` (ESM-2)
     - ``facebook/esm2_t33_650M_UR50D``
     - ``08e4846e537177426273712802403f7ba8261b6c``
     - 0–33
     - 1280
   * - ``ESM3c``
     - ``EvolutionaryScale/esmc-600m-2024-12``
     - ``e4d83bc7e10fd55c92e598e545f4a76bf04a6e5c``
     - 0–35
     - 1152
   * - ``Ankh3-Large``
     - ``ElnaggarLab/ankh3-large``
     - ``2be091622e8a393f0ef21735070084123c874b6e``
     - 0–48
     - 1536
   * - ``Prot-T5``
     - ``Rostlab/prot_t5_xl_uniref50``
     - ``973be27c52ee6474de9c945952a8008aeb2a1a73``
     - 0–24
     - 1024
   * - ``Prost-T5``
     - ``Rostlab/ProstT5``
     - ``d7d097d5bf9a993ab8f68488b4681d6ca70db9e5``
     - 0–24
     - 1024

Index 0 is the final/output representation. The reference database must contain
the requested model/layer. CPU and CUDA loaders are available, but CPU runtime
can be impractical. Model-owned input limits can still cause failures when
FANTASIA truncation is disabled.

FANTASIA resolves each recorded Hugging Face revision to an immutable local
snapshot before model and tokenizer loading. ESM3c additionally verifies the
recorded serialization SHA-256. See
`Reproducibility <../explanation/reproducibility.rst>`_.
