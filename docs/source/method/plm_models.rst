PLM Models
==========

Scope
--------
FANTASIA integrates pretrained protein language models (PLMs) through a **model registry**
that standardizes tokenization, batching, device placement, and **hidden-layer extraction**.
Models are enabled and configured at runtime via YAML (see *Defaults aligned with the bundled
lookup table* below).

Supported Embedding Models
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 14 32 10 18 36

   * - **Name (registry key)**
     - **Model ID**
     - **Params (≈)**
     - **Architecture**
     - **Notes**
   * - ESM-2
     - ``facebook/esm2_t33_650M_UR50D``
     - 650M
     - Encoder (33L)
     - Large-scale encoder without MSAs; strong accuracy across structure/function tasks.
   * - ProtT5
     - ``Rostlab/prot_t5_xl_uniref50``
     - 1.2B
     - Encoder–Decoder
     - Trained on UniRef50; robust transfer for downstream structure/function tasks.
   * - ProstT5
     - ``Rostlab/ProstT5``
     - 1.2B
     - Multi-modal T5
     - Incorporates sequence+3Di states; improves contact/function representations.
   * - Ankh3-Large
     - ``ElnaggarLab/ankh3-large``
     - 620M
     - Encoder (T5-style)
     - Fast inference with solid semantic/structural signals.
   * - ESM3c
     - ``esmc_600m``
     - 600M
     - Encoder (36L)
     - New-generation encoder trained on broad protein corpora; high precision and speed.

Default method configuration for main LookUp table
---------------------------------------------------------------------------------------
The following configuration matches the distributed setup (model keys and layer indices) and the
runner’s expectations (``distance_metric`` at the top level):

.. code-block:: yaml

    embedding:
      device: cuda
      queue_batch_size: 100
      max_sequence_length: 0
      distance_metric: cosine

      models:
        ESM-2:
          enabled: true
          batch_size: 1
          layer_index: [0]
          distance_threshold: 0

        ESM3c:
          enabled: true
          batch_size: 1
          layer_index: [0]
          distance_threshold: 0

        Ankh3-Large:
          enabled: true
          batch_size: 1
          layer_index: [0]
          distance_threshold: 0

        ProtT5:
          enabled: true
          batch_size: 1
          layer_index: [0]
          distance_threshold: 0

        ProstT5:
          enabled: true
          batch_size: 1
          layer_index: [0]
          distance_threshold: 0


Configuration Notes
-------------------
- **Registry mapping**: keys under ``embedding.models`` (e.g., ``ESM-2``, ``ProtT5``) must match the
  registry/type names used by your environment so the embedder can resolve the correct
  model/tokenizer/module.
- **Hidden-layer selection**: all indices listed under ``layer_index`` are extracted per model; each
  layer is persisted independently in HDF5 and becomes available to lookup.
- **Distance metric**: set ``distance_metric`` at the **root** of the YAML; the lookup stage reads it
  from there (not from ``embedding``).
- **Batching & device**: ``batch_size`` (per model) and global ``device`` control throughput and
  memory pressure during embedding; tune to your hardware budget.

