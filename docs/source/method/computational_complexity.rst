.. _complexity_refined:

Computational Complexity Analysis
======================================================

1. Global Parameters
--------------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - **Symbol**
     - **Meaning**
   * - :math:`S`
     - Number of input sequences.
   * - :math:`M_s`
     - Number of enabled embedding models.
   * - :math:`L_s`
     - Number of selected layers (only affects storage).
   * - :math:`L_q`
     - Average sequence length.
   * - :math:`E_L`
     - Number of reference embeddings in the lookup table.
   * - :math:`C_{\text{embed}}`
     - Inference cost per sequence/model (depends on architecture and sequence length).
   * - :math:`C_{\text{dist}}`
     - Cost of computing the distance between two embeddings.
   * - :math:`p`
     - Effective parallelism factor (processors or GPU multiprocess).


2. Stage A — Embedding Generation
---------------------------------

.. math::

   \text{Cost A} \approx O\left(\frac{S \times M_s \times L_q \times C_{\text{embed}}}{p_{\text{embed}}}\right)

* A single embedding is generated for each sequence and model.
* :math:`L_s` **does not affect** compute cost — it only impacts I/O (layer dumps in HDF5).
* The cost scales **linearly** with:

  * number of sequences :math:`S`,
  * number of models :math:`M_s`,
  * average sequence length :math:`L_q`,
  * and proportionally to model size and architecture.

.. math::

   p_{\text{embed}} \text{ represents the effective speed-up from batching, GPU cores, VRAM and I/O throughput.}

.. note::

   HDF5 writing cost is negligible compared to inference.
   The real bottleneck is the forward pass.


3. Stage B — Lookup & Distance Computation
------------------------------------------

.. math::

   \text{Lookup cost} \approx O\left(\frac{E_q \times E_L \times L_s \times C_{\text{dist}}}{p_{\text{lookup}}}\right)

where:

.. math::

   E_q = S \times M_s

* For each **selected layer** of each **enabled model**, the lookup operation must be executed independently.
* This results in a multiplication over three dimensions:

  * number of query embeddings (:math:`E_q`),
  * number of reference embeddings (:math:`E_L`),
  * number of selected layers (:math:`L_s`).

* The lookup operation itself is implemented as a **vectorized matrix multiplication** between:

  .. math::

     Q \in \mathbb{R}^{(E_q \times L_s) \times d}
     \quad \text{and} \quad
     R \in \mathbb{R}^{E_L \times d}

  where :math:`d` is the embedding dimensionality.

  This produces:

  .. math::

     \text{Similarity matrix} \in \mathbb{R}^{(E_q \times L_s) \times E_L}

* Taxonomy or redundancy filters:

  * add a small **constant overhead**,
  * reduce the effective reference size :math:`E_L^{\text{eff}}`,
  * but do not change the asymptotic order.

If filters are applied:

.. math::

   E_L^{\text{eff}} = f(E_L) \quad \text{with } f(E_L) \leq E_L

and

.. math::

   \text{Lookup cost} \approx O\left(\frac{E_q \times E_L \times L_s \times C_{\text{dist}}}{p_{\text{lookup}}}\right)

where:

.. math::

   E_q = S \times M_s

.. math::

   p_{\text{lookup}} \text{ captures the effective GPU acceleration for the vectorized kernel.}

.. note::

   Since each selected layer multiplies the number of lookup passes,
   :math:`L_s` becomes a **first-class scaling factor** in lookup complexity.
   This is typically the dominant term when using multiple models and multiple layers per model.


4. Stage C — Post-processing
----------------------------

.. math::

   \text{Cost C} \approx O\left(E_q \times K \times C_{\text{collapse}}\right)

* :math:`K`: number of retained neighbors per query embedding.
* This stage **does not scale quadratically** — its cost depends on aggregation and annotation per neighbor.
* As :math:`K` increases, selection and downstream annotation become non-negligible.

.. note::

   Post-processing is linear in :math:`K` and can significantly contribute to runtime when :math:`K` is large.


5. Overall Complexity
---------------------

.. math::

   T_{\text{total}} \approx
   O\left(\frac{S \times M_s \times L_q \times C_{\text{embed}}}{p_{\text{embed}}}\right) +
   O\left(\frac{S \times M_s \times L_s \times E_L \times C_{\text{dist}}}{p_{\text{lookup}}}\right) +
   O\left(S \times M_s \times K \times C_{\text{collapse}}\right)

* If :math:`E_L` is **large**, the lookup term dominates.
* If :math:`S` is very large and :math:`E_L` moderate, Stage A can become significant.
* Stage C grows linearly with :math:`K` and may meaningfully contribute to total runtime at high neighbor counts.


6. Dominant Factors per Stage
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - **Stage**
     - **Dominant factors**
     - **Secondary influence**
     - **Comment**
   * - A — Embedding
     - :math:`S`, :math:`M_s`, :math:`L_q`
     - Write I/O (negligible)
     - Controlled by model/layer selection and batch size.
   * - B — Lookup
     - :math:`S`, :math:`M_s`, :math:`E_L`, :math:`L_s`
     - Filters (asymptotically neutral)
     - **Quadratic** scaling — main bottleneck.
   * - C — Post
     - :math:`S`, :math:`M_s`, :math:`K`
     - —
     - Linear and predictable.


7. Practical Observations
-------------------------

* Average sequence length impacts **only Stage A**.
* Number of layers affects the lookup stage linearly — each selected layer multiplies the number of lookup operations.
* Filters help reduce memory and absolute runtime, but do not change asymptotic complexity.
* Parallelism is critical:

  * Stage A: batching + VRAM.
  * Stage B: vectorized GPU kernels.

* High values of :math:`K` also increase the cost of top-K selection and annotation in Stage C.
* For large-scale experiments with millions of reference embeddings, and especially at large :math:`K`, **Stage B (lookup) together with top-K selection becomes the asymptotic bottleneck**.
