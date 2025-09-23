Conclusions
===========

Our work leverages protein language models (PLMs) to explore functional annotation within FANTASIA.

- **Validity of the approach**
  The integration of PLMs allows the recovery of annotations with
  precision. The use of taxonomy and redundancy filters, together with
  additional metrics such as global and local identity and reliability index,
  strengthens the robustness of the method.

- **Advances in the reimplementation**
  The adoption of software engineering practices improves reproducibility and
  maintainability. The framework now supports multiple hidden states, the
  integration of newer models, and flexible filtering pipelines.

- **Limitations**
  We believe that amplifying the noise introduced by **IEA annotations** will
  not lead to reliable results in the future. For this reason, our method
  considers as *certain* only annotations with **experimental evidence**, which
  inevitably introduces a **bias toward well-studied model organisms**.

  However, by relying on annotations that we can be confident are real and
  individually validated by the scientific community, we avoid **erroneous or
  redundant transfers**. Looking ahead, we expect that the annotation databases
  will continue to grow, allowing us to keep updating our method as scientific
  knowledge in this area advances.


- **Future perspectives**
  Beyond benchmarking, FANTASIA is already being used to provide a first
  approximation of the functions of proteins that remain unannotated. This
  exploratory capacity opens the door to its integration into a wide variety of
  bioinformatics pipelines, where preliminary functional hints can guide
  downstream analyses.
  In parallel, extending the evaluation to upcoming challenges such as CAFA6
  will further strengthen the benchmarking and provide continuity with the
  international community efforts in protein function prediction.
